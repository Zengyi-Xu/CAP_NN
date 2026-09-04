"""M8190A arbitrary waveform generator control (pyvisa).

Simplified version covering the core functions of AWG_transmit / download_M8190A in the original MATLAB:
- Connect to M8190A via TCPIP socket (or VISA)
- Set sample rate, output amplitude, and DC output
- Download waveform to a segment and play
"""
import numpy as np
import pyvisa
from pathlib import Path
from typing import Optional, Tuple

import config


class M8190AController:
    """M8190A AWG controller."""

    def __init__(self,
                 visa_addr: Optional[str] = None,
                 sample_rate: float = config.AWG_SAMPLE_RATE,
                 vpp: float = config.AWG_VPP,
                 output_route: str = config.AWG_OUTPUT_ROUTE,
                 timeout_ms: int = 30000):
        """
        Args:
            visa_addr: VISA resource string, e.g. "TCPIP0::192.168.1.10::5025::SOCKET"
            sample_rate: AWG sample rate (Hz)
            vpp: output amplitude (V)
            output_route: output route, optional "DC" / "AC" / "DAC"
                - "DC": DC-coupled amplified output (default, common for baseband/DMT)
                - "AC": AC-coupled amplified output (DC-blocked, common for RF/IF)
                - "DAC": direct DAC output (unamplified, smallest amplitude)
            timeout_ms: communication timeout

        Defaults to connecting to M8190A via TCPIP port 5025 (consistent with original MATLAB).
        """
        self.visa_addr = visa_addr or config.M8190A_VISA_ADDR
        self.sample_rate = sample_rate
        self.vpp = vpp
        self.output_route = output_route.upper()
        self.rm = pyvisa.ResourceManager()
        self.inst: Optional[pyvisa.Resource] = None
        self.timeout_ms = timeout_ms

    def connect(self) -> "M8190AController":
        """Establish connection and verify device."""
        print(f"Connecting to M8190A at {self.visa_addr} ...")
        self.inst = self.rm.open_resource(self.visa_addr)
        self.inst.timeout = self.timeout_ms
        self.inst.write_termination = "\n"
        self.inst.read_termination = "\n"

        idn = self.query("*IDN?")
        print(f"  *IDN = {idn}")
        pon = self.query(":TEST:PON?")
        print(f"  :TEST:PON? = {pon}")
        return self

    def close(self) -> None:
        if self.inst is not None:
            try:
                self.inst.close()
            except Exception as e:
                print(f"Warning during close: {e}")
            self.inst = None
        self.rm.close()

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(self, cmd: str) -> None:
        if self.inst is None:
            raise RuntimeError("AWG not connected")
        self.inst.write(cmd)

    def query(self, cmd: str) -> str:
        if self.inst is None:
            raise RuntimeError("AWG not connected")
        return self.inst.query(cmd).strip()

    def reset(self) -> None:
        self.write("*RST")
        self.query("*OPC?")

    def configure(self,
                  sample_rate: Optional[float] = None,
                  vpp: Optional[float] = None,
                  channels: Tuple[int, ...] = (1, 2),
                  output_route: Optional[str] = None) -> None:
        """Configure basic AWG parameters.

        Args:
            sample_rate: sample rate (Hz)
            vpp: output amplitude (V)
            channels: tuple of channels to configure
            output_route: output route, overrides constructor setting; optional "DC" / "AC" / "DAC"
        """
        sample_rate = sample_rate or self.sample_rate
        vpp = vpp or self.vpp
        route = (output_route or self.output_route).upper()

        if route not in ("DC", "AC", "DAC"):
            raise ValueError(f"Unsupported M8190A output route: {route}. Use DC/AC/DAC.")

        # Reference clock
        self.write(":ROSC:FREQ 1e7")
        self.write(":ROSC:SOUR EXT")

        for ch in channels:
            # Sample rate
            self.write(f":FREQ:RAST {sample_rate:.15g}")
            # 12-bit wideband mode
            self.write(f":TRACe{ch}:DWIDth WSP")
            # Output route and corresponding amplitude command
            # DC/AC are amplified outputs, DAC is direct DAC output
            self.write(f":OUTP{ch}:ROUT {route}")
            self.write(f":{route}{ch}:VOLT:AMPL {vpp:.15g}")
            if route in ("DC", "AC"):
                self.write(f":DC{ch}:FORM NRZ")
            self.write(f":OUTP{ch}:NORM ON")
            self.write(f":OUTP{ch}:COMP ON")
            self.write(f":TRAC{ch}:SEL 1")

        self.query("*OPC?")
        print(f"AWG configured: fs={sample_rate/1e9:.2f} GHz, Vpp={vpp} V, route={route}")

    def _scale_to_int16(self, data: np.ndarray) -> np.ndarray:
        """Scale a floating-point waveform in [-1, 1] to the M8190A 12-bit DAC (int16, shifted left by 4 bits)."""
        data = np.asarray(data).ravel()
        if np.max(np.abs(data)) > 1.0:
            data = data / np.max(np.abs(data))
        return np.int16(np.round(8191 * data) * 4)

    def download_waveform(self,
                          data: np.ndarray,
                          channel: int = 1,
                          segment: int = 1,
                          run: bool = True) -> None:
        """Download waveform to the specified channel and segment, and optionally play immediately.

        Args:
            data: real waveform, recommended range [-1, 1]
            channel: channel 1 or 2
            segment: segment number
            run: whether to play immediately
        """
        if self.inst is None:
            raise RuntimeError("AWG not connected")

        data = self._scale_to_int16(data)
        segm_len = len(data)

        # Delete old segment and define new segment
        # M8190A ABORt / INIT:IMM take the channel number as a parameter, not appended to the command name
        self.write(f":ABORt {channel}")
        self.write(f":TRACe{channel}:DELete {segment}")
        self.write(f":TRACe{channel}:DEFine {segment},{segm_len}")

        # Download in chunks (Keysight recommends no more than 523200 int16 values per transfer)
        chunk = 523200
        offset = 0
        while offset < segm_len:
            block = data[offset:offset + chunk]
            cmd = f":TRACe{channel}:DATA {segment},{offset},"
            self.inst.write_binary_values(cmd, block, datatype="h",
                                          is_big_endian=False,
                                          header_fmt="ieee",
                                          termination=None,
                                          final_termination="\n")
            offset += chunk

        self.query("*OPC?")

        # Select segment and turn on output
        self.write(f":TRACe{channel}:SELect {segment}")
        self.write(f":FUNCtion{channel}:MODE ARBitrary")
        self.write(f":OUTPut{channel}:STATe ON")

        if run:
            # INIT:IMM also takes the channel number as a parameter, not appended to the command name
            self.write(f":INIT:IMM {channel}")
            self.query("*OPC?")
            print(f"Channel {channel} running segment {segment} ({segm_len} samples)")
        else:
            print(f"Waveform downloaded to channel {channel} segment {segment}")

    def download_iq(self,
                    iqdata: np.ndarray,
                    channel_i: int = 1,
                    channel_q: int = 2,
                    segment: int = 1,
                    run: bool = True) -> None:
        """Download real and imaginary parts of a complex waveform to two channels."""
        self.download_waveform(iqdata.real, channel=channel_i, segment=segment, run=False)
        self.download_waveform(iqdata.imag, channel=channel_q, segment=segment, run=False)
        if run:
            self.write(f":INIT:IMM {channel_i}")
            self.write(f":INIT:IMM {channel_q}")
            self.query("*OPC?")

    def stop(self, channels: Tuple[int, ...] = (1, 2)) -> None:
        for ch in channels:
            self.write(f":ABORt {ch}")

    def send_preset(self) -> None:
        """Send *RST and wait for completion."""
        self.write("*RST")
        self.query("*OPC?")


def quick_download_to_awg(data: np.ndarray,
                          sample_rate: float = config.AWG_SAMPLE_RATE,
                          vpp: float = config.AWG_VPP,
                          visa_addr: Optional[str] = None,
                          output_route: str = config.AWG_OUTPUT_ROUTE,
                          channel: int = 1) -> None:
    """Convenience function: connect-configure-download-play-close.

    Args:
        data: real waveform
        sample_rate: sample rate (Hz)
        vpp: output amplitude (V)
        visa_addr: VISA address
        output_route: output route "DC" / "AC" / "DAC"
        channel: channel 1 or 2
    """
    with M8190AController(visa_addr=visa_addr,
                          sample_rate=sample_rate,
                          vpp=vpp,
                          output_route=output_route) as awg:
        awg.configure(channels=(channel,))
        awg.download_waveform(data, channel=channel)
