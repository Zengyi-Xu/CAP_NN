"""M8190A 任意波形发生器控制 (pyvisa).

简化版，覆盖原始 MATLAB 中 AWG_transmit / download_M8190A 的核心功能：
- 通过 TCPIP socket (或 VISA) 连接 M8190A
- 设置采样率、输出幅度、DC 输出
- 下载波形到 segment 并播放
"""
import numpy as np
import pyvisa
from pathlib import Path
from typing import Optional, Tuple

import config


class M8190AController:
    """M8190A AWG 控制器."""

    def __init__(self,
                 visa_addr: Optional[str] = None,
                 sample_rate: float = config.AWG_SAMPLE_RATE,
                 vpp: float = config.AWG_VPP,
                 output_route: str = config.AWG_OUTPUT_ROUTE,
                 timeout_ms: int = 30000):
        """
        Args:
            visa_addr: VISA 资源字符串，如 "TCPIP0::192.168.1.10::5025::SOCKET"
            sample_rate: AWG 采样率 (Hz)
            vpp: 输出幅度 (V)
            output_route: 输出路径，可选 "DC" / "AC" / "DAC"
                - "DC": DC 耦合放大输出（默认，基带/DMT 常用）
                - "AC": AC 耦合放大输出（隔直，射频/IF 常用）
                - "DAC": 直接 DAC 输出（未经放大，幅度最小）
            timeout_ms: 通信超时
        """
        self.visa_addr = visa_addr or config.M8190A_VISA_ADDR
        self.sample_rate = sample_rate
        self.vpp = vpp
        self.output_route = output_route.upper()
        self.rm = pyvisa.ResourceManager()
        self.inst: Optional[pyvisa.Resource] = None
        self.timeout_ms = timeout_ms

    def connect(self) -> "M8190AController":
        """建立连接并验证设备."""
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
        """配置 AWG 基本参数.

        Args:
            sample_rate: 采样率 (Hz)
            vpp: 输出幅度 (V)
            channels: 要配置的通道元组
            output_route: 输出路径，覆盖构造时的设置；可选 "DC" / "AC" / "DAC"
        """
        sample_rate = sample_rate or self.sample_rate
        vpp = vpp or self.vpp
        route = (output_route or self.output_route).upper()

        if route not in ("DC", "AC", "DAC"):
            raise ValueError(f"Unsupported M8190A output route: {route}. Use DC/AC/DAC.")

        # 参考时钟
        self.write(":ROSC:FREQ 1e7")
        self.write(":ROSC:SOUR EXT")

        for ch in channels:
            # 采样率
            self.write(f":FREQ:RAST {sample_rate:.15g}")
            # 12 bit 宽带模式
            self.write(f":TRACe{ch}:DWIDth WSP")
            # 输出路径与对应幅度命令
            # DC/AC 为放大输出，DAC 为直接 DAC 输出
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
        """把 [-1, 1] 的浮点波形缩放到 M8190A 12-bit DAC (int16, 左移 4bit)."""
        data = np.asarray(data).ravel()
        if np.max(np.abs(data)) > 1.0:
            data = data / np.max(np.abs(data))
        return np.int16(np.round(8191 * data) * 4)

    def download_waveform(self,
                          data: np.ndarray,
                          channel: int = 1,
                          segment: int = 1,
                          run: bool = True) -> None:
        """下载波形到指定通道与 segment，并可选立即播放.

        Args:
            data: 实数波形，范围建议 [-1, 1]
            channel: 通道 1 或 2
            segment: segment 编号
            run: 是否立即播放
        """
        if self.inst is None:
            raise RuntimeError("AWG not connected")

        data = self._scale_to_int16(data)
        segm_len = len(data)

        # 删除旧 segment 并定义新 segment
        self.write(f":ABORt{channel}")
        self.write(f":TRACe{channel}:DELete {segment}")
        self.write(f":TRACe{channel}:DEFine {segment},{segm_len}")

        # 分块下载 (Keysight 推荐单次不超过 523200 个 int16)
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

        # 选择 segment 并打开输出
        self.write(f":TRAC{channel}:SEL {segment}")
        self.write(f":FUNCtion{channel}:MODE ARBitrary")
        self.write(f":OUTPut{channel} ON")

        if run:
            self.write(f":INIT:IMM{channel}")
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
        """把复数波形的实部和虚部分别下载到两个通道."""
        self.download_waveform(iqdata.real, channel=channel_i, segment=segment, run=False)
        self.download_waveform(iqdata.imag, channel=channel_q, segment=segment, run=False)
        if run:
            self.write(f":INIT:IMM{channel_i}")
            self.write(f":INIT:IMM{channel_q}")
            self.query("*OPC?")

    def stop(self, channels: Tuple[int, ...] = (1, 2)) -> None:
        for ch in channels:
            self.write(f":ABORt{ch}")

    def send_preset(self) -> None:
        """发送 *RST 并等待完成."""
        self.write("*RST")
        self.query("*OPC?")


def quick_download_to_awg(data: np.ndarray,
                          sample_rate: float = config.AWG_SAMPLE_RATE,
                          vpp: float = config.AWG_VPP,
                          visa_addr: Optional[str] = None,
                          output_route: str = config.AWG_OUTPUT_ROUTE,
                          channel: int = 1) -> None:
    """便捷函数：连接-配置-下载-播放-关闭.

    Args:
        data: 实数波形
        sample_rate: 采样率 (Hz)
        vpp: 输出幅度 (V)
        visa_addr: VISA 地址
        output_route: 输出路径 "DC" / "AC" / "DAC"
        channel: 通道 1 或 2
    """
    with M8190AController(visa_addr=visa_addr,
                          sample_rate=sample_rate,
                          vpp=vpp,
                          output_route=output_route) as awg:
        awg.configure(channels=(channel,))
        awg.download_waveform(data, channel=channel)
