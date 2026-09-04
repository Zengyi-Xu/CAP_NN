# -*- coding: utf-8 -*-
"""Keithley 2400 SourceMeter RS-232/USB control wrapper.

This module ports the Keithley 2400 driver from the IVLab project into the
DMT experiment platform. It talks to the instrument through a serial (COM)
port, which can be either a physical RS-232 port or a USB-to-RS232 adapter
(FTDI / Prolific / CH340 / CP210x, etc.).

Typical usage:

    from keithley2400_controller import Keithley2400, list_com_ports

    k = Keithley2400(port="COM3", baudrate=9600)
    k.connect()
    k.set_source_mode("voltage")      # voltage source
    k.set_compliance(0.1)             # 100 mA current limit
    k.set_output_level(1.0)           # 1 V
    k.output_on()
    print(k.measure())
    k.output_off()
    k.disconnect()
"""
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

try:
    import serial
    import serial.tools.list_ports
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "pyserial is required to control the Keithley 2400. "
        "Install it with: pip install pyserial>=3.5"
    ) from exc


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class K2400Error(Exception):
    """Base exception for Keithley 2400 operations."""
    pass


class K2400ConnectionError(K2400Error):
    """Raised when the serial connection cannot be established."""
    pass


class K2400CommandError(K2400Error):
    """Raised when a command returns an unexpected response."""
    pass


class K2400ConfigError(K2400Error):
    """Raised when an invalid configuration value is supplied."""
    pass


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logger(name: str = "k2400", level: int = logging.DEBUG) -> logging.Logger:
    """Configure a logger writing to the project log directory."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File (inside DMT_PY_NN/log)
    log_dir = Path(__file__).resolve().parent / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "keithley2400.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Port scanner
# ---------------------------------------------------------------------------
def list_com_ports() -> List[Dict]:
    """Return a list of available COM ports with metadata."""
    ports = []
    for p in serial.tools.list_ports.comports():
        ports.append({
            "port": p.device,
            "description": p.description,
            "hwid": p.hwid,
            "vid": p.vid,
            "pid": p.pid,
        })
    return ports


def find_instrument_ports(keywords: Optional[List[str]] = None) -> List[Dict]:
    """Find COM ports whose description matches common USB-serial adapter names."""
    if keywords is None:
        keywords = ["USB Serial", "FTDI", "Prolific", "CH340", "CP210", "Keithley"]
    all_ports = list_com_ports()
    matches = []
    for p in all_ports:
        desc = p["description"].upper()
        if any(kw.upper() in desc for kw in keywords):
            matches.append(p)
    return matches


# ---------------------------------------------------------------------------
# Instrument driver
# ---------------------------------------------------------------------------
class Keithley2400:
    """Keithley 2400 SourceMeter controlled over RS-232 or USB-to-RS232."""

    def __init__(self, port: str = "COM1", baudrate: int = 9600,
                 timeout: float = 5.0, logger: Optional[logging.Logger] = None):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self.connected = False
        self._source_mode = "voltage"   # "voltage" or "current"
        self._measure_func = "current"  # "current" or "voltage"
        self._debug = False
        self.logger = logger or _setup_logger("k2400")

    # --- Connection ---------------------------------------------------------
    def connect(self, retries: int = 3) -> bool:
        """Open the serial port and initialize the instrument."""
        for attempt in range(1, retries + 1):
            try:
                self.logger.info(f"[K2400] 尝试连接 {self.port} (第{attempt}次)")
                self.ser = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=self.timeout,
                    xonxoff=False,
                    rtscts=False,
                )
                time.sleep(0.5)
                self.connected = True
                self._post_connect()
                self.logger.info(f"[K2400] 连接成功")
                return True
            except Exception as e:
                self.logger.warning(f"[K2400] 连接失败: {e}")
                time.sleep(1)
        raise K2400ConnectionError(
            f"连接 {self.port} 失败，已重试{retries}次"
        )

    def disconnect(self):
        """Safely turn off output and close the serial port."""
        if self.connected:
            try:
                self.output_off()
            except Exception as e:
                self.logger.warning(f"[K2400] 断开前关闭输出失败: {e}")
            finally:
                if self.ser and self.ser.is_open:
                    self.ser.close()
                self.connected = False
                self.logger.info(f"[K2400] 已断开")

    def _post_connect(self):
        """Clear buffers and query instrument identity."""
        if self.ser:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        try:
            idn = self.idn()
            self.logger.info(f"[K2400] IDN: {idn}")
        except Exception:
            pass
        # Default to voltage source, measure current
        self.set_source_mode("voltage")
        self.set_measure_function("current")
        self.write(":SYST:RSEN OFF")          # 2-wire sensing
        self.write(":FORM:ELEM VOLT,CURR,RES,TIME,STAT")

    # --- Low-level IO -------------------------------------------------------
    def write(self, cmd: str):
        """Send an SCPI command."""
        if not self.connected or self.ser is None:
            raise K2400ConnectionError("未连接仪器")
        full_cmd = cmd + "\n"
        self.ser.write(full_cmd.encode("ascii"))
        if self._debug:
            self.logger.debug(f"[K2400] SEND: {cmd}")
        self.ser.flush()

    def read(self, timeout: Optional[float] = None) -> str:
        """Read one line from the instrument."""
        if not self.connected or self.ser is None:
            raise K2400ConnectionError("未连接仪器")
        old_timeout = self.ser.timeout
        if timeout is not None:
            self.ser.timeout = timeout
        try:
            raw = self.ser.readline()
            resp = raw.decode("ascii", errors="replace").strip()
            if self._debug:
                self.logger.debug(f"[K2400] RECV: {resp}")
            return resp
        finally:
            if timeout is not None:
                self.ser.timeout = old_timeout

    def query(self, cmd: str, timeout: Optional[float] = None) -> str:
        """Send a command and return the response."""
        self.write(cmd)
        time.sleep(0.05)
        return self.read(timeout=timeout)

    def reset(self):
        """Reset the instrument to factory defaults."""
        self.write("*RST")
        time.sleep(0.5)
        self.logger.info("[K2400] 已复位")

    def idn(self) -> str:
        """Query the instrument identification string."""
        return self.query("*IDN?")

    # --- Source / measure configuration -------------------------------------
    def set_source_mode(self, mode: str):
        """Set source mode: 'voltage' (V source, I measure) or 'current'."""
        mode = mode.lower()
        if mode == "voltage":
            self.write(":SOUR:FUNC VOLT")
            self._source_mode = "voltage"
            # When sourcing voltage, compliance is a current limit
            self._measure_func = "current"
        elif mode == "current":
            self.write(":SOUR:FUNC CURR")
            self._source_mode = "current"
            self._measure_func = "voltage"
        else:
            raise K2400ConfigError(f"不支持的源模式: {mode}")
        self.logger.info(f"[K2400] 源模式设为 {mode}")

    def get_source_mode(self) -> str:
        """Return the current source mode ('voltage' or 'current')."""
        return self._source_mode

    def set_measure_function(self, func: str):
        """Set the measurement function: 'current' or 'voltage'."""
        func = func.lower()
        if func == "current":
            self.write(':SENS:FUNC "CURR"')
            self._measure_func = "current"
        elif func == "voltage":
            self.write(':SENS:FUNC "VOLT"')
            self._measure_func = "voltage"
        else:
            raise K2400ConfigError(f"不支持的测量功能: {func}")

    def set_compliance(self, value: float):
        """Set the compliance limit (current limit in V-source, voltage limit in I-source)."""
        if self._source_mode == "voltage":
            self.write(f":SENS:CURR:PROT {value}")
        else:
            self.write(f":SENS:VOLT:PROT {value}")
        self.logger.info(f"[K2400] 合规限值设为 {value}")

    def set_nplc(self, nplc: float):
        """Set the measurement integration time in NPLC."""
        func = "CURR" if self._measure_func == "current" else "VOLT"
        self.write(f":SENS:{func}:NPLC {nplc}")
        self.logger.info(f"[K2400] NPLC设为 {nplc}")

    def set_range(self, auto: bool = True, fixed_value: Optional[float] = None):
        """Set the measurement range."""
        func = "CURR" if self._measure_func == "current" else "VOLT"
        if auto:
            self.write(f":SENS:{func}:RANG:AUTO ON")
        elif fixed_value is not None:
            self.write(f":SENS:{func}:RANG {fixed_value}")
            self.write(f":SENS:{func}:RANG:AUTO OFF")

    # --- Output control -----------------------------------------------------
    def set_output_level(self, level: float):
        """Set the source level (V in voltage mode, A in current mode)."""
        if self._source_mode == "voltage":
            self.write(f":SOUR:VOLT:LEV {level}")
        else:
            self.write(f":SOUR:CURR:LEV {level}")
        self.logger.info(f"[K2400] 输出电平设为 {level}")

    def output_on(self):
        """Turn the output on."""
        self.write(":OUTP ON")
        self.logger.info("[K2400] 输出开启")

    def output_off(self):
        """Turn the output off."""
        self.write(":OUTP OFF")
        self.logger.info("[K2400] 输出关闭")

    def measure(self) -> Dict[str, float]:
        """Trigger a measurement and return voltage/current/resistance/time."""
        resp = self.query(":READ?")
        parts = resp.split(",")
        if len(parts) >= 4:
            return {
                "voltage": float(parts[0]),
                "current": float(parts[1]),
                "resistance": float(parts[2]),
                "timestamp": float(parts[3]),
                "status": parts[4] if len(parts) > 4 else "",
            }
        raise K2400CommandError(f"测量返回格式异常: {resp}")

    def check_errors(self) -> List[str]:
        """Read the instrument error queue until empty."""
        errors = []
        for _ in range(10):
            resp = self.query(":SYST:ERR?")
            if "+0" in resp or "No error" in resp:
                break
            errors.append(resp)
        return errors

    # --- Context manager ----------------------------------------------------
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False


# ---------------------------------------------------------------------------
# Convenience helpers for the GUI
# ---------------------------------------------------------------------------
def refresh_port_list() -> List[str]:
    """Return a list of strings like 'COM3 - USB-SERIAL CH340'."""
    items = []
    for p in list_com_ports():
        desc = p["description"] or "Unknown"
        items.append(f"{p['port']} - {desc}")
    return items


def parse_port_entry(entry: str) -> str:
    """Extract the port name (e.g. COM3) from a dropdown string."""
    return entry.split(" - ", 1)[0].strip() if " - " in entry else entry.strip()
