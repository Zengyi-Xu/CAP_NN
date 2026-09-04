# -*- coding: utf-8 -*-
"""Keithley 2400 SourceMeter RS-232/USB 控制封装。

本模块将 IVLab 项目中的 Keithley 2400 驱动移植到 DMT 实验平台。
它通过串口（COM）与仪器通信，该串口可以是物理 RS-232 端口，
也可以是 USB 转 RS-232 适配器（FTDI / Prolific / CH340 / CP210x 等）。

典型用法：

    from keithley2400_controller import Keithley2400, list_com_ports

    k = Keithley2400(port="COM3", baudrate=9600)
    k.connect()
    k.set_source_mode("voltage")      # 电压源
    k.set_compliance(0.1)             # 100 mA 电流限值
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
        "控制 Keithley 2400 需要 pyserial。"
        "请使用以下命令安装：pip install pyserial>=3.5"
    ) from exc


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------
class K2400Error(Exception):
    """Keithley 2400 操作的基类异常。"""
    pass


class K2400ConnectionError(K2400Error):
    """串口连接无法建立时抛出。"""
    pass


class K2400CommandError(K2400Error):
    """命令返回意外响应时抛出。"""
    pass


class K2400ConfigError(K2400Error):
    """提供了无效配置值时抛出。"""
    pass


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def _setup_logger(name: str = "k2400", level: int = logging.DEBUG) -> logging.Logger:
    """配置一个写入项目日志目录的 logger。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件（位于 DMT_PY_NN/log 内）
    log_dir = Path(__file__).resolve().parent / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "keithley2400.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# 端口扫描
# ---------------------------------------------------------------------------
def list_com_ports() -> List[Dict]:
    """返回可用 COM 端口列表及其元数据。"""
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
    """查找描述信息匹配常见 USB 串口适配器名称的 COM 端口。"""
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
# 仪器驱动
# ---------------------------------------------------------------------------
class Keithley2400:
    """通过 RS-232 或 USB 转 RS-232 控制的 Keithley 2400 SourceMeter。"""

    def __init__(self, port: str = "COM1", baudrate: int = 9600,
                 timeout: float = 5.0, logger: Optional[logging.Logger] = None):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self.connected = False
        self._source_mode = "voltage"   # "voltage" 或 "current"
        self._measure_func = "current"  # "current" 或 "voltage"
        self._debug = False
        self.logger = logger or _setup_logger("k2400")

    # --- 连接 -----------------------------------------------------------------
    def connect(self, retries: int = 3) -> bool:
        """打开串口并初始化仪器。"""
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
        """安全关闭输出并关闭串口。"""
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
        """清空缓冲区并查询仪器标识。"""
        if self.ser:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        try:
            idn = self.idn()
            self.logger.info(f"[K2400] IDN: {idn}")
        except Exception:
            pass
        # 默认为电压源，测量电流
        self.set_source_mode("voltage")
        self.set_measure_function("current")
        self.write(":SYST:RSEN OFF")          # 2 线制检测
        self.write(":FORM:ELEM VOLT,CURR,RES,TIME,STAT")

    # --- 底层 IO ---------------------------------------------------------------
    def write(self, cmd: str):
        """发送一条 SCPI 命令。"""
        if not self.connected or self.ser is None:
            raise K2400ConnectionError("未连接仪器")
        full_cmd = cmd + "\n"
        self.ser.write(full_cmd.encode("ascii"))
        if self._debug:
            self.logger.debug(f"[K2400] SEND: {cmd}")
        self.ser.flush()

    def read(self, timeout: Optional[float] = None) -> str:
        """从仪器读取一行。"""
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
        """发送命令并返回响应。"""
        self.write(cmd)
        time.sleep(0.05)
        return self.read(timeout=timeout)

    def reset(self):
        """将仪器复位为出厂默认值。"""
        self.write("*RST")
        time.sleep(0.5)
        self.logger.info("[K2400] 已复位")

    def idn(self) -> str:
        """查询仪器标识字符串。"""
        return self.query("*IDN?")

    # --- 源/测量配置 ------------------------------------------------------------
    def set_source_mode(self, mode: str):
        """设置源模式：'voltage'（电压源，测电流）或 'current'。"""
        mode = mode.lower()
        if mode == "voltage":
            self.write(":SOUR:FUNC VOLT")
            self._source_mode = "voltage"
            # 电压源模式下，合规限值为电流限值
            self._measure_func = "current"
        elif mode == "current":
            self.write(":SOUR:FUNC CURR")
            self._source_mode = "current"
            self._measure_func = "voltage"
        else:
            raise K2400ConfigError(f"不支持的源模式: {mode}")
        self.logger.info(f"[K2400] 源模式设为 {mode}")

    def get_source_mode(self) -> str:
        """返回当前源模式（'voltage' 或 'current'）。"""
        return self._source_mode

    def set_measure_function(self, func: str):
        """设置测量功能：'current' 或 'voltage'。"""
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
        """设置合规限值（电压源时为电流限值，电流源时为电压限值）。"""
        if self._source_mode == "voltage":
            self.write(f":SENS:CURR:PROT {value}")
        else:
            self.write(f":SENS:VOLT:PROT {value}")
        self.logger.info(f"[K2400] 合规限值设为 {value}")

    def set_nplc(self, nplc: float):
        """以 NPLC 设置测量积分时间。"""
        func = "CURR" if self._measure_func == "current" else "VOLT"
        self.write(f":SENS:{func}:NPLC {nplc}")
        self.logger.info(f"[K2400] NPLC设为 {nplc}")

    def set_range(self, auto: bool = True, fixed_value: Optional[float] = None):
        """设置测量量程。"""
        func = "CURR" if self._measure_func == "current" else "VOLT"
        if auto:
            self.write(f":SENS:{func}:RANG:AUTO ON")
        elif fixed_value is not None:
            self.write(f":SENS:{func}:RANG {fixed_value}")
            self.write(f":SENS:{func}:RANG:AUTO OFF")

    # --- 输出控制 ---------------------------------------------------------------
    def set_output_level(self, level: float):
        """设置源电平（电压模式单位为 V，电流模式单位为 A）。"""
        if self._source_mode == "voltage":
            self.write(f":SOUR:VOLT:LEV {level}")
        else:
            self.write(f":SOUR:CURR:LEV {level}")
        self.logger.info(f"[K2400] 输出电平设为 {level}")

    def output_on(self):
        """打开输出。"""
        self.write(":OUTP ON")
        self.logger.info("[K2400] 输出开启")

    def output_off(self):
        """关闭输出。"""
        self.write(":OUTP OFF")
        self.logger.info("[K2400] 输出关闭")

    def measure(self) -> Dict[str, float]:
        """触发一次测量并返回电压/电流/电阻/时间。"""
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
        """读取仪器错误队列直至为空。"""
        errors = []
        for _ in range(10):
            resp = self.query(":SYST:ERR?")
            if "+0" in resp or "No error" in resp:
                break
            errors.append(resp)
        return errors

    # --- 上下文管理器 ------------------------------------------------------------
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False


# ---------------------------------------------------------------------------
# GUI 便捷辅助函数
# ---------------------------------------------------------------------------
def refresh_port_list() -> List[str]:
    """返回形如 'COM3 - USB-SERIAL CH340' 的字符串列表。"""
    items = []
    for p in list_com_ports():
        desc = p["description"] or "未知"
        items.append(f"{p['port']} - {desc}")
    return items


def parse_port_entry(entry: str) -> str:
    """从下拉列表字符串中提取端口号（如 COM3）。"""
    return entry.split(" - ", 1)[0].strip() if " - " in entry else entry.strip()
