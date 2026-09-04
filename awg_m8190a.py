"""M8190A 任意波形发生器控制（pyvisa）。

简化版本，覆盖原始 MATLAB 中 AWG_transmit / download_M8190A 的核心功能：
- 通过 TCPIP socket（或 VISA）连接 M8190A
- 设置采样率、输出幅度和 DC 输出
- 将波形下载到段并播放
"""
import numpy as np
import pyvisa
from pathlib import Path
from typing import Optional, Tuple

import config


class M8190AController:
    """M8190A AWG 控制器。"""

    def __init__(self,
                 visa_addr: Optional[str] = None,
                 sample_rate: float = config.AWG_SAMPLE_RATE,
                 vpp: float = config.AWG_VPP,
                 output_route: str = config.AWG_OUTPUT_ROUTE,
                 timeout_ms: int = 30000):
        """
        参数：
            visa_addr: VISA 资源字符串，如 "TCPIP0::192.168.1.10::5025::SOCKET"
            sample_rate: AWG 采样率（Hz）
            vpp: 输出幅度（V）
            output_route: 输出路径，可选 "DC" / "AC" / "DAC"
                - "DC"：DC 耦合放大输出（默认，常用于基带/DMT）
                - "AC"：AC 耦合放大输出（隔直，常用于 RF/IF）
                - "DAC"：DAC 直接输出（无放大，幅度最小）
            timeout_ms: 通信超时

        默认通过 TCPIP 端口 5025 连接 M8190A（与原始 MATLAB 一致）。
        """
        self.visa_addr = visa_addr or config.M8190A_VISA_ADDR
        self.sample_rate = sample_rate
        self.vpp = vpp
        self.output_route = output_route.upper()
        self.rm = pyvisa.ResourceManager()
        self.inst: Optional[pyvisa.Resource] = None
        self.timeout_ms = timeout_ms

    def connect(self) -> "M8190AController":
        """建立连接并验证设备。"""
        print(f"正在连接 M8190A（{self.visa_addr}）...")
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
                print(f"关闭时警告：{e}")
            self.inst = None
        self.rm.close()

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(self, cmd: str) -> None:
        if self.inst is None:
            raise RuntimeError("AWG 未连接")
        self.inst.write(cmd)

    def query(self, cmd: str) -> str:
        if self.inst is None:
            raise RuntimeError("AWG 未连接")
        return self.inst.query(cmd).strip()

    def reset(self) -> None:
        self.write("*RST")
        self.query("*OPC?")

    def configure(self,
                  sample_rate: Optional[float] = None,
                  vpp: Optional[float] = None,
                  channels: Tuple[int, ...] = (1, 2),
                  output_route: Optional[str] = None) -> None:
        """配置 AWG 基本参数。

        参数：
            sample_rate: 采样率（Hz）
            vpp: 输出幅度（V）
            channels: 要配置的通道元组
            output_route: 输出路径，覆盖构造函数的设置；可选 "DC" / "AC" / "DAC"
        """
        sample_rate = sample_rate or self.sample_rate
        vpp = vpp or self.vpp
        route = (output_route or self.output_route).upper()

        if route not in ("DC", "AC", "DAC"):
            raise ValueError(f"不支持的 M8190A 输出路径：{route}。请使用 DC/AC/DAC。")

        # 参考时钟
        self.write(":ROSC:FREQ 1e7")
        self.write(":ROSC:SOUR EXT")

        for ch in channels:
            # 采样率
            self.write(f":FREQ:RAST {sample_rate:.15g}")
            # 12 位宽带模式
            self.write(f":TRACe{ch}:DWIDth WSP")
            # 输出路径及对应的幅度命令
            # DC/AC 为放大输出，DAC 为 DAC 直接输出
            self.write(f":OUTP{ch}:ROUT {route}")
            self.write(f":{route}{ch}:VOLT:AMPL {vpp:.15g}")
            if route in ("DC", "AC"):
                self.write(f":DC{ch}:FORM NRZ")
            self.write(f":OUTP{ch}:NORM ON")
            self.write(f":OUTP{ch}:COMP ON")
            self.write(f":TRAC{ch}:SEL 1")

        self.query("*OPC?")
        print(f"AWG 配置完成：fs={sample_rate/1e9:.2f} GHz，Vpp={vpp} V，route={route}")

    def _scale_to_int16(self, data: np.ndarray) -> np.ndarray:
        """将 [-1, 1] 的浮点波形缩放到 M8190A 12 位 DAC（int16，左移 4 位）。"""
        data = np.asarray(data).ravel()
        if np.max(np.abs(data)) > 1.0:
            data = data / np.max(np.abs(data))
        return np.int16(np.round(8191 * data) * 4)

    def download_waveform(self,
                          data: np.ndarray,
                          channel: int = 1,
                          segment: int = 1,
                          run: bool = True) -> None:
        """将波形下载到指定通道和段，并可选立即播放。

        参数：
            data: 实数波形，建议范围 [-1, 1]
            channel: 通道 1 或 2
            segment: 段号
            run: 是否立即播放
        """
        if self.inst is None:
            raise RuntimeError("AWG 未连接")

        data = self._scale_to_int16(data)
        segm_len = len(data)

        # 删除旧段并定义新段
        # M8190A 的 ABORt / INIT:IMM 以通道号作为参数，不拼接在命令名后
        self.write(f":ABORt {channel}")
        self.write(f":TRACe{channel}:DELete {segment}")
        self.write(f":TRACe{channel}:DEFine {segment},{segm_len}")

        # 分块下载（Keysight 建议每次传输不超过 523200 个 int16 值）
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

        # 选择段并开启输出
        self.write(f":TRACe{channel}:SELect {segment}")
        self.write(f":FUNCtion{channel}:MODE ARBitrary")
        self.write(f":OUTPut{channel}:STATe ON")

        if run:
            # INIT:IMM 同样以通道号作为参数，不拼接在命令名后
            self.write(f":INIT:IMM {channel}")
            self.query("*OPC?")
            print(f"通道 {channel} 正在播放段 {segment}（{segm_len} 个采样点）")
        else:
            print(f"波形已下载到通道 {channel} 段 {segment}")

    def download_iq(self,
                    iqdata: np.ndarray,
                    channel_i: int = 1,
                    channel_q: int = 2,
                    segment: int = 1,
                    run: bool = True) -> None:
        """将复数波形的实部和虚部分别下载到两个通道。"""
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
        """发送 *RST 并等待完成。"""
        self.write("*RST")
        self.query("*OPC?")


def quick_download_to_awg(data: np.ndarray,
                          sample_rate: float = config.AWG_SAMPLE_RATE,
                          vpp: float = config.AWG_VPP,
                          visa_addr: Optional[str] = None,
                          output_route: str = config.AWG_OUTPUT_ROUTE,
                          channel: int = 1) -> None:
    """便捷函数：连接-配置-下载-播放-关闭。

    参数：
        data: 实数波形
        sample_rate: 采样率（Hz）
        vpp: 输出幅度（V）
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
