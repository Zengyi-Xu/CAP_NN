"""示波器波形读出（pyvisa，USB-B / USBTMC）。

从原始 MATLAB oscrunDMT.m / oscrunQPSK.m 改写：
- 通过 pyvisa 连接示波器 USB-B 端口（USBTMC）
- 设置采样率、时基和波形格式
- 读取指定通道波形 preamble + 数据并转换为电压
- 可选重采样到 AWG 采样率
"""
import numpy as np
import pyvisa
from pathlib import Path
from typing import Optional, Tuple

import config


class KeysightScopeUSB:
    """Keysight 示波器 USB-B（USBTMC）控制。"""

    def __init__(self,
                 resource: Optional[str] = None,
                 timeout_ms: int = 20000):
        """
        参数：
            resource: pyvisa 资源字符串，如 "USB0::0x0957::0x17A6::MY12345678::INSTR"
                      若为 None，则自动检测第一个 USB 仪器
            timeout_ms: 通信超时
        """
        self.resource = resource
        self.timeout_ms = timeout_ms
        self.rm = pyvisa.ResourceManager()
        self.inst: Optional[pyvisa.Resource] = None
        self._used_resource: Optional[str] = None

    def list_resources(self) -> Tuple[str, ...]:
        return self.rm.list_resources()

    def connect(self) -> "KeysightScopeUSB":
        """连接示波器，支持 USB 和 TCPIP。"""

        all_resources = self.list_resources()

        if self.resource is None or self.resource == "":
            # 未指定地址：优先使用 USB，未找到则报错
            usb_resources = [r for r in all_resources if r.startswith("USB")]
            if not usb_resources:
                raise RuntimeError(
                    "未找到 USB 仪器。可用资源：\n" + "\n".join(all_resources)
                )
            self._used_resource = usb_resources[0]
            print(f"自动选择示波器资源：{self._used_resource}")
        else:
            self._used_resource = self.resource

            # 若配置的地址不在列表中，尝试按 VID/PID/SN 匹配（仅限 USB）
            if self._used_resource not in all_resources and self._used_resource.startswith("USB"):
                try:
                    parts = self._used_resource.split("::")
                    target_vid = parts[1].lower()
                    target_pid = parts[2].lower()
                    target_sn = parts[3]

                    for r in all_resources:
                        if not r.startswith("USB"):
                            continue
                        r_parts = r.split("::")
                        if (r_parts[1].lower() == target_vid and
                            r_parts[2].lower() == target_pid and
                            r_parts[3] == target_sn):
                            self._used_resource = r
                            print(f"地址 {self.resource} 未找到，已匹配到 {r}")
                            break
                except Exception:
                    pass

        print(f"正在连接示波器（{self._used_resource}）...")
        self.inst = self.rm.open_resource(self._used_resource)
        self.inst.timeout = self.timeout_ms
        self.inst.write_termination = "\n"
        self.inst.read_termination = "\n"

        idn = self.query("*IDN?")
        print(f"  *IDN = {idn}")
        return self
    # def connect(self) -> "KeysightScopeUSB":
    #     """连接示波器。"""
    #     if self.resource is None or self.resource == "":
    #         usb_resources = [r for r in self.list_resources() if r.startswith("USB")]
    #         if not usb_resources:
    #             raise RuntimeError(
    #                 "未找到 USB 仪器。可用资源：\n" +
    #                 "\n".join(self.list_resources())
    #             )
    #         self._used_resource = usb_resources[0]
    #         print(f"自动选择示波器资源：{self._used_resource}")
    #     else:
    #         self._used_resource = self.resource
    #
    #     print(f"正在连接示波器（{self._used_resource}）...")
    #     self.inst = self.rm.open_resource(self._used_resource)
    #     self.inst.timeout = self.timeout_ms
    #     self.inst.write_termination = "\n"
    #     self.inst.read_termination = "\n"
    #
    #     idn = self.query("*IDN?")
    #     print(f"  *IDN = {idn}")
    #     return self

    def close(self) -> None:
        if self.inst is not None:
            try:
                self.inst.write(":RUN")
            except Exception:
                pass
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
            raise RuntimeError("示波器未连接")
        self.inst.write(cmd)

    def query(self, cmd: str) -> str:
        if self.inst is None:
            raise RuntimeError("示波器未连接")
        return self.inst.query(cmd).strip()

    def configure(self,
                  sample_rate: Optional[float] = None,
                  timebase_scale: Optional[float] = None) -> None:
        """配置采集参数。"""
        sample_rate = sample_rate or config.OSC_SAMPLE_RATE
        timebase_scale = timebase_scale or config.OSC_TIMEBASE_SCALE

        # 增大输入缓冲区，确保完整读取大波形
        try:
            self.inst.set_buffer(pyvisa.constants.VI_READ_BUF, 40_000_000)
            self.inst.set_buffer(pyvisa.constants.VI_WRITE_BUF, 1_000_000)
        except Exception:
            pass

       # self.write(":STOP")
        self.write(f":ACQUIRE:SRATE {sample_rate:.15g}")
        self.write(f":TIMEBASE:SCALE {timebase_scale:.15g}")
        # 固定采集点数以覆盖完整 DMT 波形（4M 点 @ 10 GSa/s = 400 us）
        try:
            self.write(":ACQUIRE:POINTS:AUTO OFF")
            self.write(":ACQUIRE:POINTS 4000000")
            self.write(":WAVEFORM:POINTS:MODE RAW")
            self.write(":WAVEFORM:POINTS 4000000")
        except Exception:
            pass
        self.write(":WAVEFORM:FORMAT WORD")
        self.write(":WAVEFORM:BYTEORDER LSBFirst")
        print(f"示波器配置完成：fs={sample_rate/1e9:.2f} GSa/s，时基={timebase_scale*1e6:.1f} us/div，点数=4000000")

    def read_preamble(self, channel: Optional[str] = None) -> dict:
        """读取并解析 :WAVEFORM:PREAMBLE?"""
        channel = channel or config.OSC_CHANNEL
        self.write(f":WAVEFORM:SOURCE {channel}")
        preamble_str = self.query(":WAVEFORM:PREAMBLE?")
        parts = preamble_str.split(",")
        keys = ["format", "type", "points", "count",
                "x_increment", "x_origin", "x_reference",
                "y_increment", "y_origin", "y_reference"]
        preamble = {}
        for k, v in zip(keys, parts):
            preamble[k] = float(v)
        return preamble

    def read_waveform(self,
                      channel: Optional[str] = None,
                      include_preamble: bool = True) -> Tuple[np.ndarray, dict]:
        """读取指定通道波形。

        返回：
            ydata: 电压数组（V）
            preamble: 解析后的 preamble 字典
        """
        # 新增：重启示波器并等待一次完整采集
        self.write(":RUN")
        channel = channel or config.OSC_CHANNEL
        self.write(f":WAVEFORM:SOURCE {channel}")

        if include_preamble:
            preamble = self.read_preamble(channel)
        else:
            preamble = {}

        # 读取原始 ADC 值（int16）
        self.write(":WAV:DATA?")
        raw = self.inst.read_binary_values(
            datatype="h",
            is_big_endian=False,
            header_fmt="ieee",
            expect_termination=True
        )

        # 某些固件在二进制块后会附加一个多余换行；read_binary_values 可处理大多数情况
        # 忽略残留数据
        try:
            leftover = self.inst.read_bytes(1)
        except pyvisa.errors.VisaIOError:
            leftover = b""

        raw = np.asarray(raw, dtype=np.int16)

        if include_preamble:
            yinc = preamble["y_increment"]
            yori = preamble["y_origin"]
            yref = preamble["y_reference"]
            ydata = (raw - yref) * yinc + yori
        else:
            ydata = raw.astype(float)

        return ydata, preamble

    def capture(self,
                channel: Optional[str] = None,
                sample_rate: Optional[float] = None,
                timebase_scale: Optional[float] = None,
                resample_to_awg: bool = True) -> Tuple[np.ndarray, dict]:
        """一键采集：配置 + 读取。

        参数：
            channel: 通道名
            sample_rate: 示波器采样率
            timebase_scale: 时基
            resample_to_awg: 是否重采样到 AWG 采样率

        返回：
            data: 电压波形
            preamble: 解析后的 preamble
        """
        self.configure(sample_rate, timebase_scale)
        data, preamble = self.read_waveform(channel)

        if resample_to_awg:
            fs_osc = sample_rate or config.OSC_SAMPLE_RATE
            fs_awg = config.AWG_SAMPLE_RATE
            from utils import resample_signal
            data = resample_signal(data, fs_target=fs_awg, fs_source=fs_osc)
            print(f"波形已从 {fs_osc/1e9:.2f} GSa/s 重采样到 {fs_awg/1e9:.2f} GSa/s")

        return data, preamble


def capture_to_file(out_path: Optional[Path] = None,
                    resource: Optional[str] = None,
                    channel: Optional[str] = None) -> np.ndarray:
    """便捷函数：采集一次并保存到文件。"""
    if out_path is None:
        from utils import load_txt, save_txt
        count = 0
        if config.COUNT_FILE.exists():
            try:
                count = int(load_txt(config.COUNT_FILE))
            except Exception:
                count = 0
        out_path = config.RXDATA_DIR / f"rawOSC_DMT_{count}.txt"
        save_txt(config.COUNT_FILE, np.array([count + 1]), fmt="%d")

    with KeysightScopeUSB(resource=resource) as scope:
        data, preamble = scope.capture(channel=channel)
        from utils import save_txt
        save_txt(out_path, data)
        print(f"示波器数据已保存到 {out_path}")
        return data
