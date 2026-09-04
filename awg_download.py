#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
独立运行的 M8190A 任意波形下载测试脚本。

本脚本是以下 MATLAB 代码的逐行 Python 等价实现，位于
    D:\BaiduSyncdisk\SyncWorkplace\Individual\Code\20260429_900G_testing\AWGM8190A_Auto
尤其是：
    - AWGM8190A_Auto.m      （顶层自动下载）
    - AWG_transmit.m        （基本配置与运行）
    - download.m            （参数解析与通道映射）
    - download_M8190A.m     （M8190A 专用下载）
    - xfprintf.m / xquery.m / xbinblockwrite.m  （基于 TCP 的原始 SCPI）

它使用 Python 标准 socket 库与 AWG 固件的原始 TCP 端口
（默认 5025）通信，与 MATLAB 参考实现使用 tcpclient() 的方式完全一致。

用法：
    python awg_download.py
    python awg_download.py --host 192.168.1.10 --port 5025 -f data/txdata/SNRest_QPSK.txt
    python awg_download.py --route AC --amplitude 0.3
"""

import argparse
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------
DEFAULT_HOST = "localhost"          # MATLAB：device_name = "DESKTOP-CION5EQ"
DEFAULT_PORT = 5025                 # MATLAB RX_CHANNEL_SENSING_CONFIG.txt(9)
DEFAULT_SAMPLE_RATE = 8e9           # Hz
DEFAULT_AMPLITUDE = 0.5             # Vpp
DEFAULT_WAVEFORM = Path(__file__).parent / "data" / "txdata" / "SNRest_QPSK.txt"
DEFAULT_CONFIG_TXT = "RX_CHANNEL_SENSING_CONFIG.txt"

# M8190A_12bit 模式常量（来自 loadArbConfig.m）
SEGMENT_GRANULARITY = 64
MIN_SEGMENT_SIZE = 5 * SEGMENT_GRANULARITY
MAX_SEGMENT_SIZE = 3 * 512 * 1024 * 1024
MAX_SEGMENT_NUMBER = 512 * 1024

# 输出路径选项（见 M8190A 用户手册 / gen_arb_M8190A.m）
#   "DC"  -> :OUTP1:ROUT DC  + :DC1:VOLT:AMPL  （DC 耦合放大输出）
#   "AC"  -> :OUTP1:ROUT AC  + :AC1:VOLT:AMPL  （AC 耦合放大输出）
#   "DAC" -> :OUTP1:ROUT DAC + :VOLT1:AMPL     （DAC 直接输出，幅度最小）
OUTPUT_ROUTES = ("DC", "AC", "DAC")


# =============================================================================
# 底层 SCPI 辅助函数（原始 socket，MATLAB xfprintf/xquery 的等价实现）
# =============================================================================

class ScpiSocket:
    """原始 TCP socket 封装，行为类似 MATLAB tcpclient/visadev.SOCKET。"""

    def __init__(self, host: str, port: int, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self._buf = b""

    def connect(self):
        print(f"[SCPI] 正在连接 {self.host}:{self.port} ...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))
        print(f"[SCPI] 已连接。")
        return self

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception as e:
                print(f"[SCPI] 关闭时警告：{e}")
            self.sock = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write_line(self, data: bytes | str):
        """发送以 LF 结尾的字节/字符串（MATLAB writeline）。"""
        if self.sock is None:
            raise RuntimeError("SCPI socket 未连接")
        if isinstance(data, str):
            data = data.encode("ascii")
        self.sock.sendall(data + b"\n")

    def read_line(self) -> str:
        """读取一行以 LF 结尾的数据（MATLAB readline）。"""
        if self.sock is None:
            raise RuntimeError("SCPI socket 未连接")
        while b"\n" not in self._buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                break
            self._buf += chunk
        if b"\n" not in self._buf:
            raise TimeoutError("SCPI read_line 超时")
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode("ascii", errors="replace").strip()


def xfprintf(f: ScpiSocket, cmd: str, ignore_error: bool = False) -> int:
    """发送 SCPI 命令并检查错误队列（MATLAB xfprintf）。"""
    f.write_line(cmd)
    for _ in range(50):
        try:
            err = xquery(f, ":SYST:ERR?")
        except Exception:
            err = ""
        if not err:
            print(f"[SCPI] 警告：命令 {cmd} 后 :SYST:ERR? 无响应")
            return -1
        code = 0
        try:
            code = int(err.split(",")[0])
        except ValueError:
            pass
        if code == 0:
            return 0
        if not ignore_error:
            print(f"[SCPI] 命令 '{cmd}' 后 AWG 返回错误：{err}")
            return -1
    return -1


def xquery(f: ScpiSocket, cmd: str) -> str:
    """发送 SCPI 查询并返回响应字符串（MATLAB xquery）。"""
    f.write_line(cmd)
    return f.read_line()


def xbinblockwrite(f: ScpiSocket, data: np.ndarray, fmt: str, cmd: str):
    """发送 IEEE 488.2 二进制块数据（MATLAB xbinblockwrite，用于 tcpclient）。

    参数：
        f:    SCPI socket。
        data: 一维 numpy 数组；必须与 `fmt` 匹配。
        fmt:  'int8','uint8','int16','uint16','int32','uint32' 之一。
        cmd:  SCPI 头，如 ":TRACe1:DATA 1,0,"。
    """
    dtype_map = {
        "int8": np.int8,
        "uint8": np.uint8,
        "int16": np.int16,
        "uint16": np.uint16,
        "int32": np.int32,
        "uint32": np.uint32,
    }
    if fmt not in dtype_map:
        raise ValueError(f"xbinblockwrite：不支持的格式 {fmt}")
    data = np.asarray(data, dtype=dtype_map[fmt]).ravel()
    data_bytes = data.tobytes()  # x86 / M8190A 固件为小端序
    n = len(data_bytes)
    n_str = str(n)
    header = f"{cmd}#{len(n_str)}{n_str}"
    payload = header.encode("ascii") + data_bytes + b"\n"
    f.sock.sendall(payload)


# =============================================================================
# ArbConfig 与波形加载（MATLAB makeArbConfig / readFile / loadfile）
# =============================================================================

def make_arb_config(visa_addr: str, amplitude: float, sample_rate: float, port: int):
    """构建与 makeArbConfig.m + loadArbConfig.m 一致的 arbConfig 字典。"""
    arb = {
        "model": "M8190A_12bit",
        "connectionType": "tcpip",
        "visaAddr": visa_addr,
        "ip_address": "192.168.0.100",
        "port": port,
        "LOIPAddr": "xxx.xxx.xxx.xxx",
        "SD1ModuleIndex": 0,
        "M8070ModuleID": "M2",
        "defaultFc": 0,
        "tooltips": 1,
        "DACRange": 1,
        "amplScale": 2.83,
        "amplScaleMode": "Leave Unchanged",
        "filterSettings": "None",
        "userSamplerate": sample_rate,
        "triggerMode": "Continuous",
        "amplitude": amplitude,
        "offset": 0,
        "outputType": "Single Ended",
        "clockSource": "ExtRef",       # 'Unchanged','IntRef','AxieRef','ExtRef','ExtClk'
        "clockFreq": 10_000_000,       # 10 MHz 外部参考
        "peaking": None,
        "interleaving": 0,
        "sampleMarker": "Sample rate / 4",
        "useM8192A": 0,
        "visaAddrM8192A": "TCPIP0::localhost::hislip0::INSTR",
        "timeout": 30,
    }
    # loadArbConfig.m：M8190A_12bit 参数
    arb["numChannels"] = 2
    arb["channelMask"] = np.ones(arb["numChannels"], dtype=int)
    arb["fixedSampleRate"] = 0
    arb["defaultSampleRate"] = 12e9
    arb["maximumSampleRate"] = 12e9
    arb["minimumSampleRate"] = 125e6
    arb["minimumSegmentSize"] = MIN_SEGMENT_SIZE
    arb["maximumSegmentSize"] = MAX_SEGMENT_SIZE
    arb["segmentGranularity"] = SEGMENT_GRANULARITY
    arb["maxSegmentNumber"] = MAX_SEGMENT_NUMBER
    arb["maximumModules"] = 4
    # 用户定义的默认采样率优先于一切
    arb["defaultSampleRate"] = sample_rate
    return arb


def read_file(filename: str | Path, sample_rate: float = DEFAULT_SAMPLE_RATE):
    """读取单列 ASCII 波形并准备发送到 AWG。

    这是 'Oscilloscope (.txt)' 情形下 readFile.m + loadfile.m 的
    简化 Python 版本，返回值与 MATLAB 参考实现相同。
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"未找到波形文件：{filename}")

    print(f"[LOAD] 正在从 {filename} 读取波形")
    lines = filename.read_text(encoding="utf-8", errors="ignore").splitlines()

    # 跳过表头行，直到找到数值行（与 loadfile.m 逻辑相同）
    start_idx = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if line.upper() in ("Y", "DATA") or line.upper().startswith("Y1"):
            start_idx = i + 1
            continue
        # 若该行以数字开头，则数据从这里开始
        try:
            float(line.split(",")[0].split("\t")[0])
            start_idx = i
            break
        except ValueError:
            continue
    else:
        raise ValueError("波形文件中未找到数值数据")

    # 读取数值数据（单列）
    values = []
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line.split(",")[0].split("\t")[0]))
        except ValueError:
            break
    iqdata = np.asarray(values, dtype=float).reshape(-1, 1)

    # scaleMinMax = [-1, 1, 1]  -> 对称缩放到 [-1, +1]
    scale_min, scale_max, symm = -1.0, 1.0, True
    if symm:
        max_val = np.max(np.abs(iqdata))
        if max_val > 0:
            scale = min(abs(scale_max / max_val), abs(scale_min / -max_val))
            iqdata = iqdata * scale

    # 计算重复次数以满足段粒度与最小长度要求
    # （loadfile.m 末尾部分）
    seg_gran = SEGMENT_GRANULARITY
    seg_min = MIN_SEGMENT_SIZE
    length_orig = iqdata.shape[0]
    rpt = int(np.lcm(length_orig, seg_gran) // length_orig)
    while rpt * length_orig < seg_min:
        rpt += 1

    # 双通道 M8190A 上单列实数信号的 channelMapping
    channel_mapping = np.array([[1, 0], [0, 1]], dtype=int)

    # 默认 marker：前半段高电平，后半段低电平（download.m 默认值）
    marker = np.concatenate([
        np.full(length_orig // 2, 15, dtype=np.uint16),
        np.zeros(length_orig - length_orig // 2, dtype=np.uint16)
    ])

    print(f"[LOAD] {length_orig} 个采样点，fs={sample_rate/1e9:.3f} GHz，"
          f"rpt={rpt} -> 段长度={length_orig*rpt}")
    return iqdata, sample_rate, marker, rpt, channel_mapping


# =============================================================================
# 下载逻辑（MATLAB download.m / download_M8190A.m / gen_arb_M8190A.m）
# =============================================================================

def _fixlength(x, length):
    """MATLAB fixlength 辅助函数：将向量平铺或截断到 `length` 个元素。"""
    x = np.asarray(x).ravel()
    if x.size == 0:
        return np.zeros(length)
    reps = int(np.ceil(length / x.size))
    return np.tile(x, reps)[:length]


def _do_run(f: ScpiSocket, channel_mapping: np.ndarray):
    """启动 AWG 输出（MATLAB doRun，无 M8192A 同步）。"""
    active_ch = np.where(channel_mapping.sum(axis=1) > 0)[0] + 1
    if len(active_ch) > 1:
        xfprintf(f, ":INST:COUP:STATe ON")
    for ch in active_ch:
        xfprintf(f, f":INIT:IMM{ch}")
    print(f"[RUN] 已启动通道：{list(active_ch)}")


def _gen_arb_m8190a(f: ScpiSocket, arb_config: dict, chan: int,
                    data: np.ndarray, marker: np.ndarray,
                    segm_num: int, run: int,
                    segment_length: int, segment_offset: int,
                    route: str):
    """将一路实数波形下载到指定通道/段（MATLAB gen_arb_M8190A）。"""
    if not chan:
        return
    segm_len = data.size
    if segm_len > 0:
        # 删除旧段（若不存在则忽略错误），然后定义新段
        if run >= 0 and segment_offset == 0:
            xfprintf(f, f":TRACe{chan}:DELete {segm_num}", ignore_error=True)
            xfprintf(f, f":TRACe{chan}:DEFine {segm_num},{segment_length}")

        # 缩放到 12 位 DAC：int16(round(8191 * data) * 4)
        dac_data = np.int16(np.round(8191.0 * data) * 4)

        # 添加 marker 低 2 位
        if marker is not None and marker.size:
            if marker.size != dac_data.size:
                raise ValueError("marker 长度必须等于数据长度")
            dac_data = dac_data + np.int16(np.bitwise_and(marker.astype(np.uint16), 3))

        # 以 523200 个 int16 采样点为一分块下载
        chunk = 523200
        offset = 0
        while offset < segm_len:
            block = dac_data[offset:offset + chunk]
            cmd = f":TRACe{chan}:DATA {segm_num},{offset + segment_offset},"
            xbinblockwrite(f, block, "int16", cmd)
            offset += chunk

        xquery(f, "*OPC?")

        if run >= 0 and (segment_offset + segm_len >= segment_length):
            xfprintf(f, f":TRACe{chan}:SELect {segm_num}")

    if segment_offset + segm_len >= segment_length:
        # 输出路径与幅度
        if route == "DAC":
            xfprintf(f, f":OUTPut{chan}:ROUTe DAC")
            amp_cmd = f":VOLTage{chan}:AMPLitude"
        else:
            xfprintf(f, f":OUTPut{chan}:ROUTe {route}")
            amp_cmd = f":{route}{chan}:VOLT:AMPL"
        xfprintf(f, f"{amp_cmd} {arb_config['amplitude']}")
        xfprintf(f, f":VOLTage{chan}:OFFSet {arb_config['offset']}")

        if run >= 0:
            xfprintf(f, f":FUNCtion{chan}:MODE ARBitrary")
        xfprintf(f, f":OUTPut{chan} ON")


def download_m8190a(f: ScpiSocket, arb_config: dict, fs: float,
                    data: np.ndarray, marker1: np.ndarray, marker2: np.ndarray,
                    segm_num: int, keep_open: bool,
                    channel_mapping: np.ndarray, run: int,
                    segment_length: int | None, segment_offset: int | None,
                    route: str):
    """M8190A 专用下载（MATLAB download_M8190A.m，双通道直接模式）。"""
    if segment_length is None:
        segment_length = data.shape[0]
    if segment_offset is None:
        segment_offset = 0

    # 仅保留前 2 个通道（第一个模块）
    ch_map = np.asarray(channel_mapping, dtype=int)
    if ch_map.shape[0] > 2:
        ch_map = ch_map[:2, :]

    # 查询选件以确定单通道还是双通道
    opts = xquery(f, "*opt?")
    if "001" in opts:
        num_channels = 1
        ch_map = np.vstack([ch_map[0:1, :], np.zeros((1, ch_map.shape[1]), dtype=int)])
    else:
        num_channels = 2

    # 此处省略可选的 *RST（与 AWG_transmit.m 一致，其不发送 *RST）

    # 重新配置前停止输出
    if segment_offset == 0:
        for i in range(1, num_channels + 1):
            if ch_map[i - 1, :].sum() > 0:
                xfprintf(f, f":ABORt{i}")

    # 设置采样率、12 位 WSP 模式、时钟源
    dwid = "WSPeed"  # M8190A_12bit
    clock_source = arb_config.get("clockSource", "ExtRef")
    for i in range(1, num_channels + 1):
        if ch_map[i - 1, :].sum() == 0:
            continue
        if clock_source == "ExtRef":
            cmd = (f":ROSC:SOURce EXT; :ROSC:FREQuency {arb_config['clockFreq']:.15g}; "
                   f":FREQuency:RASTer:SOURce{i} INTernal; :FREQuency:RASTer {fs:.15g};")
        else:
            cmd = f":FREQuency:RASTer {fs:.15g};"
        if num_channels == 1:
            cmd += f" :TRACe1:DWIDth {dwid};"
        else:
            cmd += f" :TRACe1:DWIDth {dwid}; :TRACe2:DWIDth {dwid};"
        if xfprintf(f, cmd) != 0:
            print("[ERROR] 设置采样率/模式失败，中止。")
            return

    # 触发/连续模式
    trigger_mode = arb_config.get("triggerMode", "Continuous")
    cont_mode = 1 if trigger_mode == "Continuous" else 0
    gate_mode = 1 if trigger_mode == "Gated" else 0
    for i in range(1, num_channels + 1):
        if ch_map[i - 1, :].sum() > 0:
            xfprintf(f, f":INIT:CONTinuous{i} {cont_mode}; GATE{i} {gate_mode}")

    # 直接模式波形下载（实数数据）
    for col in range(ch_map.shape[1] // 2):
        for ch in np.where(ch_map[:, 2 * col] > 0)[0] + 1:
            _gen_arb_m8190a(f, arb_config, int(ch), np.real(data[:, col]).ravel(),
                            marker1, segm_num, run, segment_length, segment_offset, route)
        for ch in np.where(ch_map[:, 2 * col + 1] > 0)[0] + 1:
            _gen_arb_m8190a(f, arb_config, int(ch), np.imag(data[:, col]).ravel(),
                            marker2, segm_num, run, segment_length, segment_offset, route)

    # 若完整段已下载则启动
    if segment_offset + data.shape[0] >= segment_length:
        _do_run(f, ch_map)

    if not keep_open:
        f.close()


def download(iqdata: np.ndarray, fs: float,
             channel_mapping: np.ndarray | None = None,
             segment_number: int = 1,
             marker: np.ndarray | None = None,
             arb_config: dict | None = None,
             port: int = DEFAULT_PORT,
             run: bool = True,
             route: str = "DC"):
    """顶层下载入口（MATLAB download.m 的 M8190A_12bit 简化版）。"""
    if arb_config is None:
        raise ValueError("必须提供 arb_config")

    # 确保为列向量
    if iqdata.shape[0] < iqdata.shape[1]:
        iqdata = iqdata.T

    # 双通道上单列实数信号的默认通道映射
    if channel_mapping is None:
        channel_mapping = np.array([[1, 0], [0, 1]], dtype=int)
    channel_mapping = np.asarray(channel_mapping, dtype=int)

    # 将通道映射宽度补齐到 2 * 数据列数
    target_width = 2 * iqdata.shape[1]
    if channel_mapping.shape[1] < target_width:
        pad = np.zeros((channel_mapping.shape[0], target_width - channel_mapping.shape[1]), dtype=int)
        channel_mapping = np.hstack([channel_mapping, pad])

    # 必要时归一化（read_file 已归一化，此处保留以防万一）
    scale = np.max(np.abs(iqdata))
    if scale > 1.0:
        iqdata = iqdata / scale

    # Marker：未提供时默认为方波
    n = iqdata.shape[0]
    if marker is None:
        marker = np.concatenate([
            np.full(n // 2, 15, dtype=np.uint16),
            np.zeros(n - n // 2, dtype=np.uint16)
        ])
    marker = np.asarray(marker).ravel()
    marker1 = np.bitwise_and(marker.astype(np.uint16), 3)
    marker2 = np.bitwise_and(np.right_shift(marker.astype(np.uint16), 2), 3)

    # 检查粒度
    seg_len = n
    if seg_len % SEGMENT_GRANULARITY != 0:
        raise ValueError(f"段长度 {seg_len} 必须是 {SEGMENT_GRANULARITY} 的整数倍")
    if seg_len < MIN_SEGMENT_SIZE:
        raise ValueError(f"段长度 {seg_len} 必须 >= {MIN_SEGMENT_SIZE}")
    if seg_len > MAX_SEGMENT_SIZE:
        raise ValueError(f"段长度 {seg_len} 必须 <= {MAX_SEGMENT_SIZE}")

    host = "localhost"  # download_M8190A.m 使用本地主机
    with ScpiSocket(host, port) as f:
        download_m8190a(f, arb_config, fs, iqdata, marker1, marker2,
                        segment_number, keep_open=False,
                        channel_mapping=channel_mapping, run=1 if run else -1,
                        segment_length=seg_len, segment_offset=0,
                        route=route)


# =============================================================================
# 与 main.py 集成的辅助函数
# =============================================================================

def parse_tcpip_visa(visa_addr: str) -> tuple[str, int]:
    """解析形如 'TCPIP0::host::5025::SOCKET' 的 VISA 资源字符串。

    返回：
        (host, port)
    """
    parts = visa_addr.split("::")
    if len(parts) >= 4 and parts[0].upper().startswith("TCPIP"):
        return parts[1], int(parts[2])
    raise ValueError(f"无法解析 TCPIP SOCKET 类型的 VISA 地址：{visa_addr}")


def _prepare_data_for_awg(data: np.ndarray) -> np.ndarray:
    """归一化并重复实数波形，以满足 M8190A_12bit 段规则。"""
    data = np.asarray(data, dtype=float).reshape(-1, 1)
    scale = np.max(np.abs(data))
    if scale > 1.0:
        data = data / scale
    n = data.shape[0]
    rpt = int(np.lcm(n, SEGMENT_GRANULARITY) // n)
    while rpt * n < MIN_SEGMENT_SIZE:
        rpt += 1
    return np.tile(data, (rpt, 1))


def download_to_awg(data: np.ndarray,
                    fs: float = DEFAULT_SAMPLE_RATE,
                    vpp: float = DEFAULT_AMPLITUDE,
                    host: str = DEFAULT_HOST,
                    port: int = DEFAULT_PORT,
                    route: str = "DC",
                    channel_mapping: np.ndarray | None = None) -> None:
    """配置 M8190A 并下载内存中的实数波形。

    这是 main.py 的入口：执行与 MATLAB AWG_transmit.m 相同的配置，
    然后使用经独立测试脚本验证过的原始 socket 实现下载波形。
    """
    data = _prepare_data_for_awg(data)
    if channel_mapping is None:
        channel_mapping = np.array([[1, 0], [1, 0]], dtype=int)
    awg_transmit(data, fs, vpp, host, port, route=route,
                 channel_mapping=channel_mapping)


# =============================================================================
# 顶层：先按 AWG_transmit 风格配置，再下载
# =============================================================================

def awg_transmit(iqdata: np.ndarray, fs: float, vpp: float,
                 host: str, port: int, route: str = "DC",
                 channel_mapping: np.ndarray | None = None):
    """MATLAB AWG_transmit.m 的镜像：先配置，再下载并运行。"""
    visa_addr = f"TCPIP0::{host}::{port}::SOCKET"
    arb_config = make_arb_config(visa_addr, vpp, fs, port)

    with ScpiSocket(host, port) as f:
        # 自检（MATLAB AWG_transmit.m）
        pon = xquery(f, ":TEST:PON?")
        print(f"[INFO] :TEST:PON? -> {pon}")
        if '"Selftest passed"' not in pon and "Selftest passed" not in pon:
            print("[WARN] AWG 自检未报告 'Selftest passed'")

        # 参考时钟与采样率
        xfprintf(f, ":ROSC:FREQ 1e7")
        xfprintf(f, ":ROSC:SOUR EXT")
        xfprintf(f, f":FREQ:RAST {fs:.15g}")

        # 配置两个通道（与 AWG_transmit.m 相同）
        for ch in (1, 2):
            xfprintf(f, f":TRACe{ch}:DWIDth WSP")          # 12 位宽带
            if route in ("DC", "AC"):
                xfprintf(f, f":{route}{ch}:FORM NRZ")       # 放大输出使用 NRZ 格式
            xfprintf(f, f":{route}{ch}:VOLT:AMPL {vpp:.15g}")
            xfprintf(f, f":OUTP{ch}:ROUT {route}")
            xfprintf(f, f":OUTP{ch}:NORM ON")
            xfprintf(f, f":OUTP{ch}:COMP ON")
            xfprintf(f, f":TRAC{ch}:SEL 1")

        print("[INFO] 开始加载序列...")

    # 在 download 内部重新打开 socket（download_M8190A.m 会创建自己的 tcpclient）
    download(iqdata, fs,
             channel_mapping=channel_mapping,
             segment_number=1,
             marker=None,
             arb_config=arb_config,
             port=port,
             run=True,
             route=route)

    time.sleep(3)
    print("[DONE] AWG 发送完成。")


# =============================================================================
# 命令行 / 独立运行
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="独立运行的 M8190A 波形下载测试脚本")
    p.add_argument("--host", default=DEFAULT_HOST, help="AWG 主机/IP")
    p.add_argument("--port", type=int, default=None, help="AWG 原始 TCP 端口（默认 5025）")
    p.add_argument("-f", "--waveform", default=DEFAULT_WAVEFORM, help="输入 .txt 波形文件")
    p.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE, help="采样率（Hz）")
    p.add_argument("--amplitude", type=float, default=DEFAULT_AMPLITUDE, help="输出 Vpp")
    p.add_argument("--route", default="DC", choices=OUTPUT_ROUTES,
                   help="输出路径：DC（DC 耦合放大输出），AC（AC 耦合放大输出），DAC（DAC 直接输出）")
    p.add_argument("--config-txt", default=DEFAULT_CONFIG_TXT,
                   help="MATLAB 用于读取端口的 RX_CHANNEL_SENSING_CONFIG.txt")
    p.add_argument("--no-run", action="store_true", help="仅下载，不启动输出")
    # 用 parse_known_args：当被 main.py 导入调用时，sys.argv 里是 main.py 自己的
    # 参数（--offline 等），这里忽略未知参数、只取默认值，避免 argparse 报错退出。
    args, _unknown = p.parse_known_args()
    return args


def _read_port_from_config(path: str | Path) -> int:
    """读取 RX_CHANNEL_SENSING_CONFIG.txt 的第 9 行（从 1 开始计数）。"""
    path = Path(path)
    if not path.exists():
        return DEFAULT_PORT
    nums = [float(x) for x in path.read_text().split()]
    if len(nums) >= 9:
        return int(nums[8])  # MATLAB 索引：第 9 个元素 -> Python 索引 8
    return DEFAULT_PORT


def main():
    args = _parse_args()
    port = args.port if args.port is not None else _read_port_from_config(args.config_txt)

    # 读取波形（与 MATLAB readFile 相同）
    iqdata, fs, marker, rpt, ch_map = read_file(args.waveform, args.sample_rate)

    # 重复以满足段约束（与 MATLAB AWGM8190A_Auto 相同）
    iqdata = np.tile(iqdata, (rpt, 1))
    marker = np.tile(marker, rpt)

    print(f"[MAIN] host={args.host}，port={port}，fs={fs/1e9:.3f} GHz，"
          f"Vpp={args.amplitude}，route={args.route}，波形长度={iqdata.shape[0]}")

    # 执行与 MATLAB AWG_transmit + download 相同的流程
    awg_transmit(iqdata, fs, args.amplitude, args.host, port,
                 route=args.route, channel_mapping=ch_map)


if __name__ == "__main__":
    main()
