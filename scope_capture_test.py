#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的示波器采集测试。

本脚本对应 MATLAB 的 oscrunQPSK.m / oscrunDMT.m，用于验证示波器能否
正确采集真实波形。同时支持 USB-B (USBTMC) 和 TCPIP (端口 5025) 连接。

用法：
    python scope_capture_test.py
    python scope_capture_test.py --resource "TCPIP0::192.168.1.10::5025::SOCKET" --channel CHAN2
    python scope_capture_test.py --usb --channel CHAN2 --out rxdata/scope_test.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import config
from oscilloscope import KeysightScopeUSB
from utils import save_txt, resample_signal


def plot_and_analyze(data: np.ndarray,
                     fs_osc: float,
                     fs_awg: float,
                     title: str = "示波器采集") -> None:
    """绘制时域波形、频谱，并打印基本统计信息。"""
    import matplotlib.pyplot as plt

    n = len(data)
    t = np.arange(n) / fs_osc
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs_osc))
    # MATLAB: 10*log10(abs(fft(...)))
    spec = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(data))) + 1e-12)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    ax = axes[0]
    ax.plot(t * 1e6, data, "b-")
    ax.set_title(f"{title} - 时域")
    ax.set_xlabel("时间 (us)")
    ax.set_ylabel("电压 (V)")
    ax.grid(True)

    ax = axes[1]
    ax.plot(freqs / 1e9, spec, "b-")
    ax.set_title(f"{title} - 频谱")
    ax.set_xlabel("频率 (GHz)")
    ax.set_ylabel("幅度 (dB)")
    ax.grid(True)

    fig.tight_layout()
    plt.show()

    print("\n========== 波形统计 ==========")
    print(f"  采样点数      = {n}")
    print(f"  时长          = {n/fs_osc*1e6:.2f} us")
    print(f"  最小值        = {np.min(data):.6f} V")
    print(f"  最大值        = {np.max(data):.6f} V")
    print(f"  均值          = {np.mean(data):.6f} V")
    print(f"  标准差        = {np.std(data):.6f} V")
    print(f"  峰峰值        = {np.ptp(data):.6f} V")
    print(f"  有效值        = {np.sqrt(np.mean(data**2)):.6f} V")
    print(f"  频谱最大值    = {np.max(spec):.2f} dB")
    print(f"  频谱最小值    = {np.min(spec):.2f} dB")

    # 简单的能量检查：若标准差接近 0 或频谱最大值很低，
    # 则采集到的波形很可能不包含信号。
    if np.std(data) < 1e-6:
        print("\n[警告] 波形标准差极低，未检测到信号！")
    else:
        print("\n[信息] 波形能量非零，很可能存在信号。")


def main():
    parser = argparse.ArgumentParser(description="独立的示波器采集测试")
    parser.add_argument("--resource", "-r", type=str, default=None,
                        help="VISA 资源字符串，如 'TCPIP0::ip::5025::SOCKET'")
    parser.add_argument("--usb", action="store_true",
                        help="使用第一个可用的 USB 仪器（覆盖 --resource）")
    parser.add_argument("--channel", "-c", type=str, default="CHAN2",
                        help="示波器通道（MATLAB oscrun 使用 CHAN2）")
    parser.add_argument("--sample-rate", type=float, default=config.OSC_SAMPLE_RATE,
                        help="示波器采样率，单位 Hz")
    parser.add_argument("--timebase", type=float, default=80e-6,
                        help="时基，单位 s/div")
    parser.add_argument("--resample", action="store_true", default=True,
                        help="将采集到的波形重采样到 AWG 采样率")
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="输出文件路径（默认：rxdata/scope_test_{count}.txt）")
    parser.add_argument("--no-plot", action="store_true",
                        help="不显示图形")
    args = parser.parse_args()

    # 确定资源
    resource = args.resource
    if args.usb:
        resource = None  # 自动选择第一个 USB 设备
    elif resource is None:
        # 默认：使用 config 中的配置；若 config 仍是 USB 而用户想使用 TCPIP，
        # 则应显式传入 --resource
        resource = config.OSC_VISA_ADDR

    print(f"[测试] 资源     = {resource or '自动选择 USB'}")
    print(f"[测试] 通道     = {args.channel}")
    print(f"[测试] 示波器采样率 = {args.sample_rate/1e9:.2f} GSa/s")
    print(f"[测试] 时基     = {args.timebase*1e6:.1f} us/div")

    try:
        with KeysightScopeUSB(resource=resource) as scope:
            print("\n可用资源：")
            for r in scope.list_resources():
                print(f"  {r}")

            data, preamble = scope.capture(
                channel=args.channel,
                sample_rate=args.sample_rate,
                timebase_scale=args.timebase,
                resample_to_awg=False  # 保留示波器原始采样点用于诊断
            )
    except Exception as e:
        print(f"\n[错误] 示波器采集失败: {e}")
        raise

    print("\n========== 示波器前导信息 ==========")
    for k, v in preamble.items():
        print(f"  {k:15s} = {v}")

    # 可选重采样
    if args.resample:
        data_resampled = resample_signal(data,
                                         fs_target=config.AWG_SAMPLE_RATE,
                                         fs_source=args.sample_rate)
        print(f"\n[信息] 已重采样到 {config.AWG_SAMPLE_RATE/1e9:.2f} GSa/s，"
              f"长度 {len(data_resampled)}")
    else:
        data_resampled = data

    # 保存
    if args.out is None:
        from utils import load_txt
        count = 0
        if config.COUNT_FILE.exists():
            try:
                count = int(load_txt(config.COUNT_FILE))
            except Exception:
                count = 0
        out_path = config.RXDATA_DIR / f"scope_test_{count}.txt"
        save_txt(config.COUNT_FILE, np.array([count + 1]), fmt="%d")
    else:
        out_path = Path(args.out)
    save_txt(out_path, data_resampled)
    print(f"\n[信息] 波形已保存至 {out_path}")

    # 绘制示波器原始数据
    if not args.no_plot:
        plot_and_analyze(data, args.sample_rate, config.AWG_SAMPLE_RATE,
                         title=f"示波器 {args.channel} @ {args.sample_rate/1e9:.2f} GSa/s")

        if args.resample:
            plot_and_analyze(data_resampled, config.AWG_SAMPLE_RATE, config.AWG_SAMPLE_RATE,
                             title="重采样至 AWG 采样率")


if __name__ == "__main__":
    main()
