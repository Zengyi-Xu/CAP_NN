"""DMT 完整通信流程入口.

对应原始 MATLAB 的四个步骤：
  step1: QPSK 探测发射 (SNRest TX)
  step2: QPSK 探测接收并估计每载波 SNR
  step3: Bitloading 发射
  step4: Bitloading 接收并解调/计算 BER

支持离线模式（读取已有 rxdata 文件）和在线模式（连接 AWG/Scope）。
支持虚拟信道、多种导频图案、NN 后均衡，并自动生成唯一传输记录。
"""
import numpy as np
from pathlib import Path
import argparse

import matplotlib
import config
from utils import load_txt, save_txt, save_mat, save_rx_config, sync_waveform, resample_signal
from plot_adapter import (
    CODEPLOT_DIR,
    plot_time_waveform,
    plot_spectrum,
    plot_constellation,
    plot_snrs,
    plot_tx_rx_nonlinearity,
    plot_bit_power_loading,
    plot_ser_ber_per_carrier,
    plot_constellation_density,
    plot_constellation_by_order,
)
from dmt_core import (
    generate_qpsk_tx,
    generate_bitloading_tx,
    estimate_snr_per_carrier,
    dmt_receiver,
    load_snr_table,
    assign_qam_order_from_snr,
)
from awg_download import download_to_awg, parse_tcpip_visa,_parse_args,awg_transmit,_read_port_from_config,read_file
from oscilloscope import KeysightScopeUSB
from nn_equalizer import run_nn_equalizer
from virtual_channel import VirtualChannel
from record import generate_run_id, save_record


def _resolve_offline_rx_file(pattern: str) -> Path:
    """离线模式下定位 RX 文件：先按 count.txt，没有再按修改时间找最新的匹配文件."""
    count = 0
    if config.COUNT_FILE.exists():
        count = int(load_txt(config.COUNT_FILE))
    candidate = config.RXDATA_DIR / pattern.replace("*", str(count))
    if candidate.exists():
        return candidate
    matches = sorted(config.RXDATA_DIR.glob(pattern),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        print(f"Count-based file {candidate} not found, using latest match: {matches[0]}")
        return matches[0]
    raise FileNotFoundError(
        f"Offline RX file not found: {candidate}\n"
        f"  Searched pattern: {config.RXDATA_DIR / pattern}\n"
        f"  Hint: use --use-virtual-channel 1, or provide --qpsk-rx / --bpl-rx path."
    )


def _save_virtual_rx(rx: np.ndarray, tag: str, run_id: str) -> Path:
    """保存虚拟信道生成的 RX 波形到 rxdata，便于后续复用与排查."""
    out = config.RXDATA_DIR / f"virtual_{tag}_{run_id}.txt"
    save_txt(out, rx)
    print(f"Saved virtual-channel RX to {out}")
    return out


def get_cfg():
    """组装流程配置."""
    return {
        "carrierno": config.CARRIERNO,
        "zeropad1": config.ZEROPAD1,
        "upsampleno": config.UPSAMPLENO,
        "cp": config.CP,
        "trainingno": config.TRAININGNO,
        "carrierno1": config.CARRIERNO1,
        "normalize_flag": config.NORMALIZE_FLAG,
        "pre_equ_flag": config.PRE_EQU_FLAG,
        "awg_sample_rate": config.AWG_SAMPLE_RATE,
        "pilot_pattern": config.PILOT_PATTERN,
        "pilot_value": config.PILOT_VALUE,
        "comb_start": config.PILOT_COMB_START,
        "comb_spacing": config.PILOT_COMB_SPACING,
        "mesh_start_freq": config.PILOT_MESH_START_FREQ,
        "mesh_freq_spacing": config.PILOT_MESH_FREQ_SPACING,
        "mesh_start_time": config.PILOT_MESH_START_TIME,
        "mesh_time_spacing": config.PILOT_MESH_TIME_SPACING,
    }


def step1_generate_qpsk_tx(use_awg: bool = False,
                           plot_dir: Path = None,
                           run_id: str = None):
    """STEP1: 生成 QPSK 探测波形."""
    print("\n========== STEP1: Generate QPSK TX ==========")
    tx_dict = generate_qpsk_tx(datano=config.DATANO_QPSK, cfg=get_cfg())

    # 实际发射波形：pre_equ_flag==3 时优先使用预均衡波形
    tx_out = tx_dict["tx_waveform_pre"] if tx_dict.get("tx_waveform_pre") is not None else tx_dict["tx_waveform"]

    # 保存文件（固定文件名，不编号；RX 文件在保存时附带 config）
    tx_qpsk_file = config.TX_QPSK_FILE
    tx_qpsk_pre_file = config.TX_QPSK_PRE_FILE
    origin_dec_qpsk = config.ORIGIN_DEC_DATA_QPSK
    demod_qpsk = config.DEMOD_FILE_QPSK
    bitpower_qpsk = config.BITPOWER_QPSK

    save_txt(tx_qpsk_file, tx_dict["tx_waveform"])
    if tx_dict.get("tx_waveform_pre") is not None:
        save_txt(tx_qpsk_pre_file, tx_dict["tx_waveform_pre"])
        print(f"Saved pre-equalized QPSK waveform to {tx_qpsk_pre_file}")
    save_txt(origin_dec_qpsk, tx_dict["origin_dec_data"], fmt="%d")
    save_mat(demod_qpsk,
             qamdata_final=tx_dict["qamdata"],
             AVT=tx_dict["AVT"])
    save_mat(bitpower_qpsk,
             S=np.ones(config.CARRIERNO1),
             RQ=np.full(config.CARRIERNO1, 2, dtype=int))
    save_txt(config.WAVEFORM_DUMMY_LEN,
             np.array([tx_dict["waveform_dummy_len"]]), fmt="%d")

    print(f"Saved TX QPSK waveform to {tx_qpsk_file}")
    print(f"  waveform length = {len(tx_dict['tx_waveform'])}, dummy = {tx_dict['waveform_dummy_len']}")

    # 画图
    if config.PLOT_SHOW:
        plot_time_waveform(np.arange(len(tx_dict["tx_waveform"])),
                           tx_dict["tx_waveform"],
                           f"QPSK TX Time Domain ({run_id})",
                           run_id, "SNRest_QPSK_time")
        plot_spectrum(tx_dict["tx_waveform"],
                      config.AWG_SAMPLE_RATE,
                      f"QPSK TX Spectrum ({run_id})",
                      run_id, "SNRest_QPSK_spec")
        flat_iq = tx_dict["dataIQ"].reshape(-1)
        plot_constellation(flat_iq,
                           f"QPSK TX Constellation ({run_id})",
                           run_id, "SNRest_QPSK_constellation")

    if use_awg:
        host, port = parse_tcpip_visa(config.M8190A_VISA_ADDR)
        download_to_awg(tx_out,
                        fs=config.AWG_SAMPLE_RATE,
                        vpp=config.AWG_VPP,
                        host=host,
                        port=port,
                        route=config.AWG_OUTPUT_ROUTE)

    return tx_dict


def step2_receive_qpsk(tx_dict: dict,
                       offline: bool = True,
                       rx_file: Path = None,
                       use_virtual_channel: bool = False,
                       plot_dir: Path = None,
                       run_id: str = None):
    """STEP2: 接收 QPSK 波形并估计每载波 SNR."""
    print("\n========== STEP2: Receive QPSK & Estimate SNR ==========")
    
    # 同步/参考优先使用预均衡波形（与 AWG 实际播放的信号一致）
    tx_waveform = tx_dict["tx_waveform_pre"] if tx_dict.get("tx_waveform_pre") is not None else tx_dict["tx_waveform"]
    args = _parse_args()
    port = args.port if args.port is not None else _read_port_from_config(args.config_txt)

    # Read waveform (same as MATLAB readFile)
    iqdata, fs, marker, rpt, ch_map = read_file(args.waveform, args.sample_rate)

    # Repeat to meet segment constraints (same as MATLAB AWGM8190A_Auto)
    iqdata = np.tile(iqdata, (rpt, 1))
    marker = np.tile(marker, rpt)

    print(f"[MAIN] host={args.host}, port={port}, fs={fs/1e9:.3f} GHz, "
          f"Vpp={args.amplitude}, route={args.route}, waveform_len={iqdata.shape[0]}")

    # Run the same flow as MATLAB AWG_transmit + download
    awg_transmit(iqdata, fs, args.amplitude, args.host, port,
                 route=args.route, channel_mapping=ch_map)
    rx_source = "unknown"
    if use_virtual_channel and offline:
        ch = VirtualChannel(fs=config.AWG_SAMPLE_RATE)
        rx = ch.apply(tx_waveform)
        rx_source = "virtual_channel"
        print("Generated RX via virtual channel")
        _save_virtual_rx(rx, "QPSK_SNRest", run_id)
    elif offline:
        try:
            if rx_file is None:
                rx_file = _resolve_offline_rx_file("rawOSC_QPSK_SNRest_*.txt")
            rx = load_txt(rx_file)
            rx_source = "measured_offline"
            print(f"Loaded offline RX data from {rx_file}")
        except FileNotFoundError:
            print("[WARN] Offline QPSK RX file not found, falling back to virtual channel")
            ch = VirtualChannel(fs=config.AWG_SAMPLE_RATE)
            rx = ch.apply(tx_waveform)
            rx_source = "virtual_channel"
            print("Generated RX via virtual channel")
            _save_virtual_rx(rx, "QPSK_SNRest", run_id)
    else:
        rx_source = "measured_online"
        print(f"[SCOPE] Capturing QPSK RX from {config.OSC_VISA_ADDR}, channel {config.OSC_CHANNEL}")
        with KeysightScopeUSB(resource=config.OSC_VISA_ADDR) as scope:
            rx, _ = scope.capture(channel=config.OSC_CHANNEL,
                                  sample_rate=config.OSC_SAMPLE_RATE,
                                  timebase_scale=80e-6,
                                  resample_to_awg=True)
        count = 0
        if config.COUNT_FILE.exists():
            count = int(load_txt(config.COUNT_FILE))
        rx_file = config.RXDATA_DIR / f"rawOSC_QPSK_SNRest_{count}.txt"
        save_txt(rx_file, rx)
        cfg_path = save_rx_config(
            rx_file,
            run_id=run_id,
            stage="QPSK_SNR_est",
            awg_sample_rate=config.AWG_SAMPLE_RATE,
            osc_sample_rate=config.OSC_SAMPLE_RATE,
            awg_vpp=config.AWG_VPP,
            awg_output_route=config.AWG_OUTPUT_ROUTE,
            scope_channel=config.OSC_CHANNEL,
            timebase_scale=80e-6,
            carrierno=config.CARRIERNO,
            zeropad1=config.ZEROPAD1,
            datano_qpsk=config.DATANO_QPSK,
            trainingno=config.TRAININGNO,
            pilot_pattern=config.PILOT_PATTERN,
            pre_equ_flag=config.PRE_EQU_FLAG,
            rx_length=len(rx),
            count=count,
        )
        print(f"Saved RX config to {cfg_path}")
        save_txt(config.COUNT_FILE, np.array([count + 1]), fmt="%d")

    # 同步
    rx_sync = sync_waveform(rx, tx_waveform)
    print(f"RX synced length = {len(rx_sync)}")

    # 保存 NN 数据（可选）
    save_txt(config.NN_TX_FILE, tx_waveform.ravel())
    save_txt(config.NN_RX1_FILE, rx_sync.ravel())

    # 估计 SNR
    snrs = estimate_snr_per_carrier(rx_sync, tx_dict, cfg=get_cfg())
    save_txt(config.FINAL_SNR_QPSK, snrs)
    print(f"Saved per-carrier SNR to {config.FINAL_SNR_QPSK}")
    mean_snr = np.nanmean(snrs)
    print(f"  mean SNR (linear) = {mean_snr:.4f}, mean SNR (dB) = {10*np.log10(mean_snr):.2f}")

    # 画图
    if config.PLOT_SHOW:
        plot_spectrum(rx_sync,
                      config.AWG_SAMPLE_RATE,
                      f"QPSK RX Spectrum [{rx_source}] ({run_id})",
                      run_id, "SNRest_QPSK_rx_spec")
        plot_tx_rx_nonlinearity(tx_waveform,
                                rx_sync,
                                f"QPSK TX-RX Nonlinearity [{rx_source}] ({run_id})",
                                run_id, "SNRest_QPSK_nonlinearity")
        plot_snrs(snrs, snrs,
                  f"Estimated SNR (dB) [{rx_source}] ({run_id})",
                  run_id, "SNR_QPSK")

    return snrs, rx_sync


def step3_generate_bitloading_tx(snrs: np.ndarray,
                                 constellation: str = config.CONSTELLATION_QAM,
                                 use_awg: bool = True,
                                 plot_dir: Path = None,
                                 run_id: str = None):
    """STEP3: 根据 SNR 做 bitloading 并生成发射波形."""
    print("\n========== STEP3: Generate Bitloading TX ==========")
    tx_dict = generate_bitloading_tx(snrs,
                                     constellation=constellation,
                                     datano=config.DATANO_BPL,
                                     cfg=get_cfg())

    # 实际发射波形：pre_equ_flag==3 时优先使用预均衡波形
    tx_out = tx_dict["tx_waveform_pre"] if tx_dict.get("tx_waveform_pre") is not None else tx_dict["tx_waveform"]

    # 保存（固定文件名，不编号；RX 文件在保存时附带 config）
    tx_bpl_file = config.TX_BPL_FILE
    tx_bpl_pre_file = config.TX_BPL_PRE_FILE
    origin_dec_bpl = config.ORIGIN_DEC_DATA_BPL
    demod_bpl = config.DEMOD_FILE_BPL
    bitpower_bpl = config.BITPOWER_BPL
    qamorderall = config.QAMORDERALL_FILE

    save_txt(tx_bpl_file, tx_dict["tx_waveform"])
    if tx_dict.get("tx_waveform_pre") is not None:
        save_txt(tx_bpl_pre_file, tx_dict["tx_waveform_pre"])
        print(f"Saved pre-equalized bitloading waveform to {tx_bpl_pre_file}")
    save_txt(origin_dec_bpl, tx_dict["origin_dec_data"], fmt="%d")
    save_mat(demod_bpl,
             qamdata_final=tx_dict["qamdata"],
             AVT=tx_dict["AVT"])
    save_mat(bitpower_bpl,
             S=tx_dict["S"],
             RQ=tx_dict["RQ"])
    save_txt(qamorderall, tx_dict["RQ"], fmt="%d")

    print(f"Saved TX bitloading waveform to {tx_bpl_file}")
    print(f"  waveform length = {len(tx_dict['tx_waveform'])}, dummy = {tx_dict['waveform_dummy_len']}")

    # 画图：比特功率加载
    if config.PLOT_SHOW:
        subcarriers = np.arange(config.CARRIERNO1)
        snrs_db = 10 * np.log10(np.maximum(np.nan_to_num(snrs), 1e-12))
        plot_bit_power_loading(subcarriers,
                               snrs_db,
                               tx_dict["RQ"],
                               tx_dict["S"],
                               ratio=tx_dict.get("ratio", 100),
                               rate_gbps=tx_dict.get("datarate_gbps", 0.0),
                               title=f"Bit-Power Loading ({run_id})",
                               run_id=run_id,
                               name="bit_power_loading")
        plot_time_waveform(np.arange(len(tx_dict["tx_waveform"])),
                           tx_dict["tx_waveform"],
                           f"Bitloading TX Time Domain ({run_id})",
                           run_id, "DMT_bitloading_Tx_time")
        plot_spectrum(tx_dict["tx_waveform"],
                      config.AWG_SAMPLE_RATE,
                          f"Bitloading TX Spectrum ({run_id})",
                      run_id, "DMT_bitloading_Tx_spec")


    host, port = parse_tcpip_visa(config.M8190A_VISA_ADDR)
    download_to_awg(tx_out,
                    fs=config.AWG_SAMPLE_RATE,
                    vpp=config.AWG_VPP,
                    host=host,
                    port=port,
                    route=config.AWG_OUTPUT_ROUTE)

    return tx_dict




def step4_receive_bitloading(tx_dict: dict,
                             offline: bool = True,
                             rx_file: Path = None,
                             use_nn: bool = False,
                             use_virtual_channel: bool = False,
                             plot_dir: Path = None,
                             run_id: str = None):
    """STEP4: 接收 bitloading 波形并解调/计算 BER/SER."""
    print("\n========== STEP4: Receive Bitloading & Demodulate ==========")
    
    # ✅ 使用 Step 3 生成的 bitloading 波形作为参考
    tx_waveform = tx_dict["tx_waveform_pre"] if tx_dict.get("tx_waveform_pre") is not None else tx_dict["tx_waveform"]

    # ❌ 删除：Step 3 已经下载了 bitloading 波形到 AWG，这里不需要重新下载
    # 如果确实需要重新下载（比如 AWG 被重置了），用下面这段替代：
    """
    if not offline:
        host, port = parse_tcpip_visa(config.M8190A_VISA_ADDR)
        download_to_awg(tx_waveform,
                        fs=config.AWG_SAMPLE_RATE,
                        vpp=config.AWG_VPP,
                        host=host,
                        port=port,
                        route=config.AWG_OUTPUT_ROUTE)
    """

    rx_source = "unknown"
    if use_virtual_channel and offline:
        ch = VirtualChannel(fs=config.AWG_SAMPLE_RATE)
        rx = ch.apply(tx_waveform)
        rx_source = "virtual_channel"
        print("Generated RX via virtual channel")
        _save_virtual_rx(rx, "DMT_bitloading", run_id)
    elif offline:
        try:
            if rx_file is None:
                rx_file = _resolve_offline_rx_file("rawOSC_DMT_*.txt")
            rx = load_txt(rx_file)
            rx_source = "measured_offline"
            print(f"Loaded offline RX data from {rx_file}")
        except FileNotFoundError:
            print("[WARN] Offline bitloading RX file not found, falling back to virtual channel")
            ch = VirtualChannel(fs=config.AWG_SAMPLE_RATE)
            rx = ch.apply(tx_waveform)
            rx_source = "virtual_channel"
            print("Generated RX via virtual channel")
            _save_virtual_rx(rx, "DMT_bitloading", run_id)
    else:
        rx_source = "measured_online"
        print(f"[SCOPE] Capturing bitloading RX from {config.OSC_VISA_ADDR}, channel {config.OSC_CHANNEL}")
        with KeysightScopeUSB(resource=config.OSC_VISA_ADDR) as scope:
            rx, _ = scope.capture(channel=config.OSC_CHANNEL,
                                  sample_rate=config.OSC_SAMPLE_RATE,
                                  timebase_scale=60e-6,
                                  resample_to_awg=True)
        count = 0
        if config.COUNT_FILE.exists():
            count = int(load_txt(config.COUNT_FILE))
        rx_file = config.RXDATA_DIR / f"rawOSC_DMT_{count}.txt"
        save_txt(rx_file, rx)
        cfg_path = save_rx_config(
            rx_file,
            run_id=run_id,
            stage="bitloading",
            awg_sample_rate=config.AWG_SAMPLE_RATE,
            osc_sample_rate=config.OSC_SAMPLE_RATE,
            awg_vpp=config.AWG_VPP,
            awg_output_route=config.AWG_OUTPUT_ROUTE,
            scope_channel=config.OSC_CHANNEL,
            timebase_scale=60e-6,
            carrierno=config.CARRIERNO,
            zeropad1=config.ZEROPAD1,
            datano_bpl=config.DATANO_BPL,
            trainingno=config.TRAININGNO,
            pilot_pattern=config.PILOT_PATTERN,
            pre_equ_flag=config.PRE_EQU_FLAG,
            use_nn=use_nn,
            rx_length=len(rx),
            count=count,
        )
        print(f"Saved RX config to {cfg_path}")
        save_txt(config.COUNT_FILE, np.array([count + 1]), fmt="%d")

    # 同步
    rx_sync = sync_waveform(rx, tx_waveform)
    print(f"RX synced length = {len(rx_sync)}")

    # 可选 NN 后均衡
    if use_nn:
        print("Running NN post-equalizer...")
        try:
            rx_sync = run_nn_equalizer(tx_waveform.ravel(), rx_sync.ravel())
            print(f"NN output length = {len(rx_sync)}")
        except Exception as e:
            print(f"NN post-equalizer failed: {e}. Continuing without NN.")

    # 解调
    res = dmt_receiver(rx_sync, tx_dict, cfg=get_cfg())
    mean_snr = np.nanmean(res["SNR_R"])
    print(f"Estimated BER = {res['ber']:.6e}")
    print(f"Estimated SER = {res['ser']:.6e}")
    print(f"Mean recovered SNR (linear) = {mean_snr:.4f}")

    # 保存结果
    save_txt(config.DATA_DIR / "SNR_R_recovered.txt", res["SNR_R"])
    save_txt(config.DATA_DIR / "ber_estimated.txt", np.array([res["ber"]]))
    save_txt(config.DATA_DIR / "ser_estimated.txt", np.array([res["ser"]]))

    # 画图
    if config.PLOT_SHOW:
        plot_spectrum(rx_sync,
                      config.AWG_SAMPLE_RATE,
                      f"Bitloading RX Spectrum [{rx_source}] ({run_id})",
                      run_id, "DMT_bitloading_Rx_spec")
        plot_tx_rx_nonlinearity(tx_waveform,
                                rx_sync,
                                f"Bitloading TX-RX Nonlinearity [{rx_source}] ({run_id})",
                                run_id, "DMT_bitloading_nonlinearity")
        plot_snrs(load_txt(config.FINAL_SNR_QPSK),
                  res["SNR_R"],
                  f"Estimated vs Recovered SNR (dB) [{rx_source}] ({run_id})",
                  run_id, "SNR_compare")
        plot_ser_ber_per_carrier(res["ser_per_carrier"],
                                 res["ber_per_carrier"],
                                 RQ=tx_dict.get("RQ"),
                                 title=f"SER/BER per Subcarrier [{rx_source}] ({run_id})",
                                 run_id=run_id,
                                 name="ser_ber_per_carrier")
        plot_constellation_density(res["out2"],
                                   res["in_ref"],
                                   tx_dict["RQ"],
                                   pilot_mask=res["pilot_mask"],
                                   title=f"Constellation Density [{rx_source}] ({run_id})",
                                   run_id=run_id,
                                   name="constellation_density")
        plot_constellation_by_order(res["out2"],
                                    res["in_ref"],
                                    tx_dict["RQ"],
                                    pilot_mask=res["pilot_mask"],
                                    title=f"RX Constellation by Order [{rx_source}] ({run_id})",
                                    run_id=run_id,
                                    name="constellation_by_order")

    return res
def run_full_pipeline(offline: bool = False,
                      use_awg: bool = False,
                      use_nn: bool = False,
                      use_virtual_channel: bool = False,
                      qpsk_rx_file: Path = None,
                      bpl_rx_file: Path = None):
    """运行完整 DMT 流程."""
    # 初始化计数器与本次测试唯一编号
    if not config.COUNT_FILE.exists():
        save_txt(config.COUNT_FILE, np.array([0]), fmt="%d")
    run_id = generate_run_id()
    plot_dir = None
    if config.PLOT_SAVE:
        plot_dir = config.PLOT_DIR / run_id
        plot_dir.mkdir(parents=True, exist_ok=True)
    mode_str = "OFFLINE" if offline else "ONLINE"
    print(f"\n========== DMT Pipeline [{mode_str}] ==========")
    print(f">>> Run ID: {run_id}")
    print(f">>> Mode: {mode_str} (offline={offline}, use_awg={use_awg}, use_nn={use_nn}, use_virtual_channel={use_virtual_channel})")
    if config.PLOT_SHOW:
        print(f">>> Matplotlib backend: {matplotlib.get_backend()}")
    assets_dir = CODEPLOT_DIR / run_id
    if plot_dir is not None:
        print(f">>> PNG plots will be saved to: {plot_dir}")
    print(f">>> CodePlot v5 assets will be saved to: {assets_dir}")

    # STEP1 + STEP2
    tx_qpsk = step1_generate_qpsk_tx(use_awg=use_awg,
                                     plot_dir=plot_dir,
                                     run_id=run_id)
    snrs, rx_qpsk_sync = step2_receive_qpsk(tx_qpsk,
                                            offline=offline,
                                            rx_file=qpsk_rx_file,
                                            use_virtual_channel=use_virtual_channel,
                                            plot_dir=plot_dir,
                                            run_id=run_id)

    # STEP3 + STEP4
    tx_bpl = step3_generate_bitloading_tx(snrs,
                                          constellation=config.CONSTELLATION_QAM,
                                          use_awg=use_awg,
                                          plot_dir=plot_dir,
                                          run_id=run_id)
    res = step4_receive_bitloading(tx_bpl,
                                   offline=offline,
                                   rx_file=bpl_rx_file,
                                   use_nn=use_nn,
                                   use_virtual_channel=use_virtual_channel,
                                   plot_dir=plot_dir,
                                   run_id=run_id)

    # 保存传输记录
    mean_recovered_snr = np.nanmean(res["SNR_R"])
    assets_dir = CODEPLOT_DIR / run_id
    record = {
        "pilot_pattern": config.PILOT_PATTERN,
        "use_virtual_channel": bool(use_virtual_channel),
        "virtual_channel_fc": config.VIRTUAL_CHANNEL_FC,
        "virtual_channel_snr_db": config.VIRTUAL_CHANNEL_SNR_DB,
        "virtual_channel_nonlinearity": config.VIRTUAL_CHANNEL_NONLINEARITY,
        "virtual_channel_delay": config.VIRTUAL_CHANNEL_DELAY,
        "virtual_channel_attenuation": config.VIRTUAL_CHANNEL_ATTENUATION,
        "estimated_rate_gbps": float(tx_bpl.get("datarate_gbps", 0.0)),
        "final_rate_gbps": float(tx_bpl.get("datarate_gbps", 0.0)),
        "final_ber": float(res["ber"]),
        "final_ser": float(res["ser"]),
        "mean_recovered_snr_db": float(10 * np.log10(mean_recovered_snr)),
        "ratio": int(tx_bpl.get("ratio", 100)),
        "use_nn": bool(use_nn),
        "plot_dir": str(plot_dir) if plot_dir is not None else None,
        "codeplot_assets_dir": str(assets_dir),
    }
    json_path = save_record(run_id, record)
    print(f"Saved transmission record to {json_path}")
    if config.PLOT_SHOW:
        print(f"CodePlot v5 assets saved to {assets_dir}")

    print("\n========== Pipeline Complete ==========")
    print(f"Run ID: {run_id}")
    print(f"Final estimated BER = {res['ber']:.6e}")
    print(f"Final estimated SER = {res['ser']:.6e}")
    return res


def main():
    parser = argparse.ArgumentParser(description="DMT Python Pipeline")
    parser.add_argument("--offline", type=int, default=0,
                        help="1=offline (read files), 0=online AWG+Scope (default: 0)")
    parser.add_argument("--use-awg", type=int, default=0,
                        help="1=download waveform to M8190A")
    parser.add_argument("--use-nn", type=int, default=config.USE_NN,
                        help="1=use ZY_BiGRU_GPU NN post-equalizer (default follows POSTEQ_FLAG)")
    parser.add_argument("--use-virtual-channel", type=int,
                        default=config.USE_VIRTUAL_CHANNEL,
                        help="1=offline mode uses virtual channel instead of reading files")
    parser.add_argument("--qpsk-rx", type=str, default=None,
                        help="Path to offline QPSK RX file")
    parser.add_argument("--bpl-rx", type=str, default=None,
                        help="Path to offline bitloading RX file")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "step1", "step2", "step3", "step4"],
                        help="Run specific step or full pipeline")
    args = parser.parse_args()

    offline = bool(args.offline)
    use_awg = bool(args.use_awg)
    use_nn = bool(args.use_nn)
    use_virtual_channel = bool(args.use_virtual_channel)
    qpsk_rx = Path(args.qpsk_rx) if args.qpsk_rx else None
    bpl_rx = Path(args.bpl_rx) if args.bpl_rx else None

    if not config.COUNT_FILE.exists():
        save_txt(config.COUNT_FILE, np.array([0]), fmt="%d")

    # 单步运行时不生成整套记录，只画图
    run_id = generate_run_id()
    plot_dir = None
    if config.PLOT_SAVE:
        plot_dir = config.PLOT_DIR / run_id
        plot_dir.mkdir(parents=True, exist_ok=True)

    mode_str = "OFFLINE" if offline else "ONLINE"
    print(f"\n========== DMT Pipeline [{mode_str}] ==========")

    if args.step == "all":
        run_full_pipeline(offline=offline,
                          use_awg=use_awg,
                          use_nn=use_nn,
                          use_virtual_channel=use_virtual_channel,
                          qpsk_rx_file=qpsk_rx,
                          bpl_rx_file=bpl_rx)
    elif args.step == "step1":
        step1_generate_qpsk_tx(use_awg=use_awg, plot_dir=plot_dir, run_id=run_id)
    elif args.step == "step2":
        tx_qpsk = generate_qpsk_tx(datano=config.DATANO_QPSK, cfg=get_cfg())
        step2_receive_qpsk(tx_qpsk,
                           offline=offline,
                           rx_file=qpsk_rx,
                           use_virtual_channel=use_virtual_channel,
                           plot_dir=plot_dir,
                           run_id=run_id)
    elif args.step == "step3":
        snrs = load_txt(config.FINAL_SNR_QPSK)
        step3_generate_bitloading_tx(snrs,
                                     constellation=config.CONSTELLATION_QAM,
                                     use_awg=use_awg,
                                     plot_dir=plot_dir,
                                     run_id=run_id)
    elif args.step == "step4":
        tx_bpl = generate_bitloading_tx(load_txt(config.FINAL_SNR_QPSK),
                                        constellation=config.CONSTELLATION_QAM,
                                        datano=config.DATANO_BPL,
                                        cfg=get_cfg())
        step4_receive_bitloading(tx_bpl,
                                 offline=offline,
                                 rx_file=bpl_rx,
                                 use_nn=use_nn,
                                 use_virtual_channel=use_virtual_channel,
                                 plot_dir=plot_dir,
                                 run_id=run_id)


if __name__ == "__main__":
    main()
