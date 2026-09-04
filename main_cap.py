"""UW_APSK CAP 收发机的命令行入口。

提供两种模式：
  1. 单带 CAP（等效于 oldcapAPSKTxRx20220406.m）
  2. 多带 CAP（等效于 main_CAP_3band_totalB.m）
"""
import argparse
from pathlib import Path

import numpy as np

import cap_core
import cap_rx
import channel
import config_cap as cfg
from constellation import average_power, demodulate, modulate
from equalizer import lms_equalizer, lms_volterra_equalizer
from nn_cap_equalizer import run_cap_nn_equalizer
from record import generate_run_id, save_record
from utils import load_txt, save_txt


def _align_lms_output(eq_output: np.ndarray, rx_input: np.ndarray, taps: int) -> np.ndarray:
    """提取与 rx_input 对齐的有效 LMS 中央输出段。

    LMS 均衡器会在头部和尾部用原始输入样本填充；真正的均衡后样本是
    中央的 ``len(rx_input) - taps + 1`` 个样本。
    """
    head = (taps - 1) // 2
    mm = len(rx_input) - taps + 1
    return eq_output[head : head + mm]


def run_singleband(
    numofsymbols: int = cfg.SB_NUMOFSYMBOLS,
    order: int = cfg.SB_QAMORDER,
    constellation: str = cfg.SB_CONSTELLATION,
    snr_db: float = cfg.SB_SNR_DB,
    seed: int = 100,
) -> dict:
    """运行单带 CAP 收发机离线仿真。"""
    np.random.seed(seed)

    # ---- 发射（TX）----
    dec_data = np.random.randint(0, order, size=numofsymbols)
    qam_data = modulate(dec_data, order, constellation)

    tx_signal, filter_I, filter_Q, t = cap_core.generate_singleband_cap(
        symbols=qam_data,
        fs=cfg.SB_AWG_SAMPLE_RATE,
        Rs=cfg.SB_SYMBOL_RATE,
        rolloff=cfg.SB_ALPHA,
        subcar=cfg.SB_SUBCAR,
        start_freq=cfg.SB_STARTFREQ,
        taps=cfg.SB_TAPS,
    )

    # 保存与 MATLAB 接收机兼容的发送数据文件
    name = f"data{order}{constellation}"
    save_txt(cfg.TXDATA_DIR / f"{name}.txt", tx_signal)

    # ---- 信道 ----
    if cfg.USE_VIRTUAL_CHANNEL:
        rx_signal = channel.vlc_channel(
            tx_signal,
            snr_db=snr_db,
            fs_hz=cfg.SB_CHANNEL_FS,
            factor=cfg.SB_CHANNEL_FACTOR,
            nonlinear=cfg.SB_CHANNEL_NONLINEAR,
        )
    else:
        rx_signal = load_txt(cfg.RXDATA_DIR / f"OSC_{name}.txt")

    # ---- 预均衡：波形 Volterra 均衡 ----
    tx_norm = tx_signal / np.sqrt(np.mean(tx_signal ** 2))
    rx_eq, _, _, _, _, _ = lms_volterra_equalizer(
        rx_signal,
        tx_norm,
        19,
        0.015,
        11,
        0.0004,
        cfg.SB_NUMOF_TS,
    )

    # ---- 匹配滤波 ----
    DataCapI = np.convolve(rx_eq, filter_I, mode="same")
    DataCapQ = np.convolve(rx_eq, filter_Q, mode="same")
    DataCap = DataCapI + 1j * DataCapQ
    match_data = DataCap[::cfg.SB_UPSAMPLENO]

    # ---- 后级 LMS 均衡 ----
    eq_data, _, _, _ = lms_equalizer(
        match_data,
        qam_data,
        cfg.SB_LMS_TAPS,
        cfg.SB_LMS_MU,
        cfg.SB_NUMOF_TS,
    )
    eq_valid = _align_lms_output(eq_data, match_data, cfg.SB_LMS_TAPS)

    avp = average_power(order, constellation)
    eq_valid = eq_valid / np.sqrt(np.mean(np.abs(eq_valid) ** 2)) * avp

    # ---- 解调 ----
    head = (cfg.SB_LMS_TAPS - 1) // 2
    mm = len(match_data) - cfg.SB_LMS_TAPS + 1
    decisions = demodulate(eq_valid, order, constellation)
    tx_valid = dec_data[head : head + mm]
    ser = float(np.mean(decisions != tx_valid))
    bits_per_sym = int(np.log2(order))
    ber = float(np.sum(decisions != tx_valid) * bits_per_sym / (len(tx_valid) * bits_per_sym))

    record = {
        "mode": "singleband",
        "order": order,
        "constellation": constellation,
        "snr_db": snr_db,
        "ser": ser,
        "ber": ber,
    }
    print(f"单带 CAP：符号误码率 SER={ser:.4e}  误码率 BER={ber:.4e}")
    return record


def run_multiband(
    numofsymbols: int = cfg.MB_NUMOFSYMBOLS,
    order: int = cfg.MB_M,
    constellation: str = cfg.MB_CONSTELLATION,
    snr_db: float = cfg.MB_SNR_DB,
    seed: int = 1,
    use_lms: bool = True,
    use_nn: bool = False,
) -> dict:
    """运行多带 CAP 收发机离线仿真。"""
    rng = np.random.default_rng(seed)

    # ---- 发射（TX）----
    decimal_per_band = [rng.integers(0, order, size=numofsymbols) for _ in range(cfg.MB_NUM_BANDS)]
    symbols_per_band = [modulate(dec, order, constellation) for dec in decimal_per_band]

    tx_signal, fc, gt, t, band_signals = cap_core.generate_multiband_cap(
        symbols_per_band=symbols_per_band,
        Rs=cfg.MB_RS,
        fs=cfg.MB_FS,
        rolloff=cfg.MB_ROLLOFF,
        cf=cfg.MB_CF,
        span=cfg.MB_SPAN,
        shape=cfg.MB_SHAPE,
    )
    save_txt(cfg.TXDATA_DIR / "up123_data_for_dnn.txt", tx_signal)

    # ---- 信道 ----
    if cfg.USE_VIRTUAL_CHANNEL:
        rx_signal = channel.vlc_channel(
            tx_signal,
            snr_db=snr_db,
            fs_hz=cfg.MB_CHANNEL_FS,
            factor=cfg.MB_CHANNEL_FACTOR,
            nonlinear=cfg.MB_CHANNEL_NONLINEAR,
        )
    else:
        rx_signal = load_txt(cfg.RXDATA_DIR / "rx_multiband.txt")

    # ---- 对每个子带做匹配滤波 ----
    upsampleno = int(round(cfg.MB_NUM_BANDS * cfg.MB_FS / cfg.MB_RS))
    taps = cfg.MB_SPAN * upsampleno + 1
    rx_bands = []
    for n in range(cfg.MB_NUM_BANDS):
        band_sym = cap_rx.capmatch_filter(rx_signal, gt, t, fc[n], taps, upsampleno, 0)
        rx_bands.append(band_sym)

    # 将原始接收子带保存为 NN 输入（实部/虚部交错）
    nn_input = np.hstack([np.column_stack([rb.real, rb.imag]) for rb in rx_bands])
    save_txt(cfg.NN_RX1_FILE, nn_input)

    # 保存 NN 训练用的标签
    tx_labels = np.hstack([np.column_stack([s.real, s.imag]) for s in symbols_per_band])
    save_txt(cfg.NN_TX_FILE, tx_labels)
    save_txt(cfg.DATA_DIR / "ydata_for_dnn.txt", tx_labels)

    # ---- 各子带误码率 ----
    raw_sers = []
    raw_bers = []
    eq_sers = []
    eq_bers = []

    lms_taps = 31
    lms_mu = 0.005
    train_len = min(4000, numofsymbols // 2)

    for n in range(cfg.MB_NUM_BANDS):
        # 原始匹配滤波性能
        _, ser_raw, ber_raw = cap_rx.demodulate_with_ber(
            rx_bands[n], order, constellation, symbols_per_band[n]
        )
        raw_sers.append(ser_raw)
        raw_bers.append(ber_raw)

        if use_lms:
            eq_data, _, _, _ = lms_equalizer(
                rx_bands[n],
                symbols_per_band[n],
                lms_taps,
                lms_mu,
                train_len,
            )
            eq_valid = _align_lms_output(eq_data, rx_bands[n], lms_taps)
            head = (lms_taps - 1) // 2
            mm = len(rx_bands[n]) - lms_taps + 1
            decisions = demodulate(eq_valid, order, constellation)
            tx_valid = decimal_per_band[n][head : head + mm]
            ser_eq = float(np.mean(decisions != tx_valid))
            bits_per_sym = int(np.log2(order))
            ber_eq = float(np.sum(decisions != tx_valid) * bits_per_sym / (len(tx_valid) * bits_per_sym))
            eq_sers.append(ser_eq)
            eq_bers.append(ber_eq)

    # ---- NN 后级均衡器（可选） ----
    nn_sers = []
    nn_bers = []
    if use_nn:
        try:
            nn_output = run_cap_nn_equalizer(tx_labels, nn_input)
            for n in range(cfg.MB_NUM_BANDS):
                rx_nn = nn_output[:, 2 * n] + 1j * nn_output[:, 2 * n + 1]
                # 与发送符号对齐长度（NN 因加窗会丢弃头部样本）
                valid_len = min(len(rx_nn), numofsymbols)
                decisions = demodulate(rx_nn[:valid_len], order, constellation)
                tx_valid = decimal_per_band[n][:valid_len]
                ser_nn = float(np.mean(decisions != tx_valid))
                bits_per_sym = int(np.log2(order))
                ber_nn = float(np.sum(decisions != tx_valid) * bits_per_sym / (len(tx_valid) * bits_per_sym))
                nn_sers.append(ser_nn)
                nn_bers.append(ber_nn)
            record["nn_ser"] = nn_sers
            record["nn_ber"] = nn_bers
            record["nn_ber_avg"] = float(np.mean(nn_bers))
        except Exception as exc:
            print(f"NN 均衡器已跳过/失败：{exc}")

    record = {
        "mode": "multiband",
        "order": order,
        "constellation": constellation,
        "snr_db": snr_db,
        "raw_ser": raw_sers,
        "raw_ber": raw_bers,
        "raw_ber_avg": float(np.mean(raw_bers)),
    }
    if use_lms:
        record["eq_ser"] = eq_sers
        record["eq_ber"] = eq_bers
        record["eq_ber_avg"] = float(np.mean(eq_bers))
    if nn_sers:
        record["nn_ser"] = nn_sers
        record["nn_ber"] = nn_bers
        record["nn_ber_avg"] = float(np.mean(nn_bers))

    print(f"多带 CAP 各子带原始 BER：{raw_bers}")
    print(f"多带 CAP 平均原始 BER：{np.mean(raw_bers):.4e}")
    if use_lms:
        print(f"多带 CAP LMS 均衡后各子带 BER：{eq_bers}")
        print(f"多带 CAP LMS 均衡后平均 BER：{np.mean(eq_bers):.4e}")
    if nn_sers:
        print(f"多带 CAP NN 均衡后各子带 BER：{nn_bers}")
        print(f"多带 CAP NN 均衡后平均 BER：{np.mean(nn_bers):.4e}")
    return record


def main():
    parser = argparse.ArgumentParser(description="UW_APSK CAP Python 收发机")
    parser.add_argument("--mode", choices=["singleband", "multiband"], default="singleband")
    parser.add_argument("--order", type=int, default=None)
    parser.add_argument("--constellation", choices=["APSK", "QAM"], default=None)
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--no-virtual", action="store_true", help="从文件读取接收数据，而不是使用虚拟信道")
    parser.add_argument("--no-lms", action="store_true", help="在多带模式下禁用各子带 LMS 均衡")
    parser.add_argument("--use-nn", action="store_true", help="在多带模式下运行 CAP NN 后级均衡器（需要 torch）")
    args = parser.parse_args()

    if args.no_virtual:
        cfg.USE_VIRTUAL_CHANNEL = 0

    run_id = generate_run_id()
    if args.mode == "singleband":
        order = args.order or cfg.SB_QAMORDER
        constellation = args.constellation or cfg.SB_CONSTELLATION
        snr = args.snr if args.snr is not None else cfg.SB_SNR_DB
        record = run_singleband(
            numofsymbols=cfg.SB_NUMOFSYMBOLS,
            order=order,
            constellation=constellation,
            snr_db=snr,
            seed=args.seed,
        )
    else:
        order = args.order or cfg.MB_M
        constellation = args.constellation or cfg.MB_CONSTELLATION
        snr = args.snr if args.snr is not None else cfg.MB_SNR_DB
        record = run_multiband(
            numofsymbols=cfg.MB_NUMOFSYMBOLS,
            order=order,
            constellation=constellation,
            snr_db=snr,
            seed=args.seed,
            use_lms=not args.no_lms,
            use_nn=args.use_nn,
        )

    save_record(run_id, record, cfg.RECORD_DIR)


if __name__ == "__main__":
    main()
