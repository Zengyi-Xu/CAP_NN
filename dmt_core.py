"""DMT 核心算法：调制、解调、bitloading、SNR/BER 估计."""
import numpy as np
from scipy.special import erfc
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit
from pathlib import Path
from typing import Tuple, Optional

from utils import load_txt, save_txt, save_mat, load_mat, add_dummy, center_normalize
from utils import apply_hardware_preEQ
import config


# -----------------------------------------------------------------------------
# 星座映射
# -----------------------------------------------------------------------------
def load_constellation(order: int, constellation: str, data_dir: Path = config.DATA_DIR) -> np.ndarray:
    """读取 goodGS{order}QAM{constellation}.txt 的星座点."""
    fname = data_dir / f"goodGS{order}QAM{constellation}.txt"
    tab = load_txt(fname)
    if tab.ndim == 1:
        tab = tab.reshape(-1, 2)
    return tab[:, 0] + 1j * tab[:, 1]


def qam_modulate(data_decimal: np.ndarray, order: int, constellation: str,
                 data_dir: Path = config.DATA_DIR) -> np.ndarray:
    """将十进制符号映射到星座点."""
    cons = load_constellation(order, constellation, data_dir)
    return cons[np.asarray(data_decimal, dtype=int).ravel()]


def qam_demodulate(sig: np.ndarray, order: int, constellation: str,
                   data_dir: Path = config.DATA_DIR) -> np.ndarray:
    """最小欧氏距离解调."""
    cons = load_constellation(order, constellation, data_dir)
    sig_flat = np.asarray(sig).ravel()
    idx = np.argmin(np.abs(sig_flat[:, None] - cons[None, :]), axis=1)
    return idx


# -----------------------------------------------------------------------------
# 比特加载
# -----------------------------------------------------------------------------
def load_snr_table(path: Path) -> np.ndarray:
    """读取 SNR 门限表.

    支持两种格式：
    - 单列：视为 order=1..N 对应的 SNR 门限
    - 两列：[order, snr]
    """
    tab = load_txt(path)
    if tab.ndim == 1:
        tab = np.column_stack([np.arange(1, len(tab) + 1), tab])
    return tab  # shape (N, 2)


def assign_qam_order_from_snr(snrs: np.ndarray, snr_table: np.ndarray,
                              b_max: int = 10) -> np.ndarray:
    """就低不就高地为每个子载波分配 QAM 阶数.

    Args:
        snrs: 每个子载波的 SNR（线性值）
        snr_table: 两列 [order, snr_threshold]
        b_max: 最大允许阶数

    Returns:
        每个子载波的调制阶数（比特/符号）
    """
    snrs = np.asarray(snrs).ravel()
    orders = snr_table[:, 0].astype(int)
    thresholds = snr_table[:, 1]
    out = np.zeros(len(snrs), dtype=int)
    for i, s in enumerate(snrs):
        diff = thresholds - s
        idx = np.argmin(np.abs(diff))
        if diff[idx] > 0:
            chosen = orders[idx]
        else:
            chosen = orders[idx - 1] if idx > 0 else 0
        out[i] = min(chosen, b_max)
    return out


def _truncate_bits(b: np.ndarray, b_max: int, b_min: int = 0) -> np.ndarray:
    b = np.asarray(b, dtype=float).copy()
    b[b > b_max] = b_max
    b[b < b_min] = b_min
    return b


def bit_loading_hh(snrs: np.ndarray, qam_order_all: np.ndarray,
                   snr_table: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """BL_myalgo: 自创 HH-like 比特/功率分配.

    Returns:
        S: 功率比例（需开根号后使用）
        RQ: 最终比特分配
        raise_num: 升阶子载波数
    """
    N = len(snrs)
    snrs = np.asarray(snrs).ravel().astype(float)
    qam_order_all = np.asarray(qam_order_all).ravel().astype(int)
    qam_order_all[qam_order_all == 10] = 9

    e_up = np.zeros(N)
    e_down = np.zeros(N)
    for n in range(N):
        b = qam_order_all[n]
        if b == 0:
            e_up[n] = snr_table[0, 1] / snrs[n]
            e_down[n] = 0.0
        else:
            e_up[n] = snr_table[b, 1] / snrs[n] if b < len(snr_table) else 1e6
            e_down[n] = snr_table[b - 1, 1] / snrs[n] if b - 1 >= 0 else 0.0

    order_up = np.argsort(e_up)
    order_down = np.argsort(e_down)

    E_total = np.zeros(N)
    for n in range(N):
        raise_num = n + 1
        down_num = np.setdiff1d(order_down, order_up[:raise_num])
        E_total[n] = e_up[order_up[:raise_num]].sum() + e_down[down_num].sum()

    m_idx = np.argmin(np.abs(E_total - N))
    raise_num = m_idx + 1 if E_total[m_idx] - N <= 0 else m_idx
    if raise_num < 1:
        raise_num = 1

    down_num = np.setdiff1d(order_down, order_up[:raise_num])
    index = np.concatenate([order_up[:raise_num], down_num])
    S1 = np.concatenate([e_up[order_up[:raise_num]], e_down[down_num]])
    S = np.ones(N)
    for k, idx in enumerate(index):
        S[idx] = S1[k]
    S[S == 0] = 1e-12
    S = S / S.sum() * N

    RQ = qam_order_all.copy()
    RQ[order_up[:raise_num]] += 1
    return S, RQ, raise_num


def bit_loading_lc(snrs: np.ndarray, qam_order_all: np.ndarray,
                   snr_table: np.ndarray, SE_add: float = 0.0,
                   b_max: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """BL_LC_MM4: 简化 Levin-Campello 比特/功率分配.

    Args:
        snrs: 每个子载波 SNR（线性）
        qam_order_all: 初始阶数
        snr_table: [order, threshold]
        SE_add: 总比特预算调整量（相对于 sum(qam_order_all)）
        b_max: 最大阶数

    Returns:
        S: 功率比例
        RQ: 最终比特分配
    """
    N = len(snrs)
    snrs = np.asarray(snrs).ravel().astype(float)
    qam_order_all = np.asarray(qam_order_all).ravel().astype(int)
    B = qam_order_all.sum() + SE_add

    # 初始阶数
    b_ori = assign_qam_order_from_snr(snrs, snr_table, b_max=b_max)
    b = _truncate_bits(b_ori, b_max)

    if N * b_max < B:
        print("Out of the max capacity, please enhance b_max!")
        return np.ones(N), np.full(N, b_max, dtype=int)

    # 确定全局偏移 i
    if b.sum() > B:
        i = 0
        while b.sum() > B:
            i -= 1
            b = _truncate_bits(b_ori + i, b_max)
        i += 1
    elif b.sum() < B:
        i = 0
        while b.sum() < B:
            i += 1
            b = _truncate_bits(b_ori + i, b_max)
        i -= 1
    else:
        i = 0

    b = _truncate_bits(b_ori + i, b_max)
    if b.sum() > B:
        print("The operation about i is wrong, please check!")
        return np.ones(N), qam_order_all

    # 按 SNR 降序排列
    g, idx_g = np.sort(snrs)[::-1], np.argsort(snrs)[::-1]
    b_sorted = b[idx_g].astype(int)
    lock = b_sorted >= b_max
    N_lock = int(lock.sum())

    e_now = np.zeros(N)
    e_up = np.full(N, 1e6)
    e_down = np.zeros(N)
    delta_e_up = np.zeros(N)
    delta_e_down = np.zeros(N)

    P_r = 0.0
    for n in range(N_lock):
        b_sorted[n] = b_max
        e_now[n] = snr_table[b_max - 1, 1] / g[n]
        e_up[n] = 1e6
        e_down[n] = snr_table[b_max - 2, 1] / g[n] if b_max - 2 >= 0 else 0.0
        delta_e_up[n] = e_up[n] - e_now[n]
        delta_e_down[n] = e_now[n] - e_down[n]
        P_r += 1 - e_now[n]

    for n in range(N_lock, N):
        bb = b_sorted[n]
        e_now[n] = snr_table[bb - 1, 1] / g[n] if bb > 0 else 0.0
        if bb < b_max and bb > 0:
            e_up[n] = snr_table[bb, 1] / g[n]
        elif bb == 0:
            e_up[n] = snr_table[0, 1] / g[n]
        delta_e_up[n] = e_up[n] - e_now[n]
        if bb <= 1:
            e_down[n] = e_now[n]
        else:
            e_down[n] = snr_table[bb - 2, 1] / g[n]
        delta_e_down[n] = e_now[n] - e_down[n]

    incre_num = int(B - b.sum())
    for _ in range(incre_num):
        a = int(np.argmin(delta_e_up))
        if P_r > delta_e_up[a]:
            P_r -= delta_e_up[a]
        b_sorted[a] += 1
        if b_sorted[a] == b_max:
            e_up[a] = 1e6
        else:
            e_up[a] = snr_table[b_sorted[a], 1] / g[a]
        e_now[a] = snr_table[b_sorted[a] - 1, 1] / g[a]
        if b_sorted[a] <= 1:
            e_down[a] = e_now[a]
        else:
            e_down[a] = snr_table[b_sorted[a] - 2, 1] / g[a]
        delta_e_up[a] = e_up[a] - e_now[a]
        delta_e_down[a] = e_now[a] - e_down[a]

    # 输出按原始顺序
    S = np.zeros(N)
    RQ = np.zeros(N, dtype=int)
    S[idx_g] = e_now
    RQ[idx_g] = b_sorted

    # 诊断
    if delta_e_down.max() > delta_e_up.min():
        print("Levin Campello is Not Efficient!")
    else:
        print("Levin Campello is Efficient!")
    if RQ.sum() == B:
        print("Levin Campello is B-tight!")
    else:
        print("Levin Campello is not B-tight!")
    print(f"Total Power budget of Levin Campello is {S.sum():.4f}")
    return S, RQ


# -----------------------------------------------------------------------------
# 导频图案
# -----------------------------------------------------------------------------
def create_pilot_mask(carrierno1: int,
                      datano: int,
                      pattern: str = config.PILOT_PATTERN,
                      **kwargs) -> np.ndarray:
    """创建导频位置掩码.

    Args:
        carrierno1: 有效子载波数
        datano: 符号数
        pattern: "training_only" | "comb" | "mesh"
        kwargs: 可覆盖 config 中的导频参数

    Returns:
        mask: bool 矩阵 (carrierno1, datano)，True 表示导频位置
    """
    mask = np.zeros((carrierno1, datano), dtype=bool)
    if pattern == "training_only" or pattern is None:
        return mask

    if pattern == "comb":
        start = kwargs.get("comb_start", config.PILOT_COMB_START)
        spacing = kwargs.get("comb_spacing", config.PILOT_COMB_SPACING)
        pilot_sc = np.arange(start, carrierno1, spacing)
        mask[pilot_sc, :] = True

    elif pattern == "mesh":
        f_start = kwargs.get("mesh_start_freq", config.PILOT_MESH_START_FREQ)
        f_space = kwargs.get("mesh_freq_spacing", config.PILOT_MESH_FREQ_SPACING)
        t_start = kwargs.get("mesh_start_time", config.PILOT_MESH_START_TIME)
        t_space = kwargs.get("mesh_time_spacing", config.PILOT_MESH_TIME_SPACING)
        pilot_sc = np.arange(f_start, carrierno1, f_space)
        pilot_sym = np.arange(t_start, datano, t_space)
        for sc in pilot_sc:
            mask[sc, pilot_sym] = True

    else:
        raise ValueError(f"Unknown pilot pattern: {pattern}")

    return mask


def insert_pilots(qamdata: np.ndarray,
                  AVT: np.ndarray,
                  origin_dec_data: np.ndarray,
                  mask: np.ndarray,
                  pilot_value: complex = config.PILOT_VALUE) -> tuple:
    """在指定位置插入导频.

    Returns:
        qamdata, AVT, origin_dec_data (均已修改)
    """
    if not mask.any():
        return qamdata, AVT, origin_dec_data
    qamdata = qamdata.copy()
    AVT = AVT.copy()
    origin_dec_data = origin_dec_data.copy()
    qamdata[mask] = pilot_value
    AVT[mask] = 1.0
    origin_dec_data[mask] = -1
    return qamdata, AVT, origin_dec_data


def _interp2d(pilot_sc: np.ndarray,
              pilot_sym: np.ndarray,
              values: np.ndarray,
              carrierno1: int,
              datano: int,
              fill_value: float = 0.0) -> np.ndarray:
    """使用 scipy.interpolate.griddata 做二维线性插值，边界用最近邻填充."""
    from scipy.interpolate import griddata
    grid_sc, grid_sym = np.meshgrid(np.arange(carrierno1), np.arange(datano), indexing="ij")
    interp = griddata((pilot_sc, pilot_sym), values, (grid_sc, grid_sym),
                      method="linear", fill_value=np.nan)
    # 对落在凸包外的点使用最近邻填充
    nan_mask = np.isnan(interp)
    if np.any(nan_mask):
        interp[nan_mask] = griddata((pilot_sc, pilot_sym), values,
                                    (grid_sc[nan_mask], grid_sym[nan_mask]),
                                    method="nearest")
    # 最终兜底
    interp = np.nan_to_num(interp, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return interp


def estimate_channel_from_pilots(rv_down: np.ndarray,
                                 qamdata: np.ndarray,
                                 mask: np.ndarray) -> np.ndarray:
    """利用导频估计每子载波/每符号的信道响应 H.

    - 梳状导频：每符号都有导频，采用一维频率插值（更快）
    - 网状导频：稀疏分布，采用二维时频插值

    Args:
        rv_down: 接收频域符号 (carrierno1, datano)
        qamdata: 发送参考符号 (含导频)
        mask: 导频掩码

    Returns:
        H: 信道响应矩阵 (carrierno1, datano)
    """
    carrierno1, datano = rv_down.shape
    all_freqs = np.arange(carrierno1)

    # 判断是否为梳状（每列至少有一个导频）
    is_comb_like = np.all(mask.any(axis=0))

    if is_comb_like:
        H = np.ones((carrierno1, datano), dtype=complex)
        for t in range(datano):
            pilot_freqs = np.where(mask[:, t])[0]
            if len(pilot_freqs) == 0:
                continue
            H_pilot = qamdata[pilot_freqs, t] / rv_down[pilot_freqs, t]
            if len(pilot_freqs) == 1:
                H[:, t] = H_pilot[0]
            else:
                H_real = np.interp(all_freqs, pilot_freqs, H_pilot.real)
                H_imag = np.interp(all_freqs, pilot_freqs, H_pilot.imag)
                H[:, t] = H_real + 1j * H_imag
        return H
    else:
        # 网状：二维插值
        pilot_sc, pilot_sym = np.where(mask)
        H_pilot = qamdata[pilot_sc, pilot_sym] / rv_down[pilot_sc, pilot_sym]
        H_real = _interp2d(pilot_sc, pilot_sym, H_pilot.real, carrierno1, datano, fill_value=1.0)
        H_imag = _interp2d(pilot_sc, pilot_sym, H_pilot.imag, carrierno1, datano, fill_value=0.0)
        return H_real + 1j * H_imag


def phase_recovery_from_pilots(Rx: np.ndarray,
                               Tx: np.ndarray,
                               mask: np.ndarray) -> tuple:
    """利用导频做相位恢复.

    - 梳状导频：每符号一维频率插值
    - 网状导频：二维时频插值
    """
    carrierno1, datano = Rx.shape
    all_freqs = np.arange(carrierno1)
    is_comb_like = np.all(mask.any(axis=0))

    if is_comb_like:
        phase = np.zeros((carrierno1, datano))
        for t in range(datano):
            pilot_freqs = np.where(mask[:, t])[0]
            if len(pilot_freqs) == 0:
                continue
            phase_pilot = np.angle(Rx[pilot_freqs, t] / Tx[pilot_freqs, t])
            if len(pilot_freqs) == 1:
                phase[:, t] = phase_pilot[0]
            else:
                phase[:, t] = np.interp(all_freqs, pilot_freqs, phase_pilot)
    else:
        pilot_sc, pilot_sym = np.where(mask)
        phase_pilot = np.angle(Rx[pilot_sc, pilot_sym] / Tx[pilot_sc, pilot_sym])
        phase = _interp2d(pilot_sc, pilot_sym, phase_pilot, carrierno1, datano, fill_value=0.0)

    Rx_recovery = Rx * np.exp(-1j * phase)
    return Rx_recovery, phase


# -----------------------------------------------------------------------------
# 预均衡权重生成 (port from MATLAB Pre.m, 输出 th7.txt)
# -----------------------------------------------------------------------------
def _exp2_fit(y: np.ndarray) -> np.ndarray:
    """对序列做双指数拟合 a*exp(b*x)+c*exp(d*x) (等效 MATLAB cftool 'exp2').

    与 createFit.m 一致: x 取 1..N。若拟合失败则退化为 3 次平滑样条。

    注: Pre.m 中 method 1-4 使用的 createFit_VLC / createFit_VLC_inverse
    原始文件已不可考, 这里统一用相同的 exp2 模型等效。
    """
    from scipy.interpolate import UnivariateSpline
    y = np.asarray(y, dtype=float).ravel()
    x = np.arange(1, len(y) + 1, dtype=float)

    def exp2(x, a, b, c, d):
        return a * np.exp(b * x) + c * np.exp(d * x)

    scale = float(np.max(np.abs(y))) or 1.0
    try:
        popt, _ = curve_fit(exp2, x, y,
                            p0=[scale, -5e-3, scale / 2, -7e-4],
                            maxfev=20000)
        return exp2(x, *popt)
    except Exception:
        spl = UnivariateSpline(x, y, k=3, s=len(y) * np.var(y) * 0.1)
        return spl(x)


def bridge_t_ii_1(f_center: float, f_half: float, decay_max_db: float,
                  r_ref: float = 50.0) -> Tuple[float, float, float, float, float, float]:
    """桥 T 均衡器 II 型设计 (port from MATLAB BridgeT_II_1.m).

    Args:
        f_center: 最小衰减中心频率 (Hz)
        f_half: 半衰减带宽 (Hz)
        decay_max_db: 最大衰减 (dB)
        r_ref: 参考阻抗 (Ohm)

    Returns:
        (C11, L11, R11, C22, L22, R22), 其中 C 单位为 pF, L 单位为 nH (与 MATLAB 一致)
    """
    Fref = 1e6  # 归一化参考频率 1 MHz
    Fcen_norm = f_center / Fref
    Fhalf_norm = f_half / Fref

    Decay_Line = 10 ** (decay_max_db / 10)
    Decay_Line_half = 10 ** (decay_max_db / 2 / 10)
    Decay_nat = 0.5 * np.log(Decay_Line)
    Decay_nat_half = 0.5 * np.log(Decay_Line_half)
    Fm = np.exp(2 * Decay_nat)
    Fhalf_m = np.exp(2 * Decay_nat_half)

    A2 = 1 / Fcen_norm ** 2
    y_half = -np.sqrt((Fhalf_m - 1) / (Fm - 1))
    A1 = (A2 - (1 / Fhalf_norm) ** 2) * Fhalf_norm / y_half
    r11 = np.exp(Decay_nat) - 1
    r21 = 1 / r11
    alpha11 = (A2 / A1) * r11   # a11 = b21
    beta11 = A1 / r11           # b11 = a21

    L0 = r_ref / (2 * np.pi * Fref)
    C0 = 1 / (2 * np.pi * Fref * r_ref)
    C11 = beta11 * C0 * 1e12    # pF
    L11 = alpha11 * L0 * 1e9    # nH
    C22 = alpha11 * C0 * 1e12   # pF
    L22 = beta11 * L0 * 1e9     # nH
    R11 = r11 * r_ref
    R22 = r21 * r_ref
    return C11, L11, R11, C22, L22, R22


def hardware_pre_response_db(fbegin: int = config.HW_PRE_FBEGIN,
                             adb: float = config.HW_PRE_ADB,
                             fcen_mhz: float = config.HW_PRE_FCEN_MHZ,
                             fhalf_mhz: float = config.HW_PRE_FHALF_MHZ,
                             fend: int = config.HW_PRE_FEND,
                             r0: float = config.HW_PRE_R0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算桥 T 硬件预均衡的 dB 响应 (port from Pre.m case 5).

    Returns:
        f_mhz: 1..1000 MHz 频点
        b_log: 全频段 (1 MHz-1 GHz, 1 MHz 步进) 的衰减响应 (dB, 负值)
        f_use: 截取的有效频段 b_log(fbegin : fbegin+fend) (MATLAB 1 基索引)
    """
    C11, L11, R11, _, _, _ = bridge_t_ii_1(fcen_mhz * 1e6, fhalf_mhz * 1e6, adb, r0)
    L_H = L11 * 1e-9   # nH -> H
    C_F = C11 * 1e-12  # pF -> F

    f = np.arange(1e6, 1e9 + 1, 1e6)  # 1M:1M:1G
    w = 2 * np.pi * f
    x11 = 1j * w * L_H + 1 / (1j * w * C_F)   # 串联 LC 阻抗
    z11 = R11 * x11 / (R11 + x11)             # R11 与串联 LC 并联
    b_line = np.abs((1 + z11 / r0) ** 2)
    b_log = -10 * np.log10(b_line)
    # MATLAB: f_use = b_log(Fbegin : Fbegin+Fend)  (1 基索引, 共 Fend+1 点)
    f_use = b_log[fbegin - 1: fbegin + fend]
    return f / 1e6, b_log, f_use


def generate_preemphasis_weights(channel_mag: Optional[np.ndarray] = None,
                                 method: int = config.PRE_METHOD,
                                 equal_db: float = config.PRE_EQUAL_DB,
                                 equal_db2: float = config.PRE_EQUAL_DB2,
                                 hw_params: Optional[dict] = None,
                                 n_subcarriers: Optional[int] = None,
                                 save_path: Path = config.TH7_FILE,
                                 save_curves: bool = True) -> np.ndarray:
    """生成每个子载波的预均衡幅度权重并保存到 th7.txt (port from MATLAB Pre.m).

    Args:
        channel_mag: 信道幅度响应 |ha| (如 dmt_receiver 输出 channel_response 的幅度;
                     多通道测量时可先求和, 对应 Pre.m 中 temp=ha2+ha3+ha4)。
                     method 0-4 必须提供; method 5 仅用其长度, 缺省时用 n_subcarriers。
        method: 0=normal fit; 1=inverse+normal; 2=cut off; 3=peak point;
                4=peak point fit; 5=Hardware Pre (桥T均衡器响应)
        equal_db: 截止/分段门限 (dB)
        equal_db2: 第二门限 (仅 method=4)
        hw_params: method=5 的硬件参数, 可覆盖 config 中 HW_PRE_* 键:
                   fbegin, adb, fcen_mhz, fhalf_mhz, fend, r0
        n_subcarriers: 未提供 channel_mag 时的权重长度 (默认 config.CARRIERNO1)
        save_path: th7.txt 保存路径
        save_curves: method=5 时是否同时保存 f_grid.txt / f_hardware_dB.txt

    Returns:
        temp3: 长度等于子载波数的线性幅度权重 (与 th7.txt 内容一致)
    """
    if channel_mag is not None:
        temp = np.asarray(channel_mag, dtype=float).ravel()
    else:
        n = n_subcarriers or config.CARRIERNO1
        temp = np.ones(n)  # 仅占位, method=5 只需长度
    N = len(temp)

    hw = {
        "fbegin": config.HW_PRE_FBEGIN,
        "adb": config.HW_PRE_ADB,
        "fcen_mhz": config.HW_PRE_FCEN_MHZ,
        "fhalf_mhz": config.HW_PRE_FHALF_MHZ,
        "fend": config.HW_PRE_FEND,
        "r0": config.HW_PRE_R0,
    }
    hw.update(hw_params or {})

    def _inverse_db(v: np.ndarray, ref: str = "first") -> np.ndarray:
        d = 20 * np.log10(1.0 / np.asarray(v, dtype=float))
        return d - (d[0] if ref == "first" else d[-1])

    def _cutoff_index(d: np.ndarray, thr: float) -> int:
        """MATLAB: temp_index=find(d<-thr); count_index=temp_index(1)+1 (1 基)."""
        idx = np.flatnonzero(d < -thr)
        return int(idx[0]) + 1 if len(idx) else N

    if method == 0:
        # normal: exp2 拟合信道响应后开平方
        temp2 = _exp2_fit(temp)
        temp3 = temp2 ** 0.5

    elif method == 1:
        # inverse+normal: 对归一化到末端的逆响应 (dB) 拟合
        d = _inverse_db(temp, ref="end")
        temp2 = _exp2_fit(d)
        temp2 = 1.0 / (10.0 ** (temp2 / 20))
        temp3 = np.sqrt(np.abs(temp2))

    elif method == 2:
        # cut off: 仅拟合 -equal_db 门限以上部分, 其余补最小值
        d = _inverse_db(temp, ref="first")
        count_index = _cutoff_index(d, equal_db)
        temp2 = _exp2_fit(temp[:count_index])
        temp3 = np.sqrt(np.abs(temp2))
        pad = np.full(N - count_index, np.min(temp3))
        temp3 = np.concatenate([temp3, pad])

    elif method == 3:
        # peak point: 拟合逆响应, 以门限点为界分段开 5/8 与 1/8 次方
        d = _inverse_db(temp, ref="first")
        count_index = _cutoff_index(d, equal_db)
        temp2 = _exp2_fit(d)
        temp2 = 1.0 / (10.0 ** (temp2 / 20))
        temp2 = temp2 / temp2[count_index - 1]
        temp3 = np.concatenate([temp2[:count_index - 1] ** (5 / 8),
                                temp2[count_index - 1:] ** (1 / 8)])

    elif method == 4:
        # peak point fit: 在拟合曲线上找门限点, 分段开 6/8 与 1/100 次方
        d = _inverse_db(temp, ref="first")
        temp2_db = _exp2_fit(d)
        count_index = _cutoff_index(temp2_db, equal_db)
        _cutoff_index(temp2_db, equal_db2)  # 与 MATLAB 一致仅作诊断
        temp2 = 1.0 / (10.0 ** (temp2_db / 20))
        temp2 = temp2 / temp2[count_index - 1]
        temp3 = np.concatenate([temp2[:count_index - 1] ** (6 / 8),
                                temp2[count_index - 1:] ** (1 / 100)])

    elif method == 5:
        # Hardware Pre: 桥 T 均衡器 dB 响应样条插值到子载波数
        f_mhz, b_log, f_use = hardware_pre_response_db(**hw)
        idx_src = np.arange(1, len(f_use) + 1, dtype=float)
        idx_dst = np.linspace(1, len(f_use), N)
        # MATLAB: f_use_right = spline(index_f_use, f_use, index_f_use_right)
        from scipy.interpolate import CubicSpline
        f_use_right = CubicSpline(idx_src, f_use)(idx_dst)
        temp3 = 10.0 ** (f_use_right / 20)

        if save_curves:
            save_txt(config.F_GRID_FILE, f_mhz)
            save_txt(config.HARDWARE_PRE_FILE, b_log)
    else:
        raise ValueError(f"Unknown PRE_METHOD: {method}")

    temp3 = np.asarray(temp3, dtype=float).ravel()
    save_txt(save_path, temp3)
    print(f"Pre-emphasis weights (method={method}) saved to {save_path} "
          f"(N={len(temp3)}, range=[{temp3.min():.4f}, {temp3.max():.4f}])")
    return temp3


# -----------------------------------------------------------------------------
# DMT 调制
# -----------------------------------------------------------------------------
def generate_dmt_tx(RQ: np.ndarray,
                    S: np.ndarray,
                    datano: int,
                    constellation: str,
                    cfg: Optional[dict] = None,
                    seed: int = config.RANDOM_SEED) -> dict:
    """生成 DMT 发送波形.

    Returns:
        dict 包含:
            tx_waveform, qamdata, AVT, origin_dec_data, origin_binary,
            data_final, dataIQ, dataifft1
    """
    cfg = cfg or {}
    carrierno = cfg.get("carrierno", config.CARRIERNO)
    zeropad1 = cfg.get("zeropad1", config.ZEROPAD1)
    upsampleno = cfg.get("upsampleno", config.UPSAMPLENO)
    cp = cfg.get("cp", config.CP)
    normalize_flag = cfg.get("normalize_flag", config.NORMALIZE_FLAG)
    pre_equ_flag = cfg.get("pre_equ_flag", config.PRE_EQU_FLAG)
    carrierno1 = carrierno // 2 - zeropad1

    rng = np.random.default_rng(seed)
    max_bits = int(RQ.max())
    B2data = rng.integers(0, 2, size=(carrierno1, max_bits * datano))

    origin_dec_data = np.zeros((carrierno1, datano), dtype=int)
    qamdata = np.zeros((carrierno1, datano), dtype=complex)
    AVT = np.zeros((carrierno1, datano))

    for n in range(carrierno1):
        bits = int(RQ[n])
        order = 2 ** bits if bits > 0 else 1
        if order == 1:
            origin_dec_data[n, :] = 0
            qamdata[n, :] = 0
            AVT[n, :] = 1.0
            continue

        bits_stream = B2data[n, :bits * datano]
        data_2 = bits_stream.reshape(datano, bits)
        # MSB first: b0*2^(bits-1) + ... + b_(bits-1)*2^0
        weights = 2 ** np.arange(bits)[::-1]
        dec = np.dot(data_2, weights).astype(int)
        origin_dec_data[n, :] = dec

        sym = qam_modulate(dec, order, constellation)
        qamdata[n, :] = sym

        if normalize_flag == 1:
            all_cons = load_constellation(order, constellation)
            avg_pow = np.sqrt(np.mean(np.abs(all_cons) ** 2))
            AVT[n, :] = avg_pow / S[n]
        else:
            AVT[n, :] = np.max(np.abs(sym)) / S[n]

        qamdata[n, :] = qamdata[n, :] / AVT[n, :]

    # 插入导频
    pilot_pattern = cfg.get("pilot_pattern", config.PILOT_PATTERN)
    pilot_value = cfg.get("pilot_value", config.PILOT_VALUE)
    pilot_kwargs = {
        k: cfg[k] for k in (
            "comb_start", "comb_spacing",
            "mesh_start_freq", "mesh_freq_spacing",
            "mesh_start_time", "mesh_time_spacing"
        ) if k in cfg
    }
    pilot_mask = create_pilot_mask(carrierno1, datano, pilot_pattern, **pilot_kwargs)
    qamdata, AVT, origin_dec_data = insert_pilots(
        qamdata, AVT, origin_dec_data, pilot_mask, pilot_value
    )

    # Hermitian 共轭对称
    data_final = np.zeros((carrierno, datano), dtype=complex)
    data_final[zeropad1:carrierno // 2, :] = qamdata
    data_final[carrierno // 2 + 1:carrierno - zeropad1 + 1, :] = np.conj(np.flipud(qamdata))

    data_I = data_final.real
    data_Q = data_final.imag

    # IFFT + upsample + CP
    dataIQ = data_final
    dataIQ_upsample = np.zeros((carrierno * upsampleno, datano), dtype=complex)
    dataIQ_upsample[:carrierno // 2, :] = dataIQ[:carrierno // 2, :]
    dataIQ_upsample[-carrierno // 2:, :] = dataIQ[carrierno // 2:, :]

    dataifft = np.fft.ifft(dataIQ_upsample, axis=0)
    dataifft1 = np.vstack([dataifft[-cp * upsampleno:, :], dataifft])

    data1 = np.real(dataifft1.reshape(-1, order="F").copy())

    # 输出未加 dummy 的波形（用于同步）
    dataout = center_normalize(data1)

    # 加 dummy 并再次归一化
    data1_with_dummy, dummy = add_dummy(data1, base=64)
    waveform_dummy_len = len(dummy)
    data1_with_dummy = center_normalize(data1_with_dummy)

    # 硬件预均衡
    if pre_equ_flag == 3:
        th7_path = Path(cfg.get("th7_file", config.TH7_FILE))
        if not th7_path.exists():
            # th7.txt 不存在时按 config.PRE_METHOD 自动生成 (port of MATLAB Pre.m)
            print(f"{th7_path} not found, generating with PRE_METHOD={config.PRE_METHOD}")
            generate_preemphasis_weights(
                channel_mag=cfg.get("channel_mag", None),
                method=cfg.get("pre_method", config.PRE_METHOD),
                hw_params=cfg.get("hw_pre_params", None),
                n_subcarriers=carrierno1,
                save_path=th7_path,
            )
        data_pre = apply_hardware_preEQ(np.real(dataifft1.reshape(-1, order="F")),
                                        cfg.get("awg_sample_rate", config.AWG_SAMPLE_RATE),
                                        upsampleno,
                                        th7_path)
        data_pre, dummy = add_dummy(data_pre, base=64)
        data_pre = center_normalize(data_pre)
        # 功率归一化
        data_pre = data_pre / np.sqrt(np.mean(np.abs(data_pre) ** 2))
    else:
        data_pre = None

    return {
        "tx_waveform": data1_with_dummy,
        "tx_waveform_pre": data_pre,
        "qamdata": qamdata,
        "AVT": AVT,
        "origin_dec_data": origin_dec_data,
        "origin_binary": B2data[:, :datano],
        "data_final": data_final,
        "dataIQ": dataIQ,
        "dataifft1": dataifft1,
        "data_I": data_I,
        "data_Q": data_Q,
        "waveform_dummy_len": waveform_dummy_len,
        "constellation": constellation,
        "RQ": RQ,
        "S": S,
        "pilot_mask": pilot_mask,
        "pilot_pattern": pilot_pattern,
    }


def generate_qpsk_tx(datano: int = config.DATANO_QPSK,
                     cfg: Optional[dict] = None) -> dict:
    """生成 QPSK 探测波形 (STEP1)."""
    cfg = cfg or {}
    carrierno1 = cfg.get("carrierno1", config.CARRIERNO1)
    RQ = np.full(carrierno1, 2, dtype=int)
    S = np.ones(carrierno1)
    return generate_dmt_tx(RQ, S, datano, config.CONSTELLATION_QAM, cfg=cfg)


def generate_bitloading_tx(snrs: np.ndarray,
                           constellation: str = config.CONSTELLATION_QAM,
                           datano: int = config.DATANO_BPL,
                           cfg: Optional[dict] = None) -> dict:
    """生成 bitloading 波形 (STEP3)."""
    cfg = cfg or {}
    snrs = np.asarray(snrs).ravel()
    carrierno1 = cfg.get("carrierno1", config.CARRIERNO1)

    if constellation == config.CONSTELLATION_APSK:
        snr_table = load_snr_table(config.SNR_TABLE_APSK5)
        qam_order_all = assign_qam_order_from_snr(snrs, snr_table)
        ratio = cfg.get("ratio", 100)
        S_hh, RQ_hh, raise_num = bit_loading_hh(snrs, qam_order_all, snr_table)
        SE_add = raise_num - ratio
        S, RQ = bit_loading_lc(snrs, qam_order_all, snr_table, SE_add=SE_add)
    else:
        snr_table = load_snr_table(config.SNR_TABLE_FEC4)
        qam_order_all = assign_qam_order_from_snr(snrs, snr_table)
        ratio = cfg.get("ratio", 100)
        S_hh, RQ_hh, raise_num = bit_loading_hh(snrs, qam_order_all, snr_table)
        SE_add = raise_num - ratio
        S, RQ = bit_loading_lc(snrs, qam_order_all, snr_table, SE_add=SE_add)

    # comb 导频的子载波全部用于传已知导频，不再承载数据
    pattern = cfg.get("pilot_pattern", config.PILOT_PATTERN)
    if pattern == "comb":
        from dmt_core import create_pilot_mask
        pmask = create_pilot_mask(carrierno1, datano, pattern="comb",
                                  comb_start=cfg.get("comb_start", config.PILOT_COMB_START),
                                  comb_spacing=cfg.get("comb_spacing", config.PILOT_COMB_SPACING))
        pilot_sc = np.where(pmask[:, 0])[0]
        RQ[pilot_sc] = 0
        S[pilot_sc] = 1.0
        print(f"Comb pilots on {len(pilot_sc)} carriers zeroed in bitloading")

    print(f"SE after BPL is {RQ.mean():.4f}")
    datarate = RQ.mean() * config.AWG_SAMPLE_RATE / config.UPSAMPLENO * (carrierno1 / config.CARRIERNO) / 1e9
    print(f"data rate is {datarate:.4f} Gbps")

    res = generate_dmt_tx(RQ, S, datano, constellation, cfg=cfg)
    res["RQ"] = RQ
    res["S"] = S
    res["snr_table"] = snr_table
    res["ratio"] = ratio
    res["hh_raise_num"] = raise_num
    res["datarate_gbps"] = datarate
    return res


# -----------------------------------------------------------------------------
# DMT 接收
# -----------------------------------------------------------------------------
def phase_recovery(Rx: np.ndarray, Tx: np.ndarray,
                   carriernum: int, symnum: int,
                   start: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """相位恢复 (port from MATLAB phaseRecovery.m)."""
    Rx = Rx.reshape(carriernum, symnum)
    Tx = Tx.reshape(carriernum, symnum)
    y = np.zeros((carriernum, symnum))
    Rx_recovery = np.zeros_like(Rx, dtype=complex)

    preamnum = 4
    preamble = np.floor(np.linspace(start, carriernum, preamnum)).astype(int)

    for t in range(symnum):
        phase_noise_s_temp = smooth(np.angle(Rx[:, t] / Tx[:, t]), window_len=20)
        max_p = np.max(phase_noise_s_temp[50:])
        max_idx = np.argmax(phase_noise_s_temp[50:]) + 50
        min_p = np.min(phase_noise_s_temp[50:])
        min_idx = np.argmin(phase_noise_s_temp[50:]) + 50
        delta_p = max_p - min_p

        if delta_p > 5 and min_idx - 5 > start and max_idx + 5 < carriernum:
            n1 = np.arange(start, min_idx - 5)
            nn1 = phase_noise_s_temp[start:min_idx - 5]
            n2 = np.arange(max_idx + 5, carriernum)
            nn2 = phase_noise_s_temp[max_idx + 5:carriernum] - 2 * np.pi
            n = np.concatenate([n1, n2])
            nn = np.concatenate([nn1, nn2])
            fitcurve = np.polyfit(n, nn, 1)
            fitval = np.polyval(fitcurve, np.arange(carriernum))
            y[:, t] = fitval
        else:
            end_idx = min(preamble[-1], carriernum - 1)
            n = np.arange(start, end_idx + 1)
            nn = phase_noise_s_temp[start:end_idx + 1]
            if len(n) < 2:
                y[:, t] = 0.0
            else:
                fitcurve = np.polyfit(n, nn, 1)
                fitval = np.polyval(fitcurve, np.arange(carriernum))
                y[:, t] = fitval

    y_new = np.zeros((carriernum, symnum))
    y1 = y[:, 0]
    y2 = y[:, -1]
    for i in range(carriernum):
        y_new[i, :] = np.linspace(y1[i], y2[i], symnum)
    Rx_recovery = Rx * np.exp(-1j * y_new)
    return Rx_recovery, y_new


def smooth(x: np.ndarray, window_len: int = 11) -> np.ndarray:
    """滑动平均平滑，与 MATLAB smooth(.,window_len) 近似."""
    x = np.asarray(x).ravel()
    if window_len < 3:
        return x
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    w = np.ones(window_len) / window_len
    y = np.convolve(s, w, mode="valid")
    # 让输出长度与输入一致
    pad = window_len // 2
    return y[pad:pad + len(x)]


def dmt_receiver(rx_waveform: np.ndarray,
                 tx_dict: dict,
                 cfg: Optional[dict] = None) -> dict:
    """DMT 接收处理.

    Args:
        rx_waveform: 接收到的时域波形 (1-D)
        tx_dict: generate_dmt_tx 的输出字典
        cfg: 配置

    Returns:
        dict 包含 out2(均衡并恢复相位后的频域符号), in(参考), SNR 等
    """
    cfg = cfg or {}
    carrierno = cfg.get("carrierno", config.CARRIERNO)
    zeropad1 = cfg.get("zeropad1", config.ZEROPAD1)
    upsampleno = cfg.get("upsampleno", config.UPSAMPLENO)
    cp = cfg.get("cp", config.CP)
    trainingno = cfg.get("trainingno", config.TRAININGNO)
    carrierno1 = carrierno // 2 - zeropad1
    datano = tx_dict["qamdata"].shape[1]

    rx_waveform = np.asarray(rx_waveform).ravel()

    # 去掉 dummy
    dummy_len = tx_dict["waveform_dummy_len"]
    if dummy_len > 0:
        rx_waveform = rx_waveform[:-dummy_len]

    # .reshape 成符号矩阵
    sym_len = (carrierno + cp) * upsampleno
    dataRx = rx_waveform[:sym_len * datano].reshape(sym_len, datano, order="F")
    rv_tifft = dataRx[cp * upsampleno:, :]
    rv_down1 = np.fft.fft(rv_tifft, axis=0)
    rv_down = rv_down1[zeropad1:carrierno // 2, :]

    in_ref = tx_dict["qamdata"]
    out = rv_down
    pilot_mask = tx_dict.get("pilot_mask", np.zeros((carrierno1, datano), dtype=bool))
    use_pilots = pilot_mask.any()

    if use_pilots:
        # 导频辅助信道估计
        H = estimate_channel_from_pilots(out, in_ref, pilot_mask)
        out2 = out * H
        out2_temp = out2.copy()
        # 导频辅助相位恢复
        Rx_recovery, _ = phase_recovery_from_pilots(out2, in_ref, pilot_mask)
        ha = np.mean(H, axis=1)
    else:
        # 传统 training symbol 信道估计
        h1 = in_ref[:, :trainingno] / out[:, :trainingno]
        h1 = h1.T  # (trainingno, carrierno1)
        ha = np.mean(h1, axis=0)
        ha = smooth(ha, window_len=7)
        H = np.tile(ha, (datano, 1)).T
        out2 = out * H
        out2_temp = out2.copy()
        # 相位恢复
        Rx_recovery, _ = phase_recovery(out2, in_ref, carrierno1, datano, start=50)

    # 反归一化
    AVT = tx_dict["AVT"]
    in_denorm = in_ref * AVT
    out2_denorm = Rx_recovery * AVT
    out3 = out2_temp * AVT

    # 计算 SNR（跳过导频位置）
    SNR_R = np.full(carrierno1, np.nan)
    for n in range(carrierno1):
        valid = ~pilot_mask[n, :]
        if not np.any(valid):
            continue
        sig_pow = np.mean(np.abs(out2_denorm[n, valid]) ** 2)
        err_pow = np.mean(np.abs(out2_denorm[n, valid] - in_denorm[n, valid]) ** 2)
        if err_pow == 0 or not np.isfinite(err_pow):
            SNR_R[n] = 1e12
        else:
            SNR_R[n] = sig_pow / err_pow

    # 对纯导频子载波（comb 中整列为导频），用最近邻数据子载波 SNR 填充
    valid_idx = np.where(np.isfinite(SNR_R) & (SNR_R > 0))[0]
    if len(valid_idx) > 0:
        for n in np.where(~np.isfinite(SNR_R))[0]:
            nearest = valid_idx[np.argmin(np.abs(valid_idx - n))]
            SNR_R[n] = SNR_R[nearest]
    else:
        SNR_R[:] = 1.0

    # 解调并计算误码（跳过导频位置）
    rx_dec = np.zeros_like(in_denorm, dtype=int)
    ber_total = 0.0
    bits_total = 0
    ber_per_carrier = np.zeros(carrierno1)
    ser_per_carrier = np.zeros(carrierno1)
    bits_per_carrier = np.zeros(carrierno1, dtype=int)
    RQ = tx_dict.get("RQ", None)
    for n in range(carrierno1):
        if RQ is not None:
            bits = int(RQ[n])
        else:
            bits = int(np.log2(np.max(tx_dict["origin_dec_data"][n, :]) + 1))
        bits_per_carrier[n] = bits
        if bits < 1:
            continue
        order = 2 ** bits
        dec = qam_demodulate(out2_denorm[n, :], order, tx_dict.get("constellation", "QAM"))
        rx_dec[n, :] = dec
        valid = ~pilot_mask[n, :]
        n_valid = int(np.sum(valid))
        if n_valid == 0:
            continue
        err_sym = int(np.sum(dec[valid] != tx_dict["origin_dec_data"][n, valid]))
        ser_per_carrier[n] = err_sym / n_valid
        # 精确比特错误数：对每个错误符号做 XOR 后统计 1 的个数
        diff = np.bitwise_xor(dec[valid], tx_dict["origin_dec_data"][n, valid]).astype(np.uint16)
        diff_u8 = diff.view(np.uint8)
        bits_matrix = np.unpackbits(diff_u8).reshape(-1, 16)
        bit_errs = int(bits_matrix[:, :bits].sum())
        ber_per_carrier[n] = bit_errs / (n_valid * bits)
        bits_total += bits * n_valid
        ber_total += bit_errs

    ber = ber_total / bits_total if bits_total > 0 else 0.0

    return {
        "out2": out2_denorm,
        "in_ref": in_denorm,
        "SNR_R": SNR_R,
        "rx_dec": rx_dec,
        "ber": ber,
        "ser": float(np.mean(ser_per_carrier[bits_per_carrier > 0])),
        "ber_per_carrier": ber_per_carrier,
        "ser_per_carrier": ser_per_carrier,
        "bits_per_carrier": bits_per_carrier,
        "channel_response": ha,
        "pilot_mask": pilot_mask,
    }


def estimate_snr_per_carrier(rx_waveform: np.ndarray,
                             tx_dict: dict,
                             cfg: Optional[dict] = None) -> np.ndarray:
    """QPSK 探测后估计每个子载波的 SNR (STEP2)."""
    res = dmt_receiver(rx_waveform, tx_dict, cfg=cfg)
    return res["SNR_R"]


# -----------------------------------------------------------------------------
# BER 估计
# -----------------------------------------------------------------------------
def gray_neighbor_cal(order: int, flag: int = 0) -> Tuple[float, float]:
    """Gray 邻居近似（简化）."""
    bits = int(np.log2(order))
    if flag == 0:
        return max(1.0, bits * 0.8), float(order)
    return 1.0, float(order)


def ber_est_by_snr(snr: float, qam_order: int, flag: int = 1, noise_var: float = 1.0) -> float:
    """由 SNR 估计 BER (port from MATLAB BER_est_by_SNR.m)."""
    bits = int(np.log2(qam_order))
    L = int(np.ceil(np.sqrt(qam_order)))
    if flag == 1:
        if bits == 3:
            return 10 / 16 * erfc(np.sqrt(48 / (31 * qam_order - 32) * snr))
        elif bits % 2 == 1 and bits > 1:
            Gp_N, N_N = gray_neighbor_cal(qam_order, 0)
            return Gp_N * (N_N / bits) * 0.5 * erfc(np.sqrt(48 / (31 * qam_order - 32) * snr))
        else:
            arg = 3 * bits / (L ** 2 - 1) * 2 / (snr ** (-1) * bits) / 2
            return (2 * (1 - 1 / L) / bits) * 0.5 * erfc(np.sqrt(arg))
    else:
        from scipy.special import erfc
        return 4 * 0.5 * erfc(np.sqrt(2 * noise_var))
