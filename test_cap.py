"""Unit/integration tests for UW_APSK CAP Python transceiver."""
import numpy as np

import cap_core
import cap_rx
import channel
from constellation import average_power, demodulate, modulate
from equalizer import lms_equalizer


def test_constellation_roundtrip():
    for order in [32, 64, 128]:
        for const in ["APSK", "QAM"]:
            dec = np.arange(order)
            sym = modulate(dec, order, const)
            dec2 = demodulate(sym, order, const)
            assert np.array_equal(dec, dec2), f"{order}-{const} roundtrip failed"
    print("PASS: constellation roundtrip")


def test_singleband_golden_reference():
    dec = np.loadtxt("data/decimal_32.txt").astype(int)
    matlab_tx = np.loadtxt("data/data32QAM.txt")
    sym = modulate(dec, 32, "QAM")
    tx_py, _, _, _ = cap_core.generate_singleband_cap(
        sym, 2e9, 5e8, 0.205, 0.5, 0.01, 35
    )
    m = matlab_tx / np.sqrt(np.mean(matlab_tx ** 2))
    p = tx_py / np.sqrt(np.mean(tx_py ** 2))
    max_diff = np.max(np.abs(m - p))
    assert max_diff < 1e-6, f"single-band waveform mismatch: {max_diff}"
    print(f"PASS: single-band golden reference (max diff {max_diff:.3e})")


def test_singleband_roundtrip():
    order = 16
    num = 2000
    rng = np.random.default_rng(42)
    dec = rng.integers(0, order, size=num)
    sym = modulate(dec, order, "QAM")
    tx, fI, fQ, t = cap_core.generate_singleband_cap(sym, 2e9, 5e8, 0.205, 0.5, 0.01, 35)
    rx = channel.vlc_channel(tx, 30, 100, 40, False)
    DataCapI = np.convolve(rx, fI, mode="same")
    DataCapQ = np.convolve(rx, fQ, mode="same")
    match = (DataCapI + 1j * DataCapQ)[::4]
    eq, _, _, _ = lms_equalizer(match, sym, 17, 0.005, 800)
    head = 8
    mm = len(match) - 17 + 1
    eq_valid = eq[head : head + mm]
    avp = average_power(order, "QAM")
    eq_valid = eq_valid / np.sqrt(np.mean(np.abs(eq_valid) ** 2)) * avp
    decisions = demodulate(eq_valid, order, "QAM")
    ser = np.mean(decisions != dec[head : head + mm])
    assert ser < 0.01, f"single-band roundtrip SER too high: {ser}"
    print(f"PASS: single-band roundtrip SER={ser:.4e}")


def test_multiband_modulation():
    order = 16
    num = 1024
    rng = np.random.default_rng(1)
    syms = [modulate(rng.integers(0, order, size=num), order, "QAM") for _ in range(3)]
    tx, fc, gt, t, bands = cap_core.generate_multiband_cap(
        syms, 300e6, 1.2e9, 0.2, 0.11, 8, "srrc"
    )
    assert len(tx) == num * int(round(3 * 1.2e9 / 300e6))
    assert np.isclose(np.mean(tx ** 2), 1.0, atol=1e-6)
    print("PASS: multiband modulation")


def test_multiband_separation():
    order = 16
    num = 2048
    rng = np.random.default_rng(1)
    syms = [modulate(rng.integers(0, order, size=num), order, "QAM") for _ in range(3)]
    tx, fc, gt, t, bands = cap_core.generate_multiband_cap(
        syms, 300e6, 1.2e9, 0.2, 0.11, 8, "srrc"
    )
    upsampleno = int(round(3 * 1.2e9 / 300e6))
    taps = 8 * upsampleno + 1
    for n in range(3):
        rb = cap_rx.capmatch_filter(tx, gt, t, fc[n], taps, upsampleno, 0)
        eq, _, _, _ = lms_equalizer(rb, syms[n], 31, 0.005, 1000)
        head = 15
        mm = len(rb) - 31 + 1
        eq_valid = eq[head : head + mm]
        decisions = demodulate(eq_valid, order, "QAM")
        dec_tx = np.array(
            [np.argmin(np.abs(s - modulate(np.arange(order), order, "QAM"))) for s in syms[n][head : head + mm]]
        )
        ser = np.mean(decisions != dec_tx)
        assert ser < 0.1, f"multiband band {n} SER too high: {ser}"
    print("PASS: multiband separation with LMS")


if __name__ == "__main__":
    test_constellation_roundtrip()
    test_singleband_golden_reference()
    test_singleband_roundtrip()
    test_multiband_modulation()
    test_multiband_separation()
    print("\nAll tests passed.")
