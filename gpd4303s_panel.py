# -*- coding: utf-8 -*-
"""DMT GUI 子页面：GPD-4303S 四通道直流电源控制面板.

被 ``dmt_gui.py`` 动态导入并添加为 Notebook 的一个 tab。
"""
import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Dict, Optional

import config_cap as cfg
from gpd4303s_controller import (
    GPD4303S,
    GPD4303SConnectionError,
    GPD4303SCommandError,
)
from keithley2400_controller import refresh_port_list, parse_port_entry


# 与 dmt_gui.py 中定义的配色保持一致
COLOR_BG = "#F3F5F7"
COLOR_CARD = "#FFFFFF"
COLOR_BORDER = "#E2E8F0"
COLOR_TEXT = "#1F2937"
COLOR_TEXT_DIM = "#64748B"

CHANNELS = (1, 2, 3, 4)


def make_card(parent, **pack_kwargs):
    """创建带细边框的白色卡片."""
    card = tk.Frame(parent, bg=COLOR_CARD,
                    highlightbackground=COLOR_BORDER, highlightthickness=1,
                    bd=0)
    if pack_kwargs:
        card.pack(**pack_kwargs)
    return card


class _GpdWorker(threading.Thread):
    """后台线程：负责与 GPD-4303S 通信，避免阻塞 GUI."""

    def __init__(self, port: str, baudrate: int, timeout: float,
                 state: Dict[str, Any], cmd_queue: queue.Queue):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.state = state
        self.cmd_queue = cmd_queue
        self.psu: Optional[GPD4303S] = None
        self._running = True
        self.logger = logging.getLogger("gpd4303s.worker")

    def run(self):
        try:
            self.psu = GPD4303S(self.port, baudrate=self.baudrate,
                                timeout=self.timeout)
            self.psu.connect(retries=2)
            idn = self.psu.idn()
            self.state.update({"connected": True, "idn": idn, "error": ""})
        except Exception as exc:
            self.state.update({"connected": False, "error": str(exc)})
            return

        while self._running:
            try:
                cmd = self.cmd_queue.get_nowait()
                if cmd[0] == "output":
                    if cmd[1]:
                        self.psu.output_on()
                    else:
                        self.psu.output_off()
                elif cmd[0] == "set":
                    _, ch, voltage, current = cmd
                    self.psu.set_voltage(ch, voltage)
                    self.psu.set_current(ch, current)
                elif cmd[0] == "disconnect":
                    break
            except queue.Empty:
                pass
            except Exception as exc:
                self.state["error"] = str(exc)

            try:
                status = self.psu.get_status()
                channels: Dict[int, Dict[str, Any]] = {}
                for ch in CHANNELS:
                    channels[ch] = {
                        "v_set": self.psu.get_voltage_set(ch),
                        "i_set": self.psu.get_current_set(ch),
                        "v_out": self.psu.measure_voltage(ch),
                        "i_out": self.psu.measure_current(ch),
                        "mode": status["channel_modes"].get(ch, "?"),
                    }
                raw = status["raw"]
                output_on = len(raw) >= 6 and raw[5] == "1"
                self.state.update({
                    "connected": True,
                    "output_on": output_on,
                    "status_raw": raw,
                    "channels": channels,
                    "error": "",
                })
            except Exception as exc:
                self.state["error"] = str(exc)

            time.sleep(cfg.GPD4303S_POLL_MS / 1000.0)

        if self.psu is not None:
            try:
                self.psu.output_off()
                self.psu.local()
                self.psu.disconnect()
            except Exception:
                pass
        self.state.update({"connected": False, "output_on": False})

    def stop(self):
        self._running = False
        self.cmd_queue.put(("disconnect",))


class GPD4303SPanel(ttk.Frame):
    """GPD-4303S 电源控制子页面."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.worker: Optional[_GpdWorker] = None
        self.state: Dict[str, Any] = {
            "connected": False,
            "idn": "",
            "output_on": False,
            "status_raw": "",
            "channels": {},
            "error": "",
        }
        self.cmd_queue: queue.Queue = queue.Queue()

        # 持久化设置与 config 保持一致
        self.port_var = tk.StringVar(value=cfg.GPD4303S_PORT)
        self.baud_var = tk.StringVar(value=str(cfg.GPD4303S_BAUDRATE))

        self.channel_widgets: Dict[tuple, ttk.Label] = {}
        self.channel_entries: Dict[tuple, ttk.Entry] = {}

        self._build_ui()
        self._refresh_ports()
        self._update_ui()

    # --- 界面构建 --------------------------------------------------------
    def _build_ui(self):
        self.configure(style="TFrame")

        # 标题栏
        hdr = tk.Frame(self, bg=COLOR_BG)
        hdr.pack(fill=tk.X, padx=16, pady=(14, 6))
        ttk.Label(hdr, text="GPD-4303S 四通道直流电源",
                  style="Title.TLabel").pack(side=tk.LEFT)
        self.status_lbl = ttk.Label(hdr, text="未连接", style="Pill.TLabel")
        self.status_lbl.pack(side=tk.RIGHT)

        # 连接卡片
        card = make_card(self, fill=tk.X, padx=16, pady=6)
        inner = tk.Frame(card, bg=COLOR_CARD)
        inner.pack(fill=tk.X, padx=12, pady=12)

        ttk.Label(inner, text="串口", style="Section.TLabel").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.port_combo = ttk.Combobox(inner, textvariable=self.port_var,
                                       values=[], width=40, state="readonly")
        self.port_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 8), pady=4)
        ttk.Button(inner, text="⟳ 刷新", command=self._refresh_ports).grid(
            row=0, column=2, padx=(0, 8), pady=4)

        ttk.Label(inner, text="波特率", style="Section.TLabel").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        self.baud_combo = ttk.Combobox(inner,
                                       values=[9600, 57600, 115200],
                                       textvariable=self.baud_var,
                                       width=12, state="readonly")
        self.baud_combo.grid(row=1, column=1, sticky=tk.W, padx=(0, 8), pady=4)

        self.conn_btn = ttk.Button(inner, text="连接",
                                   command=self._toggle_connect)
        self.conn_btn.grid(row=1, column=2, padx=(0, 8), pady=4)

        # IDN
        self.idn_lbl = ttk.Label(self, text="IDN: --", padding=(16, 0))
        self.idn_lbl.pack(anchor=tk.W)

        # 通道卡片区
        cards = tk.Frame(self, bg=COLOR_BG)
        cards.pack(fill=tk.BOTH, expand=True, padx=16, pady=6)

        for i, ch in enumerate(CHANNELS):
            col = i % 2
            row = i // 2
            card = ttk.LabelFrame(cards, text=f" 通道 CH{ch} ", padding=10)
            card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
            cards.columnconfigure(col, weight=1)
            cards.rowconfigure(row, weight=1)

            # 设定值输入
            ttk.Label(card, text="设定电压:").grid(
                row=0, column=0, sticky=tk.W, pady=3)
            v_entry = ttk.Entry(card, width=10, justify=tk.RIGHT)
            v_entry.grid(row=0, column=1, sticky=tk.W, padx=(8, 0), pady=3)
            ttk.Label(card, text="V").grid(row=0, column=2, sticky=tk.W, pady=3)
            self.channel_entries[(ch, "v_set")] = v_entry

            ttk.Label(card, text="设定电流:").grid(
                row=1, column=0, sticky=tk.W, pady=3)
            i_entry = ttk.Entry(card, width=10, justify=tk.RIGHT)
            i_entry.grid(row=1, column=1, sticky=tk.W, padx=(8, 0), pady=3)
            ttk.Label(card, text="A").grid(row=1, column=2, sticky=tk.W, pady=3)
            self.channel_entries[(ch, "i_set")] = i_entry

            ttk.Button(card, text="应用",
                       command=lambda c=ch: self._apply_channel(c)).grid(
                row=0, column=3, rowspan=2, padx=(12, 0), pady=3, sticky=tk.NS)

            # 实际输出
            labels = [
                ("输出电压", "v_out", "V"),
                ("输出电流", "i_out", "A"),
                ("工作模式", "mode", ""),
            ]
            for r, (name, key, unit) in enumerate(labels, start=2):
                ttk.Label(card, text=f"{name}:").grid(
                    row=r, column=0, sticky=tk.W, pady=3)
                val_lbl = ttk.Label(card, text="--", style="Value.TLabel")
                val_lbl.grid(row=r, column=1, sticky=tk.W,
                             padx=(8, 0), pady=3)
                ttk.Label(card, text=unit).grid(
                    row=r, column=2, sticky=tk.W, pady=3)
                self.channel_widgets[(ch, key)] = val_lbl

        # 输出控制 + 日志
        bottom = tk.Frame(self, bg=COLOR_BG)
        bottom.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 12))
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=2)

        out_card = make_card(bottom)
        out_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        out_inner = tk.Frame(out_card, bg=COLOR_CARD)
        out_inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        self.out_btn = ttk.Button(out_inner, text="输出 ON",
                                  command=self._toggle_output,
                                  state=tk.DISABLED)
        self.out_btn.pack(anchor=tk.W, pady=(0, 8))
        self.out_lbl = ttk.Label(out_inner, text="输出: 关",
                                 foreground="red", font=("", 10, "bold"))
        self.out_lbl.pack(anchor=tk.W)

        log_card = make_card(bottom)
        log_card.grid(row=0, column=1, sticky="nsew")
        log_inner = tk.Frame(log_card, bg=COLOR_CARD)
        log_inner.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        ttk.Label(log_inner, text="通信日志",
                  style="Section.TLabel").pack(anchor=tk.W)
        self.log_text = tk.Text(log_inner, height=8, wrap=tk.WORD,
                                bg="#FAFAFA", fg=COLOR_TEXT, relief=tk.FLAT,
                                highlightbackground=COLOR_BORDER,
                                highlightthickness=1)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.log_text.configure(state=tk.DISABLED)

    # --- 辅助 -----------------------------------------------------------
    def _log(self, text: str):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _refresh_ports(self):
        ports = refresh_port_list(interface="rs232")
        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(parse_port_entry(ports[0]))
        elif ports:
            current = parse_port_entry(self.port_var.get())
            matching = [p for p in ports if parse_port_entry(p) == current]
            if not matching:
                self.port_var.set(parse_port_entry(ports[0]))

    def _toggle_connect(self):
        if self.worker is not None and self.worker.is_alive():
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = parse_port_entry(self.port_var.get())
        if not port:
            messagebox.showwarning("提示", "请先选择串口。")
            return
        baud = int(self.baud_var.get() or cfg.GPD4303S_BAUDRATE)
        self.state["error"] = ""
        self.worker = _GpdWorker(
            port, baud, cfg.GPD4303S_TIMEOUT, self.state, self.cmd_queue
        )
        self.worker.start()
        self.conn_btn.configure(text="断开")
        self._log(f"正在连接 {port} (波特率 {baud})...")

    def _disconnect(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.conn_btn.configure(text="连接")
        self.out_btn.configure(state=tk.DISABLED, text="输出 ON")
        self._log("已断开")

    def _toggle_output(self):
        if not self.state.get("connected", False):
            self._log("未连接。")
            return
        new_state = not self.state.get("output_on", False)
        self.cmd_queue.put(("output", new_state))
        self._log(f"请求输出 {'ON' if new_state else 'OFF'}")

    def _apply_channel(self, ch: int):
        if not self.state.get("connected", False):
            messagebox.showwarning("提示", "请先连接仪器。")
            return

        v_str = self.channel_entries[(ch, "v_set")].get().strip()
        i_str = self.channel_entries[(ch, "i_set")].get().strip()
        try:
            voltage = float(v_str)
            current = float(i_str)
        except ValueError:
            messagebox.showerror("数值无效",
                                 f"CH{ch} 的电压/电流必须是数字。")
            return

        v_min, v_max = GPD4303S.VOLTAGE_RANGES.get(ch, (0.0, 32.0))
        i_min, i_max = GPD4303S.CURRENT_RANGES.get(ch, (0.0, 3.0))
        if not (v_min <= voltage <= v_max):
            messagebox.showerror(
                "越界", f"CH{ch} 电压应在 {v_min}~{v_max} V 之间"
            )
            return
        if not (i_min <= current <= i_max):
            messagebox.showerror(
                "越界", f"CH{ch} 电流应在 {i_min}~{i_max} A 之间"
            )
            return

        self.cmd_queue.put(("set", ch, voltage, current))
        self._log(f"CH{ch} 设置 V={voltage:.3f} V, I={current:.3f} A")

    def _update_ui(self):
        connected = self.state.get("connected", False)
        error = self.state.get("error", "")

        if connected:
            self.status_lbl.configure(text="已连接", foreground="green")
            self.idn_lbl.configure(text=f"IDN: {self.state.get('idn', '')}")
            self.out_btn.configure(state=tk.NORMAL)

            output_on = self.state.get("output_on", False)
            if output_on:
                self.out_btn.configure(text="输出 OFF")
                self.out_lbl.configure(text="输出: 开", foreground="green")
            else:
                self.out_btn.configure(text="输出 ON")
                self.out_lbl.configure(text="输出: 关", foreground="red")

            channels = self.state.get("channels", {})
            for ch in CHANNELS:
                data = channels.get(ch, {})
                self._set_entry(ch, "v_set", data.get("v_set"), "%.3f")
                self._set_entry(ch, "i_set", data.get("i_set"), "%.3f")
                self._set_label(ch, "v_out", data.get("v_out"), "V", "%.3f")
                self._set_label(ch, "i_out", data.get("i_out"), "A", "%.3f")
                self._set_label(ch, "mode", data.get("mode"), "", "%s")
        else:
            self.status_lbl.configure(text="未连接", foreground="gray")
            self.idn_lbl.configure(text="IDN: --")
            self.out_btn.configure(state=tk.DISABLED, text="输出 ON")
            self.out_lbl.configure(text="输出: 关", foreground="red")
            for ch in CHANNELS:
                for key in ("v_out", "i_out", "mode"):
                    self.channel_widgets[(ch, key)].configure(text="--")
                for key in ("v_set", "i_set"):
                    self.channel_entries[(ch, key)].delete(0, tk.END)

        if error and self.state.get("connected", False):
            self._log(f"错误: {error}")
            self.state["error"] = ""

        self.after(cfg.GPD4303S_POLL_MS, self._update_ui)

    def _set_label(self, ch: int, key: str, value, unit: str, fmt: str):
        lbl = self.channel_widgets[(ch, key)]
        if value is None:
            lbl.configure(text="--")
        else:
            text = f"{fmt % value} {unit}".strip()
            lbl.configure(text=text)

    def _set_entry(self, ch: int, key: str, value, fmt: str):
        entry = self.channel_entries[(ch, key)]
        if value is None:
            if self.focus_get() != entry:
                entry.delete(0, tk.END)
            return
        if self.focus_get() == entry:
            return
        text = fmt % value
        if entry.get() == text:
            return
        entry.delete(0, tk.END)
        entry.insert(0, text)

    # --- 公共接口 --------------------------------------------------------
    def on_close(self):
        """退出应用时调用，安全关闭输出."""
        self._disconnect()
