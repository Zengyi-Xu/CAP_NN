"""
CodePlot v5 — 图集排版模式
用法: python codeplot_v5.py

核心变化（相对 v4）:
  1. 单图独立编辑 — 代码只生成一张图，精细调整
  2. 图集收集 — "加入组图"把当前图保存到图集
  3. 总览排版 — 所有图按网格排列，自动加 (a)(b)(c) 序号
  4. 点击图集中的图可重新加载编辑
  5. 每个图同时保存 SVG（矢量）和 PNG（显示）

图集数据存储: .codeplot_gallery/ 目录（程序同目录）
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
matplotlib.use("TkAgg")

# ── 配置中文字体 ──
import matplotlib.font_manager as fm
_chinese_font_candidates = [
    'Microsoft YaHei', 'SimHei', 'SimSun', 'STSong',
    'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans SC'
]
_chinese_font_found = None
for _font in _chinese_font_candidates:
    try:
        fm.findfont(_font, fallback_to_default=False)
        _chinese_font_found = _font
        break
    except Exception:
        continue
if _chinese_font_found:
    matplotlib.rcParams['font.sans-serif'] = [_chinese_font_found, 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import traceback
import os
import re
import json
import uuid
import shutil

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# 组合排版引擎（compose_figure.py，与本文件同目录）
try:
    import compose_figure as cf_engine
    HAS_COMPOSE_ENGINE = True
except ImportError:
    cf_engine = None
    HAS_COMPOSE_ENGINE = False

# 组合设置中可持久化的参数键
COMPOSE_SETTING_KEYS = [
    "LABEL_FONT_FAMILY", "LABEL_FONT_SIZE", "LABEL_FONT_WEIGHT", "LABEL_COLOR",
    "LABEL_STYLE", "LABEL_BRACKETS", "LABEL_POSITION",
    "LABEL_OFFSET_X", "LABEL_OFFSET_Y", "LABEL_BACKGROUND",
    "LAYOUT", "GRID_ROWS", "GRID_COLS", "PANEL_WIDTH", "PANEL_HEIGHT",
    "H_SPACING", "V_SPACING", "MARGIN",
    "EXPORT_PDF", "EXPORT_PNG", "PNG_DPI", "BACKGROUND",
]

# ── 多段脚本模式：子图块与排版块的标记 ──
# 脚本编辑栏里可以存放多个子图块，每个块用 figN = new_figure() 开头，
# 最后是排版块，调用 compose() 把所有子图组合成总图。
COMPOSE_BLOCK_HEADER = "# ═══ 排版组合 ═══"
SUBPLOT_BLOCK_RE = re.compile(r"# ═══ 子图 (\d+)")


# ════════════════════════════════════════
# 图集管理
# ════════════════════════════════════════

class FigureGallery:
    """管理一组独立生成的图，每张图保存 SVG+PNG"""

    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".codeplot_gallery")
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.items = []          # [{id, code, title, svg_path, png_path}]
        self._load_index()

    def _index_path(self):
        return os.path.join(self.base_dir, "gallery_index.json")

    def _load_index(self):
        idx_path = self._index_path()
        if os.path.exists(idx_path):
            try:
                with open(idx_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.items = data.get("items", [])
            except Exception:
                self.items = []
        else:
            self.items = []

    def _save_index(self):
        with open(self._index_path(), "w", encoding="utf-8") as f:
            json.dump({"items": self.items}, f, ensure_ascii=False, indent=2)

    def add(self, code, fig, title=""):
        """添加一张图到图集，返回 item dict"""
        uid = str(uuid.uuid4())[:8]
        svg_path = os.path.join(self.base_dir, f"{uid}.svg")
        png_path = os.path.join(self.base_dir, f"{uid}.png")

        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=150, bbox_inches="tight")

        item = {
            "id": uid,
            "code": code,
            "title": title,
            "svg_path": svg_path,
            "png_path": png_path,
        }
        self.items.append(item)
        self._save_index()
        return item

    def remove(self, index):
        """删除指定索引的图"""
        if 0 <= index < len(self.items):
            item = self.items.pop(index)
            for p in (item.get("svg_path"), item.get("png_path")):
                if p and os.path.exists(p):
                    os.remove(p)
            self._save_index()
            return True
        return False

    def clear(self):
        """清空图集"""
        for item in self.items:
            for p in (item.get("svg_path"), item.get("png_path")):
                if p and os.path.exists(p):
                    os.remove(p)
        self.items = []
        self._save_index()

    def get_best_grid(self, n):
        """根据数量返回最佳行列数 (rows, cols)"""
        if n <= 0:
            return (0, 0)
        if n == 1:
            return (1, 1)
        if n == 2:
            return (1, 2)
        if n <= 4:
            return (2, 2)
        if n <= 6:
            return (2, 3)
        if n <= 9:
            return (3, 3)
        if n <= 12:
            return (3, 4)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        return (rows, cols)


# ════════════════════════════════════════
# 内置模板（同 v4）
# ════════════════════════════════════════

BUILTIN_TEMPLATES = {
    "基础图表": {
        "正弦波": {
            "desc": "绘制正弦曲线，展示最基本的线图用法",
            "code": """# 正弦波
x = np.linspace(0, 4*np.pi, 500)
y = np.sin(x)

ax = fig.add_subplot(111)
ax.clear()
ax.plot(x, y, color='tab:blue', linewidth=2)
ax.set_title('正弦波', fontsize=14)
ax.set_xlabel('x (rad)')
ax.set_ylabel('sin(x)')
ax.grid(True, alpha=0.3)
fig.tight_layout()
"""
        },
        "正弦+余弦波": {
            "desc": "同图绘制正弦和余弦，包含图例",
            "code": """# 正弦波 + 余弦波
x = np.linspace(0, 4*np.pi, 500)
y1 = np.sin(x)
y2 = np.cos(x)

ax = fig.add_subplot(111)
ax.clear()
ax.plot(x, y1, label='sin(x)', linewidth=2)
ax.plot(x, y2, label='cos(x)', linewidth=2, linestyle='--')
ax.set_title('正弦波与余弦波', fontsize=14)
ax.set_xlabel('x (rad)')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
"""
        },
        "随机散点图": {
            "desc": "带颜色和大小映射的散点图",
            "code": """# 随机散点图
np.random.seed(42)
x = np.random.randn(200)
y = np.random.randn(200)
colors = np.random.rand(200)
sizes = 100 * np.random.rand(200)

ax = fig.add_subplot(111)
ax.clear()
scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
ax.set_title('随机散点图', fontsize=14)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(scatter, ax=ax, label='颜色值')
fig.tight_layout()
"""
        },
        "分组柱状图": {
            "desc": "两组数据的分组柱状图对比",
            "code": """# 分组柱状图
categories = ['A', 'B', 'C', 'D', 'E']
values1 = [23, 45, 56, 78, 32]
values2 = [15, 30, 45, 60, 25]

x = np.arange(len(categories))
width = 0.35

ax = fig.add_subplot(111)
ax.clear()
ax.bar(x - width/2, values1, width, label='组1', color='tab:blue')
ax.bar(x + width/2, values2, width, label='组2', color='tab:orange')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_title('分组柱状图', fontsize=14)
ax.set_ylabel('数值')
ax.legend()
fig.tight_layout()
"""
        },
        "饼图": {
            "desc": "经典饼图，带百分比标签",
            "code": """# 饼图
labels = ['A', 'B', 'C', 'D', 'E']
sizes = [30, 25, 20, 15, 10]
explode = (0.05, 0, 0, 0, 0)

ax = fig.add_subplot(111)
ax.clear()
ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
       shadow=True, startangle=90, colors=plt.cm.Pastel1.colors)
ax.set_title('饼图', fontsize=14)
fig.tight_layout()
"""
        },
    },
    "统计图表": {
        "直方图+密度": {
            "desc": "数据分布直方图叠加核密度估计",
            "code": """# 直方图 + 密度曲线
np.random.seed(42)
data = np.random.normal(100, 15, 1000)

ax = fig.add_subplot(111)
ax.clear()
n, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.7, color='tab:blue', edgecolor='white')
mu, sigma = np.mean(data), np.std(data)
x_line = np.linspace(data.min(), data.max(), 100)
ax.plot(x_line, 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x_line-mu)**2/(2*sigma**2)),
        'r-', linewidth=2, label='正态拟合')
ax.set_title('直方图与密度曲线', fontsize=14)
ax.set_xlabel('数值')
ax.set_ylabel('密度')
ax.legend()
fig.tight_layout()
"""
        },
        "箱线图": {
            "desc": "多组数据箱线图对比",
            "code": """# 箱线图
np.random.seed(42)
data = [np.random.normal(0, std, 100) for std in range(1, 5)]
labels = ['A', 'B', 'C', 'D']

ax = fig.add_subplot(111)
ax.clear()
bp = ax.boxplot(data, labels=labels, patch_artist=True)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title('箱线图', fontsize=14)
ax.set_ylabel('数值')
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
"""
        },
    },
    "科学图表": {
        "三维曲面": {
            "desc": "3D 曲面图，展示二元函数",
            "code": """# 三维曲面图
from mpl_toolkits.mplot3d import Axes3D

X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))

ax = fig.add_subplot(111, projection='3d')
ax.clear()
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, edgecolor='none')
ax.set_title('三维曲面: z = sin(√(x²+y²))', fontsize=14)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
fig.tight_layout()
"""
        },
        "等高线图": {
            "desc": "等高线填充图，适合地形/势能面",
            "code": """# 等高线图
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2))

ax = fig.add_subplot(111)
ax.clear()
contourf = ax.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r')
contour = ax.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5)
ax.clabel(contour, inline=True, fontsize=8)
ax.set_title('等高线图: z = exp(-(x²+y²))', fontsize=14)
ax.set_xlabel('X')
ax.set_ylabel('Y')
fig.colorbar(contourf, ax=ax, label='Z 值')
fig.tight_layout()
"""
        },
        "热力图": {
            "desc": "矩阵热力图，展示相关性或数据强度",
            "code": """# 热力图
np.random.seed(42)
data = np.random.rand(10, 10)
row_labels = [f'R{i}' for i in range(1, 11)]
col_labels = [f'C{i}' for i in range(1, 11)]

ax = fig.add_subplot(111)
ax.clear()
im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
ax.set_xticks(np.arange(len(col_labels)))
ax.set_yticks(np.arange(len(row_labels)))
ax.set_xticklabels(col_labels)
ax.set_yticklabels(row_labels)
for i in range(10):
    for j in range(10):
        text = ax.text(j, i, f'{data[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)
ax.set_title('热力图', fontsize=14)
fig.colorbar(im, ax=ax, label='强度')
fig.tight_layout()
"""
        },
    },
}


# ════════════════════════════════════════
# 模板管理器（简化版）
# ════════════════════════════════════════

class TemplateManager:
    def __init__(self, filepath=None):
        if filepath is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.filepath = os.path.join(base_dir, "templates.json")
        else:
            self.filepath = filepath
        self.templates = self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["builtin"] = self._merge_builtin(data.get("builtin", {}))
                return data
            except Exception:
                pass
        data = {"builtin": dict(BUILTIN_TEMPLATES), "user": {}}
        self._save(data)
        return data

    def _merge_builtin(self, existing):
        merged = {}
        for category, items in BUILTIN_TEMPLATES.items():
            merged[category] = {}
            for name, tmpl in items.items():
                if category in existing and name in existing[category]:
                    existing_code = existing[category][name].get("code", "")
                    if existing_code == tmpl["code"]:
                        merged[category][name] = existing[category][name]
                    else:
                        merged[category][name] = existing[category][name]
                else:
                    merged[category][name] = dict(tmpl)
        return merged

    def _save(self, data=None):
        if data is None:
            data = self.templates
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_builtin(self):
        return self.templates.get("builtin", {})

    def get_user(self):
        return self.templates.get("user", {})

    def add_user_template(self, category, name, desc, code):
        if category not in self.templates["user"]:
            self.templates["user"][category] = {}
        self.templates["user"][category][name] = {"desc": desc, "code": code}
        self._save()

    def delete_user_template(self, category, name):
        user = self.templates.get("user", {})
        if category in user and name in user[category]:
            del user[category][name]
            if not user[category]:
                del user[category]
            self._save()
            return True
        return False

    def get_template_code(self, source, category, name):
        section = self.templates.get(source, {})
        cat = section.get(category, {})
        tmpl = cat.get(name, {})
        return tmpl.get("code", "")

    def get_template_info(self, source, category, name):
        section = self.templates.get(source, {})
        cat = section.get(category, {})
        return cat.get(name, {})


# ════════════════════════════════════════
# 行号画布
# ════════════════════════════════════════

class LineNumberCanvas(tk.Canvas):
    def __init__(self, master, text_widget, font_size=10, **kwargs):
        super().__init__(master, **kwargs)
        self.text_widget = text_widget
        self.font_size = font_size
        self.config(width=40, bg="#f0f0f0", highlightthickness=0)
        for ev in ("<Configure>", "<KeyRelease>", "<ButtonRelease>", "<MouseWheel>"):
            text_widget.bind(ev, lambda e: self.redraw())

    def redraw(self):
        self.delete("all")
        index = self.text_widget.index("@0,0")
        while True:
            dline = self.text_widget.dlineinfo(index)
            if dline is None:
                break
            y = dline[1]
            line_num = str(int(float(index)))
            self.create_text(35, y, anchor="ne", text=line_num,
                             font=("Consolas", self.font_size), fill="#555555")
            index = self.text_widget.index(f"{index}+1line")


# ════════════════════════════════════════
# 主应用 v5
# ════════════════════════════════════════

class CodePlotAppV5:
    def __init__(self, root):
        self.root = root
        self.root.title("CodePlot v5 — 图集排版模式")
        self.root.geometry("1700x1050")
        self.root.minsize(1200, 750)
        self.script_path = None
        self.tmpl_mgr = TemplateManager()
        self.gallery = FigureGallery()
        self._load_compose_settings()

        # 视图状态
        self.view_mode = "single"      # "single" | "overview"
        self.current_gallery_index = None   # 当前编辑的图集索引
        self._live_after_id = None

        self._build_ui()
        self._bind_shortcuts()
        self._load_default_template()

    # ────────────────── UI 构建 ──────────────────

    def _build_ui(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ══════ 左侧面板 ══════
        left_outer = ttk.Frame(main_paned)
        main_paned.add(left_outer, weight=1)

        left_paned = ttk.PanedWindow(left_outer, orient=tk.VERTICAL)
        left_paned.pack(fill=tk.BOTH, expand=True)

        # ── 模板面板 ──
        tmpl_frame = ttk.LabelFrame(left_paned, text="模板库", padding=4)
        left_paned.add(tmpl_frame, weight=1)

        tmpl_tb = ttk.Frame(tmpl_frame)
        tmpl_tb.pack(fill=tk.X, pady=(0, 2))
        ttk.Button(tmpl_tb, text="➕ 保存为模板", command=self._save_as_template).pack(side=tk.LEFT, padx=2)
        ttk.Button(tmpl_tb, text="🗑️ 删除", command=self._delete_template).pack(side=tk.LEFT, padx=2)

        self.tmpl_tree = ttk.Treeview(tmpl_frame, show="tree", selectmode="browse")
        self.tmpl_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tmpl_tree.bind("<<TreeviewSelect>>", self._on_tmpl_select)
        self.tmpl_tree.bind("<Double-1>", self._on_tmpl_double)

        tmpl_vsb = ttk.Scrollbar(tmpl_frame, orient=tk.VERTICAL, command=self.tmpl_tree.yview)
        tmpl_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tmpl_tree.config(yscrollcommand=tmpl_vsb.set)

        self._refresh_template_tree()

        ui_family = getattr(self.root, "ui_family", "Microsoft YaHei UI")
        self.tmpl_desc = ttk.Label(left_outer,
            text="提示: 单击查看描述, 双击追加为新子图",
            wraplength=300, foreground="#666666",
            font=(ui_family, -max(10, int(getattr(self.root, "ui_base", 13) * 0.75))))
        self.tmpl_desc.pack(fill=tk.X, padx=4, pady=2)

        # ── 代码编辑面板 ──
        code_frame = ttk.LabelFrame(left_paned, text="代码编辑器", padding=4)
        left_paned.add(code_frame, weight=2)

        code_tb = ttk.Frame(code_frame)
        code_tb.pack(fill=tk.X, pady=(0, 2))
        ttk.Button(code_tb, text="▶ 运行", command=self.run_code).pack(side=tk.LEFT, padx=2)
        ttk.Button(code_tb, text="💾 保存脚本", command=self.save_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(code_tb, text="📂 加载脚本", command=self.load_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(code_tb, text="🔄 重置", command=self._load_default_template).pack(side=tk.LEFT, padx=2)

        self.live_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(code_tb, text="⚡ 实时渲染", variable=self.live_var,
                        command=self._on_live_toggle).pack(side=tk.RIGHT, padx=6)

        # 快速标注工具栏
        anno_frame = ttk.LabelFrame(code_frame, text="快速标注工具（点击插入代码）", padding=2)
        anno_frame.pack(fill=tk.X, pady=(0, 2))
        anno_items = [
            ("➡️ 箭头", self._insert_arrow),
            ("⭕ 圆圈", self._insert_circle),
            ("📝 文本", self._insert_text),
            ("▭ 矩形", self._insert_rectangle),
            ("⎯ 水平线", self._insert_hline),
            ("⏐ 垂直线", self._insert_vline),
            ("🌟 高亮区", self._insert_highlight),
        ]
        for text, cmd in anno_items:
            ttk.Button(anno_frame, text=text, command=cmd).pack(side=tk.LEFT, padx=2, pady=1)

        # 代码编辑器
        editor_frame = ttk.Frame(code_frame)
        editor_frame.pack(fill=tk.BOTH, expand=True)

        # 字号用像素值（负数），以系统 UI 字体的实测行高为基准，
        # 不受 Tk 版本的高 DPI 缩放差异影响，任何环境下显示一致
        ui_base = getattr(self.root, "ui_base", 13)
        code_font_size = -max(13, int(ui_base))
        self.code_text = tk.Text(editor_frame, wrap=tk.NONE, undo=True,
                                  font=("Consolas", code_font_size), padx=6, pady=4,
                                  bg="#fafafa", fg="#333333",
                                  insertbackground="#333333",
                                  selectbackground="#b4d7ff")
        self.code_text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.code_text.bind("<KeyRelease>", self._on_code_key)

        line_canvas = LineNumberCanvas(editor_frame, self.code_text,
                                       font_size=-max(11, int(ui_base * 0.85)))
        line_canvas.config(width=max(40, int(ui_base * 2.6)))
        line_canvas.pack(side=tk.LEFT, fill=tk.Y)

        vsb = ttk.Scrollbar(code_frame, orient=tk.VERTICAL, command=self.code_text.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.code_text.config(yscrollcommand=vsb.set)

        # ══════ 右侧面板 ══════
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)

        # 图表工具栏
        ptb = ttk.Frame(right_frame)
        ptb.pack(fill=tk.X, pady=(0, 2))

        ttk.Label(ptb, text="DPI:").pack(side=tk.LEFT, padx=(4, 0))
        self.dpi_var = tk.StringVar(value="100")
        ttk.Combobox(ptb, textvariable=self.dpi_var,
                     values=["80", "100", "120", "150", "200"], width=6, state="readonly").pack(side=tk.LEFT)

        ttk.Separator(ptb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        ttk.Label(ptb, text="尺寸(英寸):").pack(side=tk.LEFT)
        self.fig_w_var = tk.StringVar(value="8")
        self.fig_h_var = tk.StringVar(value="6")
        ttk.Entry(ptb, textvariable=self.fig_w_var, width=4, justify=tk.CENTER).pack(side=tk.LEFT, padx=2)
        ttk.Label(ptb, text="×").pack(side=tk.LEFT)
        ttk.Entry(ptb, textvariable=self.fig_h_var, width=4, justify=tk.CENTER).pack(side=tk.LEFT, padx=2)
        ttk.Button(ptb, text="应用", command=self._apply_figsize).pack(side=tk.LEFT, padx=4)

        ttk.Separator(ptb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # 图集操作按钮
        ttk.Button(ptb, text="➕ 加入组图", command=self._add_to_gallery).pack(side=tk.LEFT, padx=2)
        ttk.Button(ptb, text="📋 总览排版", command=self._show_overview).pack(side=tk.LEFT, padx=2)
        ttk.Button(ptb, text="⚙ 组合设置", command=self._open_compose_settings).pack(side=tk.LEFT, padx=2)
        ttk.Button(ptb, text="🗑️ 清空图集", command=self._clear_gallery).pack(side=tk.LEFT, padx=2)

        ttk.Separator(ptb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        ttk.Button(ptb, text="📥 保存图片", command=self.save_figure).pack(side=tk.RIGHT, padx=4)
        ttk.Button(ptb, text="🔧 清除", command=self.clear_figure).pack(side=tk.RIGHT, padx=2)

        # 图集导航栏（缩略图列表）
        self.gallery_frame = ttk.LabelFrame(right_frame, text="图集（点击加载编辑）", padding=2)
        self.gallery_frame.pack(fill=tk.X, pady=(0, 2))
        self._refresh_gallery_nav()

        # 图表画布
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        nav = NavigationToolbar2Tk(self.canvas, right_frame, pack_toolbar=False)
        nav.pack(fill=tk.X)

        # 状态栏
        self.status = ttk.Label(self.root,
            text="就绪 | 运行代码 → 加入组图 → 总览排版",
            relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    # ────────────────── 图集相关 ──────────────────

    def _refresh_gallery_nav(self):
        """刷新图集导航栏缩略图"""
        for w in self.gallery_frame.winfo_children():
            w.destroy()

        if not self.gallery.items:
            ttk.Label(self.gallery_frame, text='（图集为空，运行代码后点击"加入组图"）').pack(pady=4)
            return

        self.gallery_photos = []
        for i, item in enumerate(self.gallery.items):
            frame = ttk.Frame(self.gallery_frame)
            frame.pack(side=tk.LEFT, padx=2, pady=1)

            # 加载缩略图
            photo = None
            if HAS_PIL and os.path.exists(item.get("png_path", "")):
                try:
                    img = Image.open(item["png_path"])
                    img.thumbnail((100, 75))
                    photo = ImageTk.PhotoImage(img)
                except Exception:
                    pass

            if photo:
                btn = tk.Button(frame, image=photo,
                                command=lambda idx=i: self._load_gallery_item(idx),
                                relief=tk.RIDGE, bd=2)
                btn.pack()
                self.gallery_photos.append(photo)
            else:
                ttk.Button(frame, text=f"图{i+1}",
                           command=lambda idx=i: self._load_gallery_item(idx)).pack()

            # 序号标签
            ttk.Label(frame, text=f"({chr(97+i)})", font=("Consolas", 9)).pack()

            # 删除按钮
            ttk.Button(frame, text="✕", width=2,
                       command=lambda idx=i: self._remove_gallery_item(idx)).pack()

    def _add_to_gallery(self):
        """把当前图加入图集"""
        if self.view_mode == "overview":
            messagebox.showwarning(
                "提示",
                "当前是总览排版视图，不能把总图再次加入图集。\n"
                "请先点击图集中的某张单图（或运行代码）回到单图模式。")
            return

        code = self.code_text.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("提示", "代码为空")
            return

        # 确保有图像可保存
        if len(self.fig.axes) == 0:
            messagebox.showwarning("提示", "请先运行代码生成图像")
            return

        title = f"图{len(self.gallery.items)+1}"
        item = self.gallery.add(code, self.fig, title=title)
        self._refresh_gallery_nav()
        self.status.config(text=f"已加入图集: {title} | 共 {len(self.gallery.items)} 张")

    def _load_gallery_item(self, index):
        """加载图集中的某张图进行编辑"""
        if index < 0 or index >= len(self.gallery.items):
            return
        item = self.gallery.items[index]
        self.current_gallery_index = index
        self.view_mode = "single"

        self.code_text.delete("1.0", tk.END)
        self.code_text.insert("1.0", item["code"])
        self.run_code()
        self.status.config(text=f"编辑图集项目: ({chr(97+index)}) | 修改后可重新「加入组图」")

    def _remove_gallery_item(self, index):
        """删除图集中的某张图"""
        if messagebox.askyesno("确认删除", f"确定要删除图 ({chr(97+index)}) 吗？"):
            self.gallery.remove(index)
            self._refresh_gallery_nav()
            self.status.config(text=f"已删除 | 图集剩余 {len(self.gallery.items)} 张")

    def _clear_gallery(self):
        """清空图集"""
        if not self.gallery.items:
            return
        if messagebox.askyesno("确认清空", "确定要清空整个图集吗？"):
            self.gallery.clear()
            self._refresh_gallery_nav()
            self.status.config(text="图集已清空")

    def _show_overview(self):
        """总览排版模式：把图集中所有图按网格排列，序号标在左上角"""
        n = len(self.gallery.items)
        if n == 0:
            messagebox.showinfo("提示", "图集为空，请先加入组图")
            return

        self.view_mode = "overview"
        self.current_gallery_index = None

        rows, cols = self.gallery.get_best_grid(n)

        try:
            dpi = int(self.dpi_var.get())
        except ValueError:
            dpi = 100
        try:
            w = float(self.fig_w_var.get())
            h = float(self.fig_h_var.get())
        except ValueError:
            w, h = 8, 6

        # 彻底清理旧图层（含 colorbar 等附属 axes）
        self.fig.clf()
        # 预览总图按网格放大：每个子图以接近原始尺寸显示，字体大小才一致；
        # 但不能超过画布控件像素尺寸，否则 Tk 后端 blit 会裁掉超出部分
        nat_w, nat_h = w * cols, h * rows
        try:
            cw = self.canvas.get_tk_widget().winfo_width()
            ch = self.canvas.get_tk_widget().winfo_height()
            if cw > 10 and ch > 10:
                shrink = min(1.0, cw / (nat_w * dpi), ch / (nat_h * dpi))
                nat_w, nat_h = nat_w * shrink, nat_h * shrink
        except Exception:
            pass
        self.fig.set_size_inches(nat_w, nat_h)
        self.fig.set_dpi(dpi)

        # 统一尺寸重渲染每个子图（保证组合后字体一致），失败则回退存档文件
        cache_dir = os.path.join(self.gallery.base_dir, "_overview_cache")
        fresh_svgs = []
        for i, item in enumerate(self.gallery.items):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            ax.set_axis_off()

            svg_path, png_path = self._render_item_uniform(
                item, w, h, dpi, cache_dir)
            if png_path is None:
                svg_path = item.get("svg_path", "")
                png_path = item.get("png_path", "")
            fresh_svgs.append(svg_path)

            shown = False
            if png_path and os.path.exists(png_path):
                try:
                    ax.imshow(mpimg.imread(png_path))
                    shown = True
                except Exception:
                    pass
            if not shown:
                ax.text(0.5, 0.5, f"图{i+1}", ha="center", va="center", fontsize=20)

            # 序号 (a)(b)(c) 标在子图左上角
            ax.text(0.01, 0.99, f"({chr(97 + i)})", transform=ax.transAxes,
                    fontsize=13, fontweight="bold", ha="left", va="top",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.75, pad=1.5))

        self.fig.tight_layout()
        self._redraw_canvas()

        # 同步生成矢量组合 SVG（嵌套各子图 SVG，序号在排版时加入）
        composed = os.path.join(self.gallery.base_dir, "composed.svg")
        svg_note = ""
        if self._compose_with_engine(composed, fresh_svgs):
            svg_note = f" | 矢量组图: {composed}"

        self.status.config(
            text=f"总览排版: {n} 张图 | {rows}×{cols} 网格{svg_note} | 点击单图可编辑")

    def _compose_gallery_svg(self, out_path, svg_paths=None):
        """把图集中各子图的 SVG 以纯矢量方式拼成一张总图。

        子图 SVG 内不含序号；序号 (a)(b)(c) 在组合排版时加在每张子图左上角。
        原理: 把每张子图 SVG 作为嵌套 <svg x y width height viewBox> 元素放入
        一个大的容器 SVG 中，保持矢量不栅格化。
        svg_paths: 可选，与图集条目一一对应的 SVG 路径（如统一重渲染结果）；
                   为 None 时使用图集存档的 SVG。
        """
        import re
        n = len(self.gallery.items)
        if n == 0:
            return False
        rows, cols = self.gallery.get_best_grid(n)

        entries = []
        max_w = max_h = 0.0
        for idx, item in enumerate(self.gallery.items):
            if svg_paths is not None and idx < len(svg_paths) and svg_paths[idx]:
                svg_path = svg_paths[idx]
            else:
                svg_path = item.get("svg_path", "")
            if not os.path.exists(svg_path):
                continue
            try:
                with open(svg_path, "r", encoding="utf-8") as f:
                    content = f.read()
                mw = re.search(r'width="([\d.]+)pt"', content)
                mh = re.search(r'height="([\d.]+)pt"', content)
                mv = re.search(r'viewBox="([^"]+)"', content)
                if not (mw and mh and mv):
                    continue
                w, h = float(mw.group(1)), float(mh.group(1))
                start = content.index(">", content.index("<svg")) + 1
                inner = content[start:content.rindex("</svg>")]
                entries.append({"w": w, "h": h, "vb": mv.group(1), "inner": inner})
                max_w = max(max_w, w)
                max_h = max(max_h, h)
            except Exception:
                continue

        if not entries:
            return False

        pad = 18.0        # 子图间距 (pt)
        label_fs = 16.0   # 序号字号 (pt)
        total_w = cols * max_w + (cols + 1) * pad
        total_h = rows * max_h + (rows + 1) * pad

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{total_w:.1f}pt" height="{total_h:.1f}pt" '
            f'viewBox="0 0 {total_w:.2f} {total_h:.2f}">',
            '<rect width="100%" height="100%" fill="white"/>',
        ]
        for i, e in enumerate(entries):
            r, c = divmod(i, cols)
            cell_x = pad + c * (max_w + pad)
            cell_y = pad + r * (max_h + pad)
            # 等比缩放到单元格内并居中
            scale = min(max_w / e["w"], max_h / e["h"])
            dw, dh = e["w"] * scale, e["h"] * scale
            ox = cell_x + (max_w - dw) / 2
            oy = cell_y + (max_h - dh) / 2
            parts.append(
                f'<svg x="{ox:.2f}" y="{oy:.2f}" width="{dw:.2f}" height="{dh:.2f}" '
                f'viewBox="{e["vb"]}">{e["inner"]}</svg>'
            )
            # 左上角序号（白色衬底保证在复杂图上也清晰）
            label = f"({chr(97 + i)})"
            lx, ly = ox + 2, oy + label_fs
            bw = len(label) * label_fs * 0.62 + 4
            parts.append(
                f'<rect x="{lx - 2:.2f}" y="{ly - label_fs:.2f}" '
                f'width="{bw:.2f}" height="{label_fs + 4:.2f}" '
                f'fill="white" fill-opacity="0.75"/>'
                f'<text x="{lx:.2f}" y="{ly:.2f}" '
                f'font-family="DejaVu Sans, Arial, sans-serif" '
                f'font-size="{label_fs}" font-weight="bold">{label}</text>'
            )
        parts.append("</svg>")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))
        return True

    # ────────────────── 组合排版设置（compose_figure 引擎）──────────────────

    def _compose_settings_path(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "compose_settings.json")

    def _load_compose_settings(self):
        """启动时把持久化的组合设置写回 compose_figure 引擎"""
        if not HAS_COMPOSE_ENGINE:
            return
        try:
            with open(self._compose_settings_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            for k in COMPOSE_SETTING_KEYS:
                if k in data:
                    setattr(cf_engine, k, data[k])
        except Exception:
            pass

    def _save_compose_settings(self):
        if not HAS_COMPOSE_ENGINE:
            return
        try:
            data = {k: getattr(cf_engine, k) for k in COMPOSE_SETTING_KEYS
                    if hasattr(cf_engine, k)}
            with open(self._compose_settings_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _compose_with_engine(self, out_path, svg_paths):
        """用 compose_figure 引擎组合 SVG（应用组合设置）；引擎不可用时回退内置方法"""
        if HAS_COMPOSE_ENGINE:
            try:
                cf_engine.compose(svg_paths, out_path)
                if cf_engine.EXPORT_PDF or cf_engine.EXPORT_PNG:
                    cf_engine.export_extra(out_path)
                return True
            except SystemExit as e:
                self.status.config(text=f"组合失败: {e}")
                return False
            except Exception as e:
                self.status.config(text=f"组合失败: {str(e)[:60]}")
                return False
        return self._compose_gallery_svg(out_path, svg_paths=svg_paths)

    def _open_compose_settings(self):
        """组合排版设置对话框：序号字体/位置、网格尺寸、输出选项"""
        if not HAS_COMPOSE_ENGINE:
            messagebox.showwarning("提示", "未找到 compose_figure.py（需与 codeplot_v5.py 同目录）")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("组合排版设置")
        dlg.transient(self.root)
        dlg.grab_set()

        def row(parent, r, label, widget):
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky=tk.W,
                                               padx=6, pady=2)
            widget.grid(row=r, column=1, sticky=tk.W, padx=6, pady=2)

        # ── 序号设置 ──
        f1 = ttk.LabelFrame(dlg, text="序号 (a)(b)(c)", padding=6)
        f1.pack(fill=tk.X, padx=8, pady=4)

        ui_family = getattr(self.root, "ui_family", "Microsoft YaHei UI")
        family_var = tk.StringVar(value=cf_engine.LABEL_FONT_FAMILY)
        row(f1, 0, "字体:", ttk.Combobox(f1, textvariable=family_var, width=18,
            values=[ui_family, "Arial", "Times New Roman", "HONOR Sans",
                    "Microsoft YaHei UI", "SimSun", "SimHei"]))

        size_var = tk.StringVar(value=str(cf_engine.LABEL_FONT_SIZE))
        row(f1, 1, "字号(pt):", ttk.Entry(f1, textvariable=size_var, width=8))

        weight_var = tk.StringVar(value=cf_engine.LABEL_FONT_WEIGHT)
        row(f1, 2, "字重:", ttk.Combobox(f1, textvariable=weight_var, width=8,
            values=["bold", "normal"], state="readonly"))

        color_var = tk.StringVar(value=cf_engine.LABEL_COLOR)
        row(f1, 3, "颜色:", ttk.Combobox(f1, textvariable=color_var, width=10,
            values=["black", "white", "red", "blue", "gray"]))

        style_var = tk.StringVar(value=cf_engine.LABEL_STYLE)
        row(f1, 4, "编号样式:", ttk.Combobox(f1, textvariable=style_var, width=10,
            values=["lower", "upper", "number"], state="readonly"))

        brackets_var = tk.BooleanVar(value=bool(cf_engine.LABEL_BRACKETS))
        row(f1, 5, "带括号:", ttk.Checkbutton(f1, variable=brackets_var))

        pos_var = tk.StringVar(value=cf_engine.LABEL_POSITION)
        row(f1, 6, "位置:", ttk.Combobox(f1, textvariable=pos_var, width=12,
            values=["top-left", "top-right", "bottom-left", "bottom-right"],
            state="readonly"))

        ox_var = tk.StringVar(value=str(cf_engine.LABEL_OFFSET_X))
        oy_var = tk.StringVar(value=str(cf_engine.LABEL_OFFSET_Y))
        row(f1, 7, "偏移X(pt):", ttk.Entry(f1, textvariable=ox_var, width=8))
        row(f1, 8, "偏移Y(pt):", ttk.Entry(f1, textvariable=oy_var, width=8))

        bg_var = tk.BooleanVar(value=bool(cf_engine.LABEL_BACKGROUND))
        row(f1, 9, "白色衬底:", ttk.Checkbutton(f1, variable=bg_var))

        # ── 布局设置 ──
        f2 = ttk.LabelFrame(dlg, text="网格布局（custom 布局请编辑 compose_figure.py 中 CUSTOM_PANELS）", padding=6)
        f2.pack(fill=tk.X, padx=8, pady=4)

        rows_var = tk.StringVar(value=str(cf_engine.GRID_ROWS))
        cols_var = tk.StringVar(value=str(cf_engine.GRID_COLS))
        row(f2, 0, "行数:", ttk.Entry(f2, textvariable=rows_var, width=6))
        row(f2, 1, "列数:", ttk.Entry(f2, textvariable=cols_var, width=6))

        pw_var = tk.StringVar(value=str(cf_engine.PANEL_WIDTH))
        ph_var = tk.StringVar(value=str(cf_engine.PANEL_HEIGHT))
        row(f2, 2, "子图宽(pt):", ttk.Entry(f2, textvariable=pw_var, width=8))
        row(f2, 3, "子图高(pt):", ttk.Entry(f2, textvariable=ph_var, width=8))

        hs_var = tk.StringVar(value=str(cf_engine.H_SPACING))
        vs_var = tk.StringVar(value=str(cf_engine.V_SPACING))
        mg_var = tk.StringVar(value=str(cf_engine.MARGIN))
        row(f2, 4, "列间距(pt):", ttk.Entry(f2, textvariable=hs_var, width=8))
        row(f2, 5, "行间距(pt):", ttk.Entry(f2, textvariable=vs_var, width=8))
        row(f2, 6, "边距(pt):", ttk.Entry(f2, textvariable=mg_var, width=8))

        # ── 输出设置 ──
        f3 = ttk.LabelFrame(dlg, text="输出", padding=6)
        f3.pack(fill=tk.X, padx=8, pady=4)

        pdf_var = tk.BooleanVar(value=bool(cf_engine.EXPORT_PDF))
        png_var = tk.BooleanVar(value=bool(cf_engine.EXPORT_PNG))
        dpi_var = tk.StringVar(value=str(cf_engine.PNG_DPI))
        row(f3, 0, "同时导出PDF:", ttk.Checkbutton(f3, variable=pdf_var))
        row(f3, 1, "同时导出PNG:", ttk.Checkbutton(f3, variable=png_var))
        row(f3, 2, "PNG DPI:", ttk.Entry(f3, textvariable=dpi_var, width=8))

        def apply_settings():
            try:
                cf_engine.LABEL_FONT_FAMILY = family_var.get().strip() or "Arial"
                cf_engine.LABEL_FONT_SIZE = int(float(size_var.get()))
                cf_engine.LABEL_FONT_WEIGHT = weight_var.get()
                cf_engine.LABEL_COLOR = color_var.get().strip() or "black"
                cf_engine.LABEL_STYLE = style_var.get()
                cf_engine.LABEL_BRACKETS = brackets_var.get()
                cf_engine.LABEL_POSITION = pos_var.get()
                cf_engine.LABEL_OFFSET_X = float(ox_var.get())
                cf_engine.LABEL_OFFSET_Y = float(oy_var.get())
                cf_engine.LABEL_BACKGROUND = bg_var.get()
                cf_engine.GRID_ROWS = int(float(rows_var.get()))
                cf_engine.GRID_COLS = int(float(cols_var.get()))
                cf_engine.PANEL_WIDTH = float(pw_var.get())
                cf_engine.PANEL_HEIGHT = float(ph_var.get())
                cf_engine.H_SPACING = float(hs_var.get())
                cf_engine.V_SPACING = float(vs_var.get())
                cf_engine.MARGIN = float(mg_var.get())
                cf_engine.EXPORT_PDF = pdf_var.get()
                cf_engine.EXPORT_PNG = png_var.get()
                cf_engine.PNG_DPI = int(float(dpi_var.get()))
            except ValueError as e:
                messagebox.showwarning("提示", f"数值格式有误: {e}", parent=dlg)
                return
            self._save_compose_settings()
            dlg.destroy()
            # 图集非空时立即用新设置重新组合
            if self.gallery.items:
                self._show_overview()
            self.status.config(text="组合设置已保存并应用")

        ttk.Button(dlg, text="确定", command=apply_settings).pack(pady=8)

    # ────────────────── 尺寸与缩放 ──────────────────

    def _apply_figsize(self):
        try:
            w = float(self.fig_w_var.get())
            h = float(self.fig_h_var.get())
            if w < 1 or h < 1 or w > 30 or h > 30:
                raise ValueError("尺寸范围 1~30 英寸")
        except ValueError as e:
            self.status.config(text=f"尺寸格式错误: {e}")
            return
        self.fig.set_size_inches(w, h)
        self.canvas.resize_event()
        self._redraw_canvas()
        self.status.config(text=f"Figure 尺寸已设为 {w:.1f} × {h:.1f} 英寸")

    # ────────────────── 实时渲染 ──────────────────

    def _on_live_toggle(self):
        if self.live_var.get():
            self.status.config(text="⚡ 实时渲染已开启 | 停止输入 400ms 后自动刷新")
            self._trigger_live_run()
        else:
            self.status.config(text="实时渲染已关闭 | 按 Ctrl+R 手动运行")
            if self._live_after_id is not None:
                self.root.after_cancel(self._live_after_id)
                self._live_after_id = None

    def _on_code_key(self, event=None):
        if not self.live_var.get():
            return
        if event and event.keysym in ("Shift_L", "Shift_R", "Control_L", "Control_R",
                                       "Alt_L", "Alt_R", "Up", "Down", "Left", "Right",
                                       "Prior", "Next", "Home", "End", "Caps_Lock"):
            return
        self._trigger_live_run()

    def _trigger_live_run(self):
        if self._live_after_id is not None:
            self.root.after_cancel(self._live_after_id)
        self._live_after_id = self.root.after(400, self._live_run)

    def _live_run(self):
        self._live_after_id = None
        self.run_code(silent=True)

    # ────────────────── 标注工具（fig.gca() 版本，兼容多子图）──────────────────

    def _insert_at_cursor(self, snippet):
        self.code_text.insert(tk.INSERT, snippet)
        self.status.config(text="已插入标注代码")
        if self.live_var.get():
            self._trigger_live_run()

    def _insert_arrow(self):
        snippet = """# 添加箭头标注
fig.gca().annotate('', xy=(0.5, 0.5), xytext=(0.2, 0.2),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
"""
        self._insert_at_cursor(snippet)

    def _insert_circle(self):
        snippet = """# 添加圆圈
from matplotlib.patches import Circle
circle = Circle((0.5, 0.5), 0.1, fill=False, edgecolor='red', linewidth=2)
fig.gca().add_patch(circle)
"""
        self._insert_at_cursor(snippet)

    def _insert_text(self):
        snippet = """# 添加文本标注
fig.gca().text(0.5, 0.5, '你的文本', fontsize=12, color='red',
        ha='center', va='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
"""
        self._insert_at_cursor(snippet)

    def _insert_rectangle(self):
        snippet = """# 添加矩形框
from matplotlib.patches import Rectangle
rect = Rectangle((0.3, 0.3), 0.4, 0.4, fill=False, edgecolor='blue', linewidth=2, linestyle='--')
fig.gca().add_patch(rect)
"""
        self._insert_at_cursor(snippet)

    def _insert_hline(self):
        snippet = """# 添加水平参考线
fig.gca().axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='参考线')
"""
        self._insert_at_cursor(snippet)

    def _insert_vline(self):
        snippet = """# 添加垂直参考线
fig.gca().axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='参考线')
"""
        self._insert_at_cursor(snippet)

    def _insert_highlight(self):
        snippet = """# 添加高亮区域
fig.gca().axvspan(0.3, 0.7, color='yellow', alpha=0.2, label='高亮区')
# 或水平区域: fig.gca().axhspan(ymin, ymax, ...)
"""
        self._insert_at_cursor(snippet)

    # ────────────────── 模板相关 ──────────────────

    def _refresh_template_tree(self):
        for item in self.tmpl_tree.get_children():
            self.tmpl_tree.delete(item)
        builtin_node = self.tmpl_tree.insert("", tk.END, text="📚 内置模板", open=True)
        for category, items in self.tmpl_mgr.get_builtin().items():
            cat_node = self.tmpl_tree.insert(builtin_node, tk.END, text=f"  {category}", open=False)
            for name in items:
                self.tmpl_tree.insert(cat_node, tk.END, text=f"    {name}",
                                      values=("builtin", category, name))
        user_node = self.tmpl_tree.insert("", tk.END, text="👤 我的模板", open=True)
        for category, items in self.tmpl_mgr.get_user().items():
            cat_node = self.tmpl_tree.insert(user_node, tk.END, text=f"  {category}", open=False)
            for name in items:
                self.tmpl_tree.insert(cat_node, tk.END, text=f"    {name}",
                                      values=("user", category, name))
        if not self.tmpl_mgr.get_user():
            self.tmpl_tree.insert(user_node, tk.END, text="    (暂无自定义模板)")

    def _on_tmpl_select(self, event=None):
        sel = self.tmpl_tree.selection()
        if not sel:
            return
        item = sel[0]
        vals = self.tmpl_tree.item(item, "values")
        if not vals or len(vals) < 3:
            return
        source, category, name = vals[0], vals[1], vals[2]
        info = self.tmpl_mgr.get_template_info(source, category, name)
        desc = info.get("desc", "无描述")
        self.tmpl_desc.config(text=f"【{name}】{desc}  |  双击追加为子图块")

    # ────────────────── 多段脚本（子图块）──────────────────

    def _make_subplot_block(self, index, name, code):
        """把模板代码包装成一个子图块：fig 变量改名为 figN 并注入创建语句"""
        code = re.sub(r"\bfig\b", f"fig{index}", code.strip())
        return (f"# ═══ 子图 {index}: {name} ═══\n"
                f"fig{index} = new_figure()\n"
                f"{code}\n")

    def _default_compose_block(self):
        return (
            f"{COMPOSE_BLOCK_HEADER}\n"
            "# 默认按 new_figure() 创建顺序排列；调整顺序示例: compose([fig2, fig1])\n"
            "# 常用参数: rows=2, cols=2, panel_width=420, panel_height=320,\n"
            "#           label_font_family='Arial', label_font_size=20,\n"
            "#           label_position='top-left', label_style='lower'\n"
            "compose()\n"
        )

    def _append_template_block(self, name, code):
        """把模板作为新的子图块追加到脚本中（排版块之前）"""
        text = self.code_text.get("1.0", tk.END)
        indices = [int(m.group(1)) for m in SUBPLOT_BLOCK_RE.finditer(text)]
        n = max(indices) + 1 if indices else 1
        block = self._make_subplot_block(n, name, code)
        if COMPOSE_BLOCK_HEADER in text:
            pos = text.find(COMPOSE_BLOCK_HEADER)
            line_start = text.rfind("\n", 0, pos) + 1
            new_text = text[:line_start].rstrip() + "\n\n" + block + "\n" + text[line_start:]
        else:
            new_text = text.rstrip() + "\n\n" + block + "\n" + self._default_compose_block()
        self.code_text.delete("1.0", tk.END)
        self.code_text.insert("1.0", new_text)
        return n

    def _on_tmpl_double(self, event=None):
        sel = self.tmpl_tree.selection()
        if not sel:
            return
        item = sel[0]
        vals = self.tmpl_tree.item(item, "values")
        if not vals or len(vals) < 3:
            return
        source, category, name = vals[0], vals[1], vals[2]
        code = self.tmpl_mgr.get_template_code(source, category, name)
        if code:
            n = self._append_template_block(name, code)
            self.status.config(text=f"已添加子图 {n}: {name} | Ctrl+R 运行")
            self.run_code()

    def _save_as_template(self):
        code = self.code_text.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("提示", "代码为空")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("保存为模板")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="分类:").pack(anchor=tk.W, padx=10, pady=(10, 2))
        cat_var = tk.StringVar(value="自定义")
        ttk.Combobox(dialog, textvariable=cat_var,
                     values=list(self.tmpl_mgr.get_user().keys()) + ["自定义"]).pack(fill=tk.X, padx=10)

        ttk.Label(dialog, text="模板名称:").pack(anchor=tk.W, padx=10, pady=(10, 2))
        name_var = tk.StringVar(value="我的模板")
        ttk.Entry(dialog, textvariable=name_var).pack(fill=tk.X, padx=10)

        ttk.Label(dialog, text="描述:").pack(anchor=tk.W, padx=10, pady=(10, 2))
        desc_var = tk.StringVar(value="")
        ttk.Entry(dialog, textvariable=desc_var).pack(fill=tk.X, padx=10)

        def do_save():
            cat = cat_var.get().strip()
            name = name_var.get().strip()
            desc = desc_var.get().strip() or "用户自定义模板"
            if not cat or not name:
                messagebox.showwarning("提示", "分类和名称不能为空")
                return
            self.tmpl_mgr.add_user_template(cat, name, desc, code)
            self._refresh_template_tree()
            self.status.config(text=f"模板已保存: {cat} / {name}")
            dialog.destroy()

        ttk.Button(dialog, text="保存", command=do_save).pack(pady=15)

    def _delete_template(self):
        sel = self.tmpl_tree.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选择一个模板")
            return
        item = sel[0]
        vals = self.tmpl_tree.item(item, "values")
        if not vals or len(vals) < 3:
            return
        source, category, name = vals[0], vals[1], vals[2]
        if source != "user":
            messagebox.showwarning("提示", "只能删除用户自定义模板")
            return
        if messagebox.askyesno("确认删除", f"确定要删除模板「{name}」吗？"):
            if self.tmpl_mgr.delete_user_template(category, name):
                self._refresh_template_tree()
                self.status.config(text=f"已删除模板: {name}")

    def _load_default_template(self):
        code = self.tmpl_mgr.get_template_code("builtin", "基础图表", "正弦+余弦波")
        text = (self._make_subplot_block(1, "正弦+余弦波", code)
                + "\n" + self._default_compose_block())
        self.code_text.delete("1.0", tk.END)
        self.code_text.insert("1.0", text)
        self.status.config(text="已加载默认示例 | 双击模板追加子图 | Ctrl+R 运行")
        self.run_code()

    # ────────────────── 核心逻辑 ──────────────────

    def _bind_shortcuts(self):
        self.root.bind("<Control-r>", lambda e: self.run_code())
        self.root.bind("<Control-R>", lambda e: self.run_code())
        self.root.bind("<Control-s>", lambda e: self.save_script())
        self.root.bind("<Control-S>", lambda e: self.save_script())

    def _redraw_canvas(self):
        """重绘画布。

        matplotlib 的 Tk 后端把图像写进一块 _tkphoto 位图；当 figure 从
        大尺寸缩小时，_tkphoto 仍保持旧尺寸，blit 只覆盖新图区域，
        旧像素会残留在画布上。先 blank() 清空位图再 draw 即可解决。
        """
        try:
            self.canvas._tkphoto.blank()
        except Exception:
            pass
        self.canvas.draw()

    @staticmethod
    def _make_exec_globals(fig):
        """构造用户代码的执行环境"""
        try:
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            Axes3D = None
        try:
            from scipy import stats
        except ImportError:
            stats = None
        return {
            "np": np, "fig": fig, "ax": None,
            "plt": plt, "Axes3D": Axes3D, "stats": stats,
        }

    def _render_item_uniform(self, item, w, h, dpi, cache_dir):
        """以统一尺寸重新执行图集代码，返回 (svg_path, png_path)。

        总览排版时用它保证每个子图物理尺寸一致（不加 bbox_inches='tight'），
        这样组合后各子图的字体大小完全一致。执行失败返回 (None, None)，
        调用方回退到图集存档文件。
        """
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            fig = Figure(figsize=(w, h), dpi=dpi)
            FigureCanvasAgg(fig)
            exec(item["code"], self._make_exec_globals(fig))
            if not fig.axes:
                return None, None
            os.makedirs(cache_dir, exist_ok=True)
            uid = item.get("id", "item")
            svg_path = os.path.join(cache_dir, f"{uid}.svg")
            png_path = os.path.join(cache_dir, f"{uid}.png")
            fig.savefig(svg_path, format="svg")
            fig.savefig(png_path, format="png", dpi=150)
            return svg_path, png_path
        except Exception:
            return None, None

    def _make_script_helpers(self, w, h, dpi):
        """多段脚本模式：向用户脚本提供 new_figure() 和 compose()。

        new_figure() 按当前尺寸/DPI 设置创建一张子图并登记；
        compose() 把所有（或指定的）子图保存为 SVG，经 compose_figure 引擎
        组合成总图，并在画布上显示网格预览。
        """
        app = self
        run_figs = []

        def new_figure():
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            f = Figure(figsize=(w, h), dpi=dpi)
            FigureCanvasAgg(f)
            run_figs.append(f)
            return f

        def compose(figs=None, rows=None, cols=None, **opts):
            flist = list(figs) if figs else list(run_figs)
            if not flist:
                raise ValueError("没有可组合的子图（请先用 new_figure() 创建）")
            n = len(flist)
            if rows and cols:
                r, c = int(rows), int(cols)
            else:
                r, c = app.gallery.get_best_grid(n)

            # 保存各子图为 SVG/PNG（固定尺寸，不用 tight 裁剪，保证字体一致）
            cache_dir = os.path.join(app.gallery.base_dir, "_script_cache")
            os.makedirs(cache_dir, exist_ok=True)
            svg_paths, png_paths = [], []
            for i, f in enumerate(flist):
                sp = os.path.join(cache_dir, f"script_{i}.svg")
                pp = os.path.join(cache_dir, f"script_{i}.png")
                f.savefig(sp, format="svg")
                f.savefig(pp, format="png", dpi=150)
                svg_paths.append(sp)
                png_paths.append(pp)

            out_path = os.path.join(app.gallery.base_dir, "composed.svg")
            if HAS_COMPOSE_ENGINE:
                key_map = {
                    "panel_width": "PANEL_WIDTH", "panel_height": "PANEL_HEIGHT",
                    "h_spacing": "H_SPACING", "v_spacing": "V_SPACING",
                    "margin": "MARGIN",
                    "label_font_family": "LABEL_FONT_FAMILY",
                    "label_font_size": "LABEL_FONT_SIZE",
                    "label_font_weight": "LABEL_FONT_WEIGHT",
                    "label_color": "LABEL_COLOR",
                    "label_style": "LABEL_STYLE",
                    "label_brackets": "LABEL_BRACKETS",
                    "label_position": "LABEL_POSITION",
                    "label_background": "LABEL_BACKGROUND",
                    "label_offset_x": "LABEL_OFFSET_X",
                    "label_offset_y": "LABEL_OFFSET_Y",
                }
                saved = {"GRID_ROWS": cf_engine.GRID_ROWS,
                         "GRID_COLS": cf_engine.GRID_COLS}
                cf_engine.GRID_ROWS, cf_engine.GRID_COLS = r, c
                for k, v in opts.items():
                    key = key_map.get(k, k.upper())
                    if hasattr(cf_engine, key):
                        saved.setdefault(key, getattr(cf_engine, key))
                        setattr(cf_engine, key, v)
                try:
                    labels = [cf_engine.make_label(i) for i in range(n)]
                    cf_engine.compose(svg_paths, out_path)
                    if cf_engine.EXPORT_PDF or cf_engine.EXPORT_PNG:
                        cf_engine.export_extra(out_path)
                finally:
                    for k, v in saved.items():
                        setattr(cf_engine, k, v)
            else:
                labels = [f"({chr(97 + i)})" for i in range(n)]
                app._compose_gallery_svg(out_path, svg_paths=svg_paths)

            # 画布网格预览
            app.fig.clf()
            nat_w, nat_h = w * c, h * r
            try:
                cw = app.canvas.get_tk_widget().winfo_width()
                ch = app.canvas.get_tk_widget().winfo_height()
                if cw > 10 and ch > 10:
                    shrink = min(1.0, cw / (nat_w * dpi), ch / (nat_h * dpi))
                    nat_w, nat_h = nat_w * shrink, nat_h * shrink
            except Exception:
                pass
            app.fig.set_size_inches(nat_w, nat_h)
            app.fig.set_dpi(dpi)
            for i, pp in enumerate(png_paths):
                ax = app.fig.add_subplot(r, c, i + 1)
                ax.set_axis_off()
                try:
                    ax.imshow(mpimg.imread(pp))
                except Exception:
                    ax.text(0.5, 0.5, f"图{i+1}", ha="center", va="center",
                            fontsize=20)
                ax.text(0.01, 0.99, labels[i], transform=ax.transAxes,
                        fontsize=13, fontweight="bold", ha="left", va="top",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.75, pad=1.5))
            app.fig.tight_layout()
            app._redraw_canvas()
            app.view_mode = "overview"
            app._script_composed = out_path
            return out_path

        return {"new_figure": new_figure, "compose": compose}

    def run_code(self, event=None, silent=False):
        code = self.code_text.get("1.0", tk.END).strip()
        if not code:
            if not silent:
                self.status.config(text="错误: 代码为空")
            return

        # 运行代码即回到单图编辑模式（总览图层随后会被 fig.clear() 清掉）
        self.view_mode = "single"

        if not silent:
            self.status.config(text="正在运行...")
            self.root.update_idletasks()

        try:
            dpi = int(self.dpi_var.get())
        except ValueError:
            dpi = 100

        w, h = 8.0, 6.0
        try:
            w = float(self.fig_w_var.get())
            h = float(self.fig_h_var.get())
            self.fig.set_size_inches(w, h)
        except ValueError:
            pass

        self.fig.clear()
        self.fig.set_dpi(dpi)

        exec_globals = self._make_exec_globals(self.fig)
        exec_globals.update(self._make_script_helpers(w, h, dpi))
        self._script_composed = None

        try:
            exec(code, exec_globals)
            if self._script_composed:
                # 脚本调用了 compose()：预览与状态由 compose 处理
                if not silent:
                    self.status.config(
                        text=f"组合完成: {self._script_composed} | "
                             f"修改排版块代码可调整顺序与样式")
            else:
                self._redraw_canvas()
                n_axes = len(self.fig.axes)
                size_info = f"{self.fig.get_size_inches()[0]:.1f}×{self.fig.get_size_inches()[1]:.1f}英寸"
                if n_axes > 1:
                    msg = f"运行成功 | 注意: 检测到 {n_axes} 个 axes，建议拆分为单图加入组图"
                else:
                    msg = f"运行成功 | DPI={dpi} | {size_info}"
                if not silent:
                    self.status.config(text=msg)
        except Exception as e:
            err = traceback.format_exc()
            short_err = str(e)[:80]
            if silent:
                self.status.config(text=f"⚡实时: {short_err}")
            else:
                self.status.config(text=f"错误: {short_err}")
                self._show_error(err)

    def _show_error(self, err_text):
        win = tk.Toplevel(self.root)
        win.title("运行错误")
        win.geometry("700x400")
        txt = tk.Text(win, wrap=tk.WORD, font=("Consolas", 10), bg="#fff0f0", fg="#cc0000")
        txt.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        txt.insert("1.0", err_text)
        txt.config(state=tk.DISABLED)
        ttk.Button(win, text="关闭", command=win.destroy).pack(pady=4)

    def clear_figure(self):
        self.fig.clear()
        self._redraw_canvas()
        self.status.config(text="图表已清除")

    def save_figure(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png"), ("PDF 文档", "*.pdf"),
                       ("SVG 矢量图", "*.svg"), ("所有文件", "*.*")]
        )
        if path:
            # 总览模式下保存 .svg 时导出真正的矢量组合图（嵌套子图 SVG）
            if self.view_mode == "overview" and path.lower().endswith(".svg"):
                try:
                    w = float(self.fig_w_var.get())
                    h = float(self.fig_h_var.get())
                except ValueError:
                    w, h = 8, 6
                try:
                    udpi = int(self.dpi_var.get())
                except ValueError:
                    udpi = 100
                cache_dir = os.path.join(self.gallery.base_dir, "_overview_cache")
                fresh = []
                for item in self.gallery.items:
                    sp, _ = self._render_item_uniform(item, w, h, udpi, cache_dir)
                    fresh.append(sp if sp else item.get("svg_path", ""))
                if self._compose_with_engine(path, fresh):
                    self.status.config(text=f"矢量组合图已保存: {path}")
                else:
                    self.status.config(text="无法生成矢量组合图，请检查图集")
                return
            try:
                dpi = int(self.dpi_var.get())
            except ValueError:
                dpi = 100
            self.fig.savefig(path, dpi=dpi, bbox_inches="tight")
            self.status.config(text=f"图片已保存: {path}")

    def save_script(self, event=None):
        if self.script_path:
            path = self.script_path
        else:
            path = filedialog.asksaveasfilename(
                defaultextension=".py",
                filetypes=[("Python 脚本", "*.py"), ("所有文件", "*.*")]
            )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.code_text.get("1.0", tk.END))
            self.script_path = path
            self.status.config(text=f"脚本已保存: {os.path.basename(path)}")

    def load_script(self):
        path = filedialog.askopenfilename(
            filetypes=[("Python 脚本", "*.py"), ("所有文件", "*.*")]
        )
        if path:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.code_text.delete("1.0", tk.END)
            self.code_text.insert("1.0", content)
            self.script_path = path
            self.status.config(text=f"已加载: {os.path.basename(path)}")
            self.run_code()


# ════════════════════════════════════════
# 入口
# ════════════════════════════════════════

def _enable_dpi_awareness():
    """声明 DPI 感知，让 Tkinter 按显示器原生分辨率渲染（修复高 DPI 下界面模糊）。"""
    if os.name != "nt":
        return
    try:
        import ctypes
        # Windows 10 1703+: Per-Monitor V2
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
            return
        except (AttributeError, OSError):
            pass
        # Vista+: System / Per-Monitor DPI Aware
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _get_system_ui_font():
    """查询 Windows 系统 UI 字体（NONCLIENTMETRICS.lfMessageFont）。

    HONOR 设备上通常返回 HONOR Sans 或微软雅黑，总之就是系统正在用的字体。
    """
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class LOGFONTW(ctypes.Structure):
            _fields_ = [
                ("lfHeight", wintypes.LONG), ("lfWidth", wintypes.LONG),
                ("lfEscapement", wintypes.LONG), ("lfOrientation", wintypes.LONG),
                ("lfWeight", wintypes.LONG), ("lfItalic", wintypes.BYTE),
                ("lfUnderline", wintypes.BYTE), ("lfStrikeOut", wintypes.BYTE),
                ("lfCharSet", wintypes.BYTE), ("lfOutPrecision", wintypes.BYTE),
                ("lfClipPrecision", wintypes.BYTE), ("lfQuality", wintypes.BYTE),
                ("lfPitchAndFamily", wintypes.BYTE),
                ("lfFaceName", wintypes.WCHAR * 32),
            ]

        class NONCLIENTMETRICSW(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.UINT),
                ("iBorderWidth", ctypes.c_int), ("iScrollWidth", ctypes.c_int),
                ("iScrollHeight", ctypes.c_int), ("iCaptionWidth", ctypes.c_int),
                ("iCaptionHeight", ctypes.c_int), ("lfCaptionFont", LOGFONTW),
                ("iSmCaptionWidth", ctypes.c_int), ("iSmCaptionHeight", ctypes.c_int),
                ("lfSmCaptionFont", LOGFONTW), ("iMenuWidth", ctypes.c_int),
                ("iMenuHeight", ctypes.c_int), ("lfMenuFont", LOGFONTW),
                ("lfStatusFont", LOGFONTW), ("lfMessageFont", LOGFONTW),
                ("iPaddedBorderWidth", ctypes.c_int),
            ]

        ncm = NONCLIENTMETRICSW()
        # Vista 之前结构体没有 iPaddedBorderWidth，两种大小都试
        for size in (ctypes.sizeof(NONCLIENTMETRICSW),
                     ctypes.sizeof(NONCLIENTMETRICSW) - ctypes.sizeof(ctypes.c_int)):
            ncm.cbSize = size
            if ctypes.windll.user32.SystemParametersInfoW(0x0029, size, ctypes.byref(ncm), 0):
                if ncm.lfMessageFont.lfFaceName:
                    return ncm.lfMessageFont.lfFaceName
    except Exception:
        pass
    return None


def _harmonize_fonts(root):
    """全界面统一为系统 UI 字体，并以其实测行高为基准校准像素尺寸。

    不同 Tk 版本对高 DPI 的处理不一致（有的自动放大主题字体、有的不放大），
    按 DPI 计算并不可靠。这里直接测量 TkDefaultFont 的真实渲染行高作为基准，
    Treeview 行高、编辑器等尺寸全部由它推导，任何环境下都不会重叠或过小。
    """
    import tkinter.font as tkfont
    family = _get_system_ui_font() or "Microsoft YaHei UI"
    base_font = tkfont.nametofont("TkDefaultFont")
    # 所有命名字体统一为系统字体家族（保留各自字号）
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont",
                 "TkCaptionFont", "TkSmallCaptionFont", "TkIconFont", "TkTooltipFont"):
        try:
            tkfont.nametofont(name).configure(family=family)
        except Exception:
            pass
    base = base_font.metrics()["linespace"]   # 系统 UI 字体真实行高（像素）
    # 模板树行高 = 字体行高 + 余量，从机制上杜绝重叠
    style = ttk.Style(root)
    style.configure("Treeview", rowheight=base + max(6, base // 3))
    root.ui_base = base        # 系统字体实测行高，供各控件按像素校准
    root.ui_family = family    # 系统 UI 字体家族
    return base


def main():
    _enable_dpi_awareness()
    root = tk.Tk()
    # 窗口不超出屏幕
    try:
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(1700, sw - 40)}x{min(1050, sh - 90)}")
    except Exception:
        pass
    _harmonize_fonts(root)
    app = CodePlotAppV5(root)
    root.mainloop()


if __name__ == "__main__":
    main()
