"""
CodePlot v5 — Gallery Layout Mode
Usage: python codeplot_v5.py

Core changes (from v4):
  1. Single-figure editing — code generates one figure for fine tuning
  2. Gallery collection — "Add to Gallery" saves the current figure to the gallery
  3. Overview layout — all figures arranged in a grid with automatic (a)(b)(c) labels
  4. Click a gallery thumbnail to reload and edit
  5. Each figure is saved as SVG (vector) and PNG (display)

Gallery data is stored in the .codeplot_gallery/ directory (same directory as the program)
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
matplotlib.use("TkAgg")

# ── Font configuration for Raspberry Pi / English ──
import matplotlib.font_manager as fm

# Preferred monospace font for the code editor (ships with Raspberry Pi)
EDITOR_FONT = "Liberation Mono Bold"
EDITOR_FONT_FALLBACKS = ["Liberation Mono", "DejaVu Sans Mono", "Courier"]

# Preferred UI / plot sans-serif font (also ships with Raspberry Pi)
UI_FONT = "Liberation Sans"
UI_FONT_FALLBACKS = ["DejaVu Sans", "Liberation Sans"]

def _pick_available_font(candidates):
    """Return the first font from candidates that exists on the system."""
    for name in candidates:
        try:
            fm.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return candidates[-1] if candidates else "DejaVu Sans"


EDITOR_FONT = _pick_available_font([EDITOR_FONT] + EDITOR_FONT_FALLBACKS)
UI_FONT = _pick_available_font([UI_FONT] + UI_FONT_FALLBACKS)

# Set matplotlib sans-serif fallback chain (deduplicated)
matplotlib.rcParams['font.sans-serif'] = [UI_FONT] + [f for f in UI_FONT_FALLBACKS if f != UI_FONT]
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

# Compose layout engine (compose_figure.py, same directory as this file)
try:
    import compose_figure as cf_engine
    HAS_COMPOSE_ENGINE = True
except ImportError:
    cf_engine = None
    HAS_COMPOSE_ENGINE = False

# Persistable compose-setting keys
COMPOSE_SETTING_KEYS = [
    "LABEL_FONT_FAMILY", "LABEL_FONT_SIZE", "LABEL_FONT_WEIGHT", "LABEL_COLOR",
    "LABEL_STYLE", "LABEL_BRACKETS", "LABEL_POSITION",
    "LABEL_OFFSET_X", "LABEL_OFFSET_Y", "LABEL_BACKGROUND",
    "LAYOUT", "GRID_ROWS", "GRID_COLS", "PANEL_WIDTH", "PANEL_HEIGHT",
    "H_SPACING", "V_SPACING", "MARGIN",
    "EXPORT_PDF", "EXPORT_PNG", "PNG_DPI", "BACKGROUND",
]

# ── Multi-segment script mode: subplot and compose block markers ──
# The script editor can hold multiple subplot blocks; each starts with figN = new_figure(),
# and ends with a compose block that calls compose() to combine all subplots into one figure.
COMPOSE_BLOCK_HEADER = "# ═══ Compose Layout ═══"
SUBPLOT_BLOCK_RE = re.compile(r"# ═══ Subplot (\d+)")


# ════════════════════════════════════════
# Gallery management
# ════════════════════════════════════════

class FigureGallery:
    """Manage a set of independently generated figures; each saved as SVG+PNG."""

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
        """Add a figure to the gallery and return the item dict."""
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
        """Delete the figure at the given index."""
        if 0 <= index < len(self.items):
            item = self.items.pop(index)
            for p in (item.get("svg_path"), item.get("png_path")):
                if p and os.path.exists(p):
                    os.remove(p)
            self._save_index()
            return True
        return False

    def clear(self):
        """Clear the gallery."""
        for item in self.items:
            for p in (item.get("svg_path"), item.get("png_path")):
                if p and os.path.exists(p):
                    os.remove(p)
        self.items = []
        self._save_index()

    def get_best_grid(self, n):
        """Return the best row/column count for a given number of figures."""
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
# Built-in templates (same as v4)
# ════════════════════════════════════════

BUILTIN_TEMPLATES = {
    "Basic Charts": {
        "Sine Wave": {
            "desc": "Draw a sine curve; demonstrates basic line chart usage",
            "code": """# Sine wave
x = np.linspace(0, 4*np.pi, 500)
y = np.sin(x)

ax = fig.add_subplot(111)
ax.clear()
ax.plot(x, y, color='tab:blue', linewidth=2)
ax.set_title('Sine wave', fontsize=14)
ax.set_xlabel('x (rad)')
ax.set_ylabel('sin(x)')
ax.grid(True, alpha=0.3)
fig.tight_layout()
"""
        },
        "Sine + Cosine": {
            "desc": "Plot sine and cosine on the same figure, with legend",
            "code": """# Sine wave + cosine wave
x = np.linspace(0, 4*np.pi, 500)
y1 = np.sin(x)
y2 = np.cos(x)

ax = fig.add_subplot(111)
ax.clear()
ax.plot(x, y1, label='sin(x)', linewidth=2)
ax.plot(x, y2, label='cos(x)', linewidth=2, linestyle='--')
ax.set_title('Sine and Cosine', fontsize=14)
ax.set_xlabel('x (rad)')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
"""
        },
        "Random Scatter": {
            "desc": "Scatter plot with color and size mapping",
            "code": """# Random scatter
np.random.seed(42)
x = np.random.randn(200)
y = np.random.randn(200)
colors = np.random.rand(200)
sizes = 100 * np.random.rand(200)

ax = fig.add_subplot(111)
ax.clear()
scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
ax.set_title('Random scatter', fontsize=14)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(scatter, ax=ax, label='Color value')
fig.tight_layout()
"""
        },
        "Grouped Bar Chart": {
            "desc": "Grouped bar chart comparing two data sets",
            "code": """# Grouped bar chart
categories = ['A', 'B', 'C', 'D', 'E']
values1 = [23, 45, 56, 78, 32]
values2 = [15, 30, 45, 60, 25]

x = np.arange(len(categories))
width = 0.35

ax = fig.add_subplot(111)
ax.clear()
ax.bar(x - width/2, values1, width, label='Group 1', color='tab:blue')
ax.bar(x + width/2, values2, width, label='Group 2', color='tab:orange')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_title('Grouped bar chart', fontsize=14)
ax.set_ylabel('Value')
ax.legend()
fig.tight_layout()
"""
        },
        "Pie Chart": {
            "desc": "Classic pie chart with percentage labels",
            "code": """# Pie Chart
labels = ['A', 'B', 'C', 'D', 'E']
sizes = [30, 25, 20, 15, 10]
explode = (0.05, 0, 0, 0, 0)

ax = fig.add_subplot(111)
ax.clear()
ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
       shadow=True, startangle=90, colors=plt.cm.Pastel1.colors)
ax.set_title('Pie Chart', fontsize=14)
fig.tight_layout()
"""
        },
    },
    "Statistical Charts": {
        "Histogram + KDE": {
            "desc": "Histogram of data distribution overlaid with KDE",
            "code": """# Histogram + density curve
np.random.seed(42)
data = np.random.normal(100, 15, 1000)

ax = fig.add_subplot(111)
ax.clear()
n, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.7, color='tab:blue', edgecolor='white')
mu, sigma = np.mean(data), np.std(data)
x_line = np.linspace(data.min(), data.max(), 100)
ax.plot(x_line, 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x_line-mu)**2/(2*sigma**2)),
        'r-', linewidth=2, label='Normal fit')
ax.set_title('Histogram and density curve', fontsize=14)
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.legend()
fig.tight_layout()
"""
        },
        "Box Plot": {
            "desc": "Box plot comparing multiple data sets",
            "code": """# Box Plot
np.random.seed(42)
data = [np.random.normal(0, std, 100) for std in range(1, 5)]
labels = ['A', 'B', 'C', 'D']

ax = fig.add_subplot(111)
ax.clear()
bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title('Box Plot', fontsize=14)
ax.set_ylabel('Value')
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
"""
        },
    },
    "Scientific Charts": {
        "3D Surface": {
            "desc": "3D surface plot showing a bivariate function",
            "code": """# 3D surface plot
from mpl_toolkits.mplot3d import Axes3D

X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))

ax = fig.add_subplot(111, projection='3d')
ax.clear()
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, edgecolor='none')
ax.set_title('3D Surface: z = sin(√(x²+y²))', fontsize=14)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
fig.tight_layout()
"""
        },
        "Contour Plot": {
            "desc": "Filled contour plot, suitable for terrain/potential surfaces",
            "code": """# Contour Plot
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2))

ax = fig.add_subplot(111)
ax.clear()
contourf = ax.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r')
contour = ax.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5)
ax.clabel(contour, inline=True, fontsize=8)
ax.set_title('Contour Plot: z = exp(-(x²+y²))', fontsize=14)
ax.set_xlabel('X')
ax.set_ylabel('Y')
fig.colorbar(contourf, ax=ax, label='Z value')
fig.tight_layout()
"""
        },
        "Heatmap": {
            "desc": "Matrix heatmap showing correlation or data intensity",
            "code": """# Heatmap
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
ax.set_title('Heatmap', fontsize=14)
fig.colorbar(im, ax=ax, label='Intensity')
fig.tight_layout()
"""
        },
    },
}


# ════════════════════════════════════════
# Template manager (simplified)
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
# Line-number canvas
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
                             font=(EDITOR_FONT, self.font_size), fill="#555555")
            index = self.text_widget.index(f"{index}+1line")


# ════════════════════════════════════════
# Main app v5
# ════════════════════════════════════════

class CodePlotAppV5:
    def __init__(self, root):
        self.root = root
        self.root.title("CodePlot v5 — Gallery Layout Mode")
        self.root.geometry("1700x1050")
        self.root.minsize(1200, 750)
        self.script_path = None
        self.tmpl_mgr = TemplateManager()
        self.gallery = FigureGallery()
        self._load_compose_settings()

        # View state
        self.view_mode = "single"      # "single" | "overview"
        self.current_gallery_index = None   # Currently edited gallery index
        self._live_after_id = None

        self._build_ui()
        self._bind_shortcuts()
        self._load_default_template()

    # ────────────────── UI construction ──────────────────

    def _build_ui(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ══════ Left panel ══════
        left_outer = ttk.Frame(main_paned)
        main_paned.add(left_outer, weight=1)

        left_paned = ttk.PanedWindow(left_outer, orient=tk.VERTICAL)
        left_paned.pack(fill=tk.BOTH, expand=True)

        # ── Template panel ──
        tmpl_frame = ttk.LabelFrame(left_paned, text="Template Library", padding=4)
        left_paned.add(tmpl_frame, weight=1)

        tmpl_tb = ttk.Frame(tmpl_frame)
        tmpl_tb.pack(fill=tk.X, pady=(0, 2))
        ttk.Button(tmpl_tb, text="➕ Save as Template", command=self._save_as_template).pack(side=tk.LEFT, padx=2)
        ttk.Button(tmpl_tb, text="🗑️ Delete", command=self._delete_template).pack(side=tk.LEFT, padx=2)

        self.tmpl_tree = ttk.Treeview(tmpl_frame, show="tree", selectmode="browse")
        self.tmpl_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tmpl_tree.bind("<<TreeviewSelect>>", self._on_tmpl_select)
        self.tmpl_tree.bind("<Double-1>", self._on_tmpl_double)

        tmpl_vsb = ttk.Scrollbar(tmpl_frame, orient=tk.VERTICAL, command=self.tmpl_tree.yview)
        tmpl_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tmpl_tree.config(yscrollcommand=tmpl_vsb.set)

        self._refresh_template_tree()

        ui_family = getattr(self.root, "ui_family", UI_FONT)
        self.tmpl_desc = ttk.Label(left_outer,
            text="Hint: Click to view description, double-click to append as subplot block",
            wraplength=300, foreground="#666666",
            font=(ui_family, -max(10, int(getattr(self.root, "ui_base", 13) * 0.75))))
        self.tmpl_desc.pack(fill=tk.X, padx=4, pady=2)

        # ── Code editor panel ──
        code_frame = ttk.LabelFrame(left_paned, text="Code Editor", padding=4)
        left_paned.add(code_frame, weight=2)

        code_tb = ttk.Frame(code_frame)
        code_tb.pack(fill=tk.X, pady=(0, 2))
        ttk.Button(code_tb, text="▶ Run", command=self.run_code).pack(side=tk.LEFT, padx=2)
        ttk.Button(code_tb, text="💾 Save Script", command=self.save_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(code_tb, text="📂 Load Script", command=self.load_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(code_tb, text="🔄 Reset", command=self._load_default_template).pack(side=tk.LEFT, padx=2)

        self.live_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(code_tb, text="⚡ Live Render", variable=self.live_var,
                        command=self._on_live_toggle).pack(side=tk.RIGHT, padx=6)

        # Quick annotation toolbar
        anno_frame = ttk.LabelFrame(code_frame, text="Quick Annotation Tools (click to insert code)", padding=2)
        anno_frame.pack(fill=tk.X, pady=(0, 2))
        anno_items = [
            ("➡️ Arrow", self._insert_arrow),
            ("⭕ Circle", self._insert_circle),
            ("📝 Text", self._insert_text),
            ("▭ Rectangle", self._insert_rectangle),
            ("⎯ Horizontal line", self._insert_hline),
            ("⏐ Vertical line", self._insert_vline),
            ("🌟 Highlight", self._insert_highlight),
        ]
        for text, cmd in anno_items:
            ttk.Button(anno_frame, text=text, command=cmd).pack(side=tk.LEFT, padx=2, pady=1)

        # Code editor
        editor_frame = ttk.Frame(code_frame)
        editor_frame.pack(fill=tk.BOTH, expand=True)

        # Use negative pixel values for font sizes, based on the measured line height of the system UI font,
        # so the display stays consistent across Tk versions and DPI scalings.
        ui_base = getattr(self.root, "ui_base", 13)
        code_font_size = -max(13, int(ui_base))
        self.code_text = tk.Text(editor_frame, wrap=tk.NONE, undo=True,
                                  font=(EDITOR_FONT, code_font_size), padx=6, pady=4,
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

        # ══════ Right panel ══════
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)

        # Figure toolbar
        ptb = ttk.Frame(right_frame)
        ptb.pack(fill=tk.X, pady=(0, 2))

        ttk.Label(ptb, text="DPI:").pack(side=tk.LEFT, padx=(4, 0))
        self.dpi_var = tk.StringVar(value="100")
        ttk.Combobox(ptb, textvariable=self.dpi_var,
                     values=["80", "100", "120", "150", "200"], width=6, state="readonly").pack(side=tk.LEFT)

        ttk.Separator(ptb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        ttk.Label(ptb, text="Size (inches):").pack(side=tk.LEFT)
        self.fig_w_var = tk.StringVar(value="8")
        self.fig_h_var = tk.StringVar(value="6")
        ttk.Entry(ptb, textvariable=self.fig_w_var, width=4, justify=tk.CENTER).pack(side=tk.LEFT, padx=2)
        ttk.Label(ptb, text="×").pack(side=tk.LEFT)
        ttk.Entry(ptb, textvariable=self.fig_h_var, width=4, justify=tk.CENTER).pack(side=tk.LEFT, padx=2)
        ttk.Button(ptb, text="Apply", command=self._apply_figsize).pack(side=tk.LEFT, padx=4)

        ttk.Separator(ptb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # Gallery action buttons
        ttk.Button(ptb, text="➕ Add to Gallery", command=self._add_to_gallery).pack(side=tk.LEFT, padx=2)
        ttk.Button(ptb, text="📋 Overview Layout", command=self._show_overview).pack(side=tk.LEFT, padx=2)
        ttk.Button(ptb, text="⚙ Compose Settings", command=self._open_compose_settings).pack(side=tk.LEFT, padx=2)
        ttk.Button(ptb, text="🗑️ Clear Gallery", command=self._clear_gallery).pack(side=tk.LEFT, padx=2)

        ttk.Separator(ptb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        ttk.Button(ptb, text="📥 Save Image", command=self.save_figure).pack(side=tk.RIGHT, padx=4)
        ttk.Button(ptb, text="🔧 Clear", command=self.clear_figure).pack(side=tk.RIGHT, padx=2)

        # Gallery navigation bar (thumbnail list)
        self.gallery_frame = ttk.LabelFrame(right_frame, text="Gallery (click to edit)", padding=2)
        self.gallery_frame.pack(fill=tk.X, pady=(0, 2))
        self._refresh_gallery_nav()

        # Figure canvas
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        nav = NavigationToolbar2Tk(self.canvas, right_frame, pack_toolbar=False)
        nav.pack(fill=tk.X)

        # Status bar
        self.status = ttk.Label(self.root,
            text="Ready | Run code → Add to Gallery → Overview Layout",
            relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    # ────────────────── Gallery-related ──────────────────

    def _refresh_gallery_nav(self):
        """Refresh gallery navigation thumbnails."""
        for w in self.gallery_frame.winfo_children():
            w.destroy()

        if not self.gallery.items:
            ttk.Label(self.gallery_frame, text='(Gallery is empty; run code and click "Add to Gallery")').pack(pady=4)
            return

        self.gallery_photos = []
        for i, item in enumerate(self.gallery.items):
            frame = ttk.Frame(self.gallery_frame)
            frame.pack(side=tk.LEFT, padx=2, pady=1)

            # Load thumbnail
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
                ttk.Button(frame, text=f"Fig {i+1}",
                           command=lambda idx=i: self._load_gallery_item(idx)).pack()

            # Label
            ttk.Label(frame, text=f"({chr(97+i)})", font=(EDITOR_FONT, 9)).pack()

            # Delete button
            ttk.Button(frame, text="✕", width=2,
                       command=lambda idx=i: self._remove_gallery_item(idx)).pack()

    def _add_to_gallery(self):
        """Add the current figure to the gallery."""
        if self.view_mode == "overview":
            messagebox.showwarning(
                "Notice",
                "Currently in overview layout view; cannot add the overview to the gallery again.\n"
                "Please click a single figure in the gallery (or run code) to return to single-figure mode.")
            return

        code = self.code_text.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Notice", "Code is empty")
            return

        # Ensure there is a figure to save
        if len(self.fig.axes) == 0:
            messagebox.showwarning("Notice", "Please run the code to generate a figure first")
            return

        title = f"Fig {len(self.gallery.items)+1}"
        item = self.gallery.add(code, self.fig, title=title)
        self._refresh_gallery_nav()
        self.status.config(text=f"Added to gallery: {title} | Total: {len(self.gallery.items)}")

    def _load_gallery_item(self, index):
        """Load a gallery figure for editing."""
        if index < 0 or index >= len(self.gallery.items):
            return
        item = self.gallery.items[index]
        self.current_gallery_index = index
        self.view_mode = "single"

        self.code_text.delete("1.0", tk.END)
        self.code_text.insert("1.0", item["code"])
        self.run_code()
        self.status.config(text=f"Editing gallery item: ({chr(97+index)}) | Click \"Add to Gallery\" again after editing")

    def _remove_gallery_item(self, index):
        """Delete a gallery figure."""
        if messagebox.askyesno("Confirm Delete", f"Delete figure ({chr(97+index)})?"):
            self.gallery.remove(index)
            self._refresh_gallery_nav()
            self.status.config(text=f"Deleted | {len(self.gallery.items)} figures remaining")

    def _clear_gallery(self):
        """Clear the gallery."""
        if not self.gallery.items:
            return
        if messagebox.askyesno("Confirm Clear", "Clear the entire gallery?"):
            self.gallery.clear()
            self._refresh_gallery_nav()
            self.status.config(text="Gallery cleared")

    def _show_overview(self):
        """Overview layout mode: arrange all gallery figures in a grid with labels in the top-left corner."""
        n = len(self.gallery.items)
        if n == 0:
            messagebox.showinfo("Notice", "Gallery is empty; please add figures first")
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

        # Fully clear old axes (including attached colorbars, etc.)
        self.fig.clf()
        # Scale the overview grid so each panel is near its original size, keeping font sizes consistent;
        # but do not exceed the canvas widget pixel size, or the Tk backend will clip the overflow.
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

        # Re-render each panel at a uniform size (ensures consistent fonts after composition); fall back to archived files on failure.
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
                ax.text(0.5, 0.5, f"Fig {i+1}", ha="center", va="center", fontsize=20)

            # Labels (a)(b)(c) placed in the top-left corner of each panel
            ax.text(0.01, 0.99, f"({chr(97 + i)})", transform=ax.transAxes,
                    fontsize=13, fontweight="bold", ha="left", va="top",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.75, pad=1.5))

        self.fig.tight_layout()
        self._redraw_canvas()

        # Generate a vector-combined SVG in parallel (nested subfigure SVGs; labels added during layout)
        composed = os.path.join(self.gallery.base_dir, "composed.svg")
        svg_note = ""
        if self._compose_with_engine(composed, fresh_svgs):
            svg_note = f" | Vector combined figure: {composed}"

        self.status.config(
            text=f"Overview layout: {n} figures | {rows}×{cols} grid{svg_note} | Click a figure to edit")

    def _compose_gallery_svg(self, out_path, svg_paths=None):
        """Combine each subfigure SVG in the gallery into a single figure in pure vector form.

        Subfigure SVGs do not contain labels; labels (a)(b)(c) are added at the top-left
        corner of each panel during composition.
        Principle: place each subfigure SVG as a nested <svg x y width height viewBox>
        element inside a large container SVG, preserving vector data without rasterization.
        svg_paths: optional list of SVG paths corresponding to gallery items (e.g., the
                   uniformly re-rendered results); if None, the archived gallery SVGs are used.
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

        pad = 18.0        # Panel spacing (pt)
        label_fs = 16.0   # Label font size (pt)
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
            # Scale proportionally to fit cell and center
            scale = min(max_w / e["w"], max_h / e["h"])
            dw, dh = e["w"] * scale, e["h"] * scale
            ox = cell_x + (max_w - dw) / 2
            oy = cell_y + (max_h - dh) / 2
            parts.append(
                f'<svg x="{ox:.2f}" y="{oy:.2f}" width="{dw:.2f}" height="{dh:.2f}" '
                f'viewBox="{e["vb"]}">{e["inner"]}</svg>'
            )
            # Top-left label (white backing keeps it readable over complex figures)
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

    # ────────────────── Compose layout settings (compose_figure engine) ──────────────────

    def _compose_settings_path(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "compose_settings.json")

    def _load_compose_settings(self):
        """Write persisted compose settings back to the compose_figure engine at startup."""
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
        """Combine SVGs using the compose_figure engine (applying compose settings); fall back to the built-in method when the engine is unavailable."""
        if HAS_COMPOSE_ENGINE:
            try:
                cf_engine.compose(svg_paths, out_path)
                if cf_engine.EXPORT_PDF or cf_engine.EXPORT_PNG:
                    cf_engine.export_extra(out_path)
                return True
            except SystemExit as e:
                self.status.config(text=f"Compose failed: {e}")
                return False
            except Exception as e:
                self.status.config(text=f"Compose failed: {str(e)[:60]}")
                return False
        return self._compose_gallery_svg(out_path, svg_paths=svg_paths)

    def _open_compose_settings(self):
        """Compose layout settings dialog: label font/position, grid size, and output options."""
        if not HAS_COMPOSE_ENGINE:
            messagebox.showwarning("Notice", "compose_figure.py not found (must be in the same directory as codeplot_v5.py)")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Compose Settings")
        dlg.transient(self.root)
        dlg.grab_set()

        def row(parent, r, label, widget):
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky=tk.W,
                                               padx=6, pady=2)
            widget.grid(row=r, column=1, sticky=tk.W, padx=6, pady=2)

        # ── Label settings ──
        f1 = ttk.LabelFrame(dlg, text="Labels (a)(b)(c)", padding=6)
        f1.pack(fill=tk.X, padx=8, pady=4)

        ui_family = getattr(self.root, "ui_family", UI_FONT)
        family_var = tk.StringVar(value=cf_engine.LABEL_FONT_FAMILY)
        row(f1, 0, "Font:", ttk.Combobox(f1, textvariable=family_var, width=18,
            values=[ui_family, "Arial", "Times New Roman", "HONOR Sans",
                    "Microsoft YaHei UI", "SimSun", "SimHei"]))

        size_var = tk.StringVar(value=str(cf_engine.LABEL_FONT_SIZE))
        row(f1, 1, "Size (pt):", ttk.Entry(f1, textvariable=size_var, width=8))

        weight_var = tk.StringVar(value=cf_engine.LABEL_FONT_WEIGHT)
        row(f1, 2, "Weight:", ttk.Combobox(f1, textvariable=weight_var, width=8,
            values=["bold", "normal"], state="readonly"))

        color_var = tk.StringVar(value=cf_engine.LABEL_COLOR)
        row(f1, 3, "Color:", ttk.Combobox(f1, textvariable=color_var, width=10,
            values=["black", "white", "red", "blue", "gray"]))

        style_var = tk.StringVar(value=cf_engine.LABEL_STYLE)
        row(f1, 4, "Style:", ttk.Combobox(f1, textvariable=style_var, width=10,
            values=["lower", "upper", "number"], state="readonly"))

        brackets_var = tk.BooleanVar(value=bool(cf_engine.LABEL_BRACKETS))
        row(f1, 5, "With brackets:", ttk.Checkbutton(f1, variable=brackets_var))

        pos_var = tk.StringVar(value=cf_engine.LABEL_POSITION)
        row(f1, 6, "Position:", ttk.Combobox(f1, textvariable=pos_var, width=12,
            values=["top-left", "top-right", "bottom-left", "bottom-right"],
            state="readonly"))

        ox_var = tk.StringVar(value=str(cf_engine.LABEL_OFFSET_X))
        oy_var = tk.StringVar(value=str(cf_engine.LABEL_OFFSET_Y))
        row(f1, 7, "Offset X (pt):", ttk.Entry(f1, textvariable=ox_var, width=8))
        row(f1, 8, "Offset Y (pt):", ttk.Entry(f1, textvariable=oy_var, width=8))

        bg_var = tk.BooleanVar(value=bool(cf_engine.LABEL_BACKGROUND))
        row(f1, 9, "White background:", ttk.Checkbutton(f1, variable=bg_var))

        # ── Layout settings ──
        f2 = ttk.LabelFrame(dlg, text="Grid layout (for custom layout edit CUSTOM_PANELS in compose_figure.py)", padding=6)
        f2.pack(fill=tk.X, padx=8, pady=4)

        rows_var = tk.StringVar(value=str(cf_engine.GRID_ROWS))
        cols_var = tk.StringVar(value=str(cf_engine.GRID_COLS))
        row(f2, 0, "Rows:", ttk.Entry(f2, textvariable=rows_var, width=6))
        row(f2, 1, "Cols:", ttk.Entry(f2, textvariable=cols_var, width=6))

        pw_var = tk.StringVar(value=str(cf_engine.PANEL_WIDTH))
        ph_var = tk.StringVar(value=str(cf_engine.PANEL_HEIGHT))
        row(f2, 2, "Panel width (pt):", ttk.Entry(f2, textvariable=pw_var, width=8))
        row(f2, 3, "Panel height (pt):", ttk.Entry(f2, textvariable=ph_var, width=8))

        hs_var = tk.StringVar(value=str(cf_engine.H_SPACING))
        vs_var = tk.StringVar(value=str(cf_engine.V_SPACING))
        mg_var = tk.StringVar(value=str(cf_engine.MARGIN))
        row(f2, 4, "H spacing (pt):", ttk.Entry(f2, textvariable=hs_var, width=8))
        row(f2, 5, "V spacing (pt):", ttk.Entry(f2, textvariable=vs_var, width=8))
        row(f2, 6, "Margin (pt):", ttk.Entry(f2, textvariable=mg_var, width=8))

        # ── Output settings ──
        f3 = ttk.LabelFrame(dlg, text="Output", padding=6)
        f3.pack(fill=tk.X, padx=8, pady=4)

        pdf_var = tk.BooleanVar(value=bool(cf_engine.EXPORT_PDF))
        png_var = tk.BooleanVar(value=bool(cf_engine.EXPORT_PNG))
        dpi_var = tk.StringVar(value=str(cf_engine.PNG_DPI))
        row(f3, 0, "Export PDF:", ttk.Checkbutton(f3, variable=pdf_var))
        row(f3, 1, "Export PNG:", ttk.Checkbutton(f3, variable=png_var))
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
                messagebox.showwarning("Notice", f"Invalid number format: {e}", parent=dlg)
                return
            self._save_compose_settings()
            dlg.destroy()
            # Recompose immediately with new settings when the gallery is not empty
            if self.gallery.items:
                self._show_overview()
            self.status.config(text="Compose settings saved and applied")

        ttk.Button(dlg, text="OK", command=apply_settings).pack(pady=8)

    # ────────────────── Size and scaling ──────────────────

    def _apply_figsize(self):
        try:
            w = float(self.fig_w_var.get())
            h = float(self.fig_h_var.get())
            if w < 1 or h < 1 or w > 30 or h > 30:
                raise ValueError("Size must be between 1 and 30 inches")
        except ValueError as e:
            self.status.config(text=f"Size format error: {e}")
            return
        self.fig.set_size_inches(w, h)
        self.canvas.resize_event()
        self._redraw_canvas()
        self.status.config(text=f"Figure size set to {w:.1f} × {h:.1f} inches")

    # ────────────────── Live rendering ──────────────────

    def _on_live_toggle(self):
        if self.live_var.get():
            self.status.config(text="⚡ Live render on | auto-refresh 400 ms after typing stops")
            self._trigger_live_run()
        else:
            self.status.config(text="Live render off | press Ctrl+R to run manually")
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

    # ────────────────── Annotation tools (fig.gca() version, compatible with multi-subplot figures) ──────────────────

    def _insert_at_cursor(self, snippet):
        self.code_text.insert(tk.INSERT, snippet)
        self.status.config(text="Annotation code inserted")
        if self.live_var.get():
            self._trigger_live_run()

    def _insert_arrow(self):
        snippet = """# Add arrow annotation
fig.gca().annotate('', xy=(0.5, 0.5), xytext=(0.2, 0.2),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
"""
        self._insert_at_cursor(snippet)

    def _insert_circle(self):
        snippet = """# Add circle
from matplotlib.patches import Circle
circle = Circle((0.5, 0.5), 0.1, fill=False, edgecolor='red', linewidth=2)
fig.gca().add_patch(circle)
"""
        self._insert_at_cursor(snippet)

    def _insert_text(self):
        snippet = """# Add text annotation
fig.gca().text(0.5, 0.5, 'Your text', fontsize=12, color='red',
        ha='center', va='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
"""
        self._insert_at_cursor(snippet)

    def _insert_rectangle(self):
        snippet = """# Add rectangle
from matplotlib.patches import Rectangle
rect = Rectangle((0.3, 0.3), 0.4, 0.4, fill=False, edgecolor='blue', linewidth=2, linestyle='--')
fig.gca().add_patch(rect)
"""
        self._insert_at_cursor(snippet)

    def _insert_hline(self):
        snippet = """# Add horizontal reference line
fig.gca().axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Reference line')
"""
        self._insert_at_cursor(snippet)

    def _insert_vline(self):
        snippet = """# Add vertical reference line
fig.gca().axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Reference line')
"""
        self._insert_at_cursor(snippet)

    def _insert_highlight(self):
        snippet = """# Add highlight region
fig.gca().axvspan(0.3, 0.7, color='yellow', alpha=0.2, label='Highlight area')
# Or horizontal region: fig.gca().axhspan(ymin, ymax, ...)
"""
        self._insert_at_cursor(snippet)

    # ────────────────── Templates ──────────────────

    def _refresh_template_tree(self):
        for item in self.tmpl_tree.get_children():
            self.tmpl_tree.delete(item)
        builtin_node = self.tmpl_tree.insert("", tk.END, text="📚 Built-in Templates", open=True)
        for category, items in self.tmpl_mgr.get_builtin().items():
            cat_node = self.tmpl_tree.insert(builtin_node, tk.END, text=f"  {category}", open=False)
            for name in items:
                self.tmpl_tree.insert(cat_node, tk.END, text=f"    {name}",
                                      values=("builtin", category, name))
        user_node = self.tmpl_tree.insert("", tk.END, text="👤 My Templates", open=True)
        for category, items in self.tmpl_mgr.get_user().items():
            cat_node = self.tmpl_tree.insert(user_node, tk.END, text=f"  {category}", open=False)
            for name in items:
                self.tmpl_tree.insert(cat_node, tk.END, text=f"    {name}",
                                      values=("user", category, name))
        if not self.tmpl_mgr.get_user():
            self.tmpl_tree.insert(user_node, tk.END, text="    (No custom templates)")

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
        desc = info.get("desc", "No description")
        self.tmpl_desc.config(text=f"[{name}] {desc}  |  Double-click to append as subplot block")

    # ────────────────── Multi-segment script (subplot blocks) ──────────────────

    def _make_subplot_block(self, index, name, code):
        """Wrap template code as a subplot block: rename the fig variable to figN and inject a creation statement."""
        code = re.sub(r"\bfig\b", f"fig{index}", code.strip())
        return (f"# ═══ Subplot {index}: {name} ═══\n"
                f"fig{index} = new_figure()\n"
                f"{code}\n")

    def _default_compose_block(self):
        return (
            f"{COMPOSE_BLOCK_HEADER}\n"
            "# Arranged in new_figure() creation order by default; reorder example: compose([fig2, fig1])\n"
            "# Common parameters: rows=2, cols=2, panel_width=420, panel_height=320,\n"
            "#           label_font_family='Arial', label_font_size=20,\n"
            "#           label_position='top-left', label_style='lower'\n"
            "compose()\n"
        )

    def _append_template_block(self, name, code):
        """Append the template as a new subplot block before the compose block."""
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
            self.status.config(text=f"Added subplot {n}: {name} | Ctrl+R to run")
            self.run_code()

    def _save_as_template(self):
        code = self.code_text.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Notice", "Code is empty")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Save as Template")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Category:").pack(anchor=tk.W, padx=10, pady=(10, 2))
        cat_var = tk.StringVar(value="Custom")
        ttk.Combobox(dialog, textvariable=cat_var,
                     values=list(self.tmpl_mgr.get_user().keys()) + ["Custom"]).pack(fill=tk.X, padx=10)

        ttk.Label(dialog, text="Template name:").pack(anchor=tk.W, padx=10, pady=(10, 2))
        name_var = tk.StringVar(value="My Template")
        ttk.Entry(dialog, textvariable=name_var).pack(fill=tk.X, padx=10)

        ttk.Label(dialog, text="Description:").pack(anchor=tk.W, padx=10, pady=(10, 2))
        desc_var = tk.StringVar(value="")
        ttk.Entry(dialog, textvariable=desc_var).pack(fill=tk.X, padx=10)

        def do_save():
            cat = cat_var.get().strip()
            name = name_var.get().strip()
            desc = desc_var.get().strip() or "User-defined template"
            if not cat or not name:
                messagebox.showwarning("Notice", "Category and name cannot be empty")
                return
            self.tmpl_mgr.add_user_template(cat, name, desc, code)
            self._refresh_template_tree()
            self.status.config(text=f"Template saved: {cat} / {name}")
            dialog.destroy()

        ttk.Button(dialog, text="Save", command=do_save).pack(pady=15)

    def _delete_template(self):
        sel = self.tmpl_tree.selection()
        if not sel:
            messagebox.showwarning("Notice", "Please select a template first")
            return
        item = sel[0]
        vals = self.tmpl_tree.item(item, "values")
        if not vals or len(vals) < 3:
            return
        source, category, name = vals[0], vals[1], vals[2]
        if source != "user":
            messagebox.showwarning("Notice", "Only user-defined templates can be deleted")
            return
        if messagebox.askyesno("Confirm Delete", f'Delete template "{name}"?'):
            if self.tmpl_mgr.delete_user_template(category, name):
                self._refresh_template_tree()
                self.status.config(text=f"Template deleted: {name}")

    def _load_default_template(self):
        code = self.tmpl_mgr.get_template_code("builtin", "Basic Charts", "Sine + Cosine")
        text = (self._make_subplot_block(1, "Sine + Cosine", code)
                + "\n" + self._default_compose_block())
        self.code_text.delete("1.0", tk.END)
        self.code_text.insert("1.0", text)
        self.status.config(text="Default example loaded | double-click a template to append subplot | Ctrl+R to run")
        self.run_code()

    # ────────────────── Core logic ──────────────────

    def _bind_shortcuts(self):
        self.root.bind("<Control-r>", lambda e: self.run_code())
        self.root.bind("<Control-R>", lambda e: self.run_code())
        self.root.bind("<Control-s>", lambda e: self.save_script())
        self.root.bind("<Control-S>", lambda e: self.save_script())

    def _redraw_canvas(self):
        """Redraw the canvas.

        matplotlib's Tk backend writes the image into a _tkphoto bitmap; when the
        figure shrinks from a larger size, _tkphoto keeps the old size and blit only
        covers the new figure area, leaving old pixels on the canvas. Clearing the
        bitmap with blank() before draw() fixes this.
        """
        try:
            self.canvas._tkphoto.blank()
        except Exception:
            pass
        self.canvas.draw()

    @staticmethod
    def _make_exec_globals(fig):
        """Build the execution environment for user code."""
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
        """Re-execute gallery code at a uniform size, returning (svg_path, png_path).

        Used during overview layout to keep each panel's physical size identical
        (without bbox_inches='tight'), so the fonts are consistent after composition.
        Returns (None, None) on failure; the caller falls back to the archived gallery files.
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
        """Multi-segment script mode: provide new_figure() and compose() to user code.

        new_figure() creates a subplot using the current size/DPI settings and registers it;
        compose() saves all (or specified) subplots as SVGs, combines them into a single
        figure via the compose_figure engine, and shows a grid preview on the canvas.
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
                raise ValueError("No subplots to compose (create them with new_figure() first)")
            n = len(flist)
            if rows and cols:
                r, c = int(rows), int(cols)
            else:
                r, c = app.gallery.get_best_grid(n)

            # Save each subplot as SVG/PNG at fixed size (no tight cropping, keeping fonts consistent)
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

            # Canvas grid preview
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
                    ax.text(0.5, 0.5, f"Fig {i+1}", ha="center", va="center",
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
                self.status.config(text="Error: code is empty")
            return

        # Running code returns to single-figure editing mode (overview axes will be cleared by fig.clear())
        self.view_mode = "single"

        if not silent:
            self.status.config(text="Running...")
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
                # Script called compose(): preview and status are handled by compose
                if not silent:
                    self.status.config(
                        text=f"Compose finished: {self._script_composed} | "
                             f"Edit the compose block to adjust order and style")
            else:
                self._redraw_canvas()
                n_axes = len(self.fig.axes)
                size_info = f"{self.fig.get_size_inches()[0]:.1f}×{self.fig.get_size_inches()[1]:.1f} inches"
                if n_axes > 1:
                    msg = f"Run succeeded | Note: detected {n_axes} axes, consider splitting into single figures and adding to gallery"
                else:
                    msg = f"Run succeeded | DPI={dpi} | {size_info}"
                if not silent:
                    self.status.config(text=msg)
        except Exception as e:
            err = traceback.format_exc()
            short_err = str(e)[:80]
            if silent:
                self.status.config(text=f"⚡ Live: {short_err}")
            else:
                self.status.config(text=f"Error: {short_err}")
                self._show_error(err)

    def _show_error(self, err_text):
        win = tk.Toplevel(self.root)
        win.title("Run Error")
        win.geometry("700x400")
        txt = tk.Text(win, wrap=tk.WORD, font=(EDITOR_FONT, 10), bg="#fff0f0", fg="#cc0000")
        txt.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        txt.insert("1.0", err_text)
        txt.config(state=tk.DISABLED)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=4)

    def clear_figure(self):
        self.fig.clear()
        self._redraw_canvas()
        self.status.config(text="Figure cleared")

    def save_figure(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"),
                       ("SVG Vector", "*.svg"), ("All Files", "*.*")]
        )
        if path:
            # In overview mode, saving .svg exports the real vector combined figure (nested subfigure SVGs)
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
                    self.status.config(text=f"Vector combined figure saved: {path}")
                else:
                    self.status.config(text="Cannot generate vector combined figure; please check the gallery")
                return
            try:
                dpi = int(self.dpi_var.get())
            except ValueError:
                dpi = 100
            self.fig.savefig(path, dpi=dpi, bbox_inches="tight")
            self.status.config(text=f"Image saved: {path}")

    def save_script(self, event=None):
        if self.script_path:
            path = self.script_path
        else:
            path = filedialog.asksaveasfilename(
                defaultextension=".py",
                filetypes=[("Python Script", "*.py"), ("All Files", "*.*")]
            )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.code_text.get("1.0", tk.END))
            self.script_path = path
            self.status.config(text=f"Script saved: {os.path.basename(path)}")

    def load_script(self):
        path = filedialog.askopenfilename(
            filetypes=[("Python Script", "*.py"), ("All Files", "*.*")]
        )
        if path:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.code_text.delete("1.0", tk.END)
            self.code_text.insert("1.0", content)
            self.script_path = path
            self.status.config(text=f"Loaded: {os.path.basename(path)}")
            self.run_code()


# ════════════════════════════════════════
# Entry point
# ════════════════════════════════════════

def _enable_dpi_awareness():
    """Declare DPI awareness so Tkinter renders at the monitor's native resolution (fixes blurriness on high-DPI displays)."""
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
    """Query the Windows system UI font (NONCLIENTMETRICS.lfMessageFont).

    On HONOR devices this usually returns HONOR Sans or Microsoft YaHei; basically
    whatever font the system is currently using.
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
        # The structure had no iPaddedBorderWidth before Vista; try both sizes
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
    """Unify the whole UI on the system UI font and calibrate pixel sizes from its measured line height.

    Different Tk versions handle high DPI inconsistently (some auto-scale theme fonts,
    some do not), so DPI-based calculations are unreliable. Here we measure the actual
    rendered line height of TkDefaultFont as the baseline; Treeview row height, editor
    sizes, etc. are all derived from it, preventing overlap or undersizing in any environment.
    """
    import tkinter.font as tkfont
    family = _get_system_ui_font() or UI_FONT
    base_font = tkfont.nametofont("TkDefaultFont")
    # Unify all named fonts to the system font family (keep their sizes)
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont",
                 "TkCaptionFont", "TkSmallCaptionFont", "TkIconFont", "TkTooltipFont"):
        try:
            tkfont.nametofont(name).configure(family=family)
        except Exception:
            pass
    base = base_font.metrics()["linespace"]   # Actual system UI font line height (pixels)
    # Treeview row height = line height + margin, eliminating overlap by design
    style = ttk.Style(root)
    style.configure("Treeview", rowheight=base + max(6, base // 3))
    root.ui_base = base        # Measured system font line height, used by widgets for pixel calibration
    root.ui_family = family    # System UI font family
    return base


def main():
    _enable_dpi_awareness()
    root = tk.Tk()
    # Keep window within screen bounds
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
