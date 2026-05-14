"""Terrain Mapping and Contour Visualization.

Generate an interactive 2D contour map (plus a 3D terrain surface) for a
user-supplied function f(x, y). Peaks, valleys and the steepest point are
detected from the grid and annotated on both plots, and their coordinates
are also printed to the console.
"""

import argparse
import ast
import re
import sys

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


DEFAULT_FUNCTION_EXPR = "sin(sqrt(x^2 + y^2))"
DEFAULT_X_RANGE = "-3,3"
DEFAULT_Y_RANGE = "-3,3"
DEFAULT_STEP_SIZE = 0.05
DEFAULT_CONTOUR_LEVELS = 30

SAFE_NAMES = ("sin", "cos", "tan", "exp", "log", "ln", "sqrt", "abs", "pi", "e")


# Return the natural log function with support for both log(x) and log(base, x).
def log_fn(a, b=None):
    # log(x)       -> natural log of x         (log base e = ln)
    # log(base, x) -> log base `base` of x      (= ln(x) / ln(base))
    if b is None:
        return np.log(a)
    return np.log(b) / np.log(a)


# ==========================================
# 1. CLI parsing
# ==========================================
VALUE_FLAGS = {
    "-f", "--function",
    "-x", "--x_range",
    "-y", "--y_range",
    "-s", "--step_size",
    "-k", "--contour_levels",
}


# Rewrite argv so value flags can accept arguments that begin with '-'.
def preprocess_argv(argv):
    # argparse refuses to attach values that start with "-" (e.g. "-10,10"
    # or "-x^2 + ...") to value-taking flags. Rewrite `-f -x^2+y` into
    # `-f=-x^2+y` so quoted values with a leading minus are accepted.
    out = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in VALUE_FLAGS and i + 1 < len(argv):
            out.append(f"{token}={argv[i + 1]}")
            i += 2
            continue
        out.append(token)
        i += 1
    return out


# Parse CLI flags for the function, ranges, and sampling step.
def parse_args():
    parser = argparse.ArgumentParser(
        add_help=False,
        prog="contour_map.py",
        description=(
            "Generate a contour map of f(x, y) and interpret it as terrain. "
            "Peaks, valleys and the steepest point are highlighted."
        ),
        epilog=(
            "Example:\n"
            '  python contour_map.py -f "sin(sqrt(x^2 + y^2))" '
            '-x "-10,10" -y "-10,10" -s "0.1"'
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "-f",
        "--function",
        dest="function_expr",
        default=DEFAULT_FUNCTION_EXPR,
        help=(
            'Function of two variables, e.g. "sin(sqrt(x^2 + y^2))". '
            f'Default: "{DEFAULT_FUNCTION_EXPR}".'
        ),
    )
    parser.add_argument(
        "-x",
        "--x_range",
        dest="x_range",
        default=DEFAULT_X_RANGE,
        help=f'Range of x values as "min,max". Default: "{DEFAULT_X_RANGE}".',
    )
    parser.add_argument(
        "-y",
        "--y_range",
        dest="y_range",
        default=DEFAULT_Y_RANGE,
        help=f'Range of y values as "min,max". Default: "{DEFAULT_Y_RANGE}".',
    )
    parser.add_argument(
        "-s",
        "--step_size",
        dest="step_size",
        type=float,
        default=DEFAULT_STEP_SIZE,
        help=f"Step size for x and y. Default: {DEFAULT_STEP_SIZE}.",
    )
    parser.add_argument(
        "-k",
        "--contour_levels",
        dest="contour_levels",
        type=int,
        default=DEFAULT_CONTOUR_LEVELS,
        help=f"Number of contour levels (k). Default: {DEFAULT_CONTOUR_LEVELS}.",
    )
    return parser.parse_args(preprocess_argv(sys.argv[1:]))


# Convert a "min,max" string into a validated numeric range.
def parse_range(text, name):
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError(
            f'Invalid --{name}: expected "min,max" (got "{text}").'
        )
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(
            f'Invalid --{name}: could not parse numbers in "{text}".'
        ) from exc
    if lo >= hi:
        raise ValueError(
            f'Invalid --{name}: min ({lo}) must be smaller than max ({hi}).'
        )
    return lo, hi


# Validate that the requested step fits inside the chosen axis range.
def validate_step(step, lo, hi, axis):
    if step <= 0:
        raise ValueError(f"--step_size must be positive (got {step}).")
    if step > (hi - lo):
        raise ValueError(
            f"--step_size ({step}) is larger than the {axis} range "
            f"({hi - lo}); nothing to plot."
        )


# ==========================================
# 2. Expression handling
# ==========================================
# Insert explicit multiplication where the user's expression omits '*'.
def insert_implicit_multiplication(expr):
    # "2x"   -> "2*x"   (digit followed by variable/function/opening paren)
    # "2("   -> "2*("
    # Scientific notation like "2e3" / "1.5E-4" is preserved by the
    # negative lookahead on [eE][+-]?\d.
    expr = re.sub(r"(\d)(?![eE][+-]?\d)([A-Za-z_(])", r"\1*\2", expr)
    # ")x"   -> ")*x"
    # ")("   -> ")*("
    expr = re.sub(r"(\))([A-Za-z_(])", r"\1*\2", expr)
    return expr


_RESERVED_NAMES = tuple(sorted(set(SAFE_NAMES) | {"x", "y"}, key=lambda s: -len(s)))


# Split adjacent variable names so expressions like "xy" become "x*y".
def split_adjacent_variables(expr):
    # Break run-together variable names such as "xy" into "x*y" using a
    # greedy longest-match against the reserved token list. Known tokens
    # (sin, cos, tan, log, ln, exp, sqrt, abs, pi, e, x, y) are kept
    # intact; leftover single letters are joined with "*".
    def repl(match):
        name = match.group(0)
        if name in _RESERVED_NAMES:
            return name
        parts = []
        i = 0
        while i < len(name):
            matched = None
            for token in _RESERVED_NAMES:
                if name.startswith(token, i):
                    matched = token
                    break
            if matched:
                parts.append(matched)
                i += len(matched)
            else:
                parts.append(name[i])
                i += 1
        return "*".join(parts)

    return re.sub(r"[A-Za-z]+", repl, expr)


# Normalize syntax so the expression can be safely parsed and evaluated.
def normalize_expression(expr):
    if "__" in expr:
        raise ValueError("Double underscore is not allowed in function expressions.")

    normalized = expr.replace("^", "**")

    # Strip any leading "np." on supported math names so users can write
    # either "sin(x)" or "np.sin(x)" interchangeably. Done before implicit
    # multiplication so the "." in "np." isn't disturbed.
    for name in SAFE_NAMES:
        normalized = re.sub(rf"\bnp\.{name}\b", name, normalized)

    normalized = insert_implicit_multiplication(normalized)
    normalized = split_adjacent_variables(normalized)

    return normalized


# ------------------------------------------------------------------
# Pretty LaTeX rendering of the user's expression (for the figure title)
# ------------------------------------------------------------------
_LATEX_CONSTANTS = {"pi": r"\pi", "e": "e"}
_LATEX_FUNCS = {
    "sin": r"\sin",
    "cos": r"\cos",
    "tan": r"\tan",
    "log": r"\log",
    "ln": r"\ln",
}


# Convert the parsed AST into a readable LaTeX expression for the title.
def _ast_to_latex(node, parent_prec=0):
    # Precedence levels:
    #   0: top-level,  1: +/-,  2: */,  3: unary/power base,  4: power base
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Name):
        return _LATEX_CONSTANTS.get(node.id, node.id)

    if isinstance(node, ast.UnaryOp):
        operand = _ast_to_latex(node.operand, 3)
        if isinstance(node.op, ast.USub):
            result = f"-{operand}"
        elif isinstance(node.op, ast.UAdd):
            result = f"+{operand}"
        else:
            result = operand
        if parent_prec >= 4:
            result = rf"\left({result}\right)"
        return result

    if isinstance(node, ast.BinOp):
        op = node.op
        if isinstance(op, ast.Add):
            s = f"{_ast_to_latex(node.left, 1)} + {_ast_to_latex(node.right, 1)}"
            return rf"\left({s}\right)" if parent_prec > 1 else s
        if isinstance(op, ast.Sub):
            s = f"{_ast_to_latex(node.left, 1)} - {_ast_to_latex(node.right, 2)}"
            return rf"\left({s}\right)" if parent_prec > 1 else s
        if isinstance(op, ast.Mult):
            s = rf"{_ast_to_latex(node.left, 2)} \cdot {_ast_to_latex(node.right, 2)}"
            return rf"\left({s}\right)" if parent_prec > 2 else s
        if isinstance(op, ast.Div):
            num = _ast_to_latex(node.left, 0)
            den = _ast_to_latex(node.right, 0)
            return rf"\frac{{{num}}}{{{den}}}"
        if isinstance(op, ast.Pow):
            base = _ast_to_latex(node.left, 4)
            exp_ = _ast_to_latex(node.right, 0)
            return f"{base}^{{{exp_}}}"

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fname = node.func.id
        inner = [_ast_to_latex(a, 0) for a in node.args]
        if fname == "sqrt" and len(inner) == 1:
            return rf"\sqrt{{{inner[0]}}}"
        if fname == "abs" and len(inner) == 1:
            return rf"\left|{inner[0]}\right|"
        if fname == "exp" and len(inner) == 1:
            return f"e^{{{inner[0]}}}"
        if fname == "log" and len(inner) == 2:
            return rf"\log_{{{inner[0]}}}\!\left({inner[1]}\right)"
        latex_fn = _LATEX_FUNCS.get(fname, rf"\mathrm{{{fname}}}")
        return rf"{latex_fn}\!\left({', '.join(inner)}\right)"

    try:
        return ast.unparse(node)
    except Exception:
        return "?"


# Render the user's function expression as LaTeX when possible.
def expression_to_latex(expr):
    try:
        tree = ast.parse(normalize_expression(expr), mode="eval")
        return _ast_to_latex(tree.body)
    except Exception:
        return None


# Evaluate the function expression over the full X/Y grid.
def evaluate_function_expression(function_expr, x, y):
    expr = normalize_expression(function_expr)

    safe_namespace = {
        "x": x,
        "y": y,
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": log_fn,
        "ln": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
        "e": np.e,
    }

    try:
        z = eval(expr, {"__builtins__": {}}, safe_namespace)
    except Exception as exc:
        raise ValueError(f"Invalid --function expression: {exc}") from exc

    z = np.asarray(z, dtype=float)
    if z.shape != x.shape:
        # Allow scalar expressions (e.g. "pi") by broadcasting.
        if z.ndim == 0:
            z = np.full_like(x, float(z))
        else:
            raise ValueError(
                "Function output shape must match the X/Y grid shape. "
                f"Expected {x.shape}, got {z.shape}."
            )
    return z


# ==========================================
# 3. Grid construction
# ==========================================
# Build a sample axis that honors the requested step without arange drift.
def build_axis(lo, hi, step):
    # Use linspace with a rounded sample count to avoid np.arange drift
    # while still honouring the user's requested step size.
    n = int(round((hi - lo) / step)) + 1
    n = max(n, 2)
    return np.linspace(lo, hi, n)


# ==========================================
# 4. Feature detection
# ==========================================
# Detect strict local peaks and valleys by comparing each interior cell to its neighbors.
def find_local_extrema(Z):
    # Flag every interior cell that is strictly greater (peak) or strictly
    # smaller (valley) than all 8 of its neighbors. Strict comparison means
    # flat plateaus are intentionally ignored.
    rows, cols = Z.shape
    if rows < 3 or cols < 3:
        return [], []

    interior = Z[1:-1, 1:-1]
    peak_mask = np.ones_like(interior, dtype=bool)
    valley_mask = np.ones_like(interior, dtype=bool)

    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbor = Z[1 + di:rows - 1 + di, 1 + dj:cols - 1 + dj]
            peak_mask &= interior > neighbor
            valley_mask &= interior < neighbor

    peaks = [(int(i) + 1, int(j) + 1) for i, j in np.argwhere(peak_mask)]
    valleys = [(int(i) + 1, int(j) + 1) for i, j in np.argwhere(valley_mask)]
    return peaks, valleys


def find_saddles_and_inconclusive(Z, peak_indices, valley_indices):
    saddles = []
    inconclusive = []
    rows, cols = Z.shape
    if rows < 3 or cols < 3:
        return [], []

    extrema_set = set(peak_indices) | set(valley_indices)

    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1), (1, 1), (1, 0),
        (1, -1), (0, -1)
    ]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if (i, j) in extrema_set:
                continue

            z_center = Z[i, j]
            
            all_equal = True
            diffs = []
            for di, dj in offsets:
                val = Z[i + di, j + dj]
                if val != z_center:
                    all_equal = False
                diffs.append(val - z_center)

            if all_equal:
                inconclusive.append((i, j))
                continue

            signs = []
            for d in diffs:
                if d > 0:
                    signs.append(1)
                elif d < 0:
                    signs.append(-1)

            if not signs:
                continue

            sign_changes = 0
            for k in range(len(signs)):
                if signs[k] != signs[(k + 1) % len(signs)]:
                    sign_changes += 1

            if sign_changes >= 4:
                saddles.append((i, j))

    return saddles, inconclusive


# Compute critical points and slope information for the surface.
def analyze_surface(X, Y, Z, x, y):
    dy, dx = np.gradient(Z, y, x, edge_order=2)
    slope_magnitude = np.sqrt(dx**2 + dy**2)

    peak_indices, valley_indices = find_local_extrema(Z)

    rows, cols = Z.shape
    global_peak = tuple(int(v) for v in np.unravel_index(int(np.argmax(Z)), Z.shape))
    global_valley = tuple(int(v) for v in np.unravel_index(int(np.argmin(Z)), Z.shape))
    
    # Only include global extrema if they are strictly in the interior (exclude bordering range)
    if (0 < global_peak[0] < rows - 1) and (0 < global_peak[1] < cols - 1):
        if global_peak not in set(peak_indices):
            peak_indices.append(global_peak)
            
    if (0 < global_valley[0] < rows - 1) and (0 < global_valley[1] < cols - 1):
        if global_valley not in set(valley_indices):
            valley_indices.append(global_valley)

    saddle_indices, inconc_indices = find_saddles_and_inconclusive(
        Z, peak_indices, valley_indices
    )

    def pack(idx):
        i, j = idx
        return {
            "x": float(X[i, j]),
            "y": float(Y[i, j]),
            "z": float(Z[i, j]),
            "slope": float(slope_magnitude[i, j]),
        }

    peaks = sorted((pack(idx) for idx in peak_indices), key=lambda p: -p["z"])
    valleys = sorted((pack(idx) for idx in valley_indices), key=lambda p: p["z"])
    saddles = sorted((pack(idx) for idx in saddle_indices), key=lambda p: -p["z"])
    inconclusive = sorted((pack(idx) for idx in inconc_indices), key=lambda p: -p["z"])

    return {
        "slope_magnitude": slope_magnitude,
        "peaks": peaks,
        "valleys": valleys,
        "saddles": saddles,
        "inconclusive": inconclusive,
        "global_peak": peaks[0] if peaks else None,
        "global_valley": valleys[0] if valleys else None,
    }


# ==========================================
# 5. Visualization
# ==========================================
# Add peak or valley markers to both the 2D and 3D Plotly views.
def add_extrema_traces(fig, points, category, color, symbol_2d, symbol_3d, label_prefix):
    if not points:
        return []

    indices = []

    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    zs = [p["z"] for p in points]
    slopes = [p["slope"] for p in points]
    labels = [f"{label_prefix}{i + 1}" for i in range(len(points))]
    hover_names = [
        f"{category} {label_prefix}{i + 1}" for i in range(len(points))
    ]
    customdata = np.column_stack((zs, slopes, hover_names))

    legend_name = f"{category}s ({len(points)})"

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            marker=dict(
                color=color,
                size=10,
                symbol=symbol_2d,
                line=dict(color="black", width=1),
            ),
            text=labels,
            textposition="top center",
            textfont=dict(size=10, color=color),
            name=legend_name,
            customdata=customdata,
            hovertemplate=(
                "%{customdata[2]}<br>x=%{x:.4f}<br>y=%{y:.4f}"
                "<br>z=%{customdata[0]:.4f}<br>slope=%{customdata[1]:.4f}"
                "<extra></extra>"
            ),
            legendgroup=category,
        ),
        row=1,
        col=1,
    )
    indices.append(len(fig.data) - 1)

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text",
            marker=dict(color=color, size=5, symbol=symbol_3d),
            text=labels,
            textposition="top center",
            textfont=dict(size=10, color=color),
            name=legend_name,
            customdata=customdata,
            hovertemplate=(
                "%{customdata[2]}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}"
                "<br>slope=%{customdata[1]:.4f}<extra></extra>"
            ),
            legendgroup=category,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    indices.append(len(fig.data) - 1)

    return indices


# Add a single highlighted marker, used for the steepest point.
def add_single_marker(fig, point, name, color, symbol_2d, symbol_3d):
    if point is None:
        return []

    indices = []
    label = f"{name} ({point['x']:.2f}, {point['y']:.2f}, {point['z']:.2f})"

    fig.add_trace(
        go.Scatter(
            x=[point["x"]],
            y=[point["y"]],
            mode="markers+text",
            marker=dict(
                color=color,
                size=12,
                symbol=symbol_2d,
                line=dict(color="black", width=1),
            ),
            text=[name],
            textposition="bottom center",
            name=name,
            hovertemplate=(
                f"{label}<br>slope={point['slope']:.4f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    indices.append(len(fig.data) - 1)

    fig.add_trace(
        go.Scatter3d(
            x=[point["x"]],
            y=[point["y"]],
            z=[point["z"]],
            mode="markers",
            marker=dict(color=color, size=6, symbol=symbol_3d),
            name=name,
            hovertemplate=(
                f"{label}<br>slope={point['slope']:.4f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    indices.append(len(fig.data) - 1)

    return indices


SUMMARY_MAX_COORDS = 15


# Format a limited list of coordinate lines for the HTML summary panel.
def format_coord_lines(points, prefix, limit=SUMMARY_MAX_COORDS):
    if not points:
        return [f"  (none)"]

    lines = []
    shown = points[:limit]
    for i, p in enumerate(shown, start=1):
        lines.append(
            f"  {prefix}{i:<2} ({p['x']:>7.3f}, {p['y']:>7.3f}) "
            f"z={p['z']:>7.3f}"
        )
    remaining = len(points) - len(shown)
    if remaining > 0:
        lines.append(f"  ... and {remaining} more")
    return lines


# Build the right-side HTML annotation with grid and extrema summaries.
def build_summary_html(args, x, y, analysis, slope_mag):
    peaks = analysis["peaks"]
    valleys = analysis["valleys"]
    saddles = analysis["saddles"]
    inconc = analysis["inconclusive"]
    xmin, xmax = x[0], x[-1]
    ymin, ymax = y[0], y[-1]

    parts = []
    parts.append("<b>Summary</b>")
    parts.append(f"Peaks:     {len(peaks)}")
    parts.append(f"Valleys:   {len(valleys)}")
    parts.append(f"Saddles:   {len(saddles)}")
    parts.append(f"Inconcl:   {len(inconc)}")
    parts.append(f"Slope max: {np.max(slope_mag):.4f}")
    parts.append("")
    parts.append(f"<b>Peaks (top {min(len(peaks), SUMMARY_MAX_COORDS)})</b>")
    parts.extend(format_coord_lines(peaks, "P"))
    parts.append("")
    parts.append(f"<b>Valleys (top {min(len(valleys), SUMMARY_MAX_COORDS)})</b>")
    parts.extend(format_coord_lines(valleys, "V"))
    parts.append("")
    parts.append(f"<b>Saddles (top {min(len(saddles), SUMMARY_MAX_COORDS)})</b>")
    parts.extend(format_coord_lines(saddles, "S"))
    parts.append("")
    parts.append("<b>Grid & Contours</b>")
    parts.append(f"  x = [{xmin}, {xmax}]")
    parts.append(f"  y = [{ymin}, {ymax}]")
    parts.append(f"  step = {args.step_size}")
    parts.append(f"  levels (k) = {args.contour_levels}")
    parts.append(f"  {len(x)} x {len(y)} points")
    return "<br>".join(parts)


# Create a two-state Plotly toggle button for a specific trace group.
def make_visibility_toggle(label, trace_indices):
    # Plotly button with args/args2 acts as a two-state toggle:
    #   first click   -> args  (hide)
    #   second click  -> args2 (show)
    return dict(
        label=label,
        method="restyle",
        args=[{"visible": False}, trace_indices],
        args2=[{"visible": True}, trace_indices],
    )


def build_toggle_menu(peak_idx, valley_idx, saddle_idx, inconc_idx):
    buttons = []
    if peak_idx:
        buttons.append(make_visibility_toggle("Toggle Peaks", peak_idx))
    if valley_idx:
        buttons.append(make_visibility_toggle("Toggle Valleys", valley_idx))
    if saddle_idx:
        buttons.append(make_visibility_toggle("Toggle Saddles", saddle_idx))
    if inconc_idx:
        buttons.append(make_visibility_toggle("Toggle Inconclusive", inconc_idx))
    if not buttons:
        return []

    return [
        dict(
            type="buttons",
            direction="down",
            buttons=buttons,
            x=1.20,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            showactive=True,
            pad=dict(r=6, t=6, b=6, l=6),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.25)",
            font=dict(size=11),
        )
    ]


# Assemble the full contour/surface figure and its annotations.
def build_figure(args, x, y, X, Y, Z, analysis):
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "surface"}]],
        subplot_titles=("2D Contour Terrain Map", "3D Terrain Interpretation"),
    )

    fig.add_trace(
        go.Contour(
            x=x,
            y=y,
            z=Z,
            ncontours=args.contour_levels,
            colorscale="Earth",
            contours=dict(coloring="fill", showlabels=True, labelfont=dict(size=10)),
            colorbar=dict(
                title="Elevation (Z)",
                x=0.46,
                y=0.5,
                yanchor="middle",
                len=0.85,
                thickness=14,
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="Earth",
            opacity=0.95,
            showscale=False,
            name="Terrain Surface",
        ),
        row=1,
        col=2,
    )

    peak_trace_indices = add_extrema_traces(
        fig, analysis["peaks"], "Peak", "red", "triangle-up", "diamond", "P"
    )
    valley_trace_indices = add_extrema_traces(
        fig, analysis["valleys"], "Valley", "blue", "triangle-down", "diamond", "V"
    )
    saddle_trace_indices = add_extrema_traces(
        fig, analysis["saddles"], "Saddle", "green", "circle", "circle", "S"
    )
    inconc_trace_indices = add_extrema_traces(
        fig, analysis["inconclusive"], "Inconclusive", "gray", "square", "square", "I"
    )

    slope_mag = analysis["slope_magnitude"]
    xmin, xmax = x[0], x[-1]
    ymin, ymax = y[0], y[-1]

    fig.update_xaxes(title_text="X coordinate", row=1, col=1)
    fig.update_yaxes(title_text="Y coordinate", row=1, col=1)

    latex_body = expression_to_latex(args.function_expr)
    if latex_body is not None:
        title_text = (
            "Terrain Contour + Surface (Interactive View)<br>"
            f"$\\large f(x, y) = {latex_body}$"
        )
    else:
        title_text = (
            "Terrain Contour + Surface (Interactive View)<br>"
            f"f(x, y) = {args.function_expr}"
        )

    fig.update_layout(
        title=dict(text=title_text, x=0.01, xanchor="left"),
        width=1700,
        height=820,
        margin=dict(r=430, l=60, t=110, b=100),
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Elevation (Z)"),
        updatemenus=build_toggle_menu(
            peak_trace_indices, valley_trace_indices, saddle_trace_indices, inconc_trace_indices
        ),
        legend=dict(
            orientation="h",
            xanchor="center",
            x=0.5,
            yanchor="top",
            y=-0.08,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
        annotations=[
            dict(
                text=build_summary_html(args, x, y, analysis, slope_mag),
                x=1.02,
                y=0.5,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="middle",
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.25)",
                borderwidth=1,
                borderpad=8,
                font=dict(size=11, family="Consolas, 'Courier New', monospace"),
            )
        ],
    )

    return fig


# ==========================================
# 6. Console summary
# ==========================================
# Print a tabular summary of a list of extrema to the console.
def print_extrema_table(points, prefix, title):
    if not points:
        print(f"{title}: none found")
        return

    print(f"{title} ({len(points)} found):")
    for i, p in enumerate(points, start=1):
        print(
            f"  {prefix}{i:<2} (x, y) = ({p['x']:>9.4f}, {p['y']:>9.4f})"
            f"   z = {p['z']:>10.4f}   slope = {p['slope']:.4f}"
        )


# Print the overall analysis summary before opening the figure.
def print_summary(args, x, y, analysis):
    peaks = analysis["peaks"]
    valleys = analysis["valleys"]
    saddles = analysis["saddles"]
    inconc = analysis["inconclusive"]
    slope_mag = analysis["slope_magnitude"]

    print(f"Function: f(x, y) = {args.function_expr}")
    print(
        f"Grid:     x in [{x[0]:g}, {x[-1]:g}], "
        f"y in [{y[0]:g}, {y[-1]:g}], "
        f"step = {args.step_size}  "
        f"({len(x)} x {len(y)} points)\n"
        f"Contours: k = {args.contour_levels} levels"
    )
    print()
    print_extrema_table(peaks, "P", "Peaks (local maxima, sorted by z desc)")
    print()
    print_extrema_table(valleys, "V", "Valleys (local minima, sorted by z asc)")
    print()
    print_extrema_table(saddles, "S", "Saddle points (sorted by z desc)")
    print()
    print_extrema_table(inconc, "I", "Inconclusive points (sorted by z desc)")
    print()
    print(
        f"Slope statistics: max = {np.max(slope_mag):.4f}, "
        f"mean = {np.mean(slope_mag):.4f}"
    )


# ==========================================
# 7. Entry point
# ==========================================
# Run the full workflow: parse input, analyze the surface, and show the plot.
def main():
    args = parse_args()

    try:
        xmin, xmax = parse_range(args.x_range, "x_range")
        ymin, ymax = parse_range(args.y_range, "y_range")
        validate_step(args.step_size, xmin, xmax, "x")
        validate_step(args.step_size, ymin, ymax, "y")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    x = build_axis(xmin, xmax, args.step_size)
    y = build_axis(ymin, ymax, args.step_size)
    X, Y = np.meshgrid(x, y)

    try:
        Z = evaluate_function_expression(args.function_expr, X, Y)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    analysis = analyze_surface(X, Y, Z, x, y)

    print_summary(args, x, y, analysis)

    fig = build_figure(args, x, y, X, Y, Z, analysis)
    show_figure(fig)


# Save the Plotly figure as HTML and open it in the browser.
def show_figure(fig):
    # Write a self-contained HTML file that pulls in MathJax from the CDN,
    # then open it in the default browser. This ensures the LaTeX in the
    # title renders even when the renderer that `fig.show()` would pick
    # doesn't embed MathJax by default.
    import os
    import tempfile
    import webbrowser

    html_path = os.path.join(tempfile.gettempdir(), "contour_map_plot.html")
    pio.write_html(
        fig,
        file=html_path,
        include_mathjax="cdn",
        include_plotlyjs="cdn",
        auto_open=False,
        full_html=True,
    )
    webbrowser.open_new_tab("file://" + os.path.abspath(html_path))


if __name__ == "__main__":
    main()
