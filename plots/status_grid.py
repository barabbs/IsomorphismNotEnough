"""Grid plot of (status, comment) cells per (problem, task).

Each row is a problem, each task occupies one or more columns. List-valued
``status``/``comment`` per problem can either be expanded into multiple
columns (one per element, padded with white so columns stay aligned) or
collapsed via a per-task callable that maps ``(statuses, comments)`` to a
single ``(status, comment)``. Cells are coloured by the per-task palette and
stamped with a single character from a per-task character map when a comment
is present. Rows with no data in any task are dropped, and the grid is split
across ``n_splits`` side-by-side subplots so the figure stays compact.
"""

import json
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt

# Make the sibling ``status_bars`` module importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent))
from status_bars import (
    STATUS_ALIASES, EXTERNAL_SUBDIR,
    SPEC_ISO_COLORS, PROOF_GEN_GT_COLORS, PROOF_GEN_GEN_COLORS,
    normalize_colors, lookup_color,
    resolve_list,
)

from clever_bench.benchmark import Benchmark
from core.benchmarks.clever import TASKS
from core.problem import Problem, ProblemInfo
from core import var


# ---------------------------------------------------------------- data loading
def _alias(status):
    if not status:
        return None
    return STATUS_ALIASES.get(status, status)


def _clean(comment):
    return None if comment == "" else comment


def _problem_pairs(problem):
    """Return list of ``(status, comment)`` for a problem (one entry per
    list element if status is list-valued; one entry otherwise)."""
    if problem is None:
        return None
    status = problem.metadata.get('status')
    comment = problem.metadata.get('comment')
    if problem.exception is not None and status is None:
        status = problem.exception.get('type')
    if isinstance(status, list):
        comments = comment if isinstance(comment, list) else [comment] * len(status)
        return [(_alias(s), _clean(c)) for s, c in zip(status, comments)]
    return [(_alias(status), _clean(comment))]


def _load_external_with_ids(task_key, base_dir):
    """Yield ``(problem_id, status, comment)`` from
    ``<base_dir>/clever/<task_key>.json``, preserving dict keys as ids."""
    path = Path(base_dir) / EXTERNAL_SUBDIR / f"{task_key}.json"
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        if 'status' in data and isinstance(data['status'], list):
            statuses = data['status']
            comments = data.get('comment') or [None] * len(statuses)
            for i, (s, c) in enumerate(zip(statuses, comments)):
                yield i, _alias(s), _clean(c)
            return
        for k, v in data.items():
            try:
                pid = int(k)
            except (TypeError, ValueError):
                continue
            if not isinstance(v, dict):
                continue
            record = v.get('metadata', v)
            yield pid, _alias(record.get('status')), _clean(record.get('comment'))
        return
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        record = item.get('metadata', item)
        yield i, _alias(record.get('status')), _clean(record.get('comment'))


def _load_task_pairs(task_key, ids, attempt, base_dir):
    """Return ``OrderedDict[problem_id -> list[(status, comment)] | None]``.
    ``None`` means the problem was not found (rendered as a white cell)."""
    if task_key in TASKS:
        dataset = TASKS[task_key]['dataset']
        benchmark = Benchmark(is_sample=dataset['subset'] == "sample_examples")
        benchmark.load_all()
        if ids is None:
            ids = [p.problem_id for p in benchmark.problems]
        out = OrderedDict()
        for pid in ids:
            try:
                problem = Problem.load(
                    base_dir=base_dir,
                    info=ProblemInfo(id=pid, task=task_key, **dataset, attempt=attempt),
                )
                out[pid] = _problem_pairs(problem)
            except FileNotFoundError:
                out[pid] = None
        return out
    out = OrderedDict()
    for pid, s, c in _load_external_with_ids(task_key, base_dir):
        if ids is not None and pid not in ids:
            continue
        out[pid] = [(s, c)]
    return out


def _apply_resolve(td, resolve):
    """Collapse list-typed pair-lists with ``resolve(statuses, comments) ->
    (status, comment)``. ``None`` keeps the data unchanged."""
    if resolve is None:
        return td
    out = OrderedDict()
    for pid, pairs in td.items():
        if pairs is None or len(pairs) <= 1:
            out[pid] = pairs
            continue
        statuses = [p[0] for p in pairs]
        comments = [p[1] for p in pairs]
        out[pid] = [resolve(statuses, comments)]
    return out


# ---------------------------------------------------------------- character map
def _normalize_char_dict(d):
    if not d:
        return {}
    out = {}
    for k, v in d.items():
        if isinstance(k, tuple):
            status, comment = k
            out[(status, comment if comment not in ("", None) else None)] = v
        elif ':' in k:
            s, c = k.split(':', 1)
            out[(s.strip(), c.strip() or None)] = v
        else:
            out[(k.strip(), None)] = v
    return out


def _lookup_char(cdict, status, comment):
    if (status, comment) in cdict:
        return cdict[(status, comment)]
    if (status, None) in cdict:
        return cdict[(status, None)]
    return None


# ---------------------------------------------------------------- helpers
def _split_evenly(seq, n):
    """Split ``seq`` into ``n`` near-equal chunks (earlier chunks get +1)."""
    base, extra = divmod(len(seq), n)
    out, start = [], 0
    for i in range(n):
        size = base + (1 if i < extra else 0)
        out.append(seq[start:start + size])
        start += size
    return out


def _normalize_tasks(tasks):
    """Return list of ``(task_key, label)`` tuples; allow duplicates via a
    list/tuple-of-tuples form (dict keys cannot repeat)."""
    if isinstance(tasks, dict):
        return list(tasks.items())
    out = []
    for t in tasks:
        if isinstance(t, (list, tuple)) and len(t) == 2:
            out.append(tuple(t))
        else:
            out.append((t, t))
    return out


def _broadcast(value, n, default, normalize=None):
    """Turn a scalar / list into a length-``n`` per-task list. When ``value``
    is a list shorter than ``n`` the last entry is repeated; if longer, it's
    truncated. ``normalize`` is applied to each entry."""
    if value is None:
        return [default] * n
    if isinstance(value, list):
        items = list(value)
        if len(items) < n:
            items += [items[-1]] * (n - len(items))
        items = items[:n]
    else:
        items = [value] * n
    if normalize is not None:
        items = [normalize(x) for x in items]
    return items


# ---------------------------------------------------------------- plotting
def status_grid(
        tasks,
        colors,
        characters=None,
        resolves=None,
        dividers=None,
        double_dividers=None,
        divider_gap=0.6,
        divider_color='white',
        divider_lw=0.6,
        n_splits=4,
        ids=None,
        attempt=0,
        attempts=None,
        base_dir=None,
        cell_size=0.16,
        cell_width=None,
        cell_height=None,
        cell_alphas=None,
        header_styles=None,
        breakdown_groups=None,
        breakdown_color='#F2F2F2',
        out_path=None,
        char_fontsize=6,
        id_fontsize=6,
        header_fontsize=8,
        border_color='white',
        border_lw=0.4,
        char_color='black',
        wspace=0.05,
        no_rotation=False,
        group_headers=None
):
    """Render a problems × tasks grid of ``(status, comment)`` cells.

    Parameters
    ----------
    tasks : dict[str, str] | iterable[str] | iterable[tuple[str, str]]
        Tasks in display order. Use the list-of-tuples form to repeat a task
        with different labels (e.g. one entry expanded, one collapsed).
    colors : dict | list[dict]
        Colour spec — see :func:`status_bars.normalize_colors`.
    characters : dict | list[dict] | None
        ``(status, comment) -> single character`` map(s).
    resolves : callable | list[callable | None] | None
        Per-task list-collapsing callable ``(statuses, comments) -> (status,
        comment)``. ``None`` keeps list-typed entries expanded into multiple
        columns. Pass a list to set per-task behaviour (so the same task can
        appear once expanded and once collapsed).
    dividers, double_dividers : iterable[int] | None
        Task-boundary indices at which to draw dividers. Indices in
        ``double_dividers`` reserve twice ``divider_gap``; they are also
        rendered as divider lines even when omitted from ``dividers``.
    n_splits : int
        Number of side-by-side subplots; rows are distributed near-evenly.
        Headers are repeated above each subplot.
    ids : iterable[int] | None
        Restrict to these problem IDs (default: union of IDs found).
    attempt : int
        Default attempt number used for every task column.
    attempts : int | iterable[int] | None
        Per-task attempt number(s), overriding ``attempt``. Use to compare the
        same task across repeated attempts (e.g. list the task three times with
        ``attempts=(0, 1, 2)``). Broadcast like ``colors``: a shorter list
        repeats its last entry; ``None`` falls back to ``attempt`` for all.
    no_rotation : bool | iterable[bool]
        Keep the header horizontal (no 90° rotation) for single-column tasks.
        Broadcast like ``colors``: pass a list to set it per column (shorter
        list repeats its last entry).
    group_headers : iterable[tuple[str, int, int]] | None
        One or more horizontal header layers above the per-column headers.
        A single iterable of ``(label, first_task_idx, last_task_idx)`` entries
        produces one layer. Pass an iterable of such iterables for multiple
        layers, ordered bottom-to-top. Each entry spans task columns
        ``first_task_idx``..``last_task_idx`` (inclusive); ``label`` may
        contain ``\n`` for multi-line titles. An optional fourth element is a
        dict of text properties for that label, e.g. ``{'fontweight': 'bold'}``.
    base_dir : str | Path | iterable[str | Path] | None
        Root data directory (default: ``var.DATA_DIRECTORY / "claude"``).
        Broadcast like ``colors``: pass a list to set a different data
        directory per task column (shorter list repeats its last entry).
    cell_size : float
        Default width and height of each cell in inches. Retained for
        backwards compatibility when ``cell_width`` and ``cell_height`` are
        omitted.
    cell_width : float | iterable[float] | None
        Physical cell width in inches. A scalar applies to every task; an
        iterable assigns widths per task column, including each expanded cell
        of a list-valued task. ``None`` uses ``cell_size`` throughout.
    cell_height : float | None
        Physical cell height in inches. Set it independently of
        ``cell_width`` to produce rectangular cells.
    cell_alphas : float | iterable[float] | None
        Per-task cell opacity. A scalar applies to every task; an iterable is
        broadcast per task. ``None`` keeps cells fully opaque.
    header_styles : dict | iterable[dict] | None
        Matplotlib text keyword arguments for each task header, such as
        ``{'fontstyle': 'italic', 'color': 'gray'}``.
    breakdown_groups : iterable[tuple[int, int]] | None
        Inclusive ``(overall_task_idx, last_breakdown_task_idx)`` spans. A
        continuous band covers the overall result and its breakdown, and is
        drawn only behind populated rows; empty cells leave it exposed.
    breakdown_color : color
        Background color for ``breakdown_groups``.
    """
    default_base_dir = var.DATA_DIRECTORY / "claude"
    cell_height = cell_size if cell_height is None else cell_height
    if cell_height <= 0:
        raise ValueError("cell_height must be positive")

    task_items = _normalize_tasks(tasks)
    n_tasks = len(task_items)
    if n_tasks == 0:
        raise ValueError("at least one task is required")
    per_task_cell_widths = _broadcast(
        list(cell_width) if isinstance(cell_width, (list, tuple)) else cell_width,
        n_tasks, default=cell_size)
    if any(width <= 0 for width in per_task_cell_widths):
        raise ValueError("cell_width values must be positive")
    base_cell_width = per_task_cell_widths[0]
    per_task_cell_alphas = _broadcast(
        list(cell_alphas) if isinstance(cell_alphas, (list, tuple)) else cell_alphas,
        n_tasks, default=1.0)
    if any(alpha < 0 or alpha > 1 for alpha in per_task_cell_alphas):
        raise ValueError("cell_alphas values must be between 0 and 1")
    per_task_header_styles = _broadcast(
        list(header_styles) if isinstance(header_styles, (list, tuple)) else header_styles,
        n_tasks, default={})

    per_task_colors = _broadcast(colors, n_tasks, default={},
                                 normalize=normalize_colors)
    per_task_chars = _broadcast(characters, n_tasks, default={},
                                normalize=_normalize_char_dict)
    per_task_resolves = _broadcast(resolves, n_tasks, default=None)
    per_task_attempts = _broadcast(
        list(attempts) if isinstance(attempts, (list, tuple)) else attempts,
        n_tasks, default=attempt)
    per_task_no_rotation = _broadcast(
        list(no_rotation) if isinstance(no_rotation, (list, tuple)) else no_rotation,
        n_tasks, default=False)
    per_task_base_dir = _broadcast(
        list(base_dir) if isinstance(base_dir, (list, tuple)) else base_dir,
        n_tasks, default=default_base_dir, normalize=Path)

    # Load + apply resolves.
    task_data = []
    seen_ids = []
    for ti, (task, _) in enumerate(task_items):
        key = task.upper().replace('-', '_')
        td = _load_task_pairs(key, ids, per_task_attempts[ti], per_task_base_dir[ti])
        td = _apply_resolve(td, per_task_resolves[ti])
        task_data.append(td)
        for pid in td:
            if pid not in seen_ids:
                seen_ids.append(pid)

    # Drop rows where no task has any data.
    all_ids = []
    for pid in sorted(seen_ids):
        for ti in range(n_tasks):
            pairs = task_data[ti].get(pid)
            if pairs is not None and any(p[0] is not None for p in pairs):
                all_ids.append(pid)
                break
    n_rows = len(all_ids)

    # Per-task widths (after resolve).
    task_widths = []
    for td in task_data:
        max_w = 1
        for v in td.values():
            if v is not None and len(v) > max_w:
                max_w = len(v)
        task_widths.append(max_w)
    dividers_set = set(dividers or [])
    double_dividers_set = set(double_dividers or [])
    dividers_set.update(double_dividers_set)
    col_offsets, acc = [], 0.0
    for ti, w in enumerate(task_widths):
        if ti != 0 and ti in dividers_set:
            acc += divider_gap * (2 if ti in double_dividers_set else 1)
        col_offsets.append(acc)
        acc += w * per_task_cell_widths[ti] / base_cell_width
    total_cols = acc
    breakdown_groups = list(breakdown_groups or [])
    for i0, i1 in breakdown_groups:
        if not 0 <= i0 <= i1 < n_tasks:
            raise ValueError("breakdown group indices must be valid task indices")

    breakdown_task_indices = {
        ti
        for i0, i1 in breakdown_groups
        for ti in range(i0, i1 + 1)
    }

    # Chunk rows for the side-by-side subplots.
    n_splits = max(1, min(n_splits, max(n_rows, 1)))
    chunks = _split_evenly(all_ids, n_splits)
    chunk_height = max((len(c) for c in chunks), default=0)

    # Text sizes in cell units, from real font metrics (a cell is ``cell_size``
    # inches wide/tall and the aspect is equal), so the header block hugs the
    # text instead of floating on inflated estimates.
    from matplotlib.textpath import TextPath
    line_h = header_fontsize * 1.25 / 72 / cell_height  # one text line, cell units
    header_depths = []
    for ti, (_task, label) in enumerate(task_items):
        font_size = per_task_header_styles[ti].get('fontsize', header_fontsize)
        label_line_h = font_size * 1.25 / 72 / cell_height
        if task_widths[ti] > 1 or per_task_no_rotation[ti]:
            depth = (str(label).count("\n") + 1) * label_line_h
        else:
            depth = (TextPath((0, 0), str(label).replace("\n", " "),
                              size=font_size).get_extents().width
                     / 72 / cell_height)
        header_depths.append(depth + 0.15)
    group_headers = list(group_headers or [])
    if group_headers and isinstance(group_headers[0][0], str):
        group_header_layers = [group_headers]
    else:
        group_header_layers = [list(layer) for layer in group_headers]
    group_gap = line_h * 0.25  # gap between column labels and group titles
    top_pad = 0.1
    # Build compact header stacks independently for each span. Higher layers
    # only reserve space above the columns covered by their lower layers.
    header_tops = [-0.3 - depth for depth in header_depths]
    group_header_positions = []
    for layer in group_header_layers:
        covered_tasks = {
            ti
            for _label, i0, i1, *_style in layer
            for ti in range(i0, i1 + 1)
        }
        height = max(
            ((str(label).count("\n") + 1) * line_h for label, *_ in layer),
            default=0,
        )
        y = min(header_tops[ti] for ti in covered_tasks) - group_gap - height
        positioned_layer = []
        for label, i0, i1, *style in layer:
            label_height = (str(label).count("\n") + 1) * line_h
            positioned_layer.append((label, i0, i1, y, height, label_height,
                                     style[0] if style else {}))
            # Headers on a layer share a bottom edge, but higher layers only
            # need to clear the actual text height of each covered span.
            label_top = y + height - label_height
            for ti in range(i0, i1 + 1):
                header_tops[ti] = label_top
        group_header_positions.append(positioned_layer)
    header_room = top_pad - min(header_tops)

    # Figure sizing — cells stay square via ``set_aspect('equal')``.
    left_pad_in = 0.22
    top_pad_in = header_room * cell_height + 0.1
    fig_w = (n_splits * (left_pad_in + total_cols * base_cell_width)
             + (n_splits - 1) * wspace + 0.05)
    fig_h = top_pad_in + chunk_height * cell_height + 0.05

    fig, axes = plt.subplots(1, n_splits, figsize=(fig_w, fig_h),
                             gridspec_kw={'wspace': wspace})
    if n_splits == 1:
        axes = [axes]

    for ax, chunk_ids in zip(axes, chunks):
        chunk_rows = len(chunk_ids)
        # Highlight populated breakdown rows before drawing their cells.
        breakdown_backings = {}
        for i0, i1 in breakdown_groups:
            x0 = col_offsets[i0]
            x1 = (col_offsets[i1] + task_widths[i1]
                  * per_task_cell_widths[i1] / base_cell_width)
            for ri, pid in enumerate(chunk_ids):
                has_data = any(
                    pairs is not None and any(status is not None for status, _ in pairs)
                    for pairs in (task_data[ti].get(pid) for ti in range(i0, i1 + 1))
                )
                if has_data:
                    backing = plt.Rectangle(
                        (x0, ri), x1 - x0, 1,
                        facecolor=breakdown_color, edgecolor=breakdown_color,
                        linewidth=border_lw, zorder=-1,
                    )
                    ax.add_patch(backing)
                    breakdown_backings[ri] = backing

        # Cells.
        for ti, _ in enumerate(task_items):
            td = task_data[ti]
            cmap = per_task_colors[ti]
            cdict = per_task_chars[ti]
            col_off = col_offsets[ti]
            width = task_widths[ti]
            cell_width_units = per_task_cell_widths[ti] / base_cell_width
            for ri, pid in enumerate(chunk_ids):
                pairs = td.get(pid)
                for k in range(width):
                    x = col_off + k * cell_width_units
                    y = ri
                    color, char = 'white', None
                    has_status = False
                    if pairs is not None and k < len(pairs):
                        s, c = pairs[k]
                        if s is not None:
                            has_status = True
                            color = lookup_color(cmap, s, c) or 'white'
                            char = _lookup_char(cdict, s, c) if c is not None else None
                    in_breakdown = ti in breakdown_task_indices
                    if in_breakdown and not has_status:
                        continue
                    cell = plt.Rectangle(
                        (x, y), cell_width_units, 1, facecolor=color,
                        edgecolor=breakdown_color if in_breakdown else border_color,
                        linewidth=border_lw,
                        alpha=per_task_cell_alphas[ti],
                    )
                    if in_breakdown and ri in breakdown_backings:
                        cell.set_clip_path(breakdown_backings[ri])
                    ax.add_patch(cell)
                    if char:
                        ax.text(x + cell_width_units / 2, y + 0.575, str(char),
                                ha='center', va='center', fontsize=char_fontsize,
                                color=char_color, fontweight='bold')

        # Column headers — 90° for single-column tasks, 45° when the task
        # spans multiple columns (i.e. list-valued).
        for ti, (_task, label) in enumerate(task_items):
            cx = (col_offsets[ti] + task_widths[ti]
                  * per_task_cell_widths[ti] / base_cell_width / 2)
            if task_widths[ti] > 1 or per_task_no_rotation[ti]:
                rotation, ha = 0, 'center'
            else:
                rotation, ha = 90, 'center'
            text_style = {'fontsize': header_fontsize, **per_task_header_styles[ti]}
            ax.text(cx, -0.3, str(label),
                    ha=ha, va='bottom', rotation=rotation,
                    **text_style)

        # Compact group-header layers sit above only their covered span.
        for layer in group_header_positions:
            for lbl, i0, i1, y, height, _label_height, style in layer:
                x0 = col_offsets[i0]
                x1 = (col_offsets[i1] + task_widths[i1]
                      * per_task_cell_widths[i1] / base_cell_width)
                ax.text((x0 + x1) / 2, y + height, str(lbl),
                        ha='center', va='bottom', fontsize=header_fontsize,
                        linespacing=1.25, **style)
                ax.plot([x0, x1], [y + height + group_gap / 2] * 2,
                        color='black', linewidth=0.3, solid_capstyle='butt')

        # Row headers (problem ids), repeated per subplot.
        for ri, pid in enumerate(chunk_ids):
            ax.text(-0.2, ri + 0.5, f"{pid}",
                    ha='right', va='center', fontsize=id_fontsize)

        # Vertical divider lines (clipped to this chunk's row count).
        for ti in dividers_set:
            if 0 < ti < n_tasks:
                if any(i0 < ti <= i1 for i0, i1 in breakdown_groups):
                    continue
                gap = divider_gap * (2 if ti in double_dividers_set else 1)
                x = col_offsets[ti] - gap / 2
                ax.plot([x, x], [chunk_rows + 0.05, -0.05],
                        color=divider_color, linewidth=divider_lw, zorder=5)

        # Axes framing.
        ax.set_xlim(-1.2, total_cols + 0.05)
        ax.set_ylim(chunk_height + 0.05, -header_room)
        ax.set_aspect(cell_height / base_cell_width)
        # ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0)
        print(f"Saved figure to {out_path}")
    else:
        plt.show()


# ---------------------------------------------------------------- placeholders
SPEC_ISO_CHARS = {
    ('MISMATCH', '→'): '>',
    ('MISMATCH', '/'): '·',
    ('MISMATCH', '←'): '<',
    ('ISSUE', 'PROB'): 'P',
    ('ISSUE', 'BOTH'): 'B',
    ('ISSUE', 'GEN'): 'G',
}

PROOF_GEN_CHARS = {
    ('ISSUE', 'BASE'): '*',
    ('ISSUE', 'SPEC'): 'S',
    ('ISSUE', 'BOTH'): 'B',
    ('ISSUE', 'IMPL'): 'I',
}

if __name__ == '__main__':
    status_grid(
        tasks=[
            ('SPEC_ISO_MANUAL', "Merged"),
            ('SPEC_ISO', "Simple"),
            ('SPEC_ISO_GUIDED', "Guided"),
            ('SPEC_ISO_GUIDED_MULT', "∑"),
            ('SPEC_ISO_GUIDED_MULT', "attempts"),  # list — expanded
            ('PROOF_GEN', "Ground-truth"),
            ('PROOF_GEN_GEN', "(Simple)"),
            ('PROOF_GEN_GEN_GUIDED', "(Guided)"),
        ],
        colors=[SPEC_ISO_COLORS] * 5 + [PROOF_GEN_GT_COLORS] + [PROOF_GEN_GEN_COLORS] * 2,
        characters=[SPEC_ISO_CHARS] * 5 + [PROOF_GEN_CHARS] * 3,
        resolves=[None, None, None, resolve_list, None, None, None, None],
        n_splits=3,
        dividers=[1, 4, 5],  # divider between Final (idx 3) and Ground-truth (idx 4)
        double_dividers=[5],  # double divider between Final (idx 3) and Ground-truth (idx 4)
        group_headers=[
            [("Multiple", 3, 4)],
            [("Generat.", 6, 7)],
            [("Specification", 1, 4, {'fontweight': 'bold'}), ("Implement.", 5, 7, {'fontweight': 'bold'})],
        ],
        header_fontsize=6,
        out_path='out/results_grid.png',
        cell_height = 0.12,
        cell_width=[0.16, 0.16, 0.16, 0.16, 0.12, 0.16, 0.16, 0.16],
        breakdown_groups=[(3, 4)],
        cell_alphas=[1, 1, 1, 1, 0.65, 1, 1, 1],
        no_rotation=[False, False, False, True, False, False, False, False],
        header_styles=[{'fontstyle': 'italic'}, {}, {}, {}, {}, {}, {}, {}],

    )
    # status_grid(
    #     tasks=[
    #         ('SPEC_ISO_GUIDED_MULT', "Overall"),  # collapsed via resolve_list
    #         ('SPEC_ISO_GUIDED_MULT', "Attempts"),  # list — expanded
    #     ],
    #     colors=[SPEC_ISO_COLORS, SPEC_ISO_COLORS],
    #     characters=[SPEC_ISO_CHARS, SPEC_ISO_CHARS],
    #     resolves=[resolve_list, None],
    #     dividers=[1],  # divider between Final (idx 3) and Ground-truth (idx 4)
    #     out_path='out/spec_iso_mult_grid.png',
    # )
    # status_grid(
    #     tasks=[
    #         ('SPEC_ISO_MANUAL', "SPEC Final"),  # collapsed via resolve_list
    #         ('PROOF_GEN', "IMPL Ground-truth"),  # list — expanded
    #     ],
    #     colors=[SPEC_ISO_COLORS, PROOF_GEN_GT_COLORS],
    #     characters=[SPEC_ISO_CHARS, PROOF_GEN_CHARS],
    #     out_path='out/ground_truth_grid.png',
    # )
    # status_grid(
    #     tasks=[
    #         ('SPEC_ISO', "SPEC Simple"),  # collapsed via resolve_list
    #         ('PROOF_GEN_GEN', "IMPL Gen. (Simple)"),  # list — expanded
    #     ],
    #     colors=[SPEC_ISO_COLORS, PROOF_GEN_GEN_COLORS],
    #     characters=[SPEC_ISO_CHARS, PROOF_GEN_CHARS],
    #     out_path='out/generated_simple_grid.png',
    # )
    # status_grid(
    #     tasks=[
    #         ('SPEC_ISO_GUIDED', "SPEC Guided"),  # collapsed via resolve_list
    #         ('PROOF_GEN_GEN_GUIDED', "IMPL Gen. (Guided)"),  # list — expanded
    #     ],
    #     colors=[SPEC_ISO_COLORS, PROOF_GEN_GEN_COLORS],
    #     characters=[SPEC_ISO_CHARS, PROOF_GEN_CHARS],
    #     out_path='out/generated_guided_grid.png',
    # )
    # Variance analysis: same task repeated, one column per attempt.
    # SUBSET_IDS = [3, 12, 17, 32, 36, 45, 53, 64, 92, 96, 100, 103, 118, 128, 129, 154]
    # SUBSET_IDS = [1, 3, 5, 6, 12, 13, 17, 19, 21, 23, 24, 32, 34, 36, 40, 45, 52, 53, 64, 73, 77, 78, 81, 85, 86, 92,
    #               96, 99, 100, 103, 118, 119, 123, 128, 129, 132, 151, 155, 159, 161]
    SUBSET_IDS = None

    SUBSET_TASKS = (
        # ("SPEC_GEN", "SPEC_GEN\nSimple"),
        ("SPEC_ISO", "Simple"),
        # ("SPEC_GEN_GUIDED", "SPEC_GEN\nGuided"),
        ("SPEC_ISO_GUIDED", "Guided"),
        # ("IMPL_GEN", "IMPL_GEN\nGround-truth"),
        ("PROOF_GEN", "against \nGround-truth"),
        # ("IMPL_GEN_GEN", "Generated\n(Simple)"),
        ("PROOF_GEN_GEN", "(Simple)"),
        # ("IMPL_GEN_GEN_GUIDED", "IMPL_GEN\nGenerated\n(Guided)"),
        ("PROOF_GEN_GEN_GUIDED", "(Guided)"))
    # status_grid(
    #     tasks=[('SPEC_ISO_MANUAL', "SPEC Final")] + [(i, n if j == 1 else "") for i, n in SUBSET_TASKS
    #                                                  for j in (0, 1, 2)],
    #     colors=[SPEC_ISO_COLORS] + [SPEC_ISO_COLORS] * 4 * 3 + [PROOF_GEN_GT_COLORS] * 2 * 3 + [
    #         PROOF_GEN_GEN_COLORS] * 4 * 3,
    #     characters=[SPEC_ISO_CHARS] + [SPEC_ISO_CHARS] * 4 * 3 + [PROOF_GEN_CHARS] * 6 * 3,
    #     ids=SUBSET_IDS,
    #     attempts=(0,) + (0, 1, 2) * len(SUBSET_TASKS),
    #     n_splits=1,
    #     dividers=[1 + (3 * i) for i in range(len(SUBSET_TASKS))],
    #     no_rotation=(False,) + (True,) * 3 * len(SUBSET_TASKS),
    #     header_fontsize=6,
    #     out_path='out/variance_grid.png',
    # )
    # ABLATION_CATS = ["", "-no_mcp_server", "-no_skills", "-no_mcp_server-no_skills"]
    # status_grid(
    #     tasks= [('SPEC_ISO_MANUAL', "SPEC Final")] + [(i, n.replace("\n", " ") if j == "" else f"- {j[4:]}") for i, n in SUBSET_TASKS
    #                                                  for j in ABLATION_CATS],
    #     colors=[SPEC_ISO_COLORS] + [SPEC_ISO_COLORS] * 4 * len(ABLATION_CATS) + [PROOF_GEN_GT_COLORS] * 2 * len(ABLATION_CATS) + [
    #         PROOF_GEN_GEN_COLORS] * 4 * len(ABLATION_CATS),
    #     characters=[SPEC_ISO_CHARS] + [SPEC_ISO_CHARS] * 4 * len(ABLATION_CATS) + [PROOF_GEN_CHARS] * 6 * len(ABLATION_CATS),
    #     base_dir=[var.DATA_DIRECTORY / "claude"] + [var.DATA_DIRECTORY / f"claude{c}" for c in ABLATION_CATS] * len(SUBSET_TASKS),
    #     ids=SUBSET_IDS,
    #     n_splits=1,
    #     dividers=[1 + (len(ABLATION_CATS) * i) for i in range(len(SUBSET_TASKS))],
    #     header_fontsize=6,
    #     out_path='out/ablation_grid.png',
    # )
    MODELS = ["claude", "codex", "kimi"]
    MODEL_LABELS = {"claude": "Opus 4.6", "codex": "GPT-5.4", "kimi": "Kimi K2.5"}
    status_grid(
        tasks=[('SPEC_ISO_MANUAL', "Merged (Claude)")] + [(i, MODEL_LABELS[j]) for
                                                     i, n in SUBSET_TASKS
                                                     for j in MODELS],
        colors=[SPEC_ISO_COLORS] + [SPEC_ISO_COLORS] * 2 * len(MODELS) + [PROOF_GEN_GT_COLORS] * 1 * len(MODELS) + [
            PROOF_GEN_GEN_COLORS] * 2 * len(MODELS),
        characters=[SPEC_ISO_CHARS] + [SPEC_ISO_CHARS] * 2 * len(MODELS) + [PROOF_GEN_CHARS] * 3 * len(MODELS),
        base_dir=[var.DATA_DIRECTORY / "claude"] + [var.DATA_DIRECTORY / m for m in MODELS] * len(SUBSET_TASKS),
        ids=SUBSET_IDS,
        n_splits=2,
        dividers=[1 + (len(MODELS) * i) for i in range(len(SUBSET_TASKS))],
        double_dividers=[1 + len(MODELS)*2],

        group_headers=[
            [(n, 1 + len(MODELS) * t, 1 + len(MODELS) * t + len(MODELS) - 1) for t, (i, n) in enumerate(SUBSET_TASKS)],
            [("against  Generated", len(MODELS)*3 + 1, len(MODELS)*5)],
            [("Specification", 1, len(MODELS)*2, {'fontweight': 'bold'}), ("Implementation", len(MODELS)*2 + 1, len(MODELS)*5, {'fontweight': 'bold'})],
        ],
        header_fontsize=6,
        cell_height = 0.12,
        out_path='out/models_grid.png',
        header_styles=[{'fontstyle': 'italic'}, {}, {}, {}, {}, {}, {}, {}],
    )
    FULL_SUBSET_TASKS = (
        ("SPEC_GEN", "SPEC_GEN\nSimple"),
        ("SPEC_ISO", "SPEC_ISO\nSimple"),
        ("SPEC_GEN_GUIDED", "SPEC_GEN\nGuided"),
        ("SPEC_ISO_GUIDED", "SPEC_ISO\nGuided"),
        ("IMPL_GEN", "IMPL_GEN\nGround-truth"),
        ("PROOF_GEN", "PROOF_GEN\nGround-truth"),
        ("IMPL_GEN_GEN", "IMPL_GEN\nGenerated\n(Simple)"),
        ("PROOF_GEN_GEN", "PROOF_GEN\nGenerated\n(Simple)"),
        ("IMPL_GEN_GEN_GUIDED", "IMPL_GEN\nGenerated\n(Guided)"),
        ("PROOF_GEN_GEN_GUIDED", "PROOF_GEN\nGenerated\n(Guided)"),
    )
    status_grid(
        tasks=[('SPEC_ISO_MANUAL', "Merged (Claude)")] + [(i, MODEL_LABELS[j]) for
                                                     i, n in FULL_SUBSET_TASKS
                                                     for j in MODELS],
        colors=[SPEC_ISO_COLORS] + [SPEC_ISO_COLORS] * 4 * len(MODELS) + [
            PROOF_GEN_GT_COLORS] * 2 * len(MODELS) + [PROOF_GEN_GEN_COLORS] * 4 * len(MODELS),
        characters=[SPEC_ISO_CHARS] + [SPEC_ISO_CHARS] * 4 * len(MODELS) + [
            PROOF_GEN_CHARS] * 6 * len(MODELS),
        base_dir=[var.DATA_DIRECTORY / "claude"] + [var.DATA_DIRECTORY / m for m in MODELS] * len(FULL_SUBSET_TASKS),
        ids=SUBSET_IDS,
        n_splits=2,
        dividers=[1 + len(MODELS) * i for i in range(len(FULL_SUBSET_TASKS))],
        double_dividers=[1 + len(MODELS) * 2],
        group_headers=[
            [(n, 1 + len(MODELS) * t, 1 + len(MODELS) * t + len(MODELS) - 1)
             for t, (i, n) in enumerate(FULL_SUBSET_TASKS)],
            [("against  Generated", 1 + len(MODELS) * 6, len(MODELS) * 10)],
            [("Specification", 1, len(MODELS) * 4, {'fontweight': 'bold'}),
             ("Implementation", len(MODELS) * 4 + 1, len(MODELS) * 10, {'fontweight': 'bold'})],
        ],
        header_fontsize=6,
        cell_height=0.12,
        out_path='out/models_grid-full.png',
        header_styles=[{'fontstyle': 'italic'}] + [{}] * (len(FULL_SUBSET_TASKS) * len(MODELS)),
    )
    MODELS = ["claude", "codex", "kimi", "codex-pre"]
    MODEL_LABELS = {"claude": "Opus 4.6", "codex": "GPT-5.4", "kimi": "Kimi K2.5", "codex-pre": "GPT-5.1"}
    SUBSET_IDS = [2, 20, 23, 24, 31, 37, 50, 52, 87, 91, 105, 116, 122, 131, 144, 157]
    status_grid(
        tasks=[('SPEC_ISO_MANUAL', "Merged (Claude)")] + [(i, MODEL_LABELS[j]) for
                                                     i, n in SUBSET_TASKS
                                                     for j in MODELS],
        colors=[SPEC_ISO_COLORS] + [SPEC_ISO_COLORS] * 2 * len(MODELS) + [PROOF_GEN_GT_COLORS] * 1 * len(MODELS) + [
            PROOF_GEN_GEN_COLORS] * 2 * len(MODELS),
        characters=[SPEC_ISO_CHARS] + [SPEC_ISO_CHARS] * 2 * len(MODELS) + [PROOF_GEN_CHARS] * 3 * len(MODELS),
        base_dir=[var.DATA_DIRECTORY / "claude"] + [var.DATA_DIRECTORY / m for m in MODELS] * len(SUBSET_TASKS),
        ids=SUBSET_IDS,
        n_splits=1,
        dividers=[1 + (len(MODELS) * i) for i in range(len(SUBSET_TASKS))],
        group_headers=[
            [(n, 1 + len(MODELS) * t, 1 + len(MODELS) * t + len(MODELS) - 1) for t, (i, n) in enumerate(SUBSET_TASKS)],
            [("against  Generated", len(MODELS)*3 + 1, len(MODELS)*5)],
            [("Specification", 1, len(MODELS)*2, {'fontweight': 'bold'}), ("Implementation", len(MODELS)*2 + 1, len(MODELS)*5, {'fontweight': 'bold'})],
        ],
        header_fontsize=6,
        out_path='out/models_grid_pre.png',
        header_styles=[{'fontstyle': 'italic'}] + [{}] * (len(SUBSET_TASKS) * len(MODELS)),
    )
    # MODELS = ["claude", "codex-pre"]
    # MODEL_LABELS = {"claude": "Opus 4.6", "codex-pre": "GPT-5.1"}
    # SUBSET_TASKS = (
    #     ("SPEC_GEN", "SPEC_GEN\nSimple"),
    #     ("SPEC_ISO", "SPEC_ISO\nSimple"),
    #     ("SPEC_GEN_GUIDED", "SPEC_GEN\nGuided"),
    #     ("SPEC_ISO_GUIDED", "SPEC_ISO\nGuided"),
    #     ("IMPL_GEN", "IMPL_GEN\nGround-truth"),
    #     ("PROOF_GEN", "PROOF_GEN\nGround-truth"),
    #     ("IMPL_GEN_GEN", "IMPL_GEN\nGenerated\n(Simple)"),
    #     ("PROOF_GEN_GEN", "PROOF_GEN\nGenerated\n(Simple)"),
    #     ("IMPL_GEN_GEN_GUIDED", "IMPL_GEN\nGenerated\n(Guided)"),
    #     ("PROOF_GEN_GEN_GUIDED", "PROOF_GEN\nGenerated\n(Guided)"))
    # status_grid(
    #     tasks=[('SPEC_ISO_MANUAL', "SPEC Final")] + [(i, MODEL_LABELS[j]) for
    #                                                  i, n in SUBSET_TASKS
    #                                                  for j in MODELS],
    #     colors=[SPEC_ISO_COLORS] + [SPEC_ISO_COLORS] * 2 * len(MODELS) + [PROOF_GEN_GT_COLORS] * 1 * len(MODELS) + [
    #         PROOF_GEN_GEN_COLORS] * 2 * len(MODELS),
    #     characters=[SPEC_ISO_CHARS] + [SPEC_ISO_CHARS] * 2 * len(MODELS) + [PROOF_GEN_CHARS] * 3 * len(MODELS),
    #     base_dir=[var.DATA_DIRECTORY / "claude"] + [var.DATA_DIRECTORY / m for m in MODELS] * len(SUBSET_TASKS),
    #     ids=SUBSET_IDS,
    #     n_splits=1,
    #     dividers=[1 + (len(MODELS) * i) for i in range(len(SUBSET_TASKS))],
    #     group_headers=[(n, 1 + len(MODELS) * t, 1 + len(MODELS) * t + len(MODELS) - 1)
    #                    for t, (i, n) in enumerate(SUBSET_TASKS)],
    #     header_fontsize=6,
    #     out_path='out/models_grid_pre_test.png',
    # )
