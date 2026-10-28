"""Scatter plot of (time, price) per problem across multiple task families.

The figure is laid out as a grid of ``(n_rows, n_cols)`` subplots — one row
per task family, one column per task variant. All subplots share the x and y
axes. Points are circles coloured by ``(status, comment)``. Inward-pointing
triangle markers on the axes mark each subplot's average time and price,
labelled with the value (no ``avg:`` prefix). Subplot titles only appear on
``title_rows``; the first subplot of each row carries the row's task-family
name as its y-axis title; the last subplot of each row carries the x-axis
unit. A thin horizontal separator can be drawn after any row.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

# Make sibling modules importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent))
from status_bars import (
    SPEC_ISO_COLORS, PROOF_GEN_GT_COLORS, PROOF_GEN_GEN_COLORS,
    normalize_colors, lookup_color, resolve_list,
)
from status_grid import (
    _problem_pairs, _normalize_tasks, _broadcast,
)

from clever_bench.benchmark import Benchmark
from core.benchmarks.clever import TASKS
from core.problem import Problem, ProblemInfo
from core import var


# ---------------------------------------------------------------- metrics
def _result_message(problem):
    """Return the Claude ``ResultMessage`` record for a problem, or ``None``."""
    claude_records = problem.metadata.get('claude', []) or []
    return next(
        (t for t in claude_records if t.get('type') == 'ResultMessage'), None
    )


def _extract_time(problem):
    """Wall-clock runtime in seconds, or ``None`` if missing."""
    return None if problem.timing is None else float(problem.timing)


def _extract_cost(problem):
    """Total API cost in USD, or ``None`` if missing."""
    result_msg = _result_message(problem)
    if result_msg is None:
        return None
    price = result_msg.get('total_cost_usd')
    return None if price is None else float(price)


# Token usage components summed into the total-token metric.
_TOKEN_KEYS = (
    'input_tokens', 'cache_creation_input_tokens',
    'cache_read_input_tokens', 'output_tokens',
)


def _extract_tokens(problem):
    """Total token usage (input + cache create + cache read + output), or
    ``None`` if no usage record is present."""
    result_msg = _result_message(problem)
    if result_msg is None:
        return None
    usage = result_msg.get('usage') or {}
    if not any(k in usage for k in _TOKEN_KEYS):
        return None
    return float(sum(usage.get(k, 0) or 0 for k in _TOKEN_KEYS))


def _fmt_tokens(v):
    """Compact token count: ``950``, ``12k``, ``3.4M``."""
    if v >= 1e6:
        return f'{v / 1e6:.1f}M'
    if v >= 1e3:
        return f'{v / 1e3:.0f}k'
    return f'{v:.0f}'


# name -> (extractor, axis unit label, average-value formatter)
METRICS = {
    'time': {'extract': _extract_time, 'unit': 's',
             'fmt': lambda v: f'{v:.0f}s'},
    'cost': {'extract': _extract_cost, 'unit': 'USD',
             'fmt': lambda v: f'${v:.2f}'},
    'tokens': {'extract': _extract_tokens, 'unit': 'tok',
               'fmt': _fmt_tokens},
}


def _metric(name):
    try:
        return METRICS[name]
    except KeyError:
        raise ValueError(
            f"unknown metric {name!r}; choose from {sorted(METRICS)}")


def _iter_task_problems(task_key, ids, attempt, base_dir):
    """Yield :class:`Problem` instances for a benchmark task."""
    if task_key not in TASKS:
        return
    dataset = TASKS[task_key]['dataset']
    benchmark = Benchmark(is_sample=dataset['subset'] == "sample_examples")
    benchmark.load_all()
    if ids is None:
        ids = [p.problem_id for p in benchmark.problems]
    for pid in ids:
        try:
            yield Problem.load(
                base_dir=base_dir,
                info=ProblemInfo(id=pid, task=task_key, **dataset, attempt=attempt),
            )
        except FileNotFoundError:
            continue


# Default marker cycle used when ``base_dir`` is a list of directories.
DEFAULT_MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X', '*')


def _normalize_base_dirs(base_dir):
    """Return a list of ``Path`` objects for ``base_dir``.

    ``base_dir`` may be ``None`` (default data dir), a single path-like, or
    a list/tuple of path-likes (one marker per entry, see ``markers``).
    """
    default = var.DATA_DIRECTORY / "claude"
    if base_dir is None:
        return [default]
    if isinstance(base_dir, (str, Path)):
        return [Path(base_dir)]
    return [Path(bd) for bd in base_dir]


def _normalize_attempts(attempt):
    """Return a list of attempt indices for ``attempt``.

    ``attempt`` may be a single int or a list/tuple of ints — unlike
    ``base_dir``, all attempts share the same marker (they are just pooled
    together as extra points/re-runs).
    """
    if isinstance(attempt, (list, tuple)):
        return list(attempt)
    return [attempt]


# ---------------------------------------------------------------- avg indicators
def _draw_avg_indicators(ax, avg_t, avg_p, timings=None, prices=None, *,
                         x_fmt=lambda v: f'{v:.0f}s',
                         y_fmt=lambda v: f'${v:.2f}',
                         color='black', tri_size=45, fontsize=6,
                         x_axes_y=0.04, y_axes_x=0.04,
                         x_offset_pts=5, y_offset_pts=5,
                         collision_threshold=0):
    """Triangle markers + value labels at ``avg_t`` on x and ``avg_p`` on y.
    ``x_fmt`` / ``y_fmt`` format each average into its axis label string.

    Each label switches sides when the default position would overlap data:
    - X-axis label sits above the triangle by default, moves to its right
      when too many points fall in the box around it.
    - Y-axis label is rotated 90° and sits to the right of the triangle by
      default, moves above it on collision.
    """
    trans_x = ax.get_xaxis_transform()  # (data, axes)
    trans_y = ax.get_yaxis_transform()  # (axes, data)
    ax.scatter([avg_t], [x_axes_y], marker='v', s=tri_size,
               facecolor=color, edgecolor=color, linewidths=0.5,
               transform=trans_x, clip_on=False, zorder=10)
    ax.scatter([y_axes_x], [avg_p], marker='<', s=tri_size,
               facecolor=color, edgecolor=color, linewidths=0.5,
               transform=trans_y, clip_on=False, zorder=10)

    timings = list(timings or [])
    prices = list(prices or [])

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    log_x = ax.get_xscale() == 'log'
    log_y = ax.get_yscale() == 'log'

    def _ax_to_dx(a):
        return xmin * (xmax / xmin) ** a if log_x else xmin + (xmax - xmin) * a

    def _ax_to_dy(a):
        return ymin * (ymax / ymin) ** a if log_y else ymin + (ymax - ymin) * a

    def _dx_to_ax(d):
        return (np.log(d / xmin) / np.log(xmax / xmin)) if log_x \
            else (d - xmin) / (xmax - xmin)

    def _dy_to_ax(d):
        return (np.log(d / ymin) / np.log(ymax / ymin)) if log_y \
            else (d - ymin) / (ymax - ymin)

    def _count_in_axes_box(xa_lo, xa_hi, ya_lo, ya_hi):
        if not timings:
            return 0
        x_lo, x_hi = _ax_to_dx(xa_lo), _ax_to_dx(xa_hi)
        y_lo, y_hi = _ax_to_dy(ya_lo), _ax_to_dy(ya_hi)
        return sum(1 for t, p in zip(timings, prices)
                   if x_lo <= t <= x_hi and y_lo <= p <= y_hi)

    avg_t_ax = _dx_to_ax(avg_t)
    avg_p_ax = _dy_to_ax(avg_p)

    # X-axis label — default sits above the triangle reading horizontally;
    # box matches the label footprint (narrower & shorter than y-axis side).
    x_collide = _count_in_axes_box(
        max(0, avg_t_ax - 0.04), min(1, avg_t_ax + 0.04),
        x_axes_y, x_axes_y + 0.08,
    ) > collision_threshold
    x_label = x_fmt(avg_t)
    if x_collide:
        ax.annotate(x_label,
                    xy=(avg_t, x_axes_y), xycoords=trans_x,
                    xytext=(x_offset_pts, 0), textcoords='offset points',
                    ha='left', va='center',
                    fontsize=fontsize, color=color, zorder=10)
    else:
        ax.annotate(x_label,
                    xy=(avg_t, x_axes_y), xycoords=trans_x,
                    xytext=(0, x_offset_pts), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=fontsize, color=color, zorder=10)

    # Y-axis label — default sits to the right of the triangle reading
    # horizontally, so the collision footprint is wider in x than in y.
    y_collide = _count_in_axes_box(
        max(0, y_axes_x - 0.02), y_axes_x + 0.30,
        max(0, avg_p_ax - 0.06), min(1, avg_p_ax + 0.06),
    ) > collision_threshold
    y_label = y_fmt(avg_p)
    if y_collide:
        ax.annotate(y_label,
                    xy=(y_axes_x, avg_p), xycoords=trans_y,
                    xytext=(0, y_offset_pts), textcoords='offset points',
                    ha='center', va='bottom', rotation=-90,
                    fontsize=fontsize, color=color, zorder=10)
    else:
        ax.annotate(y_label,
                    xy=(y_axes_x, avg_p), xycoords=trans_y,
                    xytext=(y_offset_pts, 0), textcoords='offset points',
                    ha='left', va='center',  # rotation=-90,
                    fontsize=fontsize, color=color, zorder=10)


# ---------------------------------------------------------------- plotting
def scatter_time_price(
        rows,
        *,
        x_metric='time',
        y_metric='cost',
        ids=None,
        attempt=0,
        base_dir=None,
        base_dir_labels=None,
        markers=DEFAULT_MARKERS,
        distinguish_attempts=False,
        out_path=None,
        subplot_size=(2.4, 2.4),
        wspace=0.0,
        hspace=0.15,
        extra_gap_at_separator=0.05,
        marker_size=12,
        edge_color='black',
        edge_lw=0.3,
        alpha=0.85,
        log_x=True,
        log_y=True,
        grid=True,
        x_unit=None,
        y_unit=None,
        unit_fontsize=7,
        tick_fontsize=5,
        title_fontsize=8,
        row_label_fontsize=8,
        title_rows=(0, 2),
        separator_after_rows=(1,),
        separator_color='lightgray',
        separator_lw=0.6,
        marker_legend_fontsize=6,
        row_groups=None,
        row_group_fontsize=9,
        row_group_gap_pt=4,
):
    """Render a (time, price) scatter plot grid.

    Parameters
    ----------
    x_metric, y_metric : str
        Metric plotted on each axis: ``'time'`` (seconds), ``'cost'`` (USD),
        or ``'tokens'`` (total token usage). Default ``x_metric='time'`` and
        ``y_metric='cost'`` reproduce the original cost-vs-time plot. Axis
        units and average-value formats follow the chosen metric unless
        overridden via ``x_unit`` / ``y_unit``.
    rows : list[dict]
        One entry per row. Each dict has:
        - ``'name'`` (str) — y-axis title for the row's first subplot.
        - ``'tasks'`` (dict[str, str]) — task key → column title.
        - ``'colors'`` (dict | list[dict]) — per-task colour spec.
        - ``'resolves'`` (callable | list | None, optional) — per-task list
          collapsing callable.
    base_dir : str | Path | list[str | Path] | None
        Data directory to read problems from. May be a list of directories
        (e.g. multiple runs/models) — each entry is then plotted with a
        different marker (cycling through ``markers``), while colouring
        (by status/comment) stays identical across entries. A single
        directory (or ``None`` for the default) keeps the original
        behaviour of plotting everything as circles.
    base_dir_labels : list[str] | None
        Labels for each ``base_dir`` entry, used in a marker legend on the
        first subplot. Defaults to each directory's ``name``. Ignored when
        ``base_dir`` is not a list.
    markers : sequence[str]
        Marker cycle used for successive ``base_dir`` entries.
    attempt : int | list[int]
        Attempt index (or list of indices) to read per problem. When a list
        is given, all attempts are pooled together and drawn with the same
        marker by default — set ``distinguish_attempts=True`` to instead
        give each attempt its own marker (see ``distinguish_attempts``).
    distinguish_attempts : bool
        When ``attempt`` is a list with more than one entry, controls
        whether attempts share a marker (``False``, default — pooled as
        repeated runs) or each get their own marker from ``markers``
        (``True``). Combines with ``base_dir``: when both are lists, every
        ``(base_dir, attempt)`` pair gets a distinct marker.
    title_rows : iterable[int]
        Row indices on which subplot titles are shown.
    separator_after_rows : iterable[int]
        Row indices after which to draw a thin horizontal separator line.
    row_groups : iterable[tuple[str, int, int]] | None
        Vertical bold headers to the left of the row titles, each spanning rows
        ``first``..``last`` (inclusive) with a vertical line alongside, e.g.
        ``[("Specification", 0, 1), ("Implementation", 2, 3)]``.
    """
    x_spec, y_spec = _metric(x_metric), _metric(y_metric)
    x_extract, y_extract = x_spec['extract'], y_spec['extract']
    if x_unit is None:
        x_unit = x_spec['unit']
    if y_unit is None:
        y_unit = y_spec['unit']

    base_dirs = _normalize_base_dirs(base_dir)
    attempts = _normalize_attempts(attempt)
    n_attempts = len(attempts)
    multi_base_dir = isinstance(base_dir, (list, tuple))
    multi_attempt = isinstance(attempt, (list, tuple)) and n_attempts > 1
    distinguish_attempts = distinguish_attempts and multi_attempt
    show_marker_legend = multi_base_dir or distinguish_attempts
    if base_dir_labels is None:
        base_dir_labels = [bd.name for bd in base_dirs]

    if not rows:
        raise ValueError("at least one row is required")
    n_rows = len(rows)
    n_cols = max(len(r['tasks']) for r in rows)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * subplot_size[0], n_rows * subplot_size[1]),
        sharex='row', sharey='row',
        gridspec_kw={'wspace': wspace, 'hspace': hspace},
    )
    # Normalize axes shape to 2D for uniform indexing.
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1) if n_rows == 1 else axes.reshape(-1, 1)

    title_rows = set(title_rows or ())
    separator_after_rows = set(separator_after_rows or ())

    plain = FuncFormatter(lambda x, _: f'{x:g}')

    # Print averages header.
    label_w = max(len(r['name']) for r in rows)
    x_head = f"avg {x_metric} ({x_unit})"
    y_head = f"avg {y_metric} ({y_unit})"
    print(f"{'task':<{label_w}}  {'col':<22}  {'n':>4}  {x_head:>13}  {y_head:>14}")
    print('-' * (label_w + 2 + 22 + 2 + 4 + 2 + 13 + 2 + 14))

    for ri, row in enumerate(rows):
        row_name = row['name']
        task_items = _normalize_tasks(row['tasks'])
        n_tasks = len(task_items)
        per_task_colors = _broadcast(row['colors'], n_tasks, default={},
                                     normalize=normalize_colors)
        per_task_resolves = _broadcast(row.get('resolves'), n_tasks,
                                       default=None)
        show_titles = ri in title_rows

        for ti, (task, label) in enumerate(task_items):
            ax = axes[ri, ti]
            key = task.upper().replace('-', '_')
            cmap = per_task_colors[ti]
            resolve = per_task_resolves[ti]
            xs, ys = [], []

            for bi, bd in enumerate(base_dirs):
                for ai, att in enumerate(attempts):
                    if distinguish_attempts:
                        marker = markers[(bi * n_attempts + ai) % len(markers)]
                    else:
                        marker = markers[bi % len(markers)]
                    for problem in _iter_task_problems(key, ids, att, bd):
                        xv, yv = x_extract(problem), y_extract(problem)
                        if xv is None or yv is None:
                            continue
                        pairs = _problem_pairs(problem)
                        if pairs is None:
                            continue
                        if resolve is not None and len(pairs) > 1:
                            statuses = [p[0] for p in pairs]
                            comments = [p[1] for p in pairs]
                            pairs = [resolve(statuses, comments)]
                        xs.append(xv)
                        ys.append(yv)
                        for s, c in pairs:
                            if s is None:
                                continue
                            color = lookup_color(cmap, s, c) or 'lightgray'
                            ax.scatter(xv, yv, marker=marker, s=marker_size,
                                       facecolor=color, edgecolor=edge_color,
                                       linewidth=edge_lw, alpha=alpha, zorder=3)

            n = len(xs)
            avg_x = sum(xs) / n if n else float('nan')
            avg_y = sum(ys) / n if n else float('nan')
            print(f"{row_name:<{label_w}}  {label[:22]:<22}  {n:>4}  "
                  f"{avg_x:>13.2f}  {avg_y:>14.4f}")

            if log_x:
                ax.set_xscale('log')
            if log_y:
                ax.set_yscale('log')
            if grid:
                ax.grid(True, which='both', alpha=0.25, linewidth=0.5)
            if show_titles:
                ax.set_title(label, fontsize=title_fontsize)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.xaxis.set_major_formatter(plain)
            ax.yaxis.set_major_formatter(plain)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

            if n:
                _draw_avg_indicators(ax, avg_x, avg_y, xs, ys,
                                     x_fmt=x_spec['fmt'], y_fmt=y_spec['fmt'])

        # Hide any unused axes in this row (when row has fewer tasks than n_cols).
        for ti in range(n_tasks, n_cols):
            axes[ri, ti].set_visible(False)

        # Y-axis title (task family name) on first subplot of the row.
        axes[ri, 0].set_ylabel(row_name, fontsize=row_label_fontsize)

        # 'USD' unit at top of first subplot (in line with y-axis spine).
        if y_unit:
            axes[ri, 0].text(0.0, 1.01, y_unit, transform=axes[ri, 0].transAxes,
                             ha='center', va='bottom', fontsize=unit_fontsize)

        # 's' unit at right end of last subplot (in line with x-axis spine).
        if x_unit:
            last = axes[ri, n_tasks - 1]
            last.text(1.01, 0.0, x_unit, transform=last.transAxes,
                      ha='left', va='center', fontsize=unit_fontsize)

    # Push rows after each separator further down so the inter-group gap is
    # larger than the intra-group ``hspace``.
    if extra_gap_at_separator:
        shift = 0.0
        for ri in range(n_rows):
            if shift > 0:
                for ax in axes[ri]:
                    pos = ax.get_position()
                    ax.set_position([pos.x0, pos.y0 - shift,
                                     pos.width, pos.height])
            if ri in separator_after_rows:
                shift += extra_gap_at_separator

    # Thin horizontal separator(s) between rows.
    for ri in separator_after_rows:
        if 0 <= ri < n_rows - 1:
            top_below = axes[ri + 1, 0].get_position().y1
            bot_above = axes[ri, 0].get_position().y0
            y = (top_below + bot_above) / 2
            x_left = axes[ri, 0].get_position().x0
            x_right = axes[ri, -1].get_position().x1
            line = plt.Line2D([x_left, x_right], [y, y],
                              color=separator_color, linewidth=separator_lw,
                              transform=fig.transFigure)
            fig.add_artist(line)

    # Vertical group headers spanning related rows, left of everything the rows
    # draw on their left (tick labels, row titles), measured after layout.
    if row_groups:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        to_fig = fig.transFigure.inverted()
        gap = row_group_gap_pt * fig.dpi / 72
        x_left = min(axes[ri, 0].get_tightbbox(renderer).x0
                     for _, r0, r1 in row_groups for ri in range(r0, r1 + 1))
        x_line = to_fig.transform((x_left - gap, 0))[0]
        x_text = to_fig.transform((x_left - 2 * gap, 0))[0]
        for label, r0, r1 in row_groups:
            y_top = axes[r0, 0].get_position().y1
            y_bot = axes[r1, 0].get_position().y0
            fig.add_artist(plt.Line2D([x_line, x_line], [y_bot, y_top],
                                      transform=fig.transFigure,
                                      color='black', linewidth=0.5))
            fig.text(x_text, (y_top + y_bot) / 2, label, rotation=90,
                     ha='right', va='center', fontsize=row_group_fontsize,
                     fontweight='bold')

    # Legend mapping markers to base_dir/attempt entries.
    if show_marker_legend:
        handles = []
        if distinguish_attempts:
            for bi, bd_label in enumerate(base_dir_labels):
                for ai, att in enumerate(attempts):
                    idx = (bi * n_attempts + ai) % len(markers)
                    label = (f"{bd_label} / attempt {att}" if multi_base_dir
                             else f"attempt {att}")
                    handles.append(plt.Line2D(
                        [], [], marker=markers[idx], linestyle='',
                        markerfacecolor='none', markeredgecolor='black',
                        markersize=6, label=label))
        else:
            for i, label in enumerate(base_dir_labels):
                handles.append(plt.Line2D(
                    [], [], marker=markers[i % len(markers)], linestyle='',
                    markerfacecolor='none', markeredgecolor='black',
                    markersize=6, label=label))
        fig.legend(handles=handles, loc='upper right',
                   fontsize=marker_legend_fontsize, frameon=False,
                   bbox_to_anchor=(1.0, 1.0))

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f"Saved figure to {out_path}")
    else:
        plt.show()


if __name__ == '__main__':
    GRID = [
        {
            'name': 'SPEC_GEN',
            'tasks': {
                'SPEC_GEN': "Simple",
                'SPEC_GEN_GUIDED': "Guided",
                'SPEC_GEN_GUIDED_MULT': "Multiple",
            },
            'colors': SPEC_ISO_COLORS,
            'resolves': [None, None, resolve_list],
        },
        {
            'name': 'SPEC_ISO',
            'tasks': {
                'SPEC_ISO': "Simple",
                'SPEC_ISO_GUIDED': "Guided",
                'SPEC_ISO_GUIDED_MULT': "Multiple",
            },
            'colors': SPEC_ISO_COLORS,
            'resolves': [None, None, resolve_list],
        },
        {
            'name': 'IMPL_GEN',
            'tasks': {
                'IMPL_GEN': "against Ground-truth",
                'IMPL_GEN_GEN': "against Generated (Simple)",
                'IMPL_GEN_GEN_GUIDED': "against Generated (Guided)",
            },
            'colors': [PROOF_GEN_GT_COLORS] + [PROOF_GEN_GEN_COLORS] * 2,
        },
        {
            'name': 'PROOF_GEN',
            'tasks': {
                'PROOF_GEN': "against Ground-truth",
                'PROOF_GEN_GEN': "against Generated (Simple)",
                'PROOF_GEN_GEN_GUIDED': "against Generated (Guided)",
            },
            'colors': [PROOF_GEN_GT_COLORS] + [PROOF_GEN_GEN_COLORS] * 2,
        },
    ]

    scatter_time_price(
        rows=GRID,
        row_groups=[("Specification", 0, 1), ("Implementation", 2, 3)],
        out_path='out/scatter-pricing-time.png',
    )
    SUBSET_GRID = [
        {
            'name': 'SPEC_GEN',
            'tasks': {
                'SPEC_GEN': "Simple",
                'SPEC_GEN_GUIDED': "Guided"
            },
            'colors': SPEC_ISO_COLORS,
            'resolves': [None, None, resolve_list],
        },
        {
            'name': 'SPEC_ISO',
            'tasks': {
                'SPEC_ISO': "Simple",
                'SPEC_ISO_GUIDED': "Guided"
            },
            'colors': SPEC_ISO_COLORS,
            'resolves': [None, None, resolve_list],
        },
        {
            'name': 'IMPL_GEN',
            'tasks': {
                'IMPL_GEN': "against Ground-truth",
                'IMPL_GEN_GEN': "against Generated (Simple)",
                'IMPL_GEN_GEN_GUIDED': "against Generated (Guided)",
            },
            'colors': [PROOF_GEN_GT_COLORS] + [PROOF_GEN_GEN_COLORS] * 2,
        },
        {
            'name': 'PROOF_GEN',
            'tasks': {
                'PROOF_GEN': "against Ground-truth",
                'PROOF_GEN_GEN': "against Generated (Simple)",
                'PROOF_GEN_GEN_GUIDED': "against Generated (Guided)",
            },
            'colors': [PROOF_GEN_GT_COLORS] + [PROOF_GEN_GEN_COLORS] * 2,
        },
    ]
    ROW_GROUPS = [("Specification", 0, 1), ("Implementation", 2, 3)]
    SUBSET_IDS = [3, 12, 17, 32, 36, 45, 53, 64, 92, 96, 100, 103, 118, 128, 129, 154]
    scatter_time_price(
        rows=SUBSET_GRID,
        ids=SUBSET_IDS,
        attempt=[0, 1, 2],
        distinguish_attempts=True,
        row_groups=ROW_GROUPS,
        out_path='out/variance-scatter-pricing-time.png',
    )
    # Total token usage (x) vs cost (y).
    scatter_time_price(
        rows=SUBSET_GRID,
        ids=SUBSET_IDS,
        x_metric='tokens',
        y_metric='cost',
        x_unit='tokens',
        row_groups=ROW_GROUPS,
        out_path='out/variance-scatter-tokens-cost.png',
    )
    scatter_time_price(
        rows=SUBSET_GRID,
        ids=SUBSET_IDS,
        row_groups=ROW_GROUPS,
        out_path='out/ablation-scatter-pricing-time.png',
        base_dir=[
            var.DATA_DIRECTORY / "claude",
            var.DATA_DIRECTORY / "claude-no_mcp_server",
            var.DATA_DIRECTORY / "claude-no_skills",
            var.DATA_DIRECTORY / "claude-no_mcp_server-no_skills",
        ],
    )
