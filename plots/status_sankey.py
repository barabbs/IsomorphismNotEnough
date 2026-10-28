"""Sankey-style variant of :mod:`status_bars`.

Each row (task) is the same stacked status bar, with the same header. Instead
of one trapezoid per status between rows, every entry gets its own slot of unit
width on the bar edges (an imaginary dot) and a curved (cubic Bezier) ribbon
joins the entry's slot on the bottom edge of one bar to its slot on the top edge
of the bar below, so the flows between classifications become visible. Within
each sub-bar, entries are ordered to keep ribbons bundled and to minimise
crossings.

Data loading, color specs and ordering conventions are shared with
``status_bars.py``; edit the calls at the bottom of this file to render.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgb

from clever_bench.benchmark import Benchmark
from core.benchmarks.clever import TASKS
from core.problem import Problem, ProblemInfo
from core import var

from status_bars import (
    EXTERNAL_SUBDIR, MISSING_STATUS, COMMENT_LABEL_ALIASES,
    get_status_comment, normalize_pair, normalize_colors, lookup_color,
    _ordered, _orders_from_colors, _is_color_split,
    LIGHT_GREEN, CONNECTOR_COLORS,
    SPEC_ISO_COLORS, SPEC_ISO_COMMENT_LABELS,
    PROOF_GEN_GT_COLORS, PROOF_GEN_GEN_COLORS, PROOF_GEN_COMMENT_LABELS,
)


# Header comment labels used where the full label does not fit its slot.
COMMENT_SHORT_ALIASES = {
    ('MISMATCH', '→'): 'Strong.',
    ('MISMATCH', '/'): 'Neith.',
    ('MISMATCH', '←'): 'Weak.',
    ('ISSUE', 'PROB'): 'GT Spec.',
    ('ISSUE', 'GEN'): 'Gen. Spec.',
    ('ISSUE', 'BASE'): 'Bench.',
    ('ISSUE', 'SPEC'): 'Spec.',
    ('ISSUE', 'IMPL'): 'Impl.',
}

# ---------------------------------------------------------------- data loading
def load_task_pairs(task, ids, attempt, base_dir):
    """Return ``{problem_id: (status, comment)}`` for one task.

    Tasks in ``TASKS`` are read from their per-problem records; anything else
    falls back to ``<base_dir>/clever/<task>.json`` (dict keyed by problem id,
    or a list/parallel arrays indexed by position).
    """
    if task in TASKS:
        dataset = TASKS[task]['dataset']
        pairs = {}
        for pid in ids:
            try:
                problem = Problem.load(
                    base_dir=base_dir,
                    info=ProblemInfo(id=pid, task=task, **dataset, attempt=attempt),
                )
            except FileNotFoundError:
                pairs[pid] = (None, None)
                continue
            pairs[pid] = get_status_comment(problem)
        return pairs

    import json
    path = Path(base_dir) / EXTERNAL_SUBDIR / f"{task}.json"
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get('status'), list):
        comments = data.get('comment') or [None] * len(data['status'])
        items = {i: {'status': s, 'comment': c}
                 for i, (s, c) in enumerate(zip(data['status'], comments))}
    elif isinstance(data, dict):
        items = {int(k): v for k, v in data.items()}
    else:
        items = dict(enumerate(data))
    pairs = {}
    for pid in ids:
        record = items.get(pid)
        if record is None:
            pairs[pid] = (None, None)
            continue
        record = record.get('metadata', record)
        pairs[pid] = normalize_pair(record.get('status'), record.get('comment'))
    return pairs


def _group_key(pair):
    status, comment = pair
    if status is None:
        return 'OK', MISSING_STATUS
    return status, comment


# ---------------------------------------------------------------- layout
def _group_order(rows, primary_colors, status_order, comment_order):
    """Global left-to-right order of ``(status, comment)`` groups."""
    color_status_order, comments_by_status = _orders_from_colors(primary_colors)
    seen = OrderedDict()
    for pairs in rows:
        for pair in pairs.values():
            seen.setdefault(_group_key(pair), None)
    statuses = _ordered(OrderedDict.fromkeys(s for s, _ in seen),
                        status_order or color_status_order)
    groups = []
    for status in statuses:
        if comment_order is None:
            co = comments_by_status.get(status)
        elif isinstance(comment_order, dict):
            co = comment_order.get(status)
        else:
            co = comment_order
        cmts = _ordered([c for s, c in seen if s == status], co)
        if MISSING_STATUS in cmts:
            cmts.remove(MISSING_STATUS)
            cmts.insert(0, MISSING_STATUS)
        groups += [(status, c) for c in cmts]
    return groups


def _row_orders(rows, ids, group_index):
    """Return, per row, the list of problem ids in left-to-right dot order.

    Row 0 is sorted by group, then by the entry's downstream path (its group in
    every following row) so entries that travel together sit together. Each
    later row is sorted by group, then by the entry's position in the row
    above, which keeps connectors between adjacent rows from crossing inside
    a bundle.
    """
    gidx = [{pid: group_index[_group_key(pairs[pid])] for pid in ids}
            for pairs in rows]
    orders = [sorted(ids, key=lambda p: tuple(g[p] for g in gidx) + (p,))]
    for r in range(1, len(rows)):
        prev_pos = {pid: k for k, pid in enumerate(orders[-1])}
        orders.append(sorted(ids, key=lambda p: (gidx[r][p], prev_pos[p])))
    return orders


def _lighten(color, amount):
    """Mix ``color`` with white; ``amount`` = 0 keeps it, 1 gives white."""
    rgb = np.array(to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def _bezier(x0, y0, x1, y1, n=40):
    """Vertical S-curve from ``(x0, y0)`` to ``(x1, y1)`` (cubic Bezier)."""
    t = np.linspace(0, 1, n)[:, None]
    ym = (y0 + y1) / 2
    p0, p1, p2, p3 = (np.array(p) for p in ((x0, y0), (x0, ym), (x1, ym), (x1, y1)))
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


# ---------------------------------------------------------------- plotting
def plot(rows, labels, colors, status_order=None, comment_order=None,
         connector_colors=None, comment_classes=None,
         comment_label_aliases=None, comment_short_aliases=None,
         link_color='source', link_lighten=0.0,
         link_alpha=0.4, bar_height=0.125, row_gap=None, major_color='#4D4D4D',
         minor_color='#808080',
         italic_labels=(), rotate_max=4, plot_width=6.0, label_margin=None,
         label_margin_word='Generated',
         out_path=None):
    """Draw stacked bars joined by per-entry ribbons.

    Parameters
    ----------
    rows : list[dict[int, (status, comment)]]
        One mapping per row, all over the same problem ids.
    labels : list[str]
        Row labels.
    colors : dict | list[dict]
        Normalized color spec, shared or one per row.
    connector_colors, comment_classes :
        As in :func:`status_bars.plot`; used here for the header fill only.
    link_color : {'source', 'target'}
        Take each ribbon's hue from the bar above or the bar below.
    link_lighten, link_alpha : float
        Ribbon tint (mix with white) and opacity.
    bar_height, row_gap : float
        Bar thickness and the vertical space left between consecutive bars
        (where the ribbons run), in data units. ``row_gap=None`` uses twice
        the bar height.
    major_color, minor_color : color
        Colours of the dividers between statuses and between comments within
        a status, used in the header, on the bar sides and between rows.
    italic_labels : iterable[str]
        Row labels to set in italics.
    plot_width, label_margin : float
        Width of the plot area and of the row-label margin to its left, in
        inches. Both are fixed and the figure is saved uncropped, so every
        figure has the same plot area at the same place (for aligning them).
        ``label_margin=None`` makes the margin exactly as wide as
        ``label_margin_word`` (the longest row-label line across the figures
        to be aligned) plus the tick-label padding.
    comment_short_aliases : dict | None
        Short forms of the header comment labels, used where the full form
        does not fit its slot (default :data:`COMMENT_SHORT_ALIASES`).
    rotate_max : float
        Header status labels whose first-row width is at most this many
        entries are rotated by 90 degrees.
    """
    n = len(rows)
    ids = sorted(rows[0])
    connector_colors = connector_colors or {}
    comment_classes = comment_classes or {}
    aliases = COMMENT_LABEL_ALIASES if comment_label_aliases is None else comment_label_aliases
    short_aliases = COMMENT_SHORT_ALIASES if comment_short_aliases is None else comment_short_aliases

    per_row_colors = list(colors) if isinstance(colors, list) else [colors] * n
    per_row_colors += [per_row_colors[-1]] * (n - len(per_row_colors))
    primary_colors = per_row_colors[0]

    groups = _group_order(rows, primary_colors, status_order, comment_order)
    group_index = {g: i for i, g in enumerate(groups)}
    orders = _row_orders(rows, ids, group_index)

    statuses = list(OrderedDict.fromkeys(s for s, _ in groups))
    all_comments = OrderedDict((s, [c for gs, c in groups if gs == s]) for s in statuses)
    labeled_comments = OrderedDict(
        (s, [c for c in cmts if c is not None and c != MISSING_STATUS])
        for s, cmts in all_comments.items()
    )

    # ---- bar layout (count units) --------------------------------------------
    # Entries are sorted by group, so entry k of a row occupies [k, k + 1] and
    # sits inside its own sub-bar: the imaginary dot at k + 0.5 on the bar edge
    # is where its ribbon attaches.
    counts = [OrderedDict((g, 0) for g in groups) for _ in rows]
    for r, pairs in enumerate(rows):
        for pid in ids:
            counts[r][_group_key(pairs[pid])] += 1
    row_boundaries, row_sub_extents = [], []
    for r in range(n):
        bounds, extents, offset = [0.0], {}, 0.0
        for status in statuses:
            for c in all_comments[status]:
                extents[(status, c)] = (offset, offset + counts[r][(status, c)])
                offset += counts[r][(status, c)]
            bounds.append(offset)
        row_boundaries.append(bounds)
        row_sub_extents.append(extents)
    xs = [{pid: k + 0.5 for k, pid in enumerate(order)} for order in orders]
    width = float(len(ids))

    bar_half = bar_height / 2
    if row_gap is None:
        row_gap = 2 * bar_height
    pitch = bar_height + row_gap
    row_y = [r * pitch for r in range(n)]
    # Header offsets are measured from the top edge of the first bar.
    y_separator_top = -bar_half - 0.07  # sub-bar separator lines top — above comment label
    y_comment_text = -bar_half - 0.15  # comment label baseline
    y_status_text = -bar_half - 0.3  # status label baseline
    y_fill_top = -bar_half - 0.4  # header fill top — above status label (acts as background)
    y_separator_div = 0.14
    y_bottom = row_y[-1] + bar_half + 0.12  # room for two-line row labels
    y_top_limit = y_fill_top - 0.05
    divider_lw = 0.5
    label_fontsize = 7  # row labels
    count_dy = 0.012  # count labels: downward shift from the bar centre (y axis points down)

    fig, ax = plt.subplots(figsize=(plot_width, 0.9 * (y_bottom - y_top_limit)))
    renderer = fig.canvas.get_renderer()
    if label_margin is None:
        probe = fig.text(0, 0, label_margin_word, fontsize=label_fontsize)
        pad_pt = plt.rcParams['ytick.major.pad']
        label_margin = probe.get_window_extent(renderer).width / fig.dpi + pad_pt / 72
        probe.remove()
    fig.set_size_inches(label_margin + plot_width, fig.get_figheight())
    # Final geometry up front, so header labels can be measured against it.
    fig.subplots_adjust(left=label_margin / fig.get_figwidth(), right=1, top=1, bottom=0)
    ax.set_xlim(0, width)

    def entry_color(r, pid):
        status, comment = _group_key(rows[r][pid])
        if comment == MISSING_STATUS:
            return lookup_color(per_row_colors[r], 'OK', None)
        return lookup_color(per_row_colors[r], status, comment)

    # ---- header: slots of row-0 status width, comment slots equally spaced ---
    header_slot_boundaries = row_boundaries[0]
    comment_slot = {}
    for i, status in enumerate(statuses):
        cmts = labeled_comments[status]
        if not cmts:
            continue
        left = header_slot_boundaries[i]
        sub_w = (header_slot_boundaries[i + 1] - left) / len(cmts)
        for k, c in enumerate(cmts):
            comment_slot[(status, c)] = (left + k * sub_w, left + (k + 1) * sub_w)

    def _split_boundaries(status, n_classes):
        """K-1 ``(header_x, row0_x)`` class boundaries, or ``None`` if class
        indices are not non-decreasing through the labeled comments."""
        classes = comment_classes[0] if isinstance(comment_classes, list) else {}
        prev_cls = -1
        for c in labeled_comments[status]:
            cls = classes.get((status, c), 0)
            if cls < prev_cls:
                return None
            prev_cls = cls
        i_status = statuses.index(status)
        boundaries = []
        for k in range(n_classes - 1):
            last_c = None
            for c in labeled_comments[status]:
                if classes.get((status, c), 0) <= k:
                    last_c = c
            if last_c is None:
                boundaries.append((header_slot_boundaries[i_status],
                                   row_boundaries[0][i_status]))
            else:
                boundaries.append((comment_slot[(status, last_c)][1],
                                   row_sub_extents[0][(status, last_c)][1]))
        return boundaries

    # ---- header fills (header -> row 0) ---------------------------------------
    for i, status in enumerate(statuses):
        fill = connector_colors.get(status)
        if fill is None:
            continue
        left = [(header_slot_boundaries[i], y_fill_top), (row_boundaries[0][i], -bar_half)]
        right = [(header_slot_boundaries[i + 1], y_fill_top),
                 (row_boundaries[0][i + 1], -bar_half)]
        boundaries = _split_boundaries(status, len(fill)) if _is_color_split(fill) else None
        if boundaries is None:
            color = fill[0] if _is_color_split(fill) else fill
            ax.add_patch(plt.Polygon(left + right[::-1], facecolor=color,
                                     edgecolor='none', zorder=0))
            continue
        paths = [left]
        for slot_x, l_x in boundaries:
            paths.append([(slot_x, y_fill_top), (slot_x, y_separator_top), (l_x, -bar_half)])
        paths.append(right)
        for k in range(len(fill)):
            ax.add_patch(plt.Polygon(paths[k] + paths[k + 1][::-1], facecolor=fill[k],
                                     edgecolor='none', zorder=0))

    def slot_px(l, r, pad_px=2):
        return ax.transData.transform((r, 0))[0] - ax.transData.transform((l, 0))[0] - pad_px

    def width_px(t):
        return t.get_window_extent(renderer).width

    for i, status in enumerate(statuses):
        l, r = header_slot_boundaries[i], header_slot_boundaries[i + 1]
        # Too narrow for a horizontal label: rotate it, centred in the header.
        narrow = r - l <= rotate_max
        t = ax.text((l + r) / 2, (y_fill_top - bar_half) / 2 if narrow else y_status_text,
                    str(status), ha='center', va='center', rotation=90 if narrow else 0,
                    fontsize=8, fontweight='bold', zorder=4)
        # A horizontal label wider than its span shrinks to fit.
        if not narrow and width_px(t) > slot_px(l, r):
            t.set_fontsize(8 * slot_px(l, r) / width_px(t))

    # Comment labels keep their full form if all of a status's labels fit their
    # slots, else they all switch to their short form; a short label still too
    # wide shrinks to fit.
    comment_fontsize = 4.5
    for status in statuses:
        items = [(c, lr) for (s_, c), lr in comment_slot.items() if s_ == status]
        texts = []
        for comment, (l, r) in items:
            full = aliases.get((status, comment), aliases.get(comment, str(comment)))
            t = ax.text((l + r) / 2, y_comment_text, full, ha='center', va='center',
                        fontsize=comment_fontsize, color='black', zorder=4)
            texts.append((t, comment, l, r, full))
        if any(width_px(t) > slot_px(l, r) for t, _, l, r, _ in texts):
            for t, comment, l, r, full in texts:
                t.set_text(short_aliases.get((status, comment), short_aliases.get(comment, full)))
                if width_px(t) > slot_px(l, r):
                    t.set_fontsize(max(comment_fontsize * slot_px(l, r) / width_px(t), 3))

    # ---- dividers -------------------------------------------------------------
    # Major dividers separate statuses, minor ones separate comments within a
    # status. The same two colours are used in the header and between rows
    # (curved like the ribbons), finely dotted. All of them sit just above the
    # header backdrop and behind everything else.
    major, minor = [], []

    # Header -> row 0.
    for x_top, x_bot in zip(header_slot_boundaries[1:-1], row_boundaries[0][1:-1]):
        major.append([(x_top, y_fill_top), (x_bot, -bar_half)])
    for i, status in enumerate(statuses):
        cmts = labeled_comments[status]
        if not any(row_sub_extents[0][(status, c)][0] != row_sub_extents[0][(status, c)][1]
                   for c in cmts):
            continue
        for comment in cmts:
            slot_r = comment_slot[(status, comment)][1]
            l_r = row_sub_extents[0][(status, comment)][1]
            if l_r == row_boundaries[0][i + 1]:
                continue  # coincides with the major divider
            minor.append([(slot_r, y_separator_top - y_separator_div),
                          (slot_r, y_separator_top), (l_r, -bar_half)])

    # Row r -> row r+1: every inner status boundary, plus the labeled-comment
    # boundaries when both rows split that status.
    for r in range(n - 1):
        y_top, y_bot = row_y[r] + bar_half, row_y[r + 1] - bar_half
        status_edges = list(zip(row_boundaries[r][1:-1], row_boundaries[r + 1][1:-1]))
        comment_edges = []
        upper_e, lower_e = row_sub_extents[r], row_sub_extents[r + 1]
        for status in statuses:
            cmts = labeled_comments[status]
            if not (any(upper_e[(status, c)][0] != upper_e[(status, c)][1] for c in cmts)
                    and any(lower_e[(status, c)][0] != lower_e[(status, c)][1] for c in cmts)):
                continue
            comment_edges += [(upper_e[(status, c)][1], lower_e[(status, c)][1]) for c in cmts]
        major += [_bezier(xu, y_top, xl, y_bot) for xu, xl in status_edges]
        minor += [_bezier(xu, y_top, xl, y_bot) for xu, xl in comment_edges
                  if (xu, xl) not in status_edges]

    for lines, color in ((minor, minor_color), (major, major_color)):
        ax.add_collection(LineCollection(lines, colors=[color], linewidths=divider_lw,
                                         linestyles=[(0, (3, 0))], capstyle='butt',
                                         zorder=0.5))

    # ---- ribbons (bottom edge of row r -> top edge of row r+1) ---------------
    # Entries adjacent in both rows and with the same colour are merged into one
    # ribbon, so bundles are drawn as single translucent bands and only genuine
    # crossings blend.
    for r in range(n - 1):
        y_top, y_bot = row_y[r] + bar_half, row_y[r + 1] - bar_half
        runs = []  # [color, src_first, src_last, tgt_first, tgt_last]
        for pid in orders[r]:
            src, tgt = xs[r][pid], xs[r + 1][pid]
            color = entry_color(r if link_color == 'source' else r + 1, pid)
            if runs and runs[-1][0] == color and tgt == runs[-1][4] + 1:
                runs[-1][2], runs[-1][4] = src, tgt
            else:
                runs.append([color, src, src, tgt, tgt])
        polys, cols = [], []
        for color, s0, s1, t0, t1 in runs:
            left = _bezier(s0 - 0.5, y_top, t0 - 0.5, y_bot)
            right = _bezier(s1 + 0.5, y_top, t1 + 0.5, y_bot)
            polys.append(np.vstack([left, right[::-1]]))
            cols.append((*_lighten(color, link_lighten), link_alpha))
        ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors='none',
                                         linewidths=0, zorder=1))

    # ---- bars + counts ---------------------------------------------------------
    for r in range(n):
        for (status, comment), (x0, x1) in row_sub_extents[r].items():
            cnt = x1 - x0
            if cnt == 0:
                continue
            missing = comment == MISSING_STATUS
            color = lookup_color(per_row_colors[r], 'OK' if missing else status,
                                 None if missing else comment)
            ax.barh(row_y[r], cnt, left=x0, color=color, edgecolor='none',
                    linewidth=0, height=bar_height, zorder=2)
            if missing:
                ax.barh(row_y[r], cnt, left=x0, facecolor='none',
                        edgecolor=LIGHT_GREEN, linewidth=0, hatch='//',
                        height=bar_height, zorder=3)
            else:
                ax.text((x0 + x1) / 2, row_y[r] + count_dy, str(int(cnt)), ha='center', va='center',
                        fontsize=6, color='black', fontweight='bold', zorder=4)

    # ---- axes ------------------------------------------------------------------
    ax.set_xlim(0, width)
    ax.set_yticks(row_y)
    # Horizontal labels: the bars are too thin for rotated multi-line ones.
    ax.set_yticklabels(labels, fontsize=label_fontsize, va='center', ha='right',
                       multialignment='right')
    for tick in ax.get_yticklabels():
        if tick.get_text() in italic_labels:
            tick.set_fontstyle('italic')
    ax.set_ylim(y_bottom, y_top_limit)
    ax.set_xticks([])
    ax.tick_params(axis='y', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.margins(x=0, y=0)
    fig.subplots_adjust(left=label_margin / fig.get_figwidth(), right=1, top=1, bottom=0)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=600)  # uncropped: keeps the plot area fixed
        plt.close(fig)
        print(f"Saved figure to {out_path}")
    else:
        plt.show()


# ---------------------------------------------------------------- top-level API
def status_sankey(tasks, colors=None, ids=None, attempt=0, base_dir=None,
                  status_order=None, comment_order=None,
                  connector_colors=None, comment_classes=None,
                  comment_label_aliases=None, out_path=None, **plot_kwargs):
    """Render stacked status bars joined by per-entry ribbons.

    ``tasks`` is an iterable of task keys or a ``{task: row_label}`` dict.
    ``ids`` restricts to these problem ids (``None`` = the whole benchmark).
    Remaining arguments match :func:`status_bars.status_bars`; extra keyword
    arguments go to :func:`plot`.
    """
    base_dir = Path(base_dir) if base_dir is not None else var.DATA_DIRECTORY / "claude"
    if isinstance(colors, list):
        colors = [normalize_colors(c) for c in colors]
    else:
        colors = normalize_colors(colors)
    items = list(tasks.items()) if isinstance(tasks, dict) else [(t, t) for t in tasks]

    if ids is None:
        benchmark = Benchmark(is_sample=False)
        benchmark.load_all()
        ids = [p.problem_id for p in benchmark.problems]
    ids = sorted(ids)

    rows = [load_task_pairs(task.upper().replace('-', '_'), ids, attempt, base_dir)
            for task, _ in items]
    plot(rows, [label for _, label in items], colors,
         status_order=status_order, comment_order=comment_order,
         connector_colors=connector_colors, comment_classes=comment_classes,
         comment_label_aliases=comment_label_aliases,
         out_path=out_path, **plot_kwargs)


if __name__ == '__main__':
    SPEC_TASKS = {'SPEC_ISO': "Simple", 'SPEC_ISO_GUIDED': "Guided",
                  'SPEC_ISO_GUIDED_MULT': "Multiple", 'SPEC_ISO_MANUAL': "Merged"}
    IMPL_TASKS = {'PROOF_GEN': "Ground-\ntruth", 'PROOF_GEN_GEN': "Generated\n(Simple)",
                  'PROOF_GEN_GEN_GUIDED': "Generated\n(Guided)"}
    for model in ("claude", "codex", "kimi"):
        # Only Claude has the multi-spec run and the manual classification.
        spec_tasks = SPEC_TASKS if model == "claude" else dict(list(SPEC_TASKS.items())[:2])
        status_sankey(
            tasks=spec_tasks,
            colors=SPEC_ISO_COLORS,
            connector_colors=CONNECTOR_COLORS,
            comment_classes=[{('ISSUE', 'BOTH'): 1, ('ISSUE', 'GEN'): 2}] * len(spec_tasks),
            comment_label_aliases=SPEC_ISO_COMMENT_LABELS,
            italic_labels={"Merged"},
            base_dir=var.DATA_DIRECTORY / model,
            out_path=f'out/sankey_spec_{model}.png',
        )
        status_sankey(
            tasks=IMPL_TASKS,
            colors=[PROOF_GEN_GT_COLORS] + [PROOF_GEN_GEN_COLORS] * 2,
            connector_colors=CONNECTOR_COLORS,
            comment_classes=[
                {('ISSUE', 'BOTH'): 1, ('ISSUE', 'IMPL'): 2},
                {('ISSUE', 'SPEC'): 2, ('ISSUE', 'BOTH'): 2, ('ISSUE', 'IMPL'): 2},
                {('ISSUE', 'SPEC'): 2, ('ISSUE', 'BOTH'): 2, ('ISSUE', 'IMPL'): 2},
            ],
            comment_label_aliases=PROOF_GEN_COMMENT_LABELS,
            base_dir=var.DATA_DIRECTORY / model,
            out_path=f'out/sankey_impl_{model}.png',
        )
