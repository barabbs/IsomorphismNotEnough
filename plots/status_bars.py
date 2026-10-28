"""Horizontal stacked bar plot of status / comment distributions across tasks.

Edit the call at the bottom of this file (or import ``status_bars`` from another
script) to control which tasks, colors, and ordering to render.
"""

import json
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt

from clever_bench.benchmark import Benchmark
from core.benchmarks.clever import TASKS
from core.problem import Problem, ProblemInfo
from core import var

STATUS_ALIASES = {'TIMEOUT': 'FAIL'}
EXTERNAL_SUBDIR = 'clever'  # data/<base_dir>/clever/<task>.json fallback location
MISSING_STATUS = "—"  # sentinel for problems with no status yet — drawn striped, leftmost
COMMENT_LABEL_ALIASES = {
    ('MISMATCH', '→'): 'STRONGER',
    ('MISMATCH', '/'): 'NEITHER',
    ('MISMATCH', '←'): 'WEAKER',
}


# ---------------------------------------------------------------- placeholders
def resolve_list(status, comment):
    """Pick a single representative {status, comment} from a list-valued entry.

    Placeholder — to be replaced with actual tie-breaking logic.
    """
    if any(s == "OK" for s in status):
        return "OK", None
    elif any(s == "ISSUE" and c in ("PROB", "BOTH") for s, c in zip(status, comment)):
        return "ISSUE", "PROB"
    elif any(s == "MISMATCH" for s in status):
        return "MISMATCH", None
    elif any(s == "ISSUE" and c == "GEN" for s, c in zip(status, comment)):
        return "ISSUE", "GEN"
    return "FAIL", None


# ---------------------------------------------------------------- data loading
def discover_problem_ids(task, dataset, attempt, base_dir):
    """List problem IDs available on disk for a given task, in order."""
    base = Path(base_dir) / dataset['dataset'] / task
    if not base.exists():
        return []
    suffix = f"-{attempt}.json"
    prefix = f"{dataset['subset']}-"
    ids = []
    for f in sorted(base.glob(f"{prefix}*{suffix}")):
        m = re.fullmatch(rf"{re.escape(prefix)}(\d+)-{attempt}", f.stem)
        if m:
            ids.append(int(m.group(1)))
    return ids


def load_task_problems(task, dataset, problem_ids, attempt, base_dir):
    benchmark = Benchmark(is_sample=dataset['subset'] == "sample_examples")
    benchmark.load_all()
    if problem_ids is None:
        problem_ids = [p.problem_id for p in benchmark.problems]
    problems = []
    for pid in problem_ids:
        try:
            problems.append(Problem.load(
                base_dir=base_dir,
                info=ProblemInfo(id=pid, task=task, **dataset, attempt=attempt),
            ))
        except FileNotFoundError:
            # Keep a placeholder so missing files surface as MISSING_STATUS bars.
            problems.append(None)
    return problems


def normalize_pair(status, comment):
    """Resolve list values, alias statuses (e.g. TIMEOUT->FAIL), and tidy empties."""
    if isinstance(status, list):
        status, comment = resolve_list(status, comment)
    if not status:
        status = None
    status = STATUS_ALIASES.get(status, status)
    if comment == "":
        comment = None
    return status, comment


def get_status_comment(problem):
    status = problem.metadata.get('status')
    comment = problem.metadata.get('comment')
    if problem.exception is not None and status is None:
        status = problem.exception.get('type')
    return normalize_pair(status, comment)


def load_external_pairs(task, base_dir):
    """Load (status, comment) pairs from ``<base_dir>/clever/<task>.json``.

    Accepted JSON shapes:
        - list of records: ``[{"status": ..., "comment": ...}, ...]``
        - dict keyed by problem id: ``{"0": {"status": ..., "comment": ...}}``
        - parallel arrays: ``{"status": [...], "comment": [...]}``
    Records may also wrap the status/comment under a ``metadata`` key.
    """
    path = Path(base_dir) / EXTERNAL_SUBDIR / f"{task}.json"
    with open(path) as f:
        data = json.load(f)

    if (isinstance(data, dict)
            and 'status' in data and isinstance(data['status'], list)):
        statuses = data['status']
        comments = data.get('comment') or [None] * len(statuses)
        return [normalize_pair(s, c) for s, c in zip(statuses, comments)]

    items = data.values() if isinstance(data, dict) else data
    pairs = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Unsupported entry in {path}: {item!r}")
        record = item.get('metadata', item)
        pairs.append(normalize_pair(record.get('status'), record.get('comment')))
    return pairs


def aggregate(pairs):
    """Return ``{status: {comment_or_None: count}}`` preserving insertion order.

    Entries with ``status is None`` are folded under ``OK`` with a sub-comment
    of :data:`MISSING_STATUS` — the plotter renders them striped so they remain
    visually distinct while sharing the OK header.
    """
    counts = OrderedDict()
    for s, c in pairs:
        if s is None:
            s, c = 'OK', MISSING_STATUS
        counts.setdefault(s, OrderedDict())
        counts[s][c] = counts[s].get(c, 0) + 1
    return counts


# ---------------------------------------------------------------- color config
def normalize_colors(colors):
    """Accept any of the following forms and return ``{(status, comment|None): color}``:

    * ``None`` -> ``{}``
    * dict with string keys ``"STATUS"`` or ``"STATUS:COMMENT"``
    * dict with tuple keys ``(status, comment_or_None)`` (already normalized)
    * dict with string keys mapping to dicts ``{comment_or_None: color, ...}``;
      a key of ``None`` (or ``""``) sets the status-level fallback
    """
    if not colors:
        return {}
    out = {}
    for key, value in colors.items():
        if isinstance(value, dict):
            for sub_key, color in value.items():
                comment = sub_key if sub_key not in ("", None) else None
                out[(key, comment)] = color
            continue
        if isinstance(key, tuple):
            status, comment = key
            out[(status, comment if comment not in ("", None) else None)] = value
            continue
        if ':' in key:
            status, comment = key.split(':', 1)
            comment = comment.strip() or None
        else:
            status, comment = key, None
        out[(status.strip(), comment)] = value
    return out


def lookup_color(colors, status, comment):
    """Return color for ``(status, comment)``, falling back to ``(status, None)``."""
    if (status, comment) in colors:
        return colors[(status, comment)]
    if (status, None) in colors:
        return colors[(status, None)]
    return None


# ---------------------------------------------------------------- plotting
def _ordered(items, order):
    """Return ``items`` reordered by ``order`` (unknown items appended at the end)."""
    if not order:
        return list(items)
    items = list(items)
    head = [x for x in order if x in items]
    tail = [x for x in items if x not in order]
    return head + tail


def _orders_from_colors(colors):
    """Derive default ``(status_order, {status: [comments...]})`` from a normalized
    color dict (iteration order of its keys)."""
    status_order = []
    comments_by_status = OrderedDict()
    for (status, comment) in colors:
        if status not in status_order:
            status_order.append(status)
        if comment is not None:
            comments_by_status.setdefault(status, []).append(comment)
    return status_order, comments_by_status


def _is_color_split(value):
    """Return True if value is a sequence of >=2 color strings (not an RGB tuple)."""
    return (isinstance(value, (tuple, list)) and len(value) >= 2
            and all(isinstance(v, str) for v in value))


# Backwards-compat alias.
_is_color_pair = _is_color_split


def plot(task_counts, colors, status_order=None, comment_order=None,
         connector_colors=None, comment_classes=None,
         comment_label_aliases=None, pct_threshold=5,
         out_path=None, show_pct=True):
    tasks = list(task_counts.keys())
    n = max(len(tasks), 1)
    connector_colors = connector_colors or {}
    comment_classes = comment_classes or {}
    aliases = COMMENT_LABEL_ALIASES if comment_label_aliases is None else comment_label_aliases

    # ``colors`` may be a single dict (shared across rows) or a list of dicts —
    # one per row, allowing different palettes per row.
    if isinstance(colors, list):
        per_row_colors = list(colors)
        if len(per_row_colors) < n:
            per_row_colors += [per_row_colors[-1]] * (n - len(per_row_colors))
        primary_colors = per_row_colors[0] if per_row_colors else {}
    else:
        primary_colors = colors or {}
        per_row_colors = [primary_colors] * n

    # Default ordering inherited from the (primary) colors dict iteration order.
    color_status_order, comments_by_status = _orders_from_colors(primary_colors)

    # Global status order across all tasks.
    seen_statuses = []
    for counts in task_counts.values():
        for s in counts:
            if s not in seen_statuses:
                seen_statuses.append(s)
    statuses = _ordered(seen_statuses, status_order or color_status_order)

    bar_height = 0.5
    bar_half = bar_height / 2  # vertical reach of bars around their row
    y_status_text = -0.65  # status label baseline
    y_comment_text = -0.45  # comment label baseline
    y_fill_top = -0.8  # status fill top — above status label (acts as background)
    y_separator_top = -0.35  # sub-bar separator lines top — above comment label
    y_separator_div = 0.175
    border_color = 'white'
    border_lw = 1

    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']

    fig, ax = plt.subplots(figsize=(8, 0.45 * n + 1.2))

    ok_color = lookup_color(primary_colors, 'OK', None)

    # ---- status-level row boundaries (cumulative widths) ---------------------
    row_boundaries = []
    for task in tasks:
        counts = task_counts[task]
        bounds = [0.0]
        for status in statuses:
            bounds.append(bounds[-1] + sum(counts.get(status, {}).values()))
        row_boundaries.append(bounds)

    # ---- header slot widths = first row's status widths ---------------------
    header_slot_boundaries = list(row_boundaries[0])
    row0_status_widths = [
        row_boundaries[0][i + 1] - row_boundaries[0][i]
        for i in range(len(statuses))
    ]

    # ---- per-status union of comments seen across rows ----------------------
    # all_comments includes None / MISSING_STATUS so the bar layout is correct;
    # labeled_comments drops them so the header has no empty / "—" slots.
    all_comments = OrderedDict()
    for status in statuses:
        if comment_order is None:
            co = comments_by_status.get(status)
        elif isinstance(comment_order, dict):
            co = comment_order.get(status)
        else:
            co = comment_order
        seen = []
        for counts in task_counts.values():
            sub = counts.get(status, {})
            for c in _ordered(sub.keys(), co):
                if c not in seen and sub[c] > 0:
                    seen.append(c)
        if MISSING_STATUS in seen:
            seen.remove(MISSING_STATUS)
            seen.insert(0, MISSING_STATUS)
        all_comments[status] = seen
    labeled_comments = OrderedDict(
        (status, [c for c in cmts if c is not None and c != MISSING_STATUS])
        for status, cmts in all_comments.items()
    )

    # ---- per-row sub-bar extents (always recorded, even when degenerate) ----
    row_sub_extents = []
    for r, task in enumerate(tasks):
        counts = task_counts[task]
        extents = {}
        offset = 0.0
        for status in statuses:
            sub = counts.get(status, {})
            for c in all_comments[status]:
                cnt = sub.get(c, 0)
                extents[(status, c)] = (offset, offset + cnt)
                offset += cnt
        row_sub_extents.append(extents)

    # ---- header comment slots: equally spaced for the labeled comments only -
    comment_slot = {}  # (status, comment) -> (x_left, x_right) in header
    for i, status in enumerate(statuses):
        cmts = labeled_comments[status]
        if not cmts:
            continue
        slot_left = header_slot_boundaries[i]
        slot_w = row0_status_widths[i]
        sub_w = slot_w / len(cmts)
        for k, c in enumerate(cmts):
            comment_slot[(status, c)] = (slot_left + k * sub_w,
                                         slot_left + (k + 1) * sub_w)

    # Per-row cumulative class-boundary offsets within each status. Used when
    # ``connector_colors[status]`` is a tuple of K colors, to split the
    # trapezoid fill at the K-1 class boundaries.
    def _class_split_offsets(row_idx, status, n_classes):
        """Return K-1 cumulative offsets (in count units) at the class
        boundaries, measured from the status' left edge in the row's bar."""
        row_label = tasks[row_idx]
        classes = comment_classes[row_idx] if isinstance(comment_classes, list) else {}
        sub = task_counts[row_label].get(status, {})
        widths = [0.0] * n_classes
        for c in all_comments[status]:
            cnt = sub.get(c, 0)
            cls = classes.get((status, c), 0)
            if 0 <= cls < n_classes:
                widths[cls] += cnt
        offsets = []
        acc = 0.0
        for k in range(n_classes - 1):
            acc += widths[k]
            offsets.append(acc)
        return offsets

    def _add_trapezoid(x_top_l, x_top_r, x_bot_l, x_bot_r, y_top, y_bot, fill,
                       split_top=None, split_bot=None):
        """Add either a single-fill trapezoid or a 2-class split trapezoid."""
        if _is_color_pair(fill) and split_top is not None and split_bot is not None:
            fill0, fill1 = fill
            ax.add_patch(plt.Polygon([
                (x_top_l, y_top), (split_top, y_top),
                (split_bot, y_bot), (x_bot_l, y_bot),
            ], facecolor=fill0, edgecolor='none', zorder=0))
            ax.add_patch(plt.Polygon([
                (split_top, y_top), (x_top_r, y_top),
                (x_bot_r, y_bot), (split_bot, y_bot),
            ], facecolor=fill1, edgecolor='none', zorder=0))
        else:
            color = fill[0] if _is_color_pair(fill) else fill
            ax.add_patch(plt.Polygon([
                (x_top_l, y_top), (x_top_r, y_top),
                (x_bot_r, y_bot), (x_bot_l, y_bot),
            ], facecolor=color, edgecolor='none', zorder=0))

    def _split_boundaries(row_idx, status, n_classes):
        """Return list of K-1 ``(slot_r, l_r)`` tuples — header & row-level x
        positions of each class boundary. Assumes class indices are
        non-decreasing through ``labeled_comments[status]``; returns ``None``
        if that assumption is violated."""
        classes = comment_classes[row_idx] if isinstance(comment_classes, list) else {}
        prev_cls = -1
        for c in labeled_comments[status]:
            cls = classes.get((status, c), 0)
            if cls < prev_cls:
                return None
            prev_cls = cls
        boundaries = []
        i_status = statuses.index(status)
        for k in range(n_classes - 1):
            last_c = None
            for c in labeled_comments[status]:
                if classes.get((status, c), 0) <= k:
                    last_c = c
            if last_c is None:
                # All comments are class > k; boundary sits at the status' left edge.
                boundaries.append((header_slot_boundaries[i_status],
                                   row_boundaries[row_idx][i_status]))
            else:
                slot_r = comment_slot[(status, last_c)][1]
                l_r = row_sub_extents[row_idx][(status, last_c)][1]
                boundaries.append((slot_r, l_r))
        return boundaries

    # ---- status trapezoid fills (header -> row 0) ---------------------------
    # Fill extends above the status label so the colour functions as a backdrop.
    # When split, each class boundary follows row-0's comment divider shape:
    # vertical from y_fill_top down to y_separator_top, then diagonal to l_r
    # at -bar_half. K colors → K polygons separated by K-1 dividers.
    for i, status in enumerate(statuses):
        fill = connector_colors.get(status)
        if fill is None:
            continue
        if _is_color_split(fill):
            n_classes = len(fill)
            boundaries = _split_boundaries(0, status, n_classes)
            if boundaries is not None:
                # Boundary paths (top -> bottom). Edges are straight lines;
                # divider boundaries follow the comment-divider shape.
                paths = [[(header_slot_boundaries[i], y_fill_top),
                          (row_boundaries[0][i], -bar_half)]]
                for slot_x, l_x in boundaries:
                    paths.append([(slot_x, y_fill_top),
                                  (slot_x, y_separator_top),
                                  (l_x, -bar_half)])
                paths.append([(header_slot_boundaries[i + 1], y_fill_top),
                              (row_boundaries[0][i + 1], -bar_half)])
                for k in range(n_classes):
                    verts = list(paths[k]) + list(reversed(paths[k + 1]))
                    ax.add_patch(plt.Polygon(verts, facecolor=fill[k],
                                             edgecolor='none', zorder=0))
                continue
        _add_trapezoid(
            header_slot_boundaries[i], header_slot_boundaries[i + 1],
            row_boundaries[0][i], row_boundaries[0][i + 1],
            y_fill_top, -bar_half, fill,
        )

    # ---- status trapezoid fills (row r -> row r+1) --------------------------
    for r in range(n - 1):
        upper = row_boundaries[r]
        lower = row_boundaries[r + 1]
        y_top = r + bar_half
        y_bot = (r + 1) - bar_half
        for i, status in enumerate(statuses):
            fill = connector_colors.get(status)
            if fill is None:
                continue
            if _is_color_split(fill):
                n_classes = len(fill)
                offs_u = _class_split_offsets(r, status, n_classes)
                offs_l = _class_split_offsets(r + 1, status, n_classes)
                paths = [[(upper[i], y_top), (lower[i], y_bot)]]
                for off_u, off_l in zip(offs_u, offs_l):
                    paths.append([(upper[i] + off_u, y_top),
                                  (lower[i] + off_l, y_bot)])
                paths.append([(upper[i + 1], y_top), (lower[i + 1], y_bot)])
                for k in range(n_classes):
                    verts = list(paths[k]) + list(reversed(paths[k + 1]))
                    ax.add_patch(plt.Polygon(verts, facecolor=fill[k],
                                             edgecolor='none', zorder=0))
                continue
            _add_trapezoid(
                upper[i], upper[i + 1],
                lower[i], lower[i + 1],
                y_top, y_bot, fill,
            )

    # ---- bars + count / percentage labels ------------------------------------
    for row, task in enumerate(tasks):
        counts = task_counts[task]
        row_total = row_boundaries[row][-1]
        offset = 0
        for status in statuses:
            sub = counts.get(status, {})
            for comment in all_comments[status]:
                cnt = sub.get(comment, 0)
                if cnt == 0:
                    continue
                missing = (comment == MISSING_STATUS)
                row_colors = per_row_colors[row]
                row_ok_color = lookup_color(row_colors, 'OK', None) if row_colors is not primary_colors else ok_color
                color = row_ok_color if missing else lookup_color(row_colors, status, comment)
                ax.barh(row, cnt, left=offset, color=color,
                        edgecolor=border_color, linewidth=border_lw,
                        height=bar_height, zorder=1)
                if missing:
                    ax.barh(row, cnt, left=offset, facecolor='none',
                            edgecolor=LIGHT_GREEN, linewidth=0, hatch='//',
                            height=bar_height, zorder=3)
                else:
                    cx = offset + cnt / 2
                    ax.text(cx, row - 0.05, str(cnt),
                            ha='center', va='center', fontsize=7,
                            color='black', fontweight='bold', zorder=4)
                    # ax.text(cx, row + 0, str(cnt),
                    #         ha='center', va='center', fontsize=7,
                    #         color='black', fontweight='bold', zorder=4)
                    if cnt >= pct_threshold and row_total:
                        pct = 100.0 * cnt / row_total
                        ax.text(cx, row + 0.10, f"{pct:.1f}%",
                                ha='center', va='center', fontsize=7,
                                color='black', zorder=4)
                        # ax.text(cx, row + 0.25, f"{pct:.1f}%",
                        #         ha='center', va='top', fontsize=7,
                        #         color='black', zorder=4)
                offset += cnt

    # ---- header status labels (centred on each row-0-width slot) ------------
    for i, status in enumerate(statuses):
        cx = (header_slot_boundaries[i] + header_slot_boundaries[i + 1]) / 2
        ax.text(cx, y_status_text, str(status),
                ha='center', va='center', fontsize=8, fontweight='bold',
                zorder=4)

    # ---- header comment labels (only for labeled comments) ------------------
    for (status, comment), (l, r) in comment_slot.items():
        text = aliases.get((status, comment), aliases.get(comment, str(comment)))
        ax.text((l + r) / 2, y_comment_text, text,
                ha='center', va='center', fontsize=5.5,  # rotation=15,
                color='black', zorder=4)

    # ---- white border lines: status boundaries, header -> row 0 -------------
    # Span the full fill height so the status colours are cleanly separated.
    for x_top, x_bot in zip(header_slot_boundaries, row_boundaries[0]):
        ax.plot([x_top, x_bot], [y_fill_top, -bar_half],
                color=border_color, linewidth=border_lw, zorder=2)

    # ---- header -> row 0: sub-bar separators --------------------------------
    # Tops sit just above the comment labels so labels stay visually separated
    # from the status label zone above. Skip when the labeled comment group is
    # empty in row 0.
    for i, status in enumerate(statuses):
        cmts = labeled_comments[status]
        if not cmts:
            continue
        row0_has = any(
            row_sub_extents[0][(status, c)][0] != row_sub_extents[0][(status, c)][1]
            for c in cmts
        )
        if not row0_has:
            continue
        for comment in cmts:
            slot_r = comment_slot[(status, comment)][1]
            l_r = row_sub_extents[0][(status, comment)][1]
            ax.plot([slot_r, slot_r, l_r], [y_separator_top - y_separator_div, y_separator_top, -bar_half],
                    color=border_color, linewidth=border_lw, zorder=2)

    # ---- white border lines: status boundaries, row r -> row r+1 ------------
    for r in range(n - 1):
        upper = row_boundaries[r]
        lower = row_boundaries[r + 1]
        for xu, xl in zip(upper, lower):
            ax.plot([xu, xl],
                    [r + bar_half, (r + 1) - bar_half],
                    color=border_color, linewidth=border_lw, zorder=2)

    # ---- row r -> row r+1: sub-bar separators -------------------------------
    # Skip when one of the two rows has all labeled comments collapsed to zero
    # (the trapezoid stays uniformly the status colour); otherwise just draw
    # the connector lines, no triangle fill.
    for r in range(n - 1):
        upper_e = row_sub_extents[r]
        lower_e = row_sub_extents[r + 1]
        for status in statuses:
            cmts = labeled_comments[status]
            if not cmts:
                continue
            upper_has = any(upper_e[(status, c)][0] != upper_e[(status, c)][1]
                            for c in cmts)
            lower_has = any(lower_e[(status, c)][0] != lower_e[(status, c)][1]
                            for c in cmts)
            if not (upper_has and lower_has):
                continue
            for comment in cmts:
                u_r = upper_e[(status, comment)][1]
                l_r = lower_e[(status, comment)][1]
                ax.plot([u_r, l_r],
                        [r + bar_half, (r + 1) - bar_half],
                        color=border_color, linewidth=border_lw, zorder=2)

    # ---- axes ---------------------------------------------------------------
    ax.set_yticks(range(n))
    ax.set_yticklabels(tasks, fontsize=7, rotation=90, va='center', ha='right',
                       multialignment='center')
    ax.set_ylim(n - 0.55, -1.05)  # reversed bounds; room for fill above status label
    ax.set_xticks([])
    ax.tick_params(axis='y', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove all margins around the axes/image.
    ax.margins(x=0, y=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0)
        print(f"Saved figure to {out_path}")
    else:
        plt.show()


# ---------------------------------------------------------------- top-level API
def status_bars(tasks, colors=None, connector_colors=None, comment_classes=None,
                ids=None, attempt=0,
                base_dir=None, status_order=None, comment_order=None,
                comment_label_aliases=None, pct_threshold=5,
                out_path=None, show_pct=True, task_labels=None):
    """Render the horizontal stacked status bar plot.

    Parameters
    ----------
    tasks : iterable[str] | dict[str, str]
        Task keys (one bar per task, in iteration order). Pass a dict to map
        each task key to a custom row label, e.g.
        ``{'PROOF_GEN': 'Proof from GT', 'PROOF_GEN_GEN_GUIDED': 'Guided'}``.
        Tasks not present in ``TASKS`` fall back to a JSON file at
        ``<base_dir>/clever/<task>.json`` (see :func:`load_external_pairs`).
    colors : dict
        Color spec — see :func:`normalize_colors`. Status-only keys colour all
        sub-bars under that status; ``(status, comment)`` keys override.
    ids : iterable[int] | None
        Restrict to these problem IDs (``None`` = all on disk; ignored for
        external JSON tasks).
    attempt : int
        Attempt index (default 0).
    base_dir : Path | None
        Root data directory. Defaults to ``data/claude``.
    status_order, comment_order : iterable[str] | None
        Left-to-right ordering. Anything not listed is appended in
        discovery order.
    out_path : Path | str | None
        File to save to; ``None`` shows interactively.
    show_pct : bool
        Append percentages to status labels.
    task_labels : dict[str, str] | None
        Alternative way to provide row labels (merged with the dict form of
        ``tasks``; ``tasks`` wins on conflict).
    """
    base_dir = Path(base_dir) if base_dir is not None else var.DATA_DIRECTORY / "claude"
    if isinstance(colors, list):
        colors = [normalize_colors(c) for c in colors]
    else:
        colors = normalize_colors(colors)

    if isinstance(tasks, dict):
        items = list(tasks.items())
    else:
        labels = task_labels or {}
        items = [(t, labels.get(t.upper().replace('-', '_'), t)) for t in tasks]

    task_counts = OrderedDict()
    for task, label in items:
        key = task.upper().replace('-', '_')
        if key in TASKS:
            dataset = TASKS[key]['dataset']
            problems = load_task_problems(
                task=key, dataset=dataset, problem_ids=ids,
                attempt=attempt, base_dir=base_dir,
            )
            pairs = [get_status_comment(p) if p is not None else (None, None)
                     for p in problems]
        else:
            pairs = load_external_pairs(key, base_dir)
        task_counts[label] = aggregate(pairs)

    plot(task_counts, colors,
         status_order=status_order,
         comment_order=comment_order,
         connector_colors=connector_colors,
         comment_classes=comment_classes,
         comment_label_aliases=comment_label_aliases,
         pct_threshold=pct_threshold,
         out_path=out_path,
         show_pct=show_pct)


# GREEN = "#B6D7A8"
# YELLOW = "#FFE598"
# ORANGE = "#F9CB9C"
# RED = "#EB9899"
# BLUE = "#A4C2F4"
# PURPLE = "#D5A6BD"
GREEN = "#6AA84E"
YELLOW = "#F1C231"
ORANGE = "#E69038"
RED = "#CC0000"
BLUE = "#3B77D8"
PURPLE = "#A64C79"
MIDDLE = "#00A2FF"  # (0, 162, 255)

SPEC_ISO_COLORS = {
    'OK': GREEN,
    'MISMATCH': YELLOW,
    ('MISMATCH', '→'): YELLOW,
    ('MISMATCH', '/'): YELLOW,
    ('MISMATCH', '←'): YELLOW,
    'ISSUE': BLUE,
    ('ISSUE', 'PROB'): BLUE,
    ('ISSUE', 'BOTH'): PURPLE,
    ('ISSUE', 'GEN'): ORANGE,  # PURPLE,
    'FAIL': RED,
    'TIMEOUT': RED,
}

SPEC_ISO_COMMENT_LABELS = {
    ('MISMATCH', '→'): 'Stronger',
    ('MISMATCH', '/'): 'Neither',
    ('MISMATCH', '←'): 'Weaker',
    ('ISSUE', 'PROB'): 'Ground-truth Specification',
    ('ISSUE', 'BOTH'): 'Both',
    ('ISSUE', 'GEN'): 'Generated Specification',
}

PROOF_GEN_GT_COLORS = {
    'OK': GREEN,
    'MISMATCH': YELLOW,
    'ISSUE': BLUE,
    ('ISSUE', 'BASE'): BLUE,
    ('ISSUE', 'SPEC'): BLUE,
    ('ISSUE', 'BOTH'): PURPLE,
    ('ISSUE', 'IMPL'): ORANGE,  # PURPLE,
    'FAIL': RED,
    'TIMEOUT': RED,
}
PROOF_GEN_GEN_COLORS = {
    'OK': GREEN,
    'MISMATCH': YELLOW,
    'ISSUE': BLUE,
    ('ISSUE', 'BASE'): BLUE,
    ('ISSUE', 'SPEC'): ORANGE,  # PURPLE,
    ('ISSUE', 'BOTH'): ORANGE,  # PURPLE,
    ('ISSUE', 'IMPL'): ORANGE,  # PURPLE,
    'FAIL': RED,
    'TIMEOUT': RED,
}

PROOF_GEN_COMMENT_LABELS = {
    ('ISSUE', 'BASE'): 'Benchmark',
    ('ISSUE', 'SPEC'): 'Specification',
    ('ISSUE', 'BOTH'): 'Spec. & Impl.',
    ('ISSUE', 'IMPL'): 'Implement.',
}

LIGHT_GREEN = "#D9EBD3"
LIGHT_YELLOW = "#FFF3CC"
LIGHT_ORANGE = "#FDE5CD"
LIGHT_RED = "#F4CCCC"
LIGHT_BLUE = "#C9DBF8"
LIGHT_PURPLE = "#EBD1DC"
LIGHT_MIDDLE = "#A7E5FF"  # (167, 229, 255)

CONNECTOR_COLORS = {
    'OK': LIGHT_GREEN,
    'MISMATCH': LIGHT_YELLOW,
    'ISSUE': (LIGHT_BLUE, LIGHT_PURPLE, LIGHT_ORANGE),
    # 'ISSUE': (LIGHT_BLUE, LIGHT_PURPLE),
    'FAIL': LIGHT_RED,
}

if __name__ == '__main__':
    status_bars(
        tasks={'SPEC_ISO': "Simple", 'SPEC_ISO_GUIDED': "Guided", 'SPEC_ISO_GUIDED_MULT': "Guided\nMulti-Spec",
               'SPEC_ISO_MANUAL': "Final\n(Manual)"},
        colors=SPEC_ISO_COLORS,
        connector_colors=CONNECTOR_COLORS,
        comment_classes=[{('ISSUE', 'BOTH'): 1, ('ISSUE', 'GEN'): 2}, ] * 4,
        comment_label_aliases=SPEC_ISO_COMMENT_LABELS,
        out_path='out/spec_iso.png',
    )
    for model in ("codex", "kimi"):
        status_bars(
            tasks={'SPEC_ISO': "Simple", 'SPEC_ISO_GUIDED': "Guided"},
            colors=SPEC_ISO_COLORS,
            connector_colors=CONNECTOR_COLORS,
            comment_classes=[{('ISSUE', 'BOTH'): 1, ('ISSUE', 'GEN'): 2}] * 2,
            comment_label_aliases=SPEC_ISO_COMMENT_LABELS,
            base_dir=var.DATA_DIRECTORY / model,
            out_path=f'out/spec_iso_{model}.png',
        )
    status_bars(
        tasks={'PROOF_GEN': "Ground-truth", 'PROOF_GEN_GEN': "Generated\n(Simple)",
               'PROOF_GEN_GEN_GUIDED': "Generated\n(Guided)"},
        # colors=PROOF_GEN_GEN_COLORS,
        colors=[PROOF_GEN_GT_COLORS, ] + [PROOF_GEN_GEN_COLORS] * 2,
        connector_colors=CONNECTOR_COLORS,
        comment_classes=[
            {('ISSUE', 'BOTH'): 1, ('ISSUE', 'IMPL'): 2},
            {('ISSUE', 'SPEC'): 2, ('ISSUE', 'BOTH'): 2, ('ISSUE', 'IMPL'): 2},
            {('ISSUE', 'SPEC'): 2, ('ISSUE', 'BOTH'): 2, ('ISSUE', 'IMPL'): 2},
        ],
        comment_label_aliases=PROOF_GEN_COMMENT_LABELS,
        out_path='out/proof_gen.png',
    )
    for model in ("codex", "kimi"):
        status_bars(
            tasks={'PROOF_GEN': "Ground-truth", 'PROOF_GEN_GEN': "Generated\n(Simple)",
                   'PROOF_GEN_GEN_GUIDED': "Generated\n(Guided)"},
            colors=[PROOF_GEN_GT_COLORS] + [PROOF_GEN_GEN_COLORS] * 2,
            connector_colors=CONNECTOR_COLORS,
            comment_classes=[
                {('ISSUE', 'BOTH'): 1, ('ISSUE', 'IMPL'): 2},
                {('ISSUE', 'SPEC'): 2, ('ISSUE', 'BOTH'): 2, ('ISSUE', 'IMPL'): 2},
                {('ISSUE', 'SPEC'): 2, ('ISSUE', 'BOTH'): 2, ('ISSUE', 'IMPL'): 2},
            ],
            comment_label_aliases=PROOF_GEN_COMMENT_LABELS,
            base_dir=var.DATA_DIRECTORY / model,
            out_path=f'out/proof_gen_{model}.png',
        )
