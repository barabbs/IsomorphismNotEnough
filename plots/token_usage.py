"""Per-run token usage across agent stacks, and a scatter grid to plot it.

Token counts come from each run's final ``ResultMessage``, i.e. the task
session only (the harness' follow-up status turn is excluded for every stack):

* Claude Code: the per-model totals in ``model_usage`` (these include
  subagents and match ``total_cost_usd``); older records without
  ``model_usage`` fall back to ``usage``, which counts the main agent only.
  Such a record is marked incomplete when its trace shows subagent activity.
* Codex CLI: ``usage.total`` (cumulative over the session).
* OpenHands: ``usage.token_usage`` (cumulative over the session).

Metrics, identical in meaning across stacks:

* ``input``: all prompt tokens processed, cached or not;
* ``cached``: the cached part of ``input`` (OpenHands reports 0 for Kimi);
* ``output``: generated tokens, reasoning included;
* ``total``: ``input + output``;
* ``cost`` (USD, Claude only) and ``time`` (wall clock, s).
"""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

sys.path.insert(0, str(Path(__file__).parent))
from status_bars import normalize_colors, lookup_color, resolve_list
from status_grid import _problem_pairs, _normalize_tasks, _broadcast
from scatter_time_price import _draw_avg_indicators, _iter_task_problems

from core import var


# ---------------------------------------------------------------- extraction
def _result_message(problem):
    """Return the run's last ``ResultMessage`` and its trace, or ``(None, None)``."""
    for trace in problem.metadata.values():
        if not isinstance(trace, list):
            continue
        rms = [t for t in trace if isinstance(t, dict) and t.get('type') == 'ResultMessage']
        if rms:
            return rms[-1], trace
    return None, None


def run_usage(problem):
    """Return ``{input, cached, output, total, cost, time, complete}`` for one
    run, or ``None`` if the record carries no usage."""
    rm, trace = _result_message(problem)
    if rm is None:
        return None
    usage = rm.get('usage') or {}
    complete = True
    if 'total' in usage:  # Codex CLI
        tot = usage['total']
        inp, cached, out = tot['input_tokens'], tot.get('cached_input_tokens', 0), tot['output_tokens']
    elif 'token_usage' in usage:  # OpenHands
        tu = usage['token_usage']
        inp, cached, out = tu['prompt_tokens'], tu.get('cache_read_tokens', 0), tu['completion_tokens']
    elif rm.get('model_usage'):  # Claude Code, all models incl. subagents
        mu = rm['model_usage'].values()
        cached = sum(m['cacheReadInputTokens'] for m in mu)
        inp = cached + sum(m['inputTokens'] + m['cacheCreationInputTokens'] for m in mu)
        out = sum(m['outputTokens'] for m in mu)
    elif 'output_tokens' in usage:  # Claude Code, main agent only
        cached = usage.get('cache_read_input_tokens', 0)
        inp = cached + usage.get('input_tokens', 0) + usage.get('cache_creation_input_tokens', 0)
        out = usage['output_tokens']
        complete = not any(isinstance(t, dict) and t.get('parent_tool_use_id') for t in trace)
    else:
        return None
    cost = rm.get('total_cost_usd')
    return {
        'input': float(inp), 'cached': float(cached), 'output': float(out),
        'total': float(inp + out),
        'cost': float(cost) if cost else None,
        'time': None if problem.timing is None else float(problem.timing),
        'complete': complete,
    }


RECORD_RE = re.compile(r"human_eval-(\d+)-(\d+)\.json")


def load_record(path):
    """Load one run record as a light object with ``metadata``, ``timing``
    and ``exception``, enough for :func:`run_usage` and :func:`run_status`."""
    with open(path) as f:
        d = json.load(f)
    return SimpleNamespace(metadata=d.get('metadata') or {}, timing=d.get('timing'),
                           exception=d.get('exception'))


def iter_records(base_dir, task, attempt=0):
    """Yield ``(problem_id, record)`` for every ``<base_dir>/clever/<task>``
    record of the given attempt, in problem order."""
    for path in sorted((Path(base_dir) / 'clever' / task).glob(f'human_eval-*-{attempt}.json')):
        m = RECORD_RE.fullmatch(path.name)
        if m:
            yield int(m.group(1)), load_record(path)


def run_status(record):
    """Return the run's ``(status, comment)``, collapsing list-valued statuses
    (multi-spec runs) with :func:`status_bars.resolve_list`."""
    pairs = _problem_pairs(record) or [(None, None)]
    if len(pairs) > 1:
        return resolve_list([p[0] for p in pairs], [p[1] for p in pairs])
    return pairs[0]


def _fmt_tokens(v):
    """Compact token count: ``950``, ``12k``, ``3.4M``."""
    if v >= 1e6:
        return f'{v / 1e6:.1f}M'
    if v >= 1e3:
        return f'{v / 1e3:.0f}k'
    return f'{v:.0f}'


# name -> (axis unit, average-value formatter)
METRICS = {
    'input': ('input tokens', _fmt_tokens),
    'cached': ('cached tokens', _fmt_tokens),
    'output': ('output tokens', _fmt_tokens),
    'total': ('tokens', _fmt_tokens),
    'cost': ('USD', lambda v: f'${v:.2f}'),
    'time': ('s', lambda v: f'{v:.0f}s'),
}


# ---------------------------------------------------------------- plotting
def scatter_grid(rows, series, *, x_metric='total', y_metric='cost',
                 color_by='status', ids=None, attempt=0, out_path=None,
                 subplot_size=(2.4, 2.4), wspace=0.0, hspace=0.3,
                 marker_size=12, edge_color='black', edge_lw=0.3, alpha=0.85,
                 tick_fontsize=5, title_fontsize=8, row_label_fontsize=8,
                 unit_fontsize=7, legend_fontsize=6, title_rows=(0,),
                 avg_labels=True):
    """Scatter one metric against another, one subplot per task.

    Parameters
    ----------
    rows : list[dict]
        One entry per row: ``name`` (row title), ``tasks`` (task key ->
        column title), ``colors`` (status colour spec, shared or per task,
        used when ``color_by='status'``), optional ``resolves`` (per-task
        callable collapsing list-valued statuses).
    series : list[dict]
        One entry per stack: ``label``, ``base_dir``, ``marker`` and, for
        ``color_by='series'``, ``color``.
    x_metric, y_metric : str
        Keys of :data:`METRICS`. Both axes are logarithmic.
    color_by : {'status', 'series'}
        Colour points by run status, or by stack (with one mean indicator
        per stack instead of one overall).
    Runs whose token totals are known to be incomplete are drawn hollow.
    """
    n_rows = len(rows)
    n_cols = max(len(r['tasks']) for r in rows)
    fig, axes = plt.subplots(
        n_rows, n_cols, squeeze=False,
        figsize=(n_cols * subplot_size[0], n_rows * subplot_size[1]),
        sharex='row', sharey='row',
        gridspec_kw={'wspace': wspace, 'hspace': hspace},
    )
    plain = FuncFormatter(lambda v, _: f'{v:g}' if v < 1e3 else _fmt_tokens(v))
    x_unit, x_fmt = METRICS[x_metric]
    y_unit, y_fmt = METRICS[y_metric]

    panel_points = {}
    print(f"{'row':<10}  {'column':<30}  {'stack':<16}  {'n':>4}  "
          f"{'mean ' + x_metric:>14}  {'mean ' + y_metric:>14}  incomplete")
    for ri, row in enumerate(rows):
        task_items = _normalize_tasks(row['tasks'])
        colors = _broadcast(row.get('colors'), len(task_items), default={},
                            normalize=normalize_colors)
        resolves = _broadcast(row.get('resolves'), len(task_items), default=None)
        for ti, (task, label) in enumerate(task_items):
            ax = axes[ri, ti]
            ax.set_xscale('log')
            ax.set_yscale('log')
            key = task.upper().replace('-', '_')
            points = []  # (series index, x, y, colour, complete)
            for si, s in enumerate(series):
                for problem in _iter_task_problems(key, ids, attempt, Path(s['base_dir'])):
                    u = run_usage(problem)
                    if u is None or not u[x_metric] or not u[y_metric]:
                        continue
                    if color_by == 'series':
                        color = s['color']
                    else:
                        pairs = _problem_pairs(problem) or [(None, None)]
                        if resolves[ti] is not None and len(pairs) > 1:
                            pairs = [resolves[ti]([p[0] for p in pairs], [p[1] for p in pairs])]
                        color = lookup_color(colors[ti], *pairs[0]) or 'lightgray'
                    points.append((si, u[x_metric], u[y_metric], color, u['complete']))
            panel_points[(ri, ti)] = points

            for si, s in enumerate(series):
                pts = [p for p in points if p[0] == si]
                for complete in (True, False):
                    sel = [p for p in pts if p[4] == complete]
                    if not sel:
                        continue
                    cols = [p[3] for p in sel]
                    ax.scatter([p[1] for p in sel], [p[2] for p in sel],
                               marker=s['marker'], s=marker_size,
                               facecolor=cols if complete else 'none',
                               edgecolor=edge_color if complete else cols,
                               linewidth=edge_lw if complete else 0.6,
                               alpha=alpha, zorder=3)
                xs, ys = [p[1] for p in pts], [p[2] for p in pts]
                n_inc = sum(not p[4] for p in pts)
                print(f"{row['name']:<10}  {label.replace(chr(10), ' ')[:30]:<30}  "
                      f"{s['label']:<16}  {len(pts):>4}  "
                      f"{np.mean(xs) if xs else float('nan'):>14.1f}  "
                      f"{np.mean(ys) if ys else float('nan'):>14.4f}  {n_inc}")

            ax.grid(True, which='both', alpha=0.25, linewidth=0.5)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.xaxis.set_major_formatter(plain)
            ax.yaxis.set_major_formatter(plain)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            if ri in title_rows:
                ax.set_title(label, fontsize=title_fontsize)

        for ti in range(len(task_items), n_cols):
            axes[ri, ti].set_visible(False)
        axes[ri, 0].set_ylabel(row['name'], fontsize=row_label_fontsize)
        axes[ri, 0].text(0.0, 1.01, y_unit, transform=axes[ri, 0].transAxes,
                         ha='center', va='bottom', fontsize=unit_fontsize)
        last = axes[ri, len(task_items) - 1]
        last.text(1.01, 0.0, x_unit, transform=last.transAxes,
                  ha='left', va='center', fontsize=unit_fontsize)

    # Mean indicators need the final (shared) axis limits, so draw them last.
    no_label = lambda v: ''
    for (ri, ti), points in panel_points.items():
        ax = axes[ri, ti]
        if color_by == 'series':
            groups = [([p for p in points if p[0] == si], s['color'])
                      for si, s in enumerate(series)]
        else:
            groups = [(points, 'black')]
        for pts, color in groups:
            if not pts:
                continue
            xs, ys = [p[1] for p in pts], [p[2] for p in pts]
            _draw_avg_indicators(ax, np.mean(xs), np.mean(ys), xs, ys,
                                 x_fmt=x_fmt if avg_labels else no_label,
                                 y_fmt=y_fmt if avg_labels else no_label,
                                 color=color)

    if len(series) > 1 or color_by == 'series':
        handles = [plt.Line2D([], [], marker=s['marker'], linestyle='',
                              markerfacecolor=s.get('color', 'none') if color_by == 'series' else 'none',
                              markeredgecolor='black', markeredgewidth=0.4,
                              markersize=5, label=s['label']) for s in series]
        fig.legend(handles=handles, loc='lower center', ncol=len(handles),
                   fontsize=legend_fontsize, frameon=False,
                   bbox_to_anchor=(0.5, axes[0, 0].get_position().y1 + 0.02))

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f"Saved figure to {out_path}")
    else:
        plt.show()
