"""Token usage per pipeline run and stack, as mirrored bars.

One group per run (Simple, Guided, Multi-spec; implementation from
ground-truth, simple-generated and guided-generated specs), one bar per
stack. Each run sums its generation and proving stages per problem; its
status is that of the proving stage. Output tokens grow upwards from the
x axis, input tokens downwards, each on its own log scale. Bars show the
median, whiskers the 25th-75th percentiles, and coloured dots the median of
each status class (dot area grows with the class size). The hatched bottom
part of a bar is the generation stage's share of all tokens in that run, as a
share of the bar's length. Problems with a
missing stage or with incomplete token counts are left out.

Also writes the per-problem data (``*_runs.csv``) and summary statistics
per run, stack and status class (``*_summary.csv``).
"""

import csv
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).parent))
from status_bars import (
    SPEC_ISO_COLORS, PROOF_GEN_GT_COLORS, PROOF_GEN_GEN_COLORS,
    normalize_colors, lookup_color,
    GREEN, YELLOW, BLUE, PURPLE, ORANGE, RED,
)
from token_usage import iter_records, run_status, run_usage, _fmt_tokens

from core import var

# Grey shades keep the stacks apart without clashing with the status colours.
STACKS = [
    ('claude', "Claude Code + Opus 4.6", '#D9D9D9'),
    ('codex', "Codex CLI + GPT-5.4", '#ABABAB'),
    ('kimi', "OpenHands + Kimi K2.5", '#7D7D7D'),
]

# (label, part, stages summed per problem, status colours, stacks that ran it)
GROUPS = [
    ("Simple", "Specification", ('SPEC_GEN', 'SPEC_ISO'), SPEC_ISO_COLORS, None),
    ("Guided", "Specification", ('SPEC_GEN_GUIDED', 'SPEC_ISO_GUIDED'), SPEC_ISO_COLORS, None),
    ("Multi", "Specification", ('SPEC_GEN_GUIDED_MULT', 'SPEC_ISO_GUIDED_MULT'),
     SPEC_ISO_COLORS, ('claude',)),
    ("Ground-truth", "Implementation", ('IMPL_GEN', 'PROOF_GEN'), PROOF_GEN_GT_COLORS, None),
    ("Generated\n(Simple)", "Implementation", ('IMPL_GEN_GEN', 'PROOF_GEN_GEN'),
     PROOF_GEN_GEN_COLORS, None),
    ("Generated\n(Guided)", "Implementation", ('IMPL_GEN_GEN_GUIDED', 'PROOF_GEN_GEN_GUIDED'),
     PROOF_GEN_GEN_COLORS, None),
]

# Status classes, left to right inside each bar, as (colour, legend label).
STATUS_CLASSES = [
    (GREEN, "OK"),
    (YELLOW, "MISMATCH"),
    (BLUE, "ISSUE (benchmark)"),
    (PURPLE, "ISSUE (both)"),
    (ORANGE, "ISSUE (generated)"),
    (RED, "FAIL"),
]

METRICS = ('input', 'cached', 'output', 'total', 'cost', 'time')


def collect(groups=GROUPS, stacks=STACKS, data_dir=var.DATA_DIRECTORY, attempt=0):
    """Return ``{(group, stack): [run, ...]}`` with one summed run per problem."""
    data = OrderedDict()
    dropped = 0
    for label, _, stages, colors, only in groups:
        colors = normalize_colors(colors)
        for key, _, _ in stacks:
            if only and key not in only:
                continue
            per_stage = [dict(iter_records(Path(data_dir) / key, task, attempt)) for task in stages]
            runs = []
            for pid in sorted(set.intersection(*(set(s) for s in per_stage))):
                usages = [run_usage(s[pid]) for s in per_stage]
                if any(u is None or not u['complete'] for u in usages):
                    dropped += 1
                    continue
                status, comment = run_status(per_stage[-1][pid])
                run = {'problem_id': pid, 'status': status, 'comment': comment,
                       'color': lookup_color(colors, status, comment)}
                for m in METRICS:
                    vals = [u[m] for u in usages]
                    run[m] = None if any(v is None for v in vals) else sum(vals)
                for prefix, task, u in zip(('gen', 'proof'), stages, usages):
                    run[f'{prefix}_task'] = task
                    run[f'{prefix}_input'], run[f'{prefix}_output'] = u['input'], u['output']
                runs.append(run)
            data[(label, key)] = runs
    print(f"Dropped {dropped} problem(s) with incomplete token counts.")
    return data


def _stats(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(v):
        return None
    return {'n': len(v), 'mean': v.mean(), 'median': np.median(v),
            'p25': np.percentile(v, 25), 'p75': np.percentile(v, 75),
            'min': v.min(), 'max': v.max()}


def write_csvs(data, stem):
    """Per-problem rows and summary statistics per (group, stack, class)."""
    runs_path, summary_path = Path(f'{stem}_runs.csv'), Path(f'{stem}_summary.csv')
    runs_path.parent.mkdir(parents=True, exist_ok=True)

    run_fields = ['group', 'stack', 'problem_id', 'status', 'comment', *METRICS,
                  'gen_task', 'gen_input', 'gen_output', 'proof_task', 'proof_input', 'proof_output']
    with open(runs_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=run_fields, extrasaction='ignore')
        w.writeheader()
        for (group, stack), runs in data.items():
            for r in runs:
                w.writerow({'group': group.replace('\n', ' '), 'stack': stack, **r})

    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['group', 'stack', 'status_class', 'metric',
                                          'n', 'mean', 'median', 'p25', 'p75', 'min', 'max'])
        w.writeheader()
        for (group, stack), runs in data.items():
            classes = [('all', runs)] + [
                (f"{s}:{c}" if c else s, [r for r in runs if (r['status'], r['comment']) == (s, c)])
                for s, c in OrderedDict.fromkeys((r['status'], r['comment']) for r in runs)]
            for cls, sel in classes:
                for m in METRICS:
                    st = _stats(r[m] for r in sel)
                    if st:
                        w.writerow({'group': group.replace('\n', ' '), 'stack': stack,
                                    'status_class': cls, 'metric': m, **st})
    print(f"Saved {runs_path} and {summary_path}")


def plot(data, groups=GROUPS, stacks=STACKS, out_path=None, bar_width=0.27,
         dot_size=1.2, figsize=(8, 3)):
    bases = {'output': 1e3, 'input': 1e5}  # where the bars start, at the x axis
    hatch_color = '#555555'
    plt.rcParams['hatch.linewidth'] = 0.4
    fig, (ax_out, ax_in) = plt.subplots(2, 1, sharex=True, figsize=figsize,
                                        gridspec_kw={'hspace': 0})
    stack_color = {k: c for k, _, c in stacks}

    for gi, (label, _, _, _, _) in enumerate(groups):
        keys = [k for k, _, _ in stacks if (label, k) in data]
        for bi, key in enumerate(keys):
            runs = data[(label, key)]
            x = gi + (bi - (len(keys) - 1) / 2) * bar_width
            for ax, metric in ((ax_out, 'output'), (ax_in, 'input')):
                st = _stats(r[metric] for r in runs)
                if not st:
                    continue
                base = bases[metric]
                # Generation share of all tokens in this run, drawn as that
                # share of the bar's visual length (on the log scale the split
                # is a proportion, not a token value).
                gen = sum(r[f'gen_{metric}'] for r in runs)
                share = gen / sum(r[metric] for r in runs)
                y_split = base * (st['median'] / base) ** share
                w = bar_width * 0.92
                ax.bar(x, y_split - base, bottom=base, width=w, color=stack_color[key],
                       edgecolor=hatch_color, linewidth=0, hatch='//////', zorder=1)
                ax.bar(x, st['median'] - y_split, bottom=y_split, width=w,
                       color=stack_color[key], edgecolor='none', zorder=1)
                # Status-class medians on the bar axis, largest class at the back.
                classes = [(c, [r[metric] for r in runs if r['color'] == c])
                           for c, _ in STATUS_CLASSES]
                classes = sorted((cv for cv in classes if cv[1]), key=lambda cv: -len(cv[1]))
                for k, (color, vals) in enumerate(classes):
                    ax.scatter(x, np.median(vals), s=dot_size * (4 + len(vals)) ** 0.9,
                               facecolor=color, edgecolor='none', zorder=3 + k * 0.01)
                ax.errorbar(x, st['median'], yerr=[[st['median'] - st['p25']],
                                                   [st['p75'] - st['median']]],
                            fmt='none', ecolor='black', elinewidth=0.6, capsize=1.5,
                            capthick=0.6, zorder=4)

    # Each axis starts at an arbitrary baseline on the shared x axis; its tick
    # label is left out so the two scales do not collide there.
    for ax, label, base in ((ax_out, "output tokens", bases['output']),
                            (ax_in, "input tokens", bases['input'])):
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _, base=base: '' if np.isclose(v, base) else _fmt_tokens(v)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis='y', labelsize=6)
        ax.set_ylabel(label, fontsize=7)
        ax.grid(True, axis='y', which='major', alpha=0.3, linewidth=0.5, zorder=0)
        for side in ('top', 'right', 'bottom'):
            ax.spines[side].set_visible(False)
        ax.set_ylim(bottom=base)
    ax_in.invert_yaxis()  # input bars hang down from the shared x axis
    fig.align_ylabels([ax_out, ax_in])
    ax_out.axhline(ax_out.get_ylim()[0], color='black', linewidth=0.6, zorder=5)

    # Part labels above the group labels, with the Specification | Implementation split.
    ax_in.set_xticks([])
    for gi, (label, *_) in enumerate(groups):
        ax_out.text(gi, 1.01, label, transform=ax_out.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=7, multialignment='center')
    parts = OrderedDict()
    for gi, (_, part, *_) in enumerate(groups):
        parts.setdefault(part, []).append(gi)
    for part, idx in parts.items():
        # Fixed offset in points, clear of the two-line group labels at any figsize.
        ax_out.annotate(part, xy=(np.mean(idx), 1.0), xycoords=ax_out.get_xaxis_transform(),
                        xytext=(0, 24), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
    for part, idx in list(parts.items())[1:]:
        for ax in (ax_out, ax_in):
            ax.axvline(idx[0] - 0.5, color='#999999', linewidth=0.5, linestyle=':')
    ax_in.set_xlim(-0.5, len(groups) - 0.5)

    stack_handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=lab) for _, lab, c in stacks]
    stage_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=hatch_color, linewidth=0,
                      hatch='//////', label="generation stage"),
        plt.Rectangle((0, 0), 1, 1, facecolor='#D9D9D9', edgecolor='none',
                      label="proving stage"),
    ]
    status_handles = [plt.Line2D([], [], marker='o', linestyle='', markersize=4,
                                 markerfacecolor=c, markeredgecolor='none', label=lab)
                      for c, lab in STATUS_CLASSES]
    leg = fig.legend(handles=stack_handles + stage_handles,
                     loc='upper center', ncol=len(stack_handles) + len(stage_handles),
                     fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.075))
    fig.legend(handles=status_handles, loc='upper center', ncol=len(status_handles),
               fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.035), handletextpad=0.1)
    fig.subplots_adjust(bottom=0.1)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f"Saved figure to {out_path}")
    else:
        plt.show()


if __name__ == '__main__':
    data = collect()
    write_csvs(data, 'out/token_bars')
    plot(data, out_path='out/token_bars.png')
