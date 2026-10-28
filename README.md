# Isomorphism Is Not Enough!

Code, prompts, run records and figure scripts supporting the paper
[*Isomorphism Is Not Enough: An Agentic Audit of Verified Code Generation in Lean*](https://arxiv.org/abs/2605.23772)
by Alessandro Sosso, Akhil Arora and Bas Spitters.

The paper runs agentic coding setups on the full
[CLEVER](https://github.com/trishullab/clever) pipeline for verified code
generation in Lean 4: the four tasks `SPEC_GEN`, `SPEC_ISO`, `IMPL_GEN` and
`PROOF_GEN`, with several prompt variants and with ground-truth or generated
upstream inputs. The main results use Claude Opus 4.6 through the Claude Agent
SDK. The cross-stack comparison runs the same tasks and prompts with GPT-5.4
(Codex CLI) and Kimi K2.5 (OpenHands SDK). A contamination control uses GPT-5.1
(Codex CLI) on a 16-entry pilot.

This release contains:

- the harness that drives the runs, and the prompts given to the agents;
- the fork of CLEVER used as the evaluation harness;
- every per-entry record of the reported runs (status flag, annotation,
  timing, validation outcome), with the solutions and agent traces replaced by
  SHA-256 commitments (see [Redacted data](#redacted-data));
- the complete per-entry findings of the benchmark audit;
- the scripts behind the paper's figures.


## Repository layout

```
core/                         Harness (Python package `core`)
├── problem.py                Problem / ProblemInfo: JSON record I/O
├── var.py                    Paths, Lean version, constants
├── utils.py                  Git-status helper (checks the lean4-skills pin)
├── benchmarks/clever.py      TASKS: definition of every task variant
└── models/                   Prover backends
    ├── base_prover.py        Abstract prover, multi-process executor
    ├── base_agentic.py       Per-attempt sandbox, skills copy, retries
    ├── claude.py             Claude Agent SDK      (-p claude)
    ├── codex.py              Codex CLI             (-p codex, -p codex-pre)
    └── openhands.py          OpenHands SDK         (-p kimi, -p qwen, ...)

scripts/
├── clever.py                 CLI: run tasks, evaluate, write summary CSVs
├── display.py                Terminal browser for the JSON records
├── sample_ids.py             Seeded draw of the 16-entry pilot
└── redact.py                 Redaction of the run records, and its verification

plots/                        Scripts for the paper figures (write to ./out/)
├── status_grid.py            Per-entry result grids
├── status_sankey.py          Flag-transition Sankey diagrams
├── token_bars.py             Token usage per stack and task
├── scatter_time_price.py     Cost against runtime (Claude)
└── status_bars.py, token_usage.py   Shared helpers

agents/
├── contexts/clever/          Prompt (context) file of each task variant
└── envs/clever/              Lean 4 project template copied into each attempt

benchmarks/clever/            Fork of CLEVER (installable as `clever-bench`)
├── src/clever_bench/         Python evaluation API
├── src/lean4/human_eval/     The 161 problem files, with the fork's fixes
└── ...                       Upstream files (reference solutions, tests, ...)

findings/                     Per-entry audit findings (full appendix tables)
├── reference_spec_issues.csv  Reference-specification ISSUE findings (80 entries)
├── mismatches.csv            MISMATCH findings (13 entries)
└── base_issues.csv           Benchmark-base ISSUE findings (4 entries)

data/                         Run records, redacted (see below)
pyproject.toml, uv.lock       uv-managed Python project
LICENSE, LICENSE-CC-BY-4.0    MIT (code), CC BY 4.0 (data, findings, prompts)
```


## Requirements

- Python 3.13.12 and [uv](https://docs.astral.sh/uv/). `uv sync` installs the
  harness and the fork of CLEVER (editable, from `benchmarks/clever`).
- elan and Lean 4 v4.24.0 with Mathlib (see `agents/envs/clever/lean-toolchain`
  and `lake-manifest.json`).
- `lean-lsp-mcp` and `leanclient`, launched through `uvx` at run time.
- `lean4-skills`, cloned at the pinned commit (next section).
- A backend for the runs to reproduce:
  - `-p claude`: Claude Agent SDK (a core dependency) and a logged-in Claude
    account; model `claude-opus-4-6[1m]`.
  - `-p codex`, `-p codex-pre`: `uv sync --extra codex` plus a `codex` CLI on
    `PATH`; models GPT-5.4 and GPT-5.1. `codex-pre` reads an API key from
    `agents/keys/openai`.
  - `-p kimi`: `uv sync --extra openhands` plus an OpenAI-compatible
    endpoint serving the model. The base URL and key are read from a two-line
    file in `agents/keys/` (base URL, then key): `agents/keys/openhands` for
    `-p kimi`, `agents/keys/vllm` for `-p qwen`.

  Credentials are not part of the repository; `agents/keys/` is git-ignored.
  A backend whose SDK is missing only disables its own `-p` option.

### lean4-skills

Every backend loads the `lean4` plugin of
[lean4-skills](https://github.com/cameronfreer/lean4-skills) from
`agents/skills/freer-skills`. Before copying it into an attempt, the harness
checks that this directory is a clean git checkout on branch `main` at commit
`a1cef7dc21d9b68b79311d57cd78ecf6b7ca22bf`, and refuses to run otherwise:

```bash
git clone https://github.com/cameronfreer/lean4-skills agents/skills/freer-skills
git -C agents/skills/freer-skills checkout -B main a1cef7dc21d9b68b79311d57cd78ecf6b7ca22bf
```


## Setup

```bash
uv sync                           # harness + clever-bench
uv sync --extra codex             # optional: Codex backend
uv sync --extra openhands         # optional: OpenHands backend
(cd agents/envs/clever && lake exe cache get)   # Mathlib cache for the template
```


## Running the harness

The entry point is `scripts/clever.py`. Tasks are the keys of `TASKS` in
`core/benchmarks/clever.py`; the backend is selected with `-p` (default
`claude`).

```bash
uv run python scripts/clever.py -t SPEC_GEN -w 2               # all 161 entries, 2 workers
uv run python scripts/clever.py -p codex -t SPEC_ISO -i 0 1 2  # subset of entries
uv run python scripts/clever.py -t SPEC_GEN --dry-run /tmp/in  # write the inputs only
uv run python scripts/clever.py -p claude --evaluate -t PROOF_GEN
uv run python scripts/clever.py -p claude --summary status -t SPEC_ISO PROOF_GEN
```

Each attempt runs in a temporary copy of `agents/envs/clever` under
`/tmp/appv` (override with `APPV_TMPDIR` or `--tmp-dir`). The generation
timeout is 1 hour per attempt for Claude and the Codex CLI, and 4 hours for
OpenHands. `--evaluate` checks each solution with the CLEVER harness and stores
the result under `metadata.validation`.

### Task variants

| `TASKS` key | Run in the paper | Prompt (`agents/contexts/clever/`) | Upstream input |
|---|---|---|---|
| `SPEC_GEN` | Simple specification generation | `spec-gen.md` | NL description |
| `SPEC_GEN_GUIDED` | Guided specification generation | `spec-gen-guided.md` | NL description + tests |
| `SPEC_GEN_GUIDED_MULT` | Multiple specifications (Claude only) | `spec-gen-guided-multiple.md` | NL description + tests |
| `SPEC_ISO` | Isomorphism proof, Simple specs | `spec-iso.md` | `SPEC_GEN` |
| `SPEC_ISO_GUIDED` | Isomorphism proof, Guided specs | `spec-iso-guided.md` | `SPEC_GEN_GUIDED` |
| `SPEC_ISO_GUIDED_MULT` | Isomorphism proof, Multiple specs | `spec-iso-guided-multiple.md` | `SPEC_GEN_GUIDED_MULT` |
| `IMPL_GEN` | Implementation from the reference spec | `impl-gen.md` | reference `problem_spec` |
| `IMPL_GEN_GEN` | Implementation from Simple specs | `impl-gen.md` | `SPEC_GEN` |
| `IMPL_GEN_GEN_GUIDED` | Implementation from Guided specs | `impl-gen-guided.md` | `SPEC_GEN_GUIDED` |
| `PROOF_GEN` | Correctness proof, reference spec | `proof-gen.md` | `IMPL_GEN` |
| `PROOF_GEN_GEN` | Correctness proof, Simple specs | `proof-gen.md` | `IMPL_GEN_GEN` |
| `PROOF_GEN_GEN_GUIDED` | Correctness proof, Guided specs | `proof-gen-guided.md` | `IMPL_GEN_GEN_GUIDED` |

`SPEC_ISO*` and `PROOF_GEN*` drive the agent with the lean4-skills command
`/lean4:autoprove problem.lean --deep=stuck`. `TASKS` also defines an
`END_TO_END` variant that is not part of the paper; its prompt is not included.


## Data

```
data/
├── claude/                       Claude Opus 4.6 (Claude Agent SDK)
│   ├── clever/<TASK>/            12 task variants (161 entries; 105 for the two *_MULT tasks)
│   ├── clever/SPEC_ISO_MANUAL.json    Merged SPEC_ISO annotation (65 OK / 81 ISSUE / 13 MISMATCH / 2 FAIL)
│   └── claude-full.csv                Status snapshot (2026-08-30)
├── codex/                        GPT-5.4 (Codex CLI): 10 tasks x 161 entries, + codex-full.csv snapshot
├── kimi/                         Kimi K2.5 (OpenHands): 10 tasks x 161 entries, + kimi-full.csv snapshot
└── codex-pre/                    GPT-5.1 (Codex CLI): 10 tasks x 16 pilot entries, + summary CSV
```

Each `data/<stack>/clever/<TASK>/human_eval-<id>-0.json` is one attempt
(attempt 0; all reported results are single-attempt) serialized from
`core/problem.py`:

```jsonc
{
  "text":     "<Lean file given to the agent>",   // redacted for SPEC_ISO*, IMPL_GEN_GEN*, PROOF_GEN_GEN*
  "info":     {"dataset": "clever", "subset": "human_eval", "id": 42,
               "task": "PROOF_GEN", "attempt": 0},
  "solution": "sha256:…",                // Lean file returned by the agent
  "timing":   123.45,                    // wall-clock seconds
  "metadata": {
    "prompt":     "...",                 // task prompt sent to the agent
    "<stack>":    "sha256:…",            // full agent trace, key = claude | codex | kimi | codex-pre
    "status":     "OK" | "FAIL" | "ISSUE" | "MISMATCH" | "TIMEOUT",
    "comment":    "...",                 // ISSUE source / MISMATCH direction, when set
    "result":     "sha256:…",            // final agent reply
    "validation": [ {"problem_id": "42", "compilation_ok": true,
                     "isomorphism_ok": ..., "correctness_ok": ...,
                     "error_message": "sha256:…", "lean_code": "sha256:…"} ]
  },
  "exception": null
}
```

For `*_MULT` tasks, `status` and `comment` are lists with one item per
candidate specification (and so were the original `solution` values).

### Redacted data

Publishing the agents' specifications, implementations and proofs would leak
solutions to the CLEVER problems and contaminate the benchmark. Every field
that carries them is therefore replaced by a SHA-256 commitment to its
original value:

| Field | Content |
|---|---|
| `solution` | Lean file(s) returned by the agent |
| `metadata.<stack>` | full agent trace |
| `metadata.result` | final agent reply |
| `metadata.validation[].lean_code` | Lean file checked by the CLEVER harness |
| `metadata.validation[].error_message` | Lean/harness output of that check |
| `text` (`SPEC_ISO*`, `IMPL_GEN_GEN*`, `PROOF_GEN_GEN*` only) | task input, which embeds the specifications or implementations generated upstream |

All other fields are unchanged. The input `text` of the remaining tasks
contains only benchmark material and is kept in clear.

Each redacted value `v` (a string, list or object) becomes the string

```
"sha256:" + hex(SHA-256(UTF-8(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":")))))
```

The digests are deterministic: anyone given the original records can recompute
them and check that they match the published ones.
[`scripts/redact.py`](scripts/redact.py) implements both directions:

```bash
python scripts/redact.py redact ORIG_DATA_DIR OUT_DATA_DIR       # produce the published records
python scripts/redact.py verify ORIG_DATA_DIR REDACTED_DATA_DIR  # check originals against them
```

The complete records, with solutions and traces, are available to researchers
on request: contact Alessandro Sosso at <sosso@cs.au.dk>.

### Status flags

| Flag | Meaning |
|---|---|
| `OK` | Task completed; no `sorry`; the file compiles |
| `FAIL` | `sorry` remains or the file does not compile (`TIMEOUT`: budget exhausted) |
| `ISSUE` | Task impossible because of a bug in the inputs, with an argument. Source: `PROB` / `GEN` / `BOTH` (SPEC_ISO); `SPEC` / `IMPL` / `BOTH` / `BASE` (PROOF_GEN) |
| `MISMATCH` | Both inputs are valid readings of the NL description but are not equivalent. Direction (SPEC_ISO): `→` generated stronger, `←` weaker, `/` neither |


## Reproducing the figures

| Paper item | Source |
|---|---|
| Result grids, Sankey diagrams, token usage, cost against runtime (figures) | `plots/` (below) |
| Complete finding summaries appendix | `findings/*.csv` |
| Pilot sample (16 entries, seed 50550) | `PYTHONPATH=. python scripts/sample_ids.py 16` |

The CSVs in `findings/` hold the complete per-entry findings behind the
appendix tables, row for row: for each entry, the error type, a short summary,
and the full description with the supporting argument or counterexample, as
submitted to the benchmark authors for verification. Columns: `id`,
`taxonomy` (several types joined with `; `), `summary`, `description`
(UTF-8 text; Lean code in Markdown backticks).

The figure scripts import `core` and `clever_bench` and need `matplotlib` and
`numpy`. Run them from `plots/`; they write PNGs to `plots/out/`:

```bash
cd plots
uv run --with matplotlib --with numpy python status_grid.py         # results_grid, models_grid, models_grid_pre
uv run --with matplotlib --with numpy python status_sankey.py       # sankey_{spec,impl}_{claude,codex,kimi}
uv run --with matplotlib --with numpy python token_bars.py          # token_bars (+ CSV summaries)
uv run --with matplotlib --with numpy python scatter_time_price.py  # scatter-pricing-time
```

`token_bars.py` and `scatter_time_price.py` read token counts and costs from
the agent traces, and the trace view of `scripts/display.py` needs them too, so
these run only on the complete records.


## Citation

```bibtex
@misc{sosso2026agentic,
  title         = {Isomorphism Is Not Enough: An Agentic Audit of Verified Code Generation in Lean},
  author        = {Sosso, Alessandro and Arora, Akhil and Spitters, Bas},
  year          = {2026},
  eprint        = {2605.23772},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2605.23772}
}
```


## License

- The code (`core/`, `scripts/`, `plots/`, `agents/envs/`) is released under
  the MIT license; see [`LICENSE`](LICENSE).
- The run records (`data/`), the audit findings (`findings/`) and the prompts
  (`agents/contexts/`) are released under the
  [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
  license (CC BY 4.0); see [`LICENSE-CC-BY-4.0`](LICENSE-CC-BY-4.0).
- The fork of CLEVER in `benchmarks/clever/` keeps its upstream MIT license
  ([`benchmarks/clever/LICENSE`](benchmarks/clever/LICENSE)).
