from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal, Iterable, Callable

from pathlib import Path
from core.problem import Problem

from clever_bench.task import TaskComponent
from clever_bench.lean_problem import LeanProblemView

DEFAULT_LABELS = {
    "function_signature": "FUNCTION_SIGNATURE",
    "problem_spec_nl": "NL_SPECIFICATION",
    "helper_definitions": "HELPER_DEFINITIONS",
    "problem_spec_formal_ground_truth": "FORMAL_SPECIFICATION",
    "problem_spec_formal_generated": "GENERATED_SPECIFICATION",
    "problem_spec_formal_generated_body": "SPECIFICATION_BODY",
    "isomorphism_helper_lemmas": "ISOMORPHISM_HELPER_LEMMAS",
    "isomorphism_theorem": "ISOMORPHISM_THEOREM",
    "isomorphism_proof": "ISOMORPHISM_PROOF",
    "implementation_signature": "IMPLEMENTATION_SIGNATURE",
    "implementation": "IMPLEMENTATION",
    "test_cases_lean": "TEST_CASES",
    "correctness_helper_lemmas": "CORRECTNESS_HELPER_LEMMAS",
    "correctness_theorem": "CORRECTNESS_THEOREM",
    "correctness_proof": "CORRECTNESS_PROOF",
}
DEFAULT_PROOFS = {
    "problem_spec_formal_generated_body": "sorry",
    "correctness_proof": "by sorry",
    "implementation": "sorry",
    "isomorphism_proof": "by sorry",
}
DEFAULT_LISTS = [
    "helper_definitions",
    "isomorphism_helper_lemmas",
    "correctness_helper_lemmas"
]
DEFAULT_HANDLERS = {
    "problem_spec_nl": lambda s: f"/--\n{s}\n-/",
    "helper_definitions": lambda s: re.sub(r'^/--\n.*?-/\n*', '', s, flags=re.DOTALL),
}
def _parse_nl(s: str) -> str:
    """Extract the docstring body from the NL section, tolerating drift.

    Models occasionally rewrite the ``/--`` doc comment as a plain ``/-`` block
    (or add stray whitespace); fall back to a lenient extraction with a warning
    instead of crashing the whole evaluation run.
    """
    m = re.fullmatch(r'\n*/--\n(.*?)\n-/\n*', s, flags=re.DOTALL)
    if m is not None:
        return m.group(1)
    m = re.search(r'/--?\s*\n?(.*?)\n?\s*-/', s, flags=re.DOTALL)
    if m is not None:
        logging.warning("problem_spec_nl: nonstandard comment block, parsed leniently")
        return m.group(1)
    logging.warning("problem_spec_nl: no comment block found, using raw section")
    return s.strip()


PARSE_HANDLERS = {
    "problem_spec_nl": _parse_nl
}


@dataclass
class BlockSpec:
    field: Literal[
        "function_signature",
        "problem_spec_nl",
        "helper_definitions",
        "problem_spec_formal_ground_truth",
        "problem_spec_formal_generated",
        "problem_spec_formal_generated_body",
        "isomorphism_helper_lemmas",
        "isomorphism_theorem",
        "isomorphism_proof",
        "implementation_signature",
        "implementation",
        "test_cases_lean",
        "correctness_helper_lemmas",
        "correctness_theorem",
        "correctness_proof",
    ]
    label: str | None = ""
    proof: str | None = None
    hide: bool = False
    handler: Callable[[str], str] | None = None

    def __post_init__(self):
        if self.label == "":
            self.label = DEFAULT_LABELS.get(self.field)
        if (self.proof is None) and (self.field in DEFAULT_PROOFS.keys()):
            self.proof = DEFAULT_PROOFS[self.field]
        if self.handler is None:
            self.handler = DEFAULT_HANDLERS.get(self.field, lambda s: s)


def format_lean(problem: LeanProblemView, blocks: Iterable[BlockSpec], add_imports=False) -> str | list[str]:
    lean = list()
    for prob in (problem,) if not isinstance(problem, list) else problem:
        if add_imports:
            parts = ["import Imports.AllImports"]
        else:
            parts = ["import Mathlib\nimport Mathlib.Algebra.Polynomial.Basic\nimport Std.Data.HashMap"]

        for spec in blocks:
            if spec.hide:
                prob.hide_params(spec.field, )
            content = getattr(prob, spec.field)
            if spec.proof is not None and content is None:
                content = spec.proof

            if content is not None:
                if spec.label is not None and not (spec.field == "helper_definitions" and len(content) == 0):
                    parts.append(f"\n-- [{spec.label}]")
                if not isinstance(content, list):
                    content = [content]
                parts.extend(spec.handler(c) for c in content)
        lean.append("\n\n".join(parts) + "\n")
    return lean if len(lean) > 1 else lean[0]


SECTION_TAG = r'\n-- \[([A-Z_]+)\]\n'
SECTION_TAG_NC = r'\n-- \[(?:[A-Z_]+)\]\n'


def parse_view(problem: Problem, task_component: TaskComponent, blocks: Iterable[BlockSpec], handlers=PARSE_HANDLERS) -> \
        list[LeanProblemView]:
    """Parse a formatted Lean file back into labeled sections."""
    views = list()
    solutions = problem.solution if isinstance(problem.solution, list) else [problem.solution]
    for solution in solutions:
        sections = {
            label.strip(): content.strip("\n")
            for label, content in re.findall(
                rf'{SECTION_TAG}(.*?)(?={SECTION_TAG_NC}|$)',
                solution,
                re.DOTALL
            )
        }
        view = LeanProblemView(
            problem_id=problem.info.id,
            task_component=task_component,
        )
        for block in blocks:
            if block.label and block.label in sections:
                setattr(view, block.field, handlers.get(block.field, lambda s: s)(sections[block.label]))
        views.append(view)
    return views


########  TASKS  ########

DATASET = {'dataset': "clever", "subset": "human_eval"}

TASKS = dict()

MANUAL_PROMPTS = {
    "SPEC_ISO": "Read {filepath} and fill the [ISOMORPHISM_PROOF] section faithfully following the prior instructions.\n"
                "Consult the lean4 skill for Lean 4 conventions, strategies and available tools. Complete the task autonomously, do not return before completion.",

    "PROOF_GEN": "Read {filepath} and fill the [CORRECTNESS_PROOF] section faithfully following the prior instructions.\n" \
                 "Consult the lean4 skill for Lean 4 conventions, strategies and available tools. Complete the task autonomously, do not return before completion."
}

# SPEC GEN

TASKS['SPEC_GEN'] = {
    'dataset': DATASET,
    'config': {
        'context': {'src': Path("clever/spec-gen.md")},
        'prompt': "Read {filepath} and fill the [SPECIFICATION_BODY] section faithfully following the prior instructions.\n"
                  "Consult the lean4 skill for Lean 4 conventions, strategies and available tools. Fix any compilation errors"
    },
    'task_component': TaskComponent.SPEC_GENERATION,
    'blocks': (
        BlockSpec(field="problem_spec_nl"),
        BlockSpec(field="helper_definitions"),
        BlockSpec(field="problem_spec_formal_generated", label="SPECIFICATION_SIGNATURE"),
        BlockSpec(field="problem_spec_formal_generated_body", hide=True),
    )
}
TASKS['SPEC_GEN_GUIDED'] = {
    'dataset': DATASET,
    'config': {
        'context': {'src': Path("clever/spec-gen-guided.md")},
        'prompt': "Read {filepath} and fill the [SPECIFICATION_BODY] section faithfully following the prior instructions.\n"
                  "Consult the lean4 skill for Lean 4 conventions, strategies and available tools. Fix any compilation errors"
    },
    'task_component': TaskComponent.SPEC_GENERATION,
    'blocks': (
        BlockSpec(field="problem_spec_nl"),
        BlockSpec(field="helper_definitions"),
        BlockSpec(field="problem_spec_formal_generated", label="SPECIFICATION_SIGNATURE"),
        BlockSpec(field="problem_spec_formal_generated_body", hide=True),
        BlockSpec(field="test_cases_lean", handler=lambda s: re.sub(r'^#test', '-- #test', s, flags=re.MULTILINE)),
    ),
    'load_kwargs': {
        'tests_in_nl_spec': True,
    }
}
TASKS['SPEC_GEN_GUIDED_MULT'] = {
    'dataset': DATASET,
    'config': {
        'context': {'src': Path("clever/spec-gen-guided-multiple.md")},
        'prompt': "Read {filepath} and create .lean files in the `solutions` subdirectory filling the [SPECIFICATION_BODY] section faithfully following the prior instructions.\n"
                  "Consult the lean4 skill for Lean 4 conventions, strategies and available tools. Fix any compilation errors",
        'env_template': "clever-mult",
        'output': "solutions/*.lean",
    },
    'task_component': TaskComponent.SPEC_GENERATION,
    'blocks': (
        BlockSpec(field="problem_spec_nl"),
        BlockSpec(field="helper_definitions"),
        BlockSpec(field="problem_spec_formal_generated", label="SPECIFICATION_SIGNATURE"),
        BlockSpec(field="problem_spec_formal_generated_body", hide=True),
        BlockSpec(field="test_cases_lean", handler=lambda s: re.sub(r'^#test', '-- #test', s, flags=re.MULTILINE)),
    ),
    'load_kwargs': {
        'tests_in_nl_spec': True,
    }
}

# SPEC ISO

TASKS['SPEC_ISO'] = {
    'dataset': DATASET,
    'config': {
        'context': {'src': Path("clever/spec-iso.md")},
        'prompt': "/lean4:autoprove {filepath} --deep=stuck",
        'status_prompt': 'mismatch-spec',
    },
    'task_component': TaskComponent.SPEC_ISOMORPHISM,
    'blocks': (
        BlockSpec(field="problem_spec_nl"),
        BlockSpec(field="helper_definitions"),
        BlockSpec(field="problem_spec_formal_ground_truth", label="GROUND_TRUTH_SPECIFICATION"),
        BlockSpec(field="problem_spec_formal_generated"),
        BlockSpec(field="problem_spec_formal_generated_body", label=None),
        BlockSpec(field="isomorphism_helper_lemmas", hide=True),
        BlockSpec(field="isomorphism_theorem"),
        BlockSpec(field="isomorphism_proof", hide=True),
    ),
    'previous': {
        'task': 'SPEC_GEN',
        'task_component': TASKS['SPEC_GEN']['task_component'],
        'blocks': TASKS['SPEC_GEN']['blocks'],
        'fields': ("problem_spec_formal_generated_body",)
    }
}
TASKS['SPEC_ISO_GUIDED'] = {
    'dataset': DATASET,
    'config': {
        'context': {'src': Path("clever/spec-iso-guided.md")},
        'prompt': "/lean4:autoprove {filepath} --deep=stuck",
        'status_prompt': 'mismatch-spec',
    },
    'task_component': TaskComponent.SPEC_ISOMORPHISM,
    'blocks': (
        BlockSpec(field="problem_spec_nl"),
        BlockSpec(field="helper_definitions"),
        BlockSpec(field="problem_spec_formal_ground_truth", label="GROUND_TRUTH_SPECIFICATION"),
        BlockSpec(field="problem_spec_formal_generated"),
        BlockSpec(field="problem_spec_formal_generated_body", label=None),
        BlockSpec(field="isomorphism_helper_lemmas", hide=True),
        BlockSpec(field="isomorphism_theorem"),
        BlockSpec(field="isomorphism_proof", hide=True),
        BlockSpec(field="test_cases_lean", handler=lambda s: re.sub(r'^#test', '-- #test', s, flags=re.MULTILINE)),
    ),
    'previous': {
        'task': 'SPEC_GEN_GUIDED',
        'task_component': TASKS['SPEC_GEN_GUIDED']['task_component'],
        'blocks': TASKS['SPEC_GEN_GUIDED']['blocks'],
        'fields': ("problem_spec_formal_generated_body",)
    },
    'load_kwargs': {
        'tests_in_nl_spec': True,
    }
}
TASKS['SPEC_ISO_GUIDED_MULT'] = {
    'dataset': DATASET,
    'config': {
        'context': {'src': Path("clever/spec-iso-guided-multiple.md")},
        'prompt': "/lean4:autoprove {filepath} --deep=stuck",
        'status_prompt': 'mismatch-spec-mult',
        'input': "problem_{i}.lean",
    },
    'task_component': TaskComponent.SPEC_ISOMORPHISM,
    'blocks': (
        BlockSpec(field="problem_spec_nl"),
        BlockSpec(field="helper_definitions"),
        BlockSpec(field="problem_spec_formal_ground_truth", label="GROUND_TRUTH_SPECIFICATION"),
        BlockSpec(field="problem_spec_formal_generated"),
        BlockSpec(field="problem_spec_formal_generated_body", label=None),
        BlockSpec(field="isomorphism_helper_lemmas", hide=True),
        BlockSpec(field="isomorphism_theorem"),
        BlockSpec(field="isomorphism_proof", hide=True),
        BlockSpec(field="test_cases_lean", handler=lambda s: re.sub(r'^#test', '-- #test', s, flags=re.MULTILINE)),
    ),
    'previous': {
        'task': 'SPEC_GEN_GUIDED_MULT',
        'task_component': TASKS['SPEC_GEN_GUIDED_MULT']['task_component'],
        'blocks': TASKS['SPEC_GEN_GUIDED_MULT']['blocks'],
        'fields': ("problem_spec_formal_generated_body",)
    },
    'load_kwargs': {
        'tests_in_nl_spec': True,
    }
}

TASKS['SPEC_ISO_GUIDED_MANUAL'] = {
    'dataset': DATASET,
    'config': {
        'context': {'src': Path("clever/spec-iso-guided-manual.md")},
        'prompt': MANUAL_PROMPTS['SPEC_ISO'],
        'status_prompt': 'mismatch-spec',
    },
    'task_component': TaskComponent.SPEC_ISOMORPHISM,
    'blocks': (
        BlockSpec(field="problem_spec_nl"),
        BlockSpec(field="helper_definitions"),
        BlockSpec(field="problem_spec_formal_ground_truth", label="GROUND_TRUTH_SPECIFICATION"),
        BlockSpec(field="problem_spec_formal_generated"),
        BlockSpec(field="problem_spec_formal_generated_body", label=None),
        BlockSpec(field="isomorphism_helper_lemmas", hide=True),
        BlockSpec(field="isomorphism_theorem"),
        BlockSpec(field="isomorphism_proof", hide=True),
        BlockSpec(field="test_cases_lean", handler=lambda s: re.sub(r'^#test', '-- #test', s, flags=re.MULTILINE)),
    ),
    'previous': {
        'task': 'SPEC_GEN_GUIDED',
        'task_component': TASKS['SPEC_GEN_GUIDED']['task_component'],
        'blocks': TASKS['SPEC_GEN_GUIDED']['blocks'],
        'fields': ("problem_spec_formal_generated_body",)
    },
    'load_kwargs': {
        'tests_in_nl_spec': True,
    }
}

# IMPL GEN

TASKS['IMPL_GEN'] = {
    'dataset': DATASET,
    'config': {
        'context': {'src': Path("clever/impl-gen.md")},
        'prompt': "Read {filepath} and fill the [IMPLEMENTATION] section faithfully following the prior instructions.\n"
                  "Consult the lean4 skill for Lean 4 conventions, strategies and available tools. Uncomment the test cases upon completing the implementation, and fix any compilation or test errors"
    },
    'task_component': TaskComponent.IMPL_GENERATION,
    'blocks': (
        BlockSpec(field="problem_spec_nl"),
        BlockSpec(field="helper_definitions"),
        BlockSpec(field="problem_spec_formal_ground_truth"),
        BlockSpec(field="implementation_signature"),
        BlockSpec(field="implementation", hide=True),
        BlockSpec(field="test_cases_lean",
                  handler=lambda s: "-- Uncomment the #test lines below after completing the implementation\n" +
                                    re.sub(r'^#test', '-- #test', s, flags=re.MULTILINE)
                  ),
    )
}

TASKS['IMPL_GEN_GEN'] = {
    'dataset': DATASET,
    'config': {
        'context': {'src': Path("clever/impl-gen.md")},
        'prompt': "Read {filepath} and fill the [IMPLEMENTATION] section faithfully following the prior instructions.\n"
                  "Consult the lean4 skill for Lean 4 conventions, strategies and available tools. Uncomment the test cases upon completing the implementation, and fix any compilation or test errors"
    },
    'task_component': TaskComponent.IMPL_GENERATION,
    'blocks': (
        BlockSpec(field="problem_spec_nl"),
        BlockSpec(field="helper_definitions"),
        BlockSpec(field="problem_spec_formal_generated", label="FORMAL_SPECIFICATION",
                  handler=lambda s: re.sub(r'^def\s+generated_spec', 'def problem_spec', s, flags=re.MULTILINE)
                  ),
        BlockSpec(field="problem_spec_formal_generated_body", label=None),
        BlockSpec(field="implementation_signature"),
        BlockSpec(field="implementation", hide=True),
        BlockSpec(field="test_cases_lean",
                  handler=lambda s: "-- Uncomment the #test lines below after completing the implementation\n" +
                                    re.sub(r'^#test', '-- #test', s, flags=re.MULTILINE)
                  ),
    ),
    'previous': {
        'task': 'SPEC_GEN',
        'task_component': TASKS['SPEC_GEN']['task_component'],
        'blocks': TASKS['SPEC_GEN']['blocks'],
        'fields': ("problem_spec_formal_generated", "problem_spec_formal_generated_body")
    },
}

TASKS['IMPL_GEN_GEN_GUIDED'] = {
    'dataset': DATASET,
    'config': {
        'context': {'src': Path("clever/impl-gen-guided.md")},
        'prompt': "Read {filepath} and fill the [IMPLEMENTATION] section faithfully following the prior instructions.\n"
                  "Consult the lean4 skill for Lean 4 conventions, strategies and available tools. Uncomment the test cases upon completing the implementation, and fix any compilation or test errors"
    },
    'task_component': TaskComponent.IMPL_GENERATION,
    'blocks': (
        BlockSpec(field="problem_spec_nl"),
        BlockSpec(field="helper_definitions"),
        BlockSpec(field="problem_spec_formal_generated", label="FORMAL_SPECIFICATION",
                  handler=lambda s: re.sub(r'^def\s+generated_spec', 'def problem_spec', s, flags=re.MULTILINE)
                  ),
        BlockSpec(field="problem_spec_formal_generated_body", label=None),
        BlockSpec(field="implementation_signature"),
        BlockSpec(field="implementation", hide=True),
        BlockSpec(field="test_cases_lean",
                  handler=lambda s: "-- Uncomment the #test lines below after completing the implementation\n" +
                                    re.sub(r'^#test', '-- #test', s, flags=re.MULTILINE)
                  ),
    ),
    'previous': {
        'task': 'SPEC_GEN_GUIDED',
        'task_component': TASKS['SPEC_GEN_GUIDED']['task_component'],
        'blocks': TASKS['SPEC_GEN_GUIDED']['blocks'],
        'fields': ("problem_spec_formal_generated", "problem_spec_formal_generated_body")
    },
    'load_kwargs': {
        'tests_in_nl_spec': True,
    }
}

# PROOF GEN

TASKS['PROOF_GEN'] = {
    'dataset': DATASET,
    'config': {
        'context': {'src': Path("clever/proof-gen.md")},
        'prompt': "/lean4:autoprove {filepath} --deep=stuck",
        'status_prompt': 'mismatch-impl',
    },
    'task_component': TaskComponent.PROOF_GENERATION,
    'blocks': (
        BlockSpec(field="problem_spec_nl"),
        BlockSpec(field="helper_definitions"),
        BlockSpec(field="problem_spec_formal_ground_truth"),
        BlockSpec(field="implementation_signature", label="IMPLEMENTATION"),
        BlockSpec(field="implementation", label=None),
        BlockSpec(field="test_cases_lean",
                  handler=lambda s: re.sub(r'^--\s*#test', '#test', s, flags=re.MULTILINE)
                  ),
        BlockSpec(field="correctness_helper_lemmas", hide=True),
        BlockSpec(field="correctness_theorem"),
        BlockSpec(field="correctness_proof", hide=True),
    ),
    'previous': {
        'task': 'IMPL_GEN',
        'task_component': TASKS['IMPL_GEN']['task_component'],
        'blocks': TASKS['IMPL_GEN']['blocks'],
        'fields': ("implementation",)
    }
}

TASKS['PROOF_GEN_GEN'] = {
    'dataset': DATASET,
    'config': {
        'context': {'src': Path("clever/proof-gen.md")},
        'prompt': "/lean4:autoprove {filepath} --deep=stuck",
        'status_prompt': 'mismatch-impl',
    },
    'task_component': TaskComponent.PROOF_GENERATION,
    'blocks': (
        BlockSpec(field="problem_spec_nl"),
        BlockSpec(field="helper_definitions"),
        BlockSpec(field="problem_spec_formal_generated", label="FORMAL_SPECIFICATION"),
        BlockSpec(field="implementation_signature", label="IMPLEMENTATION"),
        BlockSpec(field="implementation", label=None),
        BlockSpec(field="test_cases_lean",
                  handler=lambda s: re.sub(r'^--\s*#test', '#test', s, flags=re.MULTILINE)
                  ),
        BlockSpec(field="correctness_helper_lemmas", hide=True),
        BlockSpec(field="correctness_theorem"),
        BlockSpec(field="correctness_proof", hide=True),
    ),
    'previous': {
        'task': 'IMPL_GEN_GEN',
        'task_component': TASKS['IMPL_GEN_GEN']['task_component'],
        'blocks': TASKS['IMPL_GEN_GEN']['blocks'],
        'fields': ("problem_spec_formal_generated", "implementation",)
    },
}

TASKS['PROOF_GEN_GEN_GUIDED'] = {
    'dataset': DATASET,
    'config': {
        'context': {'src': Path("clever/proof-gen-guided.md")},
        'prompt': "/lean4:autoprove {filepath} --deep=stuck",
        'status_prompt': 'mismatch-impl',
    },
    'task_component': TaskComponent.PROOF_GENERATION,
    'blocks': (
        BlockSpec(field="problem_spec_nl"),
        BlockSpec(field="helper_definitions"),
        BlockSpec(field="problem_spec_formal_generated", label="FORMAL_SPECIFICATION"),
        BlockSpec(field="implementation_signature", label="IMPLEMENTATION"),
        BlockSpec(field="implementation", label=None),
        BlockSpec(field="test_cases_lean",
                  handler=lambda s: re.sub(r'^--\s*#test', '#test', s, flags=re.MULTILINE)
                  ),
        BlockSpec(field="correctness_helper_lemmas", hide=True),
        BlockSpec(field="correctness_theorem"),
        BlockSpec(field="correctness_proof", hide=True),
    ),
    'previous': {
        'task': 'IMPL_GEN_GEN_GUIDED',
        'task_component': TASKS['IMPL_GEN_GEN_GUIDED']['task_component'],
        'blocks': TASKS['IMPL_GEN_GEN_GUIDED']['blocks'],
        'fields': ("problem_spec_formal_generated", "implementation",)
    },
    'load_kwargs': {
        'tests_in_nl_spec': True,
    }
}

# END-TO-END

TASKS['END_TO_END'] = {
    'dataset': DATASET,
    'config': {
        'context': {'src': Path("clever/end-to-end.md")},
        'prompt': "Read {filepath} and fill the [SPECIFICATION_BODY], [IMPLEMENTATION], and [CORRECTNESS_PROOF] sections faithfully following the prior instructions.\n"
                  "Consult the lean4 skill for Lean 4 conventions, strategies and available tools, and use the lean4 skill '/lean4:autoprove --deep=stuck' command to prove the correctness theorem. Fix any compilation errors"
    },
    'task_component': TaskComponent.PROOF_GENERATION,
    'blocks': (
        BlockSpec(field="problem_spec_nl"),
        BlockSpec(field="helper_definitions"),
        BlockSpec(field="problem_spec_formal_generated", label="SPECIFICATION_SIGNATURE",
                  handler=lambda s: re.sub(r'^def\s+generated_spec', 'def problem_spec', s, flags=re.MULTILINE)),
        BlockSpec(field="problem_spec_formal_generated_body", hide=True, label="SPECIFICATION"),
        BlockSpec(field="implementation_signature"),
        BlockSpec(field="implementation", hide=True),
        BlockSpec(field="test_cases_lean",
                  handler=lambda s: "-- Uncomment the #test lines below after completing the implementation\n" +
                                    re.sub(r'^#test', '-- #test', s, flags=re.MULTILINE)
                  ),
        BlockSpec(field="correctness_helper_lemmas", hide=True),
        BlockSpec(field="correctness_theorem"),
        BlockSpec(field="correctness_proof", hide=True),
    ),
    'load_kwargs': {
        'tests_in_nl_spec': True,
    }
}
