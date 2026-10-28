# Specifications Isomorphism

## Task
**SETTING**: You will access a Lean 4 file containing:
1. a natural language specification of a function (formatted as a Python-style docstring)
2. a ground truth problem specification in Lean 4
3. another problem specification in Lean 4 generated from the same NL specification

**OBJECTIVE**: Your objective is to write a formal proof in Lean 4 that both the specifications are equivalent.

**DETAILS**: The formal specifications take in an implementation and program input as parameters and hold true for all possible correct implementations.

**NOTE**: Try as hard as possible to prove equivalence, but note that some specifications may be buggy or incorrect.
If you believe the proof to be impossible due to a mismatch between specifications, your thesis must be supported by rigorous arguments such as concrete counterexamples.

**REFERENCE**: This task description is always available at `task.md` in the current working directory. Refer to it whenever uncertain about the task requirements or format.

## Format

### Input
The `.lean` file will be formatted with the following sections:

* The natural language specification
```
-- [NL_SPECIFICATION]

/--
def <function_name>(<input_type>) -> <output_type>
"""
<NL Description>
"""
-/
```

* The ground truth specification
```
-- [GROUND_TRUTH_SPECIFICATION]

def problem_spec
-- function signature
(impl: <function_signature>)
-- inputs
(<input_name>: <input_type>) :=
<ground_truth_specification>
```
* The generated specification
```
-- [GENERATED_SPECIFICATION]

def generated_spec
-- function signature
(impl: <function_signature>)
-- inputs
(<input_name>: <input_type>) : Prop :=
<generated_specification>
```
* The isomorphism theorem statement
```
-- [ISOMORPHISM_THEOREM]

theorem spec_isomorphism:
∀ impl,
(∀ <input_name>, problem_spec impl <input_name>) ↔
(∀ <input_name>, generated_spec impl <input_name>) :=
```

An optional `[HELPER_DEFINITIONS]` section will contain definitions useful to the problem.

### Output
Insert the generated proof in the `[ISOMORPHISM_PROOF]` section:
```
-- [ISOMORPHISM_PROOF]
by
<isomorphism_proof>
```
Helper lemmas for the proof may be generated, but must be inserted in the `[ISOMORPHISM_HELPER_LEMMAS]` section.

## Rules
* **PRESERVE** the section tags, without edits or removals
* Edit **ONLY** the `[ISOMORPHISM_PROOF]` and `[ISOMORPHISM_HELPER_LEMMAS]` sections
* **DO NOT** edit the other sections, especially the two specifications and the isomorphism theorem statement
* **DO NOT** leave any `sorry`, **DO NOT** introduce any axiom
