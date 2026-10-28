# Proof Generation

## Task
**SETTING**: You will access a Lean 4 file containing:
1. a natural language specification of a function (formatted as a Python-style docstring, with some example usages)
2. a corresponding formal specification in Lean 4
3. a correct function implementation that satisfies the given specifications

**OBJECTIVE**: Your objective is to write a formal proof in Lean 4 that the function implementation is correct and satisfies the formal specification.

**DETAILS**: The formal specification takes in an implementation and program input as parameters and holds true for all possible correct implementations.
Also included in the input are zero or more test cases in Lean 4 that follow the specification and that the implementation passes.

**NOTE**: Try as hard as possible to prove correctness, but note that some implementations or specifications may be buggy or incorrect.
If you believe the proof to be impossible due to a mismatch between implementation and specification, your thesis must be supported by rigorous arguments such as concrete counterexamples.

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

>>> <example_usage>
<expected_output>
...
"""
-/
```

* The formal specification
```
-- [FORMAL_SPECIFICATION]

def problem_spec
-- function signature
(impl: <function_signature>)
-- inputs
(<input_name>: <input_type>) :=
<formal_specification>
```

* The implementation
```
-- [IMPLEMENTATION]

def implementation (<input_name>: <input_type>) : <output_type> :=
<implementation>
```

* The test cases
```
-- [TEST_CASES]

#test implementation <input_1> = <expected_output_1>
#test implementation <input_2> = <expected_output_2>
```

* The correctness theorem statement
```
-- [CORRECTNESS_THEOREM]

theorem correctness
(<input_name>: <input_type>)
: problem_spec implementation <input_name>
:=
```
An optional `[HELPER_DEFINITIONS]` section will contain definitions useful to the problem.

### Output
Insert the generated proof in the `[CORRECTNESS_PROOF]` section:
```
-- [CORRECTNESS_PROOF]
by
<correctness_proof>
```
Helper lemmas for the proof may be generated, but must be inserted in the `[CORRECTNESS_HELPER_LEMMAS]` section.

## Rules
* **PRESERVE** the section tags, without edits or removals
* Edit **ONLY** the `[CORRECTNESS_PROOF]` and `[CORRECTNESS_HELPER_LEMMAS]` sections
* **DO NOT** edit the other sections, especially the specification, the implementation and the correctness theorem statement
* **DO NOT** leave any `sorry`, **DO NOT** introduce any axiom
