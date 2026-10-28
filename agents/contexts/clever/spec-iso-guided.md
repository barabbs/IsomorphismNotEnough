# Specifications Isomorphism

## Task
**SETTING**: You will access a Lean 4 file containing:
1. a natural language specification of a function (formatted as a Python-style docstring, with some example usages)
2. a ground truth problem specification in Lean 4
3. another problem specification in Lean 4 generated from the same NL specification

**OBJECTIVE**: Your objective is to write a formal proof in Lean 4 that both the specifications are equivalent.

**DETAILS**: The formal specifications take in an implementation and program input as parameters and hold true for all possible correct implementations.
Also included in the input for context are zero or more test cases in Lean 4 that the implementation is expected to pass and that your specification should be consistent with.

**NOTE**: Try as hard as possible to prove equivalence, but note that some specifications may be inconsistent or incorrect.
If you believe the proof to be impossible due to a mismatch between specifications, your thesis must be supported by rigorous arguments such as concrete counterexamples.
In this case, try to determine if the two specifications are inconsistent yet both valid interpretations of the NL description and the test cases, or if the mismatch is indeed due to any of the two being erroneous.

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

* The ground truth specification
```
-- [GROUND_TRUTH_SPECIFICATION]

def problem_spec
-- function signature
(impl: <function_signature>)
-- inputs
(<input_name>: <input_type>) :=
  -- spec
  let spec (result : <output_type>) :=
    [<precondition> → <precondition> → ... → ]
    <property> ∧
    <property> ∧                                                                                 
    ... ;
  -- program termination
  ∃ result, impl <input_name> = result ∧
  spec result
```

* The generated specification
```
-- [GENERATED_SPECIFICATION]

def generated_spec
-- function signature
(impl: <function_signature>)
-- inputs
(<input_name>: <input_type>) : Prop :=
  -- spec
  let spec (result : <output_type>) :=
    [<precondition> → <precondition> → ... → ]
    <property> ∧
    <property> ∧                                                                                 
    ... ;
  -- program terminates
  ∃ result, impl <input_name> = result ∧
  -- return value satisfies spec
  spec result
```

* The isomorphism theorem statement
```
-- [ISOMORPHISM_THEOREM]

theorem spec_isomorphism:
∀ impl,
(∀ <input_name>, problem_spec impl <input_name>) ↔
(∀ <input_name>, generated_spec impl <input_name>) :=
```

* The formal test cases (commented out)
```
-- [TEST_CASES]

-- #test implementation <input_1> = <expected_output_1>
-- #test implementation <input_2> = <expected_output_2>
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
