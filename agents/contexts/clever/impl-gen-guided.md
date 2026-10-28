# Implementation Generation

## Task
**SETTING**: You will access a Lean 4 file containing:
1. a natural language specification of a function (formatted as a Python-style docstring, with some example usages)
2. a corresponding formal specification in Lean 4

**OBJECTIVE**: Your objective is to write a formal function implementation in Lean 4 that matches the natural language and formal specifications in the input.

**DETAILS**: The formal specification takes in an implementation and program input as parameters and holds true for all possible correct implementations.
A signature is provided for your implementation. Also included in the input are zero or more test cases in Lean 4 that follow the specification and that your implementation should pass.

**NOTE**: Try as hard as possible to generate a faithful implementation, but note that some specifications may be buggy or incorrect.
If you believe the implementation to be impossible due to an erroneous specification, your thesis must be supported by rigorous arguments such as concrete counterexamples.

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

* The implementation signature
```
-- [IMPLEMENTATION_SIGNATURE]

def implementation (<input_name>: <input_type>) : <output_type> :=
```

* The test cases (commented out)
```
-- [TEST_CASES]

-- Uncomment the #test lines below after completing the implementation
-- #test implementation <input_1> = <expected_output_1>
-- #test implementation <input_2> = <expected_output_2>
```

An optional `[HELPER_DEFINITIONS]` section will contain definitions useful to the problem.

### Output
Insert the generated implementation in the `[IMPLEMENTATION]` section:
```
-- [IMPLEMENTATION]
<implementation>
```
Upon completing the implementation, uncomment the `-- #test` lines in the `[TEST_CASES]` section (by removing the leading `-- `), to test the correctness of the implementation.

## Rules
* **PRESERVE** the section tags, without edits or removals
* Edit **ONLY** the `[IMPLEMENTATION]` section, and the `[TEST_CASES]` section only to uncomment the test cases
* **DO NOT** edit the other sections, especially the specification and the implementation signature
* **DO NOT** leave any `sorry`, **DO NOT** introduce any axiom
