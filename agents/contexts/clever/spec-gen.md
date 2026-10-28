# Specification Generation

## Task
**SETTING**: You will access a Lean 4 file containing:
1. a natural language specification of a function (formatted as a Python-style docstring)

**OBJECTIVE**: Your objective is to write a proposition in Lean 4 formalizing the specification.

**DETAILS**: The proposition takes in an implementation and program input as parameters.
The proposition should hold true for all possible inputs in the domain, which means any preconditions should be mentioned in the specification
to ensure that those cases are handled appropriately and hence the proposition is always valid if the implementation is correct.

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
* The specification signature
```
-- [SPECIFICATION_SIGNATURE]

def generated_spec
-- function signature
(impl: <function_signature>)
-- inputs
(<input_name>: <input_type>) : Prop :=
```

An optional `[HELPER_DEFINITIONS]` section will contain definitions useful to the problem.

### Output
Insert the generated specification in the `[SPECIFICATION_BODY]` section, and follow this format:
```
-- [SPECIFICATION_BODY]
∀ (x : <input_type>), <precondition> → <postcondition>
```

## Rules
* **PRESERVE** the section tags, without edits or removals
* Edit **ONLY** the `[SPECIFICATION_BODY]` section, and follow the format closely
* **DO NOT** edit the other sections, especially the specification signature
* **DO NOT** leave any `sorry`, **DO NOT** introduce any axiom
