# Specification Generation

## Task
**SETTING**: You will access a Lean 4 file containing:
1. a natural language specification of a function (formatted as a Python-style docstring, with some example usages)

**OBJECTIVE**: Your objective is to write a proposition in Lean 4 formalizing the specification.

**DETAILS**: The proposition takes in an implementation and program input as parameters.
The proposition should hold true for all possible inputs in the domain, which means any preconditions should be mentioned in the specification
to ensure that those cases are handled appropriately and hence the proposition is always valid if the implementation is correct.
Also included in the input for context are zero or more test cases in Lean 4 that the implementation is expected to pass and that your specification should be consistent with.

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
* The specification signature
```
-- [SPECIFICATION_SIGNATURE]

def generated_spec
-- function signature
(impl: <function_signature>)
-- inputs
(<input_name>: <input_type>) : Prop :=
```

* The formal test cases (commented out)
```
-- [TEST_CASES]

-- #test implementation <input_1> = <expected_output_1>
-- #test implementation <input_2> = <expected_output_2>
```

An optional `[HELPER_DEFINITIONS]` section will contain definitions useful to the problem.

### Output
Insert the generated specification in the `[SPECIFICATION_BODY]` section, and follow this format:
```
-- [SPECIFICATION_BODY]
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

`<precondition>`: optional domain guards that restrict the valid input domain (as explicitly stated in the description or naturally implied by the described behavior).
  Not to be included if the function is well-defined and expected to correctly handle any input of type `<input_type>`

`<property>`: relational properties specified for the output, can be a mix of:
  - universally holding properties `<universal_property>`
  - case-specific properties expressed with constructs such as `(<case> → <case_property>)`, `(<case> ↔ <case_property>)`, `if/then/else`, `match` and `∨` case-split
    (e.g. if different input-case behaviors are described, or if a given property only applies to qualifying output elements)

## Guidelines

Be sure to adhere to the following guidelines:

* **Specification structure: _existential-result_ and _preconditions_**

  Strictly follow the provided structure for the specification:
  - The **existential pattern** naturally expresses that the _implementation terminates_ and returns a value _satisfying the spec_.
  - Encode **input constraints** stated or implied by the NL specification as **implication guards** inside the `spec` body. This makes the spec _vacuously true_ for inputs _outside the valid domain_, which is the intended behavior.
  - Indeed, be **permissive** on _edge cases_ and _degenerate inputs_: the implementation may return anything for invalid inputs.


* **State properties _relationally_, not _functionally_**
    
  The specification must characterize the result by the **properties** it must satisfy, **NOT** by the **exact computation** of the result (e.g. by asserting the exact result like `impl args = <computed_value>`).
  This avoids over-constraining specs that fix arbitrary implementations details and reject different yet correct implementations.
  See some examples of common patterns below.

    | Pattern         | Relational (GOOD)                                                        | Functional (BAD)               |
    |-----------------|--------------------------------------------------------------------------|--------------------------------|
    | **List result** | membership + count/permutation equality (multiplicity preservation)      | `= list.map ... `              |
    | **Filtering**   | bi-directional membership + count preservation + containment constraints | `= list.filter ...`            |
    | **Sorting**     | `List.Sorted` + permutation/count-based equality                         | `= list.mergeSort ...`         |
    | **Searching**   | condition satisfaction + optimality of found element                     | `= list.find? ...`             |
    | **String ops**  | char-level properties like `∀ i, i < result.length → ...`                | `result.data = s.data.map ...` |
    | **Numeric**     | use defining properties (divisibility / modular arithmetic)              | `result = <formula>`           |


* **Recursive/inductive specifications**                                                                        
                                                                                                                  
  When the function has a natural recursive decomposition, the spec may characterize the result **inductively** by relating it to `impl` applied to a reduced input.
  Structure as case-split property conjuncts:
  - _base case_: `(<base_condition> → result = <base_value>)`
  - _recursive step_: `(<step_condition> → <relation between result and impl(<reduced_input>)>)`


* **Lean 4 _syntax_ recommendations**
  - You may define helper definitions `let <optional_helper> := <definition>;` both before and inside the `spec` body, as needed
  - Prefer `get!` with `Nat` indices preceded by explicit bounds (`i < list.length →`) over `get?`/`Fin`-based access
  - Watch operator precedence: `∧` binds tighter than `→` and `↔`; when mixing these, use explicit parentheses
  - When the return type is `Bool`, use `result = true ↔ <condition>` or `if result then <P> else ¬<P>` inside the spec predicate, guarded by any necessary preconditions
  

## Rules
* **PRESERVE** the section tags, without edits or removals
* Edit **ONLY** the `[SPECIFICATION_BODY]` section, and follow the format and guidelines closely
* **DO NOT** edit the other sections, especially the specification signature
* **DO NOT** leave any `sorry`, **DO NOT** introduce any axiom
