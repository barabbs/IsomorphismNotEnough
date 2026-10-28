import Imports.AllImports

-- start_def problem_details
/--
function_signature: "def find_zero(xs: list)"
docstring: |
    xs are coefficients of a polynomial.
    find_zero find x such that poly(x) = 0.
    find_zero returns only only zero point, even if there are many.
    Moreover, find_zero only takes list xs having even number of coefficients
    and largest non zero coefficient as it guarantees a solution.
    Note(George): This problem has been modified from the original HumanEval spec because of Real is not a computable type, but a zero does not necessarily exist over the rationals.
test_cases:
  - input: [1, 2]
    output: -0.5
  - input: [-6, 11, -6, 1]
    output: 3.0
-/
-- end_def problem_details

-- start_def problem_spec
def problem_spec
-- function signature
(implementation: List Rat → Rat)
-- inputs
(xs: List Rat) :=
-- spec
let spec (result: Rat) :=
  let eps := (1: Rat) / 1000000;
  xs.length ≥ 1 → xs.length % 2 = 0 →
  ∀ poly : Polynomial Rat,
    poly.degree = some (xs.length - 1) →
    (∀ i, i ≤ xs.length - 1 → poly.coeff i = xs[i]!) →
    |poly.eval result| ≤ eps;
-- program termination
∃ result,
  implementation xs = result ∧
  spec result
-- end_def problem_spec

-- start_def generated_spec
def generated_spec
-- function signature
(implementation: List Rat → Rat)
-- inputs
(xs: List Rat) : Prop :=
-- end_def generated_spec
-- start_def generated_spec_body
sorry
-- end_def generated_spec_body

-- start_def spec_isomorphism
theorem spec_isomorphism:
∀ implementation,
(∀ xs, problem_spec implementation xs) ↔
(∀ xs, generated_spec implementation xs) :=
-- end_def spec_isomorphism
-- start_def spec_isomorphism_proof
sorry
-- end_def spec_isomorphism_proof

-- start_def implementation_signature
def implementation (xs: List Rat) : Rat :=
-- end_def implementation_signature
-- start_def implementation
let rec poly (xs: List Rat) (x: Rat) := xs.reverse.foldl (λ acc a => acc * x + a) 0;
let rec eps := (1: Rat) / 1000000;
let lc := |xs.getLast!|;
let initBound : Rat := if lc = 0 then 1
  else 1 + xs.dropLast.foldl (fun acc a => acc + |a|) 0 / lc;
let rec findBound (xs: List Rat) (b: Rat) (fuel: Nat) : Rat :=
  match fuel with
  | 0 => b
  | fuel' + 1 =>
    if poly xs b * poly xs (-b) ≤ 0 then b
    else findBound xs (b * 2) fuel';
let rec bisect (xs: List Rat) (lo hi: Rat) (fuel: Nat) : Rat :=
  match fuel with
  | 0 => (lo + hi) / 2
  | fuel' + 1 =>
    let mid := (lo + hi) / 2;
    if |poly xs mid| ≤ eps then mid
    else if poly xs lo * poly xs mid ≤ 0 then bisect xs lo mid fuel'
    else bisect xs mid hi fuel';
let bound := findBound xs initBound 100;
let coeffBits := xs.foldl (fun acc a => acc + a.num.natAbs.size + a.den.size) 0;
let fuel := 1000000 + coeffBits * (xs.length + 1) * 40;
bisect xs (-bound) bound fuel
-- end_def implementation

-- Uncomment the following test cases after implementing the function
-- start_def test_cases
#test |implementation [1, 2] - (-1/2)| ≤ 1/1000000
#test |implementation [-6, 11, -6, 1] - 1| ≤ 1/1000000 ∨ |implementation [-6, 11, -6, 1] - 2| ≤ 1/1000000 ∨ |implementation [-6, 11, -6, 1] - 3| ≤ 1/1000000
-- end_def test_cases

-- start_def correctness_definition
theorem correctness
(xs: List Rat)
: problem_spec implementation xs
:=
-- end_def correctness_definition
-- start_def correctness_proof
by
unfold problem_spec
let result := implementation xs
use result
simp [result]
intros h1 h2 poly h3 h4
repeat sorry
-- end_def correctness_proof
