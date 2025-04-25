#import "@preview/ctheorems:1.1.3": *
#show: thmrules

#let theorem = thmplain(
  "theorem",
  "Theorem",
  base_level: 0,
  titlefmt: strong
)

#let definition = thmplain(
  "theorem",
  "Definition",
  base_level: 0,
  titlefmt: strong
)

#let proposition = thmplain(
  "theorem",
  "Proposition",
  base_level: 0,
  titlefmt: strong
)

#let lemma = thmplain(
  "theorem",
  "Lemma",
  base_level: 0,
  titlefmt: strong
)

#let corollary = thmplain(
  "theorem",
  "Corollary",
  base_level: 0,
  titlefmt: strong
)

#let proof = thmproof(
  "proof",
  "Proof"
)

#let sol = thmproof(
  "sol",
  "Solution"
)

#let claim = thmproof(
  "claim",
  "Claim"
)

#set par(
  justify: true,
)


Increasing fraction of yes/no instances
how does distribution of negative / posiitve rows change with input yes/no instances

people have done the min non-zero params, each cluase has one row


(Another) goal:
Embed inputs (x_i -> vector). This will smear inputs onto 
C: W1 = C, weight column i corresponds to feature i
DC: L1 on D, doesn't necessarily corresond to inputs


Each neuron corresonds to a feautre (other rows 0) -> monosemanticity
Basic initial expariemnts
- mono-semantic? (single row per clause)
- do experiments well


$ x -> E x -> (C D) E x $

$W_1 = C D$, do L1 on W1

Then:
- impact of embedding inputs
- seems less likely L1 on D will push to monosemantic




Next steps:
- How does ratio of positive / neg examples in training impact \# positives in $W_2$
4p, 4n, 3p1n, etc. 
- understand current experiments better
    - \# neurons >= \# clauses: monosemantic?
    - \#neurons < \#clauses: poly? does loss go to 0?
- if ^ done, embed inputs x-> Ex, L1 on W1 = C D, then look at resulting C (= W1 \@ D^-1)
    - E = Hadamard matrix
    - E = random boolean matrix
