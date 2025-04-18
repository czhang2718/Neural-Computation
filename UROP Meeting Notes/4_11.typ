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

Goal

- Understand training of NN, playing with para

- increasing x_sum worse than decreasing x_sum
- increasing / decreasing y about the same
- random shuffle works better

Only vary initialization of W2
- all negative init
- all pos init
- random even init
- train longer and larger dataset

L1 vs. L2 vs. no regularization

most prevalent patterns vs. regularization (fix epoch and all other params)

== percent 1s (again)
+ graph of patterns vs. percent 1s
+ Percent of positive values in W2

Why almost same number of positive and negative W2 entries


[Not now] Big question: More data, more features (scaling laws). This not fair to assess models because data ordering/distribution affects it. So let's try trainin on subsets of data, each adding a feature, see how model changes. \~ continuous learning, we worry about forgetting.