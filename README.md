# droppedaneuralnet: my solution + agent run notes

This is my solution repo for **dropped a neural network**.

In `human/droppedaneuralnet_noah.ipynb`, I have my direct human solution where I investigated pairing/order signals, ruled out weaker hypotheses, and iterated to a working arrangement.

The other half of this repo is an automated research agent (`agent.py`, `agent_core.py`, `agent_actions.py`, `evaluate.py`) that runs iterative solver experiments, logs outcomes, and keeps trying to improve the permutation.

The narrative below is the agent's process from its full run, where I let it go until I ran out of free API credits. At the end I give an analysis of what could have been done better if I were to do it again. 

---

## TLDR

The agent found a strong early baseline (**iter 2, MSE = 0.52010363**), correctly identified basic structure (48 in-layers, 48 out-layers, 1 last layer), and did a lot of diagnostics.
However, it repeatedly got trapped in two failure modes:

- catastrophic wrong permutations (~664 MSE), and
- stable local-optimum permutations (~0.766 MSE).

It eventually diagnosed multiple implementation/process issues, but never converted that into a final successful solve before credits ran out.

---

## Stage 0: Initial orientation (iter 0)

The agent started correctly by investigating first.  
It read the data and pieces, and got the core dimensions right:

- 97 total layers
- one `(1, 48)` piece (last layer candidate)
- 48 pieces of `(48, 96)` and 48 pieces of `(96, 48)`

So from the start, it understood this was a permutation/reconstruction problem over alternating linear layers.

---

## Stage 1: First solver + first bug + immediate recovery (iter 1-2)

At iter 1 it attempted a baseline solver and immediately hit a practical bug:

- treated loaded `.pth` objects as tensors, but they were `OrderedDict` state dicts.

At iter 2 it fixed that and got MSE of 0.52010363

This was the strongest concrete success in the whole run unfortunately.

---

## Stage 2: Early exploration and tooling friction (iter 3-16)

After the baseline, the agent shifted into feature/statistics analysis:

- weight moments,
- spectral norms,
- structure checks,
- pairing correlations.

There were some runtime/pandas issues, but they were mostly recoverable.

The problem in this stage was direction control, it started generating many analyses without a tight “this directly changes solver behavior” understanding each time.

---

## Stage 3: Strategy churn and repeated hard failures (iter ~17-40)

This is where the run got noisier.

The agent cycled between:

- INVESTIGATE
- THINK
- SOLUTION

with many pivots (hill climbing, annealing variants, structure assumptions), but no sustained improvement beyond 0.52. Some attempts were technically rejected, and a few failed by runtime/API response issues. 

---

## Stage 4: Mid-run conceptual correction (iter ~41-57)

This stage had the agent finally realize that search space was pairing + ordering, not trivial matching.

While a useful realization and central to finding the solution (at least how I did it in my own solution), several difficult barriers remained that the agent was not able to overcome.

---

## Stage 5: Hungarian/SVD/trace push and repeated collapse modes (iter ~58-89)

After the correction, the agent doubled down on assignment-based solvers:

- SVD distance cost matrices,
- trace-based scores,
- Hungarian matching,
- then search wrappers around those.

Two recurring outcomes:

1. catastrophic MSE (~664) from clearly broken constructions, and  
2. non-catastrophic but stuck MSE (~0.7659) from local-optimum assignments.

It also experienced several “empty assistant output” failures, which were non-logic failures but still consumed iterations. I suspect there was an issue with the cerebras api and the specific model I was using.

What it learned correctly:

- trace had signal, but not enough to recover exact true pairing globally via Hungarian.

What was weak:

- too many near-duplicate reruns with slightly different wrapper text,
- not enough hard guardrails to preserve known best state.

---

## Stage 6: Framework debugging and SHA/MSE consistency checks (iter ~90-113)

Late run, it started getting skeptical and audited the solver pipeline itself.

It checked:

- whether best permutation files were being loaded,
- SHA consistency across outputs,
- whether fallback logic actually took effect,
- whether “initialize from best” was real or overwritten.

It found at least one concrete bug class:

- list preallocation / indexing issues in piece loading (`list assignment index out of range` style failure).

However before its full investigation finished, the api credits ran out and the script terminated.

---
## Final thoughts and future improvements

Setting up this research agent, it quickly became a kitchen sink of various ideas I had and I think it suffered for that. If I were to do it again, I would simplify things with a centralized context file that is modified in a controlled way alongside the research log itself, with no other bloat. Moreover, I would make the script evolutionary where if a new script is being written it uses the old one as a template, such that once the interaction layer and boilerplate python is established there won't be repeated errors as it tries to re-invent the wheel every single time. Perhaps adding an overarching orchestration agent, with multiple instances of the sub-investigation agents would have let to better outcomes, with each agent specializing in a different aspect (like one focusing on theoretical decompositions, etc). Anyway, research agents are very exciting, and theres much that can be done!
