# Default-matcher selection protocol (MATCH-03/04/05)

**Status: FROZEN. Committed before the evaluation was written or run.**
Check `git log` on this file against `reports/default_matcher_selection.json`:
this document's commit must precede the result's. If a threshold below is ever
edited after results exist, the experiment is void and must be re-run on new
seeds.

---

## 1. Why this experiment exists

`configs/regimes.yaml` ships `default_matcher: aliked-lightglue`. It was
accepted against PLAN.md P2-T03's bar:

> *"the chosen default runs on CPU and GPU, returns a conforming MatchSet, and
> beats SIFT on the synthetic illumination-shifted pair."*

**That bar cannot discriminate.** SIFT fails outright — zero successful seeds —
at every sun-azimuth difference of 30 degrees or more. Beating a method that
scores zero is automatic, so the bar was passed by a candidate that was never
compared against any alternative capable of passing it too.

This does **not** mean the current default is wrong. It means nothing in the
repository establishes that it is right. This protocol decides the question.

**What this experiment must not become:** a search for a justification to ship
XoFTR. An earlier 3-seed benchmark (`reports/rift_benchmark.json`, produced for
MATCH-06) suggested XoFTR outperforms the current default. That table was
inspected before this protocol was written, which is exactly why:

- every threshold in section 4 is derived from a source **external** to that
  table — the problem statement, the research document, or a constant already
  committed to this repository before this experiment began;
- the decision is made on **fresh seeds** (section 5) that no one has looked at;
- the old 3-seed table is demoted to a **consistency check** and has no vote.

---

## 2. Candidate pool (frozen)

The six matchers currently declared shippable, plus the three cross-modal
candidates, so that methods confined to the cross-modal path are finally
measured on the same-modality path they were never tested against.

| Candidate | Source | Currently |
|---|---|---|
| `aliked-lightglue` | vismatch | **the incumbent default** |
| `disk-lightglue` | vismatch | shippable |
| `eloftr` | vismatch | shippable |
| `xfeat` | vismatch | shippable, CPU path |
| `sift-lightglue` | vismatch | shippable |
| `sift-nn` | OpenCV | the classical floor (MATCH-01) |
| `xoftr` | vismatch | cross-modal candidate only |
| `minima-loftr` | vismatch | cross-modal candidate only |
| `matchanything-eloftr` | vismatch | cross-modal candidate only |

**RIFT2 is excluded as a default candidate.** MATCH-06 established its role as
an independent cross-check, and a method used to check the default cannot also
be the default without destroying that independence.

`sift-nn` is included as the floor. It is not expected to qualify; it is there
so the eligibility rules are demonstrably capable of rejecting something.

---

## 3. Evaluation, identical for every candidate

Nothing gets a private path. One code path, one generator, one gate.

| | |
|---|---|
| Regimes | the same 8 as `scripts/bench_rift.py` |
| Seeds | **10 fresh seeds: 101–110** (section 5) |
| Generator | `chandralign.synth.make_pair`, unchanged |
| Estimator | `estimate.robust.estimate`, repository defaults |
| Quality gate | `evaluate.quality.assess`, repository defaults |
| Error | RMSE over a 16x16 grid against the known transform |
| Timing | wall clock around the match call only, same device |

Ground truth grades runs. It is never an input to any candidate.

### Regime classes

Defined here **by what the product must do**, not by what any candidate scored.

**CORE** — every one corresponds to a real pairing this pipeline produces:

| Regime | Why it is core |
|---|---|
| `same_lighting` | the ordinary case |
| `lighting_+30` | routine between-orbit sun drift |
| `cross_modal` | OHRC-to-IIRS is the problem statement's own hardest named pairing |
| `resolution_2x` | TMC-2 to SELENE TC, a real shipped route |
| `low_texture` | lunar mare; failure mode #1 |

**STRETCH** — extreme illumination, at or past the azimuth difference our own
router already declares `unsolved`:

`lighting_+90`, `lighting_opposite`, `cross_modal+opposite`

Stretch regimes **cannot make a candidate eligible or ineligible**. They are
scored, and they break ties. A regime the product does not claim to support must
not decide what ships — otherwise the winner is whoever fails most gracefully at
something we already tell users we cannot do.

---

## 4. Eligibility (a gate, not a score)

Every rule has a source outside this experiment's data. Fail any one and the
candidate is out, however well it scores elsewhere.

### 4.1 Licence and shippability — absolute

Must pass `matching/licence.py` with `ship_mode: true`, and its weights must be
redistributable. **Source:** CHECK-10, failure mode #14. Verified against each
model's actual upstream licence, not inferred from its name. A non-commercial
licence is disqualifying at any score.

### 4.2 Success rate on CORE regimes — at least 90%

A run succeeds when the estimator **accepts** it and the transform is within
**2.0 px** of truth.

**Source of 2.0 px:** the accuracy bar already used throughout this repository
before this experiment (`BAD_PX` in `scripts/bench_rift.py`,
`scripts/bench_estimators.py`). Not chosen here.

**Source of 90%:** a default is what runs when nothing else is selected. One
core failure in ten on the pairings we actually ship is the most that can be
called a working default.

### 4.3 False-confidence rate — at most 2% of all runs

A false confidence is a run the quality gate **accepted** while the transform
was more than 2.0 px wrong.

**Source:** failure modes #12 and #19. This is the failure our control gates
exist for, and MATCH-06 measured that they catch only 2 of 12 such cases,
because consistent errors are invisible to any check reading one method's own
output. A default that lies confidently is worse than one that fails loudly, so
the tolerance is near-zero: with 80 runs per candidate, **at most 1**.

### 4.4 Runtime — under 5 s per 512 px pair

**Source:** research document section 38, the 3-minute live demo. A matcher that
cannot complete a tile in seconds cannot be the default regardless of accuracy.

---

## 5. Selection among the eligible (priority order, fixed here)

Applied in order; each step only breaks ties left by the one above.

1. **Lowest false-confidence rate.** A confident wrong answer is the worst
   outcome this pipeline can produce.
2. **Lowest median error on CORE regimes.**
3. **Highest STRETCH coverage** — regimes solved on a majority of seeds.
4. **Lowest median runtime.**

**Not** *"XoFTR wins if it beats ALIKED."* The rule admits the possibility that
no candidate qualifies, that the incumbent qualifies and wins, or that several
qualify and the incumbent loses on criterion 1.

**Seeds 101–110** were chosen as the ten integers following 100, before any run.
The previous benchmark used 3, 11 and 29; none recurs.

---

## 6. Real-data check on the leading candidate

Synthetic evidence alone promotes nothing. The leader is then run on the real
pairs already registered in this repository:

`TMC-2 -> SELENE TC` · `OHRC -> TMC-2` · `OHRC -> SELENE TC`

**This is a behaviour and consistency check, not a second accuracy table.**
These pairs have no exact ground truth — TMC-2 to TC is scored on SYSTEM corners
only — so the questions are: does it produce a result at all, does it pass the
control gates, and does it agree with the registration already established? A
candidate that wins on synthetic data and misbehaves here is not promoted.

---

## 7. If the default changes

Rerun **only** results the old default directly produced. A result that named
its matcher explicitly — MIND, RIFT2, SIFT, or any explicit `model_name` — does
not depend on the default and is not re-run. The dependency list is produced by
inspecting the call sites, not by assumption.

---

## 8. Standing commitment

No threshold, eligibility rule, candidate or priority order in this document may
be changed once results exist. The recommendation follows from the rule
mechanically. **The decision to change the shipped default is the team's, not
this experiment's** — the experiment produces a recommendation and its evidence.
