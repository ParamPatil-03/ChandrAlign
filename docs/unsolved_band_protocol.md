# Unsolved-illumination band: re-derivation protocol (MATCH-09 / P2-T10)

**Status: FROZEN. Committed before the sweep was written or run.**
Check `git log` on this file against `reports/illumination_sweep.json`: this
document's commit must precede the result's. If any threshold, matcher, azimuth
sample or seed below is edited after results exist, the experiment is void and
must be re-run on new seeds under a new protocol.

Frozen against `origin/main` at `4962176`.

---

## 1. Why this exists

`configs/regimes.yaml` ships:

```yaml
unsolved_illumination:
  min_d_azimuth_deg: 60.0
  evidence: reports/synthetic_matcher_bench.json
  note: >
    All four matchers failed across three seeds above 60 deg of sun azimuth
    difference. ...
```

`matching/regime.py` reads that single number and stamps `expectation:
"unsolved"` on **every** pair whose sun azimuths differ by 60 degrees or more,
so no caller treats the returned transform as supported.

**The band was fitted on a pool that excluded the methods that work.** Its
evidence is a four-matcher benchmark — `sift`, `xfeat`, `aliked-lightglue`,
`eloftr` — across three seeds. All four fail above 60 degrees, so the band
records where *those four* stop, not where the system stops. It is the third
instance of the same defect this project has now found in its own criteria:

| Criterion | Why it could not discriminate |
|---|---|
| "default matcher must beat SIFT on the illumination-shifted pair" | SIFT scores zero at ≥30 deg, so everything passes |
| "MAGSAC beats plain RANSAC at 70% outliers" | every estimator succeeds on scattered outliers, so nothing separates |
| "unsolved above 60 deg" (this one) | every matcher in the pool fails above 60 deg, so the pool fixed the answer |

**The band also contradicts the file it lives in.** Forty lines above it, the
`representation_bands` comment states the measured finding plainly:

> The relationship is NOT monotonic in d_azimuth — 90 deg is the worst case,
> not 180 deg — so a single "more difference -> more invariance" rule is wrong.

`regime.py`'s own module docstring repeats it. A `min_d_azimuth_deg` threshold
is exactly the monotonic rule both of them say is wrong: it necessarily calls
180 degrees at least as hard as 90.

**Consequence:** the selector may be marking solvable pairs as unsupported.
That is not a cosmetic error. `Decision.supported` is what tells a caller
whether the system can handle a pair at all.

## 2. Disclosure: results are already known, and they do not vote

Before this protocol was written, the task that commissioned it supplied
headline figures from `reports/default_matcher_selection.json` (720 runs, seeds
101–110, branch `part2/default-matcher`):

```
lighting_opposite (180 deg)   xoftr 10/10,  minima-loftr 10/10
lighting_+90      (90 deg)    minima-loftr 7/10,  everything else 0/10
```

That is prior knowledge, and pretending otherwise would be worse than naming
it. It is handled the way `docs/default_matcher_protocol.md` §1 handled the
XoFTR table it had already seen:

- every threshold in §7 comes from a source **external** to those figures;
- the decision runs on **fresh seeds 201–210** (§6), which nothing has run;
- the prior figures are a **consistency check with no vote**. If the sweep
  disagrees with them, the sweep is the result and the disagreement is
  reported, not reconciled.

The 7/10 at 90 degrees is the specific hazard: it sits near any plausible
threshold, so a threshold chosen after seeing it would be a threshold chosen to
place it. §7 fixes two thresholds from two pre-existing committed sources and
accepts wherever 7/10 falls.

## 3. Scope

**In scope:** same-modality pairs, differing only in sun azimuth.

**Out of scope, and unchanged by this experiment:** cross-modality routing,
the `extreme_scale` cascade, the `representation_bands` table, the
`d_incidence` and `incidence >= 70` rules, and `default_matcher`. A separate
frozen experiment (MATCH-03/04/05) owns the default and is not yet resolved;
nothing here may move it.

---

## 4. Membership: which matchers can make a band "solved"

**A band is SOLVED if at least one MEMBER matcher meets §5's criterion on it.**
Not "the default solves it". The band is a support and warning layer, not a
matcher: its job is to tell a caller whether the system can handle a pair at
all, and the selector can then route to whichever member handles it.

The two readings give different answers and the difference is the whole point.
At 180 degrees the default (`aliked-lightglue`) is expected to fail while other
members succeed. Under "solved by the default" the band would stay `unsolved`
and callers would keep being told the system cannot do something it can.

**MEMBERSHIP RULE, frozen:** a matcher is a member iff
`chandralign.matching.licence.restriction_reason(name) is None` — licence-clean
under the component gate.

**Source:** this is the same gate `scripts/select_default_matcher.py` applies to
this same candidate pool under `docs/default_matcher_protocol.md` §4.1, which is
committed and frozen. It excludes exactly what the rule is meant to exclude:
non-commercial and share-alike weights (`superpoint`, `superglue`, `r2d2`, and
every compound name containing them), and everything on
`regimes.yaml: benchmark_only_matchers`.

**A recorded caveat, not a loophole.** `licence.shippable()` is a *narrower*
list — the six names declared in `regimes.yaml: shippable_matchers` — and
`xoftr` and `minima-loftr` are not yet on it. They are licence-clean; they are
simply awaiting a promotion decision that MATCH-03/04/05 owns and has not made.
So:

- the report and the config **must name which matcher solved each band**
  (§8), so that a band resting on a not-yet-promoted matcher is visible;
- if that promotion is refused, any band solved only by a refused matcher must
  be re-derived. The config carries this dependency in its comment.

## 5. What counts as "solved" for one matcher on one azimuth

A run SUCCEEDS iff the estimator **accepts** it and the recovered transform is
within **2.0 px** of the known truth, measured as RMSE over a 16×16 grid of
source pixels.

**Source of 2.0 px:** `BAD_PX = 2.0` in `scripts/bench_matchers.py:48` ("above
this, a transform is wrong enough to matter"), committed before this protocol.
It is the same definition `docs/default_matcher_protocol.md` §4.2 uses. Not
chosen here, and not adjustable here.

Ground truth is the exact `H` returned by `chandralign.synth.make_pair`. It
grades runs and is never an input to any matcher.

---

## 6. The sweep, fixed in advance

### 6.1 Matchers

| Matcher | Why it is in |
|---|---|
| `xoftr` | reported to solve 180 deg; cross-modal transformer |
| `minima-loftr` | reported to solve 180 deg and to be the only method scoring above zero at 90 deg |
| `aliked-lightglue` | the configured default; the band must be honest about what the default does |
| `eloftr` | in the original four-matcher pool that produced the 60 deg band |
| `sift-nn` | the classical floor, and the control: it is **expected to fail** above 30 deg. If it does not, the harness is wrong, not SIFT |

All five are licence-clean, so all five are members under §4.

### 6.2 Azimuth differences

**0, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180 degrees.** Twelve samples.

Denser than the existing 0/30/60/90/120/150/180 bands, deliberately and stated
up front: 15-degree steps from 30 to 180 resolve the transition around 60–120
degrees, which is where `representation_bands` places the hard region and where
a boundary has to be drawn. A 30-degree grid cannot locate an edge finer than
30 degrees.

Sun elevation is held at **45 degrees for every sample**, so azimuth difference
is the only variable. `make_pair`'s reference default `sun_ref = (135.0, 45.0)`
is kept; `sun_src = (135.0 + d_az, 45.0)`.

*Note for anyone comparing tables:* `scripts/select_default_matcher.py`'s
`lighting_+30` regime uses `sun_src=(165, 40)`, which moves elevation as well.
Its numbers and these are therefore not directly comparable at 30 degrees, and
neither is wrong — they measure different things.

### 6.3 Everything else held fixed

`out_shape=(512, 512)`, `n_craters=60`, `rot_deg=8.0`, `cross_modal=False`,
`scale=1.0`, `device="cuda"`. Estimation via `estimate.robust.estimate` at
repository defaults. One code path for every matcher; `sift-nn` goes through
`classical.match`, the rest through `adapter.match`, exactly as
`scripts/select_default_matcher.py` does.

### 6.4 Seeds

**Seeds 201–210**, ten consecutive integers, fixed here before any run.

Fresh by inspection of every seed this repository has spent: `0–9`
(`reports/estimator_benchmark.json`), `3/11/29` (the `tiers.low` quality-gate
fit in `scripts/bench_matchers.py`), `101–110` (`docs/default_matcher_protocol.md`),
`301–315` (reserved by `docs/experiment3_protocol.md`). None recurs.

Ten seeds is chosen so that both thresholds in §7 land on integers: 9/10 and
6/10.

### 6.5 Size

5 matchers × 12 azimuths × 10 seeds = **600 runs**, all synthetic.

Feasibility was checked before freezing, at 0 degrees only — a timing and
"do the weights load" check, not an accuracy reading on any contested band.
Warm cost per pair: `sift-nn` 0.09 s, `eloftr` 0.13 s, `aliked-lightglue`
0.20 s, `minima-loftr` 0.24 s, `xoftr` 0.30 s on an RTX 4050.

---

## 7. The band verdicts, and where the numbers come from

For each azimuth sample, let `best = max over member matchers of (successes /
10 seeds)`.

| Verdict | Condition | What it tells a caller |
|---|---|---|
| `solved` | `best >= 9/10` | supported: some member handles this reliably |
| `degraded` | `6/10 <= best < 9/10` | runs, often works, not reliable — check the quality gate |
| `unsolved` | `best < 6/10` | do not read a returned transform as supported |

**Source of 9/10.** `docs/default_matcher_protocol.md` §4.2 sets the success
bar at 90%, justified there as: *"a default is what runs when nothing else is
selected. One core failure in ten on the pairings we actually ship is the most
that can be called a working default."* The same sentence defines what this
band is for. A regime we tell callers is **supported** must work nine times in
ten, for the identical reason.

**Source of 6/10.** The same document, §5 criterion 3: *"Highest STRETCH
coverage — regimes solved on a majority of seeds."* Its STRETCH class is
defined as *"extreme illumination, at or past the azimuth difference our own
router already declares unsolved"* and names `lighting_+90` and
`lighting_opposite` explicitly. That is this experiment's subject matter, and a
majority of seeds is the bar the house already applies to it.

**Why three verdicts and not two.** `regime.Expectation` already has exactly
this vocabulary — `"solved" | "degraded" | "unsolved" | "unknown"` — and
`_expectation_for` already returns `degraded` for the 30–60 degree range today.
This protocol populates an existing four-valued field; it does not invent a
scale. Collapsing to two would force every band that works most of the time
into one of two lies.

**These two thresholds were taken from two pre-existing committed sources
before the sweep was written.** Whatever 7/10 does against them, it does by
arithmetic. It is not adjusted, and §10 forbids adjusting it afterwards.

### 7.1 Turning per-sample verdicts into bands

Adjacent samples sharing a verdict merge into one band, written as
`max_d_azimuth_deg` so the shape matches `representation_bands` in the same
file.

**An azimuth between two samples with different verdicts takes the WORSE of the
two.** Fixed here, before any result: interpolation must never promote an
unmeasured angle. Worse is ordered `solved` < `degraded` < `unsolved`.

`d_azimuth = None` (sun geometry unknown) remains `unknown` and is untouched by
this experiment.

---

## 8. What the report must contain

`reports/illumination_sweep.json`, with `"source": "synthetic"`.

Per azimuth sample, **named attribution is mandatory**: which matcher or
matchers reached the verdict, and the per-seed score for every matcher. A band
marked `solved` with no attribution is useless to the selector and is treated
as a failed run of this protocol, not as a result.

Also recorded per run: matches, inliers, RMSE, accepted, tier, seconds, and any
exception, verbatim and never suppressed.

The default matcher's own per-band score is reported alongside, whether or not
it is the one that solved the band, because a band solved only by a non-default
matcher means the selector must route to reach it.

---

## 9. What this cannot establish

**9.1 It is synthetic.** `synth.make_pair` renders one hillshaded fractal-plus-
craters height field under a Lambertian model the Moon does not obey. Every
number here carries `"source": "synthetic"`. `regime.py`'s existing
`_SYNTHETIC_EVIDENCE` string stays attached to every decision, and this
experiment does not remove it.

**9.2 Both views come from one height field.** Azimuth difference is simulated
by re-lighting the same terrain, so the two images differ in illumination and
in nothing else. Real pairs differ in sensor, resolution, noise, atmosphere-free
but time-separated surface state, and geolocation error as well. A band solved
here is not thereby solved on real data.

**9.3 It does not validate against the real illumination ladder.** The stated
next step for these bands is the real LRO NAC ladder (9 products, incidence
2.2–81.8 deg) named in `configs/regimes.yaml`. This experiment does not do it
and does not discharge it.

**9.4 One elevation.** Everything is measured at 45 degrees solar elevation.
Low-sun behaviour — the single most load-bearing lunar-specific finding in the
research document §6, that SIFT/AKAZE degrade at the poles while SuperGlue-class
methods do not — is a different axis and is not measured here.

**9.5 Azimuth difference is not the whole regime.** `d_incidence` and
`max_incidence` continue to route independently. This experiment re-derives one
band, not the selector.

**9.6 A `solved` verdict is an expectation, not a guarantee.** It says some
member matcher met the bar on synthetic terrain at those seeds. The quality gate
re-decides every real pair from the actual match evidence, and the
`expectation` flag is advisory only. That relationship is unchanged.

**9.7 It cannot promote a matcher.** Whether `xoftr` and `minima-loftr` join
`shippable_matchers` is MATCH-03/04/05's decision. If a band here rests on one
of them and that promotion is refused, the band is void (§4).

---

## 10. Standing commitment

No threshold, matcher, azimuth sample, seed, success definition, membership rule
or verdict boundary in this document may be changed once results exist. If one
is changed, the experiment is void and must be re-run on fresh seeds under a new
frozen document, and the void run reported as void rather than quietly dropped.

This includes, explicitly:

- the 2.0 px success bar;
- the 9/10 and 6/10 verdict boundaries, and the three-verdict structure;
- membership = licence-clean, and the decision that **one** member suffices;
- the twelve azimuth samples and the fixed 45 degree elevation;
- seeds 201–210;
- the worse-of-two rule for unmeasured angles between samples.

The verdicts follow from the rule mechanically. The rule can return the status
quo: if no member matcher clears 6/10 above 60 degrees, the existing band is
confirmed and stays. It can also return a band that is harder in the middle than
at the end, which is what the rest of `configs/regimes.yaml` predicts and what
the current shape cannot express.
