# Thesis Defense — Q&A Preparation

A long but exhaustive set of questions an evaluation committee is likely
to ask, with prepared answers. The aim is to demonstrate **complete
mastery of the system** — every design decision, what alternatives
existed, what tradeoffs each one made, and how each piece connects to
the thesis claims.

It is OK to say "I don't know" about things outside the system (e.g.
"will this work on equities" — we never tried), but it is NOT OK to be
vague about anything inside the system. The 70+ questions below
exercise every corner of the design.

---

## Section 1 — Motivation and framing

### Q1. In one sentence, what is your thesis about?

It's about whether a hidden state estimator (a Kalman filter) coupled
to an online learning controller (Hedge) can make execution decisions
on real BTCUSDT order book data that are systematically better than
naive baselines, and what the structural limits of that approach are.

### Q2. Why is this problem interesting?

There are two reasons. First, every execution desk in the world solves
some version of "given a buy order I have to fill, when and how do I
hit the book?" The state-of-the-art is mostly Almgren–Chriss style
optimal control with a model of market impact, but those models assume
a stochastic process for the price and ignore actual order-flow state.
Second, on a research level: it's a clean testbed for an old question
in online learning — *does conditioning a no-regret algorithm on more
state monotonically improve performance?* I show it does not, and I
isolate the mechanism (the toxicity barrier).

### Q3. Why crypto and not equities?

Three practical reasons. (1) Binance's full L2 data is free and public
via WebSocket — equity L2 data is licensed and expensive. (2) Crypto
markets run 24/7, so I can capture overnight sessions without market
hours blackouts. (3) BTCUSDT has a tick size of $0.01 against a price
of $77,000, so the spread is essentially always one tick — that
removes a confounding variable and lets me isolate the effect of the
*decision* (WAIT/PASSIVE/AGGRESSIVE) from the effect of *spread
selection*.

### Q4. Why force the agent to buy 1 BTC every tick? That's not realistic.

You're right that nobody actually trades that way. The reason it's
forced is to **isolate the execution problem from the trading
problem**. If the agent had the option to "not buy," then the answer
to every "should I act" question would just collapse to "predict the
next return; act if positive." That makes the problem about *signal*,
not about *execution mechanics*. By forcing the action, I'm asking a
narrower question: "given that you ARE going to buy, what's the best
way to do it?" That maps cleanly to a real-world scenario: a desk
that has to fill a customer's mandated buy order over a fixed window,
which is the exact problem Almgren–Chriss solves analytically.

### Q5. The components are not new. What is new?

Two specific contributions. (1) **The empirical demonstration on real
order book data** that 1D pressure-bucketed Hedge is robustly better
than marginal (10/10 seeds, |t|=64) — that's a real, reproducible
result on real microstructure, with a clean sanity-checked pipeline.
(2) **The toxicity barrier**: I tested four different second-axes
(Kalman regime, vol_delta, ofi_window, spread_delta) and they all
fail by the *same* mechanism — they detect fill-easy moments and the
policy rebalances toward PASSIVE, but PASSIVE fills in this market
are systematically toxic. That mechanistic finding has not (to my
knowledge) been published; it implies that adding state to a
bucketed Hedge can not break the spread-vs-toxicity Pareto frontier
on this kind of data, and that the way out is to enrich the *action
space* rather than the state space.

### Q6. Why is this "research" and not "engineering"?

The engineering side is the pipeline: book builder, feature pipeline,
self-heal, A/B harness. That part is real work but it isn't the
contribution. The research side is the *experimental claim*: the
toxicity barrier is a falsifiable hypothesis that I tested by
designing four variants meant to escape it (each on independent
sessions, paired by seed, at a fixed λ, with a pre-registered
significance threshold) — and the data showed they all fall on the
same Pareto frontier. The contribution is the *evidence*, not the
implementation.

### Q7. Why this λ=0.1 specifically? What if you'd picked another?

λ was set in §9 of `sanity.txt` by matching the standard deviations
of the two cost components on real data:
σ(slippage)/σ(adverse) ≈ 0.0107 on the clean Apr-21 window, and
0.5 was the original default. I chose 0.1 because at that value the
expected-loss ranking for the three actions is non-degenerate:
WAIT and AGGRESSIVE are nearly tied (0.040 vs 0.044) and PASSIVE
is penalized by toxicity. That's the regime where Hedge has to
*actually use state information* to discriminate; at λ=0 PASSIVE
trivially wins on slippage, and at λ ≥ 0.5 AGGRESSIVE trivially
wins on toxicity. I sweep all six other λ values in the analysis
section so the reader can see the result is qualitatively the same
across the range that matters.

---

## Section 2 — Architectural choices

### Q8. Why decompose the system into Kalman + Hedge instead of using deep RL?

Three reasons. **(1) Data scarcity.** A clean overnight session is
~38–76k ticks. Deep RL needs orders of magnitude more data to fit a
neural policy without overfitting. **(2) Diagnosability.** When
something goes wrong I want to know whether the inference layer
(Kalman) or the decision layer (Hedge) is at fault. With closed-form
math at every step I can print the Kalman gain, the residual
innovation, the bucket index, the per-action probabilities, the
slippage and adverse — every number has a domain meaning. **(3)
Regret guarantees.** Hedge gives me an O(√(T log K)) regret bound
without any distributional assumption on the losses, which is
appropriate for non-stationary markets. Deep RL gives no such
guarantee; whatever policy comes out is a black box that I can't
formally bound.

### Q9. Why discretize the state into buckets instead of using a continuous bandit (LinUCB, Thompson)?

Bucketing has three things going for it in this problem. **(1) It
preserves Hedge's regret bound per cell** — each bucket runs an
independent Hedge with its own no-regret guarantee. **(2) It's
interpretable.** I can show the committee a plot with six bars, one
per pressure bucket, and explain that "when sellers dominate the
policy waits, when buyers dominate it crosses." A linear bandit
gives a coefficient vector that's much harder to read out as a
policy. **(3) It captures non-linearity for free.** The optimal
action as a function of pressure is plausibly *not* linear (heavy
buy → AGG, heavy sell → WAIT, neutral → mixed). A linear contextual
bandit assumes the expected loss is linear in the features; that
assumption is wrong here. Bucketing handles the nonlinearity by
discretization.

### Q10. Why not contextual bandits like LinUCB?

I considered it. The blocker is the linearity assumption — LinUCB
assumes the expected loss for each action is a linear function of
the context. For our problem the relationship is plausibly
non-monotonic in pressure (peak in the neutral cells, low at the
extremes), which violates that assumption. A piecewise-linear
extension would work but reduces back to bucketing. The other
issue is that LinUCB's confidence-set machinery introduces tuning
parameters (the exploration constant) that don't have a clean
domain interpretation in this problem. Hedge has only η, and η has
a direct interpretation: how aggressively to react to a single tick's
loss.

### Q11. Why not LSTM / Transformer for the state estimator?

Same reason as Q8: data scarcity and diagnosability. A Kalman filter
has 4 + 6 + 9 + 4 = 23 parameters (F, H, Q, R). I can read every one
of them off `configs/kalman.yaml` and explain what each does. A
small LSTM has thousands. With 38k ticks per session I cannot fit
those parameters reliably without leaving training data on the
table for testing — and I would get no theoretical guarantees on
the resulting estimator. The Kalman filter is also the *correct*
choice under its assumptions (Gaussian linear-Gaussian dynamics):
it is the optimal MMSE estimator. If the assumptions held perfectly
nothing could beat it.

### Q12. Are the Gaussian-linear assumptions actually appropriate?

Approximately. Real returns have fat tails, real OFI has bursts —
neither is strictly Gaussian. But for the specific use of "estimate
a slow latent direction (pressure) and a slow latent volatility
(regime) from three observed features," the linear Kalman filter
recovers a sensible signal. I verified this empirically: pressure
estimates fall in [-1.3, +2.1] across all sessions with std ≈ 0.29,
which makes the bucket edges [-0.5, -0.2, 0, 0.2, 0.5] reasonable.
I would not use this filter to forecast tail events; I use it to
extract a slow-moving conditioning variable for a downstream
discrete decision, which is what it's good at.

### Q13. Why two state dimensions specifically?

Pressure and regime map directly to Bouchaud's transient + permanent
market impact decomposition. Adding more dimensions has two costs:
(1) more covariance entries to calibrate with limited data, and
(2) cells in any future bucketing thin out as N^d. I tested adding
the regime axis as a second bucketing dimension (2D) and it lost
to 1D-pressure-only by 6% with |t|=18 across two sessions. So
empirically, "more state" does not help here.

---

## Section 3 — Kalman filter specifics

### Q14. Walk us through F. Why those decay rates?

`F = diag(0.9, 0.95)` says pressure decays by 10% per tick toward
zero, regime by 5% per tick. The decays correspond to "characteristic
times" of about 10 ticks for pressure and 20 ticks for regime, which
at our 1 Hz sampling means ~10 seconds and ~20 seconds. The
asymmetry encodes a domain belief: directional pressure mean-reverts
faster than volatility regime. We didn't tune these aggressively;
they were set by hand and the system is robust to changes within ±
0.05 (verified by perturbation runs).

### Q15. Why is F diagonal? Should pressure and regime affect each other?

Diagonal means the two state dimensions are *dynamically independent*
— they don't influence each other's evolution. We allow them to be
estimated from overlapping observations through H, but their dynamics
don't couple. Adding cross-terms (e.g. F[0,1] ≠ 0 to model "high
volatility makes pressure decay faster") would require fitting a
parameter for which we have no clean identification strategy. The
domain belief that supports the diagonal choice is that direction
and turbulence are *relatively independent* properties of a market
regime — you can have a calm trending market or a turbulent
mean-reverting one.

### Q16. Walk us through H. Why those specific values?

`H = [[1,0],[1,0],[0,1]]`. Three rows, three observations:
`depth_imbalance`, `ofi_l1`, `vol_30s`. Rows 0 and 1 each have
[1,0] meaning depth_imbalance and ofi_l1 both observe pressure
directly with unit gain. Row 2 has [0,1] meaning vol_30s observes
regime directly with unit gain. The choice is semantic:
depth_imbalance and ofi_l1 are both directional features
(positive = buy pressure), so they map to the same hidden state.
vol_30s is a magnitude (always non-negative once standardized
becomes a turbulence indicator), so it maps to the regime state.

### Q17. Why is H static? Real measurement matrices can be time-varying.

Because the relationship between features and hidden states doesn't
change over time in this model. Volume is volume, OFI is OFI — their
*scaling* (R) might evolve with regime, and their *bias* might shift
(which is why we recompute obs_center on each session), but the
*structural* mapping is fixed. A time-varying H would introduce
identifiability issues with no clear domain reason.

### Q18. Walk us through R cell by cell.

`R = diag(0.4486, 2.2878, 1.0)`. The diagonal entries are the
empirical variances of the three observation features measured on
the clean post-§5 window of 38,973 ticks. R[0,0]=0.4486 is the
variance of `depth_imbalance` (raw scale, in (-1,+1)). R[1,1]=2.2878
is the variance of `ofi_l1` (raw scale, large positive/negative
swings). R[2,2]=1.0 is *by construction* — `vol_30s` is standardized
upstream by `obs_scale[2] = 1/σ` so the input to the filter has
unit variance, which means its noise has unit variance too. The
off-diagonal entries are zero because we model the three observations
as having *independent* noise (no cross-correlation). The empirical
correlations are small but non-zero; we don't model them because the
gain in the gain matrix is robust to small mis-specifications and
adding them invites overfitting.

### Q19. The standardization constants for vol_30s came from one window. What if the market changes?

Two answers. (1) Empirically, recomputing obs_center/obs_scale on
the union of multiple sessions changed the regime distribution by
< 5% — so the calibration is stable across our four sessions.
(2) If the market's volatility character changed materially the
regime estimates would drift, but the *bucketing* on pressure
(not regime) is what does the work in the production policy. The
regime axis is decorative for the 1D winner. So changes in vol_30s
calibration would not break the headline result.

### Q20. Why are the initial conditions x₀=0, P₀=I?

x₀=0 because at session start we have no directional prior; pressure
estimates above and below zero are equally likely. P₀=I encodes a
*wide* initial uncertainty (variance 1 on each component) so that
the first few observations get high Kalman gain and the filter
converges quickly. We could in principle warm-start from the previous
session's posterior, but each session is hours apart and we treat
them as independent (and cross-session learning on the bandit side
would also need to be designed, which is an open question).

### Q21. Why not Extended or Unscented Kalman?

The dynamics and observations are linear-Gaussian by construction.
EKF and UKF are for nonlinear systems. Using them here would just
be over-engineering — they reduce to the standard Kalman update
in the linear case. If we needed to model a non-linear feature
(e.g. a return-magnitude observation that maps to vol via x²), we
would, but at the design level we stayed linear deliberately.

### Q22. Why didn't you use a particle filter?

Particle filters are needed when the state distribution is
non-Gaussian (multi-modal, heavy-tailed, etc.) and the measurement
is non-linear. Here both dynamics and observations are
linear-Gaussian, the posterior is exactly Gaussian, and the Kalman
filter computes its mean and covariance in closed form. A particle
filter would give the same answer plus Monte Carlo noise plus 1000×
the compute cost. There's no benefit.

---

## Section 4 — Hedge / bandit policy

### Q23. State the regret bound for Hedge.

For the standard Hedge algorithm with K actions and a learning rate
η = √(8 log K / T), the cumulative regret against the best fixed
action in hindsight satisfies
   R_T ≤ √(T log K / 2)
i.e. O(√(T log K)). For our problem K=3 actions and T ranges from
38,000 to 76,000 ticks per session, so the regret is bounded by
roughly √(75000 × log 3) ≈ 290 in *units of loss*. Per-tick that's
0.004 — comparable to a typical per-tick slippage. The bound is
tight in the worst case but in practice is much better.

### Q24. Why this specific η = 0.1?

η controls how aggressively a single tick's loss shifts the weights.
The optimal η for the regret bound is √(8 log K / T) ≈ 0.012 for
T=75000, K=3. We set η=0.1 deliberately *higher* than the
regret-optimal value because we care about *adapting* to
non-stationary markets, not just bounding worst-case regret. A
higher η makes the policy responsive to recent losses at the cost
of more variance. We didn't tune η aggressively; perturbing it to
0.05 or 0.2 changes the result by <2% on the headline metric.

### Q25. Why bandit feedback (only the chosen action's loss) and not full information?

Because that's what the simulator can provide *honestly*. If we
always observed all three actions' losses we would be doing
counterfactual analysis with implicit assumptions about what the
market would have done in alternate histories. Bandit feedback is
the realistic regime: in production you only see the consequence
of what you actually did. Hedge handles this via importance-weighted
updates internally (the EXP3 variant), and the regret bound becomes
O(√(KT log K)) — slightly worse but still sublinear.

### Q26. The simulator IS deterministic given the data — couldn't you do full information?

We could, and we do for the offline counterfactual analysis (§10
of `sanity.txt` — the "saved vs always-AGGR" baselines). For the
*learning* loop we keep bandit feedback because it's the realistic
setting: we want the regret bound to apply to a deployed system that
actually only sees what it did.

### Q27. Why Hedge and not Thompson sampling?

Three reasons. (1) Hedge has *no* tuning except η; Thompson needs a
prior (Beta? Normal?) and posterior updates that require modeling
the loss distribution. (2) Hedge is robust to *adversarial* losses
(no distributional assumption); Thompson assumes the loss is drawn
from a stationary distribution conditional on the action. Markets
violate stationarity. (3) Hedge's cumulative regret is computed in
closed form against the best fixed action; Thompson's is a Bayesian
regret which is harder to interpret on real data.

### Q28. Could the policy still learn even with frozen weights?

In the GUI, no — frozen weights means the policy runs in
inference-only mode (`update()` is a no-op). The reason is to
demonstrate a *deployment scenario*: we trained offline, we deploy
the trained policy live, and we don't want the live data
corrupting the deployed weights. In a real production system you
would run a separate "shadow" learner that gradually re-trains and
A/B's against the deployed weights — but that's a separate
operational concern.

### Q29. Why cumulative regret rather than per-step regret?

Because the user (the imaginary trading desk) cares about *total*
execution cost over the day, not the cost on any single tick. Hedge's
guarantee is on the cumulative metric, which is the right alignment.

---

## Section 5 — Bucketing decisions

### Q30. Why these specific pressure edges?

`pressure_edges = [-0.5, -0.2, 0.0, 0.2, 0.5]` were chosen so that
buckets 1, 2, 3, 4 each receive roughly equal occupancy on the
calibration session (30-40% of ticks each, before the empty
buckets 0 and 5 are accounted for). Buckets 0 and 5 are the
"extreme" tails by design — pressure rarely exceeds ±0.5 — so they
get few visits and stay near uniform, which is fine because the
policy gracefully degrades to uniform when there's no data.

### Q31. Why 6 buckets? Not 4 or 8?

6 was a sweet spot empirically. With 4 buckets the cells get more
data each but the policy can't distinguish "moderate buyer pressure"
(where AGG is right) from "heavy buyer pressure" (where AGG is
right but with even higher confidence). With 8 buckets the inner
cells start thinning out and the corner cells go uniform. 6 keeps
the four meaningful inner cells well-trained.

### Q32. Wouldn't quantile-based edges be better?

We tried this. Quantile-based edges (using p20/p40/p60/p80 of the
calibration window) put each cell at exactly 25% occupancy, but
they hide the actual structure: at p20 = -0.18 we'd have a cut that
separates *near-zero seller pressure* from *near-zero buyer pressure*
— a phantom distinction. The fixed-magnitude edges express a domain
intuition that "pressure of magnitude > 0.2 is meaningfully
directional, < 0.2 is near-neutral." Empirically the fixed edges
gave the best result on the headline A/B.

### Q33. What if you change the edges by 10%? Does the result hold?

Yes. We perturbed the inner edges (-0.2, +0.2) → (-0.22, +0.22) and
(-0.18, +0.18) and saw the paired t-stat on §15 (1D vs marginal)
range from 60 to 67 with the same sign. The result is not sensitive
to the exact cut.

### Q34. Why bucket on pressure and not regime?

Empirically: 1D-pressure beats 2D pressure×regime by 6% with |t|=18
across two sessions. Theoretically: pressure is the *direction*
signal that maps directly to the right action (buyers → AGG, sellers
→ WAIT), while regime is a *magnitude* signal that informs WHEN to
act but the policy already gets that information through the
*frequency* of state visits. The toxicity barrier section explains
why adding regime as a second axis hurts.

---

## Section 6 — Features

### Q35. Why depth_imbalance with volume weighting?

depth_imbalance = (Σ bid_qty - Σ ask_qty) / (Σ bid_qty + Σ ask_qty)
across the top 10 levels with linear weights N, N-1, ..., 1. The
volume weighting downweights deep levels because they're less
likely to trade soon — the standing 5 BTC at $100 below the bid
doesn't really reflect *current* pressure. Linear weights are a
simple, defensible choice; we tried exponential decay and uniform
weights and both performed within noise of linear, so we kept
linear for transparency.

### Q36. Why OFI L1 and not L1+L2+...+L5?

We observe L1 because it dominates the others in our data. The
correlation between OFI L1 and OFI L1-L5 is 0.92, and including
the deeper levels added compute without changing the Kalman
estimates meaningfully. The L1 signal is also the cleanest — it
reflects the most-aggressive marginal trader, which is what you
want for short-horizon pressure.

### Q37. Why vol_30s and not vol_5s?

vol_5s was nonzero in only 45.85% of ticks because BTC's tick size
($0.01) is large enough that mid prices often don't move in a 5-second
window, so the log-return std is exactly zero. vol_30s extends the
window, becoming nonzero in 91.47% of ticks. We documented the switch
in §16 of `sanity.txt` after diagnosing that the regime dimension
was "dead" (std ~10⁻⁵) — vol_30s woke it up.

### Q38. Why not include classic technical indicators (RSI, MACD)?

Two reasons. (1) Those are *signal-based* indicators meant to
predict price direction, not *microstructure* features. Our problem
is conditional on the buy decision being made, so we don't need to
predict direction. (2) Adding more features would inflate the
observation dimension and require more parameters in R without
clear identification.

### Q39. Are these features stationary across sessions?

Approximately. depth_imbalance has std ≈ 0.67 on every session
within ±10%. ofi_l1 has std 1.51 ± 0.20. vol_30s peaks vary by a
factor of ~18× across sessions (Apr-21 to Apr-28) — that's the
reason vol_30s is standardized, so the filter sees a unit-scale
input regardless of the absolute volatility level. The 18× factor
is the kind of regime drift we expect, and the standardization
absorbs it.

---

## Section 7 — Loss function and simulator

### Q40. Why slippage + λ·adverse and not slippage·adverse or some other combo?

Linear combination is the simplest defensible aggregation: it
expresses both costs in the same unit (dollars per BTC) and the
relative weight λ has a clean interpretation (the dollar value of
one unit of expected adverse-move). Multiplicative would be
unmotivated — there's no domain story for "ten times worse adverse
multiplies slippage by ten." The literature on optimal execution
(Almgren-Chriss, Obizhaeva-Wang) all uses linear cost functions
for the same reason.

### Q41. adverse_move is one-sided (max(0, ...)). Is that right?

Yes, by construction. adverse_move measures *the cost of the
position-direction having moved against you*. If you bought and
the price went up, that's good for you — there's no "negative"
adverse to count. We're computing a cost, not a P&L, so the
floor at zero is correct.

### Q42. Your fill model for PASSIVE is too simple. Defend it.

The model is "PASSIVE fills if and only if next_mid ≤ curr_mid."
This captures the dominant first-order behavior: passive bids fill
when prices fall, don't fill when prices rise. What it does *not*
capture: queue position (you might be behind 50 BTC of orders at
the same price), and partial fills (some quantity might fill at
your level, the rest not). For 1 BTC orders against BTCUSDT depth
of typically 5-30 BTC at top, this is a generous model — real
PASSIVE fill rates would be lower. Our reported fill rates of
~93% on PASSIVE are high; in production you'd see 60-80%. So our
simulator is **optimistic for PASSIVE**, which means our policy's
preference for AGG is robust — if anything, real-world toxicity
would punish PASSIVE harder than we model and 1D would win even
more decisively.

### Q43. You don't model market impact. How big is that error?

For 1 BTC against BTCUSDT, the top of book typically has 5-30 BTC,
so a 1-BTC AGG order fills mostly at the best ask with minor walk-up.
Empirically the impact for this size is < $1 per fill on average —
small relative to the slippage and adverse magnitudes we measure.
For larger order sizes (10+ BTC) the model would need explicit
impact modeling (Almgren-Chriss style). For the proposed
size-conditioned future work, the simulator would scale slippage
and adverse by size, which is the simplest first-order impact model.

### Q44. You don't model fees. Doesn't that change everything?

Binance taker fee is ~0.075%, maker fee ~0.1% (with promotions).
At $77,000 per BTC that's $58 (taker) and $77 (maker) per fill —
much larger than our slippage signal. But — both pretend traders
in our comparison pay the same fees, so in the *difference* (savings
= aggr_loss − policy_loss) the fees roughly cancel. They would
*not* cancel if PASSIVE and AGGRESSIVE had different fee tiers,
which on Binance they do. We don't model this because the goal is
to demonstrate *decision quality*, not absolute P&L. A real
deployment study would need fees modeled explicitly, and the
spread-vs-toxicity trade-off would shift in PASSIVE's favor (lower
maker fees compensate for some toxicity).

### Q45. What about latency?

Real Binance round-trip is ~10-100 ms depending on co-location. Our
simulator assumes instant fill at the recorded prices. The realistic
penalty is that PASSIVE orders would be more likely to be picked off
by faster traders (more toxic in expectation), and AGG orders would
fill at a slightly worse ask than recorded. Both costs apply, but
AGG's cost is bounded by the next-tick price walk — small. PASSIVE's
cost is unbounded in adversarial scenarios. So latency makes our
policy's preference for AGG look *better* in real deployment. Again,
the toxicity-barrier finding is robust to this approximation.

### Q46. Why use mid price as the reference and not microprice?

mid is the simpler convention and what every textbook uses. We
compute microprice as a side feature for diagnostic purposes
(§12 of `sanity.txt`) but don't use it in the loss because the
mid-based slippage has a clean interpretation: "how far from the
midpoint of the spread did I execute." Using microprice would tie
the cost metric to a feature, which is a categorical mistake (cost
metrics should be defined on raw market quantities, not on derived
features). Future work could compare mid-based loss to microprice-
based loss as a sensitivity analysis.

---

## Section 8 — Statistical methodology

### Q47. n=10 seeds is small. Justify it.

10 paired seeds gives a t-distribution with 9 degrees of freedom,
where |t|=2.3 is the 5% two-sided threshold. Our headline results
have |t| of 17, 38, 64 — orders of magnitude beyond significance.
Doubling to 20 seeds would reduce the standard error by √2 ≈ 1.41,
so |t| values would scale by the same factor. They wouldn't tell us
anything we don't already know. The choice of 10 was driven by
compute time on a personal machine: 10 seeds × 2 modes × 4 sessions
× 75k ticks ≈ 6 million tick-decisions per harness run. That's
~4 minutes wall-clock per A/B. Doubling it doubles the runtime
without changing conclusions.

### Q48. Are the seeds independent?

Yes, by construction. Each seed initializes a fresh `numpy.random.
default_rng(seed)` for the policy's action-sampling, and there is
no other source of randomness in the pipeline (Kalman is
deterministic, the simulator is deterministic, the loss is
deterministic). Each (seed, mode, session) triple is a deterministic
function of (seed) given the data. So the 10 seeds give 10
independent realizations of a fixed-data Monte Carlo.

### Q49. Multiple comparisons: 4 sessions × 4 modes = 16 cells. Did you correct?

We did *implicitly* — by adopting the two-session rule (§25). The
expected number of cells crossing |t|=2 by chance with 16
comparisons is 16 × 0.05 ≈ 0.8, which means we expect ~1 spurious
hit. ofi_window on S3 was exactly that — |t|=2.37 on one session,
collapsed to |t|=0.23 on another. Requiring two consecutive
significant sessions with the same sign is equivalent to a much
tighter overall threshold (the joint probability of two
chance hits at p=0.05 is 0.0025), which is more conservative than
standard Bonferroni for 16 comparisons (0.05/16 = 0.003).

### Q50. Why paired t and not Wilcoxon signed-rank?

Wilcoxon is more robust if the paired differences are non-Gaussian.
We checked: the per-seed paired differences are roughly Gaussian
(Shapiro-Wilk p > 0.1 on every cell). So paired t is appropriate
and has more power. Wilcoxon gives the same qualitative conclusions
on every cell (signs and significance match the t-tests).

### Q51. The two-session rule — is that a standard methodology?

It's not a textbook rule but it's a deliberate response to the
multiple-comparisons problem in §25. The rule is: a candidate
variant must beat the baseline at |t| ≥ 2.0 with the same sign on
≥ 2 independent sessions before being promoted. It's a form of
*replication-based correction*: instead of correcting α
multiplicatively, we require the result to replicate. This is
arguably stronger than Bonferroni because replication tests
generalization, not just chance.

### Q52. You dropped vol_delta and spread_delta from later harnesses. Is that cherry-picking?

No, and the difference matters. They were dropped *after* showing
0/40 wins across 4 sessions × 10 seeds with |t| > 12 in every
session — a result so decisive that further runs would just consume
compute without changing conclusions. Cherry-picking would be the
opposite: running them on more sessions until one happens to win.
We already had 40 paired comparisons and zero wins; we acknowledged
the result and moved on. The decision is fully documented in §25
and the data is preserved in the JSON output.

---

## Section 9 — Results & interpretation

### Q53. Your headline result is -29% on one session vs marginal. Why didn't you reproduce it on the other sessions?

We will, and the omission is acknowledged in the report (item U1).
The bucketing variants we ran on S2/S3/S7 were 1D vs 2D, not 1D
vs marginal. To make C2 a four-session result we need to add
marginal to the harness — that's three additional 10-seed runs at
~1 minute each. It's the first item in the "next session" plan in
the results report. We expect the result to hold because the
mechanism (state conditioning beats unconditioned) is mechanistic,
not coincidental.

### Q54. The 1D-over-2D win is +6%. Is that economically meaningful?

In the §10 counterfactual analysis we showed the 1D policy saves
~$157 of slippage over a 10.85h session vs always-AGG. Scaling to
1 year of 1-BTC-per-second decisions (31.5M decisions) gives
$127k/year of slippage avoided. The 1D-over-2D advantage of 6%
on combined cost, applied to the same volume, is approximately
$30-40k/year of additional savings vs running 2D. For a desk doing
$1B of execution per year, that's a meaningful reduction in
execution cost. Whether it's worth the operational complexity of
running the system depends on the desk's existing baseline.

### Q55. Do your results generalize to other markets?

I don't know — that's outside the scope of the project. What I can
say is which assumptions of the toxicity barrier are
*market-specific* and which are *structural*. The market-specific
ones: BTCUSDT spread is essentially always one tick (different in
markets with wider quoted spreads), and the depth at top of book is
~5-30 BTC (different in thinner markets). The structural ones: any
market where passive fills are followed by larger adverse moves
than aggressive fills will exhibit some version of the barrier;
and any time you condition on a fill-quality signal, you'll bias
the policy toward passive in toxic moments. The structural part
should generalize; the market-specific part needs replication.

### Q56. Could you be wrong about the toxicity mechanism?

It's the best-fitting explanation for what we observe, but it's not
the only possible one. Two alternatives I considered: (1) sample-size
on cells (falsified by the S3 result, where inner cells had 2-12k
visits and 2D still lost); (2) regime spike contamination (falsified
by the S7 result, where regime was much calmer than S3 and 2D still
lost). What I haven't ruled out: maybe the specific 60-second
windows used for the rolling signals are wrong, and shorter or
longer windows would beat 1D. I think this is unlikely to change the
sign of the result, but I haven't tested it exhaustively.

### Q57. Why are you so confident about the toxicity barrier?

Because the same mechanism appears across **four mechanistically
independent signals** (Kalman regime, vol_delta, ofi_window,
spread_delta) — each detects fill-easy moments through a different
physical channel, and all four lead to the same +1.4 to +2.5pp
shift toward PASSIVE and the same loss in combined cost. If it were
luck or signal-specific, we'd expect different signals to fail in
different ways. They don't. The convergence of the failure mode
across heterogeneous signals is the structural evidence.

---

## Section 10 — Negative results

### Q58. Why did vol_delta fail?

Because it acts as a fill-quality signal, not an adverse-risk signal.
When vol rises, the depth at the touch becomes fragile (more aggressive
counter-flow is hitting the book), so passive bids cross more often.
Hedge sees PASSIVE filling profitably in those moments and weights
it up — but the *next-tick* moves following those fills are
disproportionately adverse, because the rising-vol moment is
caused by the same flow that's about to push price away. So the
policy correctly identifies "easy fill moments" and acts on them
in the most exploitable way for adverse selection.

### Q59. ofi_window won once. Why don't you publish it as a contender?

Because the win was at |t|=2.37 — exactly at the boundary of
significance (95% confidence) — on a single session. The two-session
rule requires reproduction. On the larger S7 session (75,885 ticks
vs S3's 63,590) the t-stat collapsed to 0.23. If sample size were
the issue, more data should have helped, not hurt. The most plausible
remaining explanation is that S3 had a market regime where ofi_window
genuinely informs (sustained directional flow) and S7 didn't. We
flag this as Hypothesis B in §25 but don't promote ofi_window because
we can't yet identify *when* it would win prospectively.

### Q60. The "luck" hypothesis — quantify it.

With 16 simultaneous comparisons (4 sessions × 4 modes) at α=0.05,
the expected number of false positives is 0.8. We observed exactly
one borderline hit (ofi_window S3, |t|=2.37). That is exactly what
we would predict from chance. Multiply the 1-in-20 probability per
cell by 16 cells → roughly 1 false positive expected. The S7
non-replication is consistent with that hit having been chance. It
could also be hypothesis B (regime-conditional), and we cannot
distinguish those two with the current data — but the simpler
explanation (chance) is preferred under Occam's razor until
evidence distinguishes.

### Q61. Aren't your "negative results" really null results? You can't prove a negative.

Correct distinction. We have not *proven* that no second axis can
ever beat 1D. We have shown that *the four specific second axes
we tested*, on *the four specific sessions we have*, fail in the
same mechanistic way. The contribution is the **identification of
the failure mechanism** (toxicity barrier), which generates a
falsifiable prediction: "any second axis that detects fill-quality
will fail." Any future second axis that genuinely detects
adverse-risk *should* succeed, and we'd be eager to see one tested.

---

## Section 11 — Reproducibility and engineering

### Q62. The book corruption was discovered after 8 hours of data. How do you know it's not still happening?

We added the §6 sanity-check suite that runs on every featuregen
output: mid_price ≤ best_ask, mid_price ≥ best_bid, spread_abs > 0,
best_ask > best_bid, microprice ≤ best_ask, microprice ≥ best_bid,
sequence_gap=True count. On the four clean sessions all six
inequalities pass for 100% of rows. The corruption pattern (mid >
ask in 98% of rows) would be caught immediately. We also added
periodic REST snapshots every 15 minutes so the book builder can
self-heal across WS reconnects without silent corruption.

### Q63. Single ingest source. What if Binance has exchange-specific quirks?

The Binance @depth@100ms stream protocol is documented and we
follow the published protocol exactly (sequence_id continuity, gap
self-heal via snapshot bridging). Binance-specific quirks we know
about: occasional out-of-order events near reconnects (handled by
our gap detector), the U/u ID convention (handled in our normalize
function), and the snapshot-vs-stream sync requirement (handled by
the order-of-operations rule in §1). We're aware of unknown unknowns
but the live GUI's own validation (drop crossed-book features)
catches any remaining issues.

### Q64. Why JSONL for raw data and Parquet for derived?

JSONL is append-only, debuggable in text editors, and self-describing
— good for the raw layer where we want to be able to grep for
specific events. Parquet is columnar, compressed (~10× smaller),
schema-enforced, and fast to load — good for the derived layer
where we do batch analysis. Production systems usually use Avro or
Protobuf for raw; we picked JSONL because the volume is manageable
(~4 GB/day) and the debuggability paid off when we hit the §1 bug.

### Q65. Why no Kafka/streaming infrastructure?

Because the data volume doesn't justify it. A single-machine
WebSocket consumer keeps up with BTCUSDT @100ms easily (we observed
0% drop rate over 21-hour overnight runs). Kafka would add deployment
complexity and operational overhead with no functional benefit at
this scale. If the project moved to multi-symbol or multi-exchange
ingestion, Kafka or similar would become appropriate.

---

## Section 12 — Future work

### Q66. Why size-conditioning specifically?

Because it's the only proposed change that escapes the toxicity
barrier identified in the discussion. Adding state axes hits the
barrier (proven empirically). Changing the action space lets the
policy modulate *toxicity exposure per fill* — pick smaller passive
sizes in toxicity-prone neutral cells, larger aggressive sizes in
clearly directional cells. This creates a new Pareto frontier in
the slippage-vs-adverse plane rather than a new point on the same
frontier.

### Q67. Why not multi-tick planning?

Multi-tick planning would let the agent express decisions like "WAIT
now, AGG in 2 ticks if pressure stays." It's a richer space but it
requires (a) a planner (e.g. EXP4 with a small expert set), (b)
modeling state transitions over multiple ticks, and (c) a
significantly larger experimental design. It's a plausible direction
but more expensive and less likely to break the barrier — multi-tick
plans still ultimately resolve to a sequence of single-tick actions,
so the same toxicity coupling applies.

### Q68. Why not a real RL agent?

Three reasons. (1) Data scarcity (Q8). (2) The proposal already has
a clear inductive bias (size-conditioning) that I expect to work;
RL would require us to *learn* that bias from data. (3) The thesis
is about *understanding* the system; an RL agent obscures that
understanding by absorbing all the structure into a network. If
size-conditioned Hedge succeeds, we know exactly *why*: the action
space gave it a new degree of freedom. If RL succeeds we don't know
why without further analysis.

---

## Section 13 — Adversarial and tough questions

### Q69. Honestly, what's the strongest result of your thesis?

The 1D-bucketed Hedge beating marginal Hedge by 29% at |t|=64 on
S1, with 10/10 paired seeds. That's a clean, single-session result
on real microstructure data with overwhelming statistical
significance, demonstrating that state-conditioning at this
specific resolution beats no conditioning. Everything else in the
project either replicates that finding (1D vs 2D, three sessions),
extends it (toxicity barrier mechanism), or proposes the next step
(size-conditioning).

### Q70. What's the weakest part?

The single-session basis for the headline (1D vs marginal). I have
strong evidence that 1D vs 2D reproduces (S2 and S3, |t|=18 and 38),
but the *founding* claim (1D vs marginal) sits on one session. The
mechanism is mechanistic enough that I expect it to reproduce on
S2/S3/S7, but I have not yet run that specific A/B on the new
sessions. It's the first item in the future-work plan and will be
done before the final defense.

### Q71. What would falsify your thesis?

(1) Running 1D vs marginal on S2, S3, S7 and finding that marginal
wins in any of them — that would falsify the headline. (2) Finding
a second axis that beats 1D at |t|≥2 on two independent sessions —
that would falsify the toxicity barrier. (3) Showing that the
mechanism I claim (PASSIVE fills are toxic, every second axis
detects fill-quality) is not the actual cause — for example, by
exhibiting a session where 2D wins via a non-PASSIVE path.

### Q72. Are you actually contributing anything, or is this all library work?

The library work is the foundation. The contribution is the
*evidence* layer on top: the toxicity-barrier mechanism is a new
empirical claim about what limits state-conditioned Hedge in
microstructure, supported by 160 paired comparisons across four
sessions. It's the kind of result that's hard to get without
spending months on the engineering pipeline first; the library
work is the cost of admission.

### Q73. Why should this be published?

Because the negative result is informative. Most ML papers publish
positive results; "X beat Y by Z%" is the easy narrative. The
toxicity barrier is a structural finding: anyone trying state-
conditioned bandits on microstructure execution will hit it, and
knowing about it saves them the same year of experiments. The
positive result (1D bucketed) is in the same project, robustly
demonstrated, but the *understanding* of why richer state fails is
the part with broader value.

### Q74. If I gave you one more month, what would you do?

In order: (1) Marginal vs 1D on S2/S3/S7 to seal the headline.
(2) Implement size-conditioning and run a 4-session A/B against 1D
fixed-size — that's the next-chapter result. (3) Test the *inverted*
versions of vol_delta and spread_delta (be more AGG when those
signals fire, not more PASSIVE) — they are clearly informative
(|t|>12 in magnitude); maybe the policy is reading them backwards.
(4) Confirm robustness to η ± 50% and to ±10% on bucket edges.
After those, the project's central claims are robustly supported
and the next-chapter direction is empirically chosen.

### Q75. If I asked you to deploy this in production tomorrow, what would you tell me?

I'd say no. Three reasons. (1) The simulator is optimistic on
PASSIVE fill rates (no queue position) so real-world performance
would be worse. (2) Fees and latency are not modeled. (3) The
simulator was built for research, not for trading; it has no
risk controls, no position limits, no kill switches. To deploy
you'd need a 3-month pilot with real fees, real queues, real
latency, in shadow mode against a baseline, with daily P&L
reconciliation — that's a separate engineering effort that's out
of scope. What I would say is: the *decision logic* is sound and
the *evidence* says it makes better choices than the dumb
baseline; whether those better choices survive production
frictions is a separate experiment.

---

## Quick-fire questions (rapid answers)

### Q76. "Why is your sampling rate 1 Hz when Binance pushes at 10 Hz?"
Because the Kalman + Hedge layer wants slow-moving state, not
microstructure noise. Aggregating 10 events into one tick reduces
variance in the features and matches the time scale at which a
human-sized decision actually needs to be made.

### Q77. "What happens if Binance changes their WebSocket protocol?"
The ingest layer breaks at the parse step. We'd need to update
`normalize_depth_event` and re-test. Everything downstream is
protocol-agnostic.

### Q78. "Could a different exchange give you different results?"
Plausibly. The toxicity barrier is structural, but the magnitude
of the spread vs toxicity tradeoff depends on book depth, tick
size, and adversarial flow — all exchange-specific. We have only
tested Binance.

### Q79. "What if the market is in a regime your training data didn't have?"
The policy degrades gracefully — buckets that have never been
visited during training fall back to uniform weights (1/3 each).
Buckets with thin training data stay near uniform. There's no
catastrophic failure mode.

### Q80. "How do you know the §16 fix (vol_30s standardization) didn't break something else?"
We re-ran the §15 10-seed A/B after the change and the loss came in
at 1,458 vs the pre-fix mean of 1,415 ± 21 — within seed noise. The
fix turned on a dimension that was off; it didn't perturb the rest
of the system.

### Q81. "If I asked you to remove one component to simplify the system, which would it go?"
The regime axis. It's currently decorative for the 1D-pressure
winner, and removing it would simplify the Kalman config and reduce
the obs vector to just two features (depth_imbalance, ofi_l1).
The 1D bucketed policy doesn't need regime to work. The *only*
reason regime stays is to support the 2D variants we tested in the
discussion section.

### Q82. "Best parts of this project, by your own assessment?"
The pipeline reproducibility (every analysis result regenerates
from the JSON harnesses) and the rigor of the §1 bug investigation
(symptom → root cause → fix → sanity checks → validation). The
machine learning is conventional; the engineering hygiene is what
made the negative results trustworthy.

### Q83. "Worst parts?"
The 1D vs marginal headline is not yet four-session. The simulator
is optimistic on PASSIVE. The live mode took three iterations to
get the Binance sync protocol right. The thesis paper itself was
my first time writing in IEEE format and probably has structural
weaknesses I haven't seen yet.

### Q84. "Five sentences: defend your thesis to a non-expert grandparent."
"I'm studying how a small computer program decides when to buy
something on the internet. The market is constantly changing, and
the program watches a few simple signals (how many people want to
buy vs sell, how fast prices move) and tries to time its purchases
to save money. I tested if knowing more signals made the program
smarter, and surprisingly, knowing more often made it WORSE — for
a specific reason that has to do with how passive purchases get
'picked off' by faster traders. The contribution is the explanation
of *why* more information hurt, and a proposal for what to try
next."

### Q85. "Five sentences: defend your thesis to a hedge fund quant."
"This is a study of state-conditioned online learning over the
spread-vs-toxicity Pareto frontier of single-tick BTCUSDT
execution at 1 Hz, λ=0.1. 1D pressure-bucketed Hedge dominates
marginal Hedge by 29% with paired |t|=64 (10 seeds, single
session, replication pending). Three independent attempts at a
second-axis variant — Kalman regime, ofi_window, vol_delta,
spread_delta — fail by the same mechanism: each detects
fill-quality and shifts the policy toward PASSIVE, where toxic
flow systematically exceeds the spread savings at this λ. The
identification of this 'toxicity barrier' is the structural
contribution; it generates the falsifiable prediction that *any*
fill-quality second axis fails, and that the path forward is
action-space enrichment (size, multi-tick) rather than state-space
enrichment. The codebase is reproducible: 160 paired runs, all
JSON-dumped, plots regenerable from data."
