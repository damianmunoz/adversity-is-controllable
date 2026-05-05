# The math, explained from zero

A walkthrough of what every number in the project means, where it comes
from, and how we go from "the policy lost 1415 in session S1" to "we
can claim this policy is genuinely better."

No proofs. No jargon. Just concrete numbers, and an analogy whenever
the abstraction gets in the way.

---

## Part 1 — What do the numbers in the table mean?

You saw this table:

| Sesión | marginal | 1D bucketed | Mejora |
|---|---|---|---|
| S1 | 1985 | 1415 | −28.7% |
| S2 | 2131 | 1562 | −26.7% |
| S3 | 3817 | 2500 | −34.5% |
| S7 | 4155 | 2869 | −31.0% |

**The first thing to internalize: those numbers are dollars.** They are
not "regret values." They are not abstract scores. They are
**cumulative dollar cost** that the policy paid over the entire session.

`1985` means: "if a real trader had run the marginal policy across
the 38,972 ticks of session S1, they would have paid a total cost of
1985 USDT (roughly 1985 dollars)."

`1415` means the same thing but for the 1D bucketed policy on the same
session.

`1985 - 1415 = 570` is the savings. Per tick that's `570 / 38,972 ≈
$0.015` per decision. Over a year of trading at 1 BTC/second pace that
adds up to ~$127,000.

**That's it.** All those numbers in the table are dollars of cumulative
loss. The "improvement %" is how much the dollar number went down.

---

## Part 2 — Where one of those dollars comes from

Let's walk through a single tick — just one second of one session —
and follow exactly how the dollar cost is computed. Once you see how
ONE tick works, the rest is just adding 38,972 of them.

### A concrete tick

At some moment during S1, the order book looked like this:

```
Best bid: 76,230.69 USDT (someone willing to BUY 1 BTC at this price)
Best ask: 76,230.70 USDT (someone willing to SELL 1 BTC at this price)
Mid price: 76,230.695 USDT (average of bid and ask — the "fair" price)
```

The next tick (one second later), the mid price moved to:
```
Next mid: 76,230.65 USDT (price went DOWN by half a cent)
```

The policy has to choose one of three actions for this tick. Let's
compute the cost for each one:

#### Action 1: AGGRESSIVE (cross the spread, buy at best ask)

You pay 76,230.70 USDT to get 1 BTC.

- **Slippage** = fill_price − mid = 76,230.70 − 76,230.695 = **+$0.005**
  (You paid half a cent above the fair price. That's the cost of cutting in line.)

- **Adverse move** = max(0, fill_price − next_mid) = max(0, 76,230.70 − 76,230.65) = **+$0.05**
  (After you bought, the price dropped by 5 cents. You bought right
   before a drop. That's bad luck = the "adverse move".)

- **Loss** = slippage + λ × adverse_move = 0.005 + 0.1 × 0.05 = **+$0.010**

#### Action 2: PASSIVE (post a limit buy at best bid)

You posted an order at 76,230.69. Will it fill?

- The next tick's mid is 76,230.65, which is **lower** than the current
  mid 76,230.695. So sellers came down to your price. **Yes, it fills.**

- **Slippage** = fill_price − mid = 76,230.69 − 76,230.695 = **−$0.005**
  (You paid HALF A CENT BELOW the fair price. That's a gain, hence
   negative slippage.)

- **Adverse move** = max(0, fill_price − next_mid) = max(0, 76,230.69 − 76,230.65) = **+$0.04**
  (You got filled at 76,230.69 and the price dropped to 76,230.65.
   You overpaid by 4 cents relative to where the market went.)

- **Loss** = −0.005 + 0.1 × 0.04 = **−$0.001**
  (Net win of one-tenth of a cent on this tick.)

#### Action 3: WAIT (do nothing)

- **Slippage** = $0 (no fill happened).

- **Adverse move** = max(0, next_mid − mid) = max(0, 76,230.65 − 76,230.695) = **$0**
  (The price went DOWN. Waiting saved us from buying high. So no penalty.)

- **Loss** = 0 + 0.1 × 0 = **$0**

#### What the policy actually does at that tick

The policy looks at the Kalman state, finds itself in some bucket
(say bucket 2 — pressure slightly negative), looks up the bucket's
weights (say WAIT=0.67, PASSIVE=0.14, AGG=0.19), and **rolls a die
weighted by those weights**. If the die lands on PASSIVE, the policy
incurs a loss of −$0.001 for this tick.

Now repeat 38,971 more times. Each tick the loss is some small number
(usually between −$0.05 and +$0.10). Add them all up. That sum is
your `total_loss` for the session.

### Why are the totals like 1415 and not just $30 or $400?

Because `total_loss` includes **the adverse term scaled by λ=0.1**. The
adverse_move per tick is on the order of dollars (sometimes much more,
when the price jumps), not cents.

Look at `sanity.txt §13`: 38,972 ticks have an average per-tick adverse
of ~0.475. So total adverse over a session = 38,972 × 0.475 ≈ 18,500
units. With λ=0.1, that's ~1,850 units of loss from the adverse term
alone. Add the slippage component (~$50) and you get the ~1,900 we see.

**Net summary for Part 2:** The 1985 and 1415 numbers are sums of
~38,972 small per-tick numbers. Each per-tick number is computed
from the actual recorded order book using the formula `slippage + 0.1 ×
adverse_move`. The dollar units are real; the math is mechanical.

---

## Part 3 — What the three features actually measure

Before we go to "how the Kalman filter works," let's nail what its
inputs are.

### Feature 1: depth_imbalance

**What it asks:** *"At this moment, is there more BTC sitting on the
buy side or the sell side of the order book?"*

**How we compute it:**
```
depth_imbalance = (Σ bid_qty - Σ ask_qty) / (Σ bid_qty + Σ ask_qty)
```
where the sums are over the top 10 price levels, with linear weights
that down-weight deeper levels.

**Range:** strictly between −1 and +1.

**Concrete examples:**

| Bid total | Ask total | depth_imbalance | Plain meaning |
|---|---|---|---|
| 50 BTC | 50 BTC | 0.00 | balanced |
| 80 BTC | 20 BTC | +0.60 | "buyers stacked" |
| 5 BTC | 95 BTC | −0.90 | "sellers stacked" |
| 99 BTC | 1 BTC | +0.98 | "buyers ready to pounce" |

**Why we use it:** The depth tells you *which side has more "fuel" to
move the price.* If 99% of the standing liquidity is on the bid side,
the next price move is more likely to be down (the asks are thin and
will get eaten through quickly).

### Feature 2: ofi_l1 (Order Flow Imbalance, level 1)

**What it asks:** *"Did the bid grow or shrink between the last tick
and this one? What about the ask?"*

**How we compute it:**
```
ofi_l1 = (bid_qty_now - bid_qty_prev) - (ask_qty_now - ask_qty_prev)
```
at the top of book.

**Range:** typically −30 to +30 BTC.

**Concrete examples:**

| Δ bid | Δ ask | ofi_l1 | Plain meaning |
|---|---|---|---|
| +5 BTC | 0 | +5 | "more buyers showed up at the top" |
| 0 | +5 BTC | −5 | "more sellers showed up at the top" |
| 0 | −10 BTC | +10 | "asks are vanishing — buyers are eating them" |
| −10 BTC | 0 | −10 | "bids are vanishing — sellers are eating them" |

**Why we use it:** depth_imbalance is a *static* picture (where is the
liquidity right now). OFI is a *dynamic* picture (where is liquidity
*changing*). It captures momentum — who is being more active.

### Feature 3: vol_30s (realized volatility over 30 seconds)

**What it asks:** *"How much has the price been jumping around in the
last 30 seconds?"*

**How we compute it:**
```
For each tick s in the last 30 seconds:
   r_s = ln(mid_s / mid_{s-1})            # the log-return
vol_30s = standard_deviation(those r_s values)
```

**Range:** 0 to about 0.0007 (very small numbers, because the price
changes are very small relative to the price).

**Concrete examples:**

| Recent moves | vol_30s | Plain meaning |
|---|---|---|
| Mid stuck at 76,230 for 30s | 0.000000 | dead market |
| Mid hopping ±$0.01 each tick | 0.000003 | normal BTCUSDT |
| Mid hopping ±$0.10 each tick | 0.00003 | active |
| Mid hopping ±$1 each tick | 0.0003 | turbulent |
| News-driven jump | 0.001+ | regime spike |

**Why we use it:** This is the *turbulence* number. depth_imbalance
and OFI both tell you DIRECTION. vol_30s tells you MAGNITUDE — how
crazy is the market right now?

### So why three features?

Because we want to estimate two things and we need at least one
observation for each, and ideally redundancy:

- **Pressure** (direction) = informed by depth_imbalance + ofi_l1
- **Regime** (turbulence) = informed by vol_30s

Two angles on direction (one static, one dynamic) lets the Kalman
filter average them and reduce noise. Volatility is its own channel.

---

## Part 4 — Why we normalize (obs_scale)

Now we hit the part you said felt confusing. Bear with me — once you
see the actual numbers, it becomes obvious.

### The problem

Look at the **typical magnitudes** of the three features:

| Feature | Typical magnitude | Standard deviation |
|---|---|---|
| depth_imbalance | ±0.5 | 0.67 |
| ofi_l1 | ±5 | 1.51 |
| vol_30s | ~0.00003 | 0.000027 |

Notice anything? **vol_30s is a million times smaller than ofi_l1.**
It's literally a different order of magnitude.

Now the Kalman filter wants to combine these three numbers using its
H matrix:

```
H = [1  0]    (depth_imbalance contributes to pressure)
    [1  0]    (ofi_l1 contributes to pressure)
    [0  1]    (vol_30s contributes to regime)
```

The way the math works inside the filter, *how much each observation
moves the hidden state estimate is proportional to its raw magnitude*.

If vol_30s is in the range 0 to 0.0007, and the filter receives it
through a unit-gain entry (H[2,1] = 1), then the regime estimate is
*literally bounded above by 0.0007*. No matter how much we tune the
noise covariance R, regime can never have a useful dynamic range
because **its observation is too small**.

### The proof from the data

Before the fix, the regime estimate had:
- standard deviation ≈ 0.00003 (almost zero)
- max value ≈ 0.0007 (almost zero)

It was effectively dead. The filter "saw" vol_30s but the signal was
so small it produced no meaningful output. We documented this in
`sanity.txt §16` — we called it the "dead regime dimension."

### The fix: standardize before feeding to the filter

We transform each observation BEFORE the Kalman filter sees it:

```
z_normalized = (z_raw - obs_center) * obs_scale
```

For depth_imbalance and ofi_l1: `obs_center = 0`, `obs_scale = 1`.
This means **no transformation** — they already live on reasonable
scales.

For vol_30s:
- `obs_center[2] = 3.1658e-5` (the empirical mean of vol_30s on a
  clean 39k-tick window)
- `obs_scale[2] = 36876.637` (which is `1/std`, since
  std = 2.7117e-5)

What this does: it shifts vol_30s so its mean is zero, then scales it
so its standard deviation is 1. **Now vol_30s is in the same
range as the other two features.**

Concrete examples:

| Raw vol_30s | After normalization | Plain meaning |
|---|---|---|
| 0.000031658 (the mean) | 0 | "exactly average turbulence" |
| 0.000058775 (1 std above) | +1 | "1 std above average" |
| 0 (silent moment) | −1.17 | "below average — calm" |
| 0.000570 (a spike) | +19.8 | "huge spike, far above average" |

After this fix, the regime estimate has:
- standard deviation ≈ 0.37 (alive!)
- max value ≈ 4.09 (real range!)

The information was always there. We just couldn't see it through
the filter without normalizing first.

### Why we don't normalize depth_imbalance and ofi_l1

Because they already live on sensible scales (`std ≈ 0.67` and `1.51`
respectively, comparable to each other). And — important — the
**bucket edges** for the policy are calibrated to the **raw** pressure
scale (edges = `[-0.5, -0.2, 0, +0.2, +0.5]`). If we normalized
depth_imbalance, the pressure estimate would be on a different scale
and our bucket edges would not separate the data the way they do now.

So the rule is: only normalize what *needs* normalizing.
depth_imbalance and ofi_l1 don't need it. vol_30s does.

---

## Part 5 — How we ascertain the results

Here's where you go from "the policy paid 1415 dollars in S1" to "I
am confident this policy is genuinely better than marginal." This is
the statistics half.

### The basic idea: paired comparison

Suppose you want to test whether shoe brand A is faster than shoe
brand B for running. The wrong way: have 10 different people run in A,
10 different people run in B, compare averages. The problem is some
people are naturally faster runners — that noise drowns out the shoe
effect.

The right way: have the SAME 10 people run in BOTH shoes (alternate
days). For each person, compute `time_A - time_B`. Now you have 10
paired differences that *all measure the shoe effect*, with the
runner's natural ability cancelled out.

We do exactly the same thing.

### What the "seed" is

The Hedge algorithm has one source of randomness: the dice roll that
samples an action from the bucket's weights. Each "seed" is a
specific seed for the random number generator. With seed=42, the
policy will roll the same sequence of dice every time you replay.

So a "seed" is like a "specific person" in the running analogy.

### What we actually do

For each seed in {0, 1, 2, ..., 9}:
1. Replay session S1 with the **marginal** policy and that seed.
   Compute total_loss → call it `M_seed`.
2. Replay session S1 with the **1D bucketed** policy and the **same
   seed**. Compute total_loss → call it `B_seed`.
3. Compute the paired difference `d_seed = B_seed - M_seed`.

For S1, those 10 pairs gave us:

| seed | marginal | 1D bucketed | difference |
|---|---|---|---|
| 0 | 1970.85 | 1438.99 | −531.86 |
| 1 | 1999.87 | 1415.54 | −584.33 |
| 2 | 1962.24 | 1410.02 | −552.22 |
| 3 | 1969.10 | 1386.16 | −582.94 |
| 4 | 1957.32 | 1438.15 | −519.17 |
| 5 | 1989.80 | 1403.10 | −586.70 |
| 6 | 2028.42 | 1441.05 | −587.37 |
| 7 | 1955.70 | 1391.72 | −563.98 |
| 8 | 2007.93 | 1397.58 | −610.35 |
| 9 | 2012.29 | 1430.71 | −581.58 |

All 10 differences are **negative**, meaning 1D bucketed lost less
than marginal in every paired comparison.

### Computing the t-statistic

The mean of those 10 differences:
```
mean(d) = (−531.86 − 584.33 − 552.22 − ... − 581.58) / 10 = −570.05
```

The standard deviation of those 10 differences:
```
std(d) ≈ 28.10
```

Then the t-statistic is just:
```
t = mean / (std / sqrt(n))
  = -570.05 / (28.10 / sqrt(10))
  = -570.05 / 8.886
  = -64.15
```

### What does t = -64 mean in plain words?

The t-statistic measures **how many standard errors the mean is away
from zero**. Or in plainer language:

> "If there were really NO difference between the two policies, how
> surprising would it be to see a paired-difference average this far
> from zero?"

A t of 1 means "not surprising at all, this could easily be noise."
A t of 2 means "kind of surprising, about 5% chance of happening by
luck."
A t of 3 means "really surprising, about 0.3% chance by luck."
A t of 10 means "essentially impossible by luck."
A t of 64 means "the universe ends before this happens by luck."

So when we see |t|=64, we're saying:
> The signal (the systematic 570-dollar advantage of 1D over marginal)
> is 64 times bigger than the noise we'd expect from random sampling
> across seeds. We are extraordinarily confident the difference is real.

### Why we replicate across sessions

A single t=64 result on a single session could still be wrong if there
was something weird about that specific session. So we do the same
test on S2, S3, S7. Look at the t-statistics:

| Sesión | mean diff | t-stat |
|---|---|---|
| S1 | −570 | 64.2 |
| S2 | −570 | 36.8 |
| S3 | −1316 | 95.1 |
| S7 | −1286 | 105.3 |

All four are way beyond significance, all in the same direction.
**Same effect, four different days, four different markets.** That's
how we go from "interesting result" to "robust scientific claim."

### The savings number on the savings curve

When the GUI shows you the cumulative savings going up to $+838.64
on S7, that's:

```
savings = total_aggressive_baseline_loss - total_policy_loss
        = 3728 - 2891
        = 837 (≈ 838 with seed-noise)
```

It's a real dollar number, computed exactly the same way as the loss
totals — just subtracting.

---

## Part 6 — Putting it all together

Here's the entire pipeline in sequence, with the dollar interpretation
at each step:

1. **Order book event arrives** from Binance. Just a raw record of
   "at this microsecond, the bid moved from X to Y."

2. **Order book is updated** in our local copy via `book_builder.py`.

3. **Once per second**, the feature pipeline computes:
   - `depth_imbalance` (a number between −1 and +1)
   - `ofi_l1` (a number, typically −5 to +5)
   - `vol_30s` (a small positive number, typically 0 to 0.0007)

4. **The Kalman filter normalizes vol_30s** by `(z − 3.1658e-5) ×
   36876.637` so all three observations live on similar scales.

5. **The Kalman filter combines** the three observations into two
   estimates: pressure (between roughly −1.5 and +2.5) and regime
   (between roughly −0.7 and +4).

6. **The Hedge policy takes the pressure estimate**, looks up which
   bucket it's in (one of 6 buckets between −∞ and +∞), and reads
   off the weight distribution for that bucket (e.g.
   `[WAIT=0.89, PASSIVE=0.05, AGG=0.06]`).

7. **The policy rolls a weighted die** and picks an action: WAIT,
   PASSIVE, or AGGRESSIVE.

8. **The simulator computes the dollar consequence** of that action
   using the next tick's mid:
   - WAIT: usually $0 cost.
   - PASSIVE: maybe −$0.005 (gain) if filled, $0 if not, plus possible
     adverse move.
   - AGGRESSIVE: +$0.005 cost, plus possible adverse move.

9. **The Hedge update rule** multiplies the action's weight by
   `exp(−η × loss)` so that high-loss actions lose probability over
   time.

10. **At the end of the session**, we sum up all those per-tick
    dollar amounts. That sum is the `total_loss` you see in the
    table.

11. **Across 10 seeds**, we compare two policies' totals on the SAME
    session. The mean difference, divided by its standard error, is
    the t-statistic.

12. **Across 4 sessions**, all t-stats point the same way → robust
    conclusion: 1D bucketed Hedge consistently beats marginal Hedge.

That's the entire chain. There's no magic anywhere. Every number is
either a dollar amount, a probability, or a t-statistic measuring how
many noise units a dollar amount differs from zero.

---

## Cheat sheet for the defense

When someone asks you in the defense:

**"What does 1985 mean?"**
→ "It's the cumulative cost in USDT that the marginal policy paid
across 38,972 ticks of session 1. Each tick contributes a small dollar
amount = slippage + 0.1 × adverse_move. The 1985 is the sum."

**"What is obs_scale?"**
→ "vol_30s is on the order of 10⁻⁵ while the other two features are
on the order of 1. Without normalizing, the regime estimate of the
Kalman filter is bounded by the magnitude of vol_30s, so it stays
near zero. obs_scale = 1/std rescales vol_30s to unit variance so the
filter can use it. Empirically, the regime std went from 10⁻⁵
to 0.37 after this fix."

**"What is t = 64?"**
→ "It's how many standard errors the mean paired difference is from
zero. We computed 10 paired differences (one per seed), got a mean
of −570 dollars and a std of 28. The t = 570/(28/√10) = 64. For
context, a t of 2 is the conventional significance threshold; we're
at 64. The probability of seeing this by chance is essentially zero."

**"How do you know the result generalizes?"**
→ "We replicated the same test on four independent sessions —
different days, different market conditions. All four showed the
same direction with t-statistics from 36 to 105. The systematic
1D-vs-marginal advantage is present in every session we have data
for."

**"Are those numbers real?"**
→ "The market data is real (recorded from Binance). The math
that turns a market state into a dollar cost is correct (slippage
and adverse are computed against actual recorded next-tick prices).
The number is *real* in the sense that it's a fair counterfactual
comparison of two policies on the same recorded data. It's NOT a
forecast of P&L on a real deployment, because we're not modeling
fees, queue position, or latency. But the *relative* comparison is
fair."

---

## Part 7 — The Hedge algorithm, from zero

Words like "distribution," "regret," and "regret bound" sound
intimidating until you see what they really mean. Let me strip the
jargon away.

### A toy version (no math yet)

Imagine you've moved to a new city for 30 days. There are three
restaurants near your apartment. You don't know which is best for
you. You have to eat somewhere every night.

A reasonable strategy:

- **Night 1.** Roll a 3-sided die. Whichever lands, that's where you
  eat. After dinner, give yourself a small "cost" number — 0 if you
  loved it, 1 if it was OK, 2 if you hated it.

- **Night 2 onward.** Use a *weighted* die. Restaurants that gave
  low cost get bigger slices on the die; restaurants that gave high
  cost get smaller slices. You don't fully commit to one place
  (tonight's bad meal might be a fluke), but you slowly bias toward
  what's been working.

- **After 30 days.** The die is heavily biased toward whichever
  restaurant has been giving the most consistent low-cost meals.

That's Hedge, in restaurant form. The "weighted die" is the
**distribution**. The way you nudge the bias after each night is the
**update rule**.

### Strip three jargon words

**Distribution.** A list of numbers between 0 and 1 that sums to 1.
For three actions:
```
{WAIT: 0.45, PASSIVE: 0.15, AGGRESSIVE: 0.40}      ← sums to 1
```
That's it. Each number is a probability for an outcome. Nothing
fancier than that.

**Sampling from a distribution.** Roll a uniform random number
$u \in [0,1)$. Walk along the distribution adding up probabilities
until your number is exceeded:

| Random $u$ | Cumulative interval it falls in | Action |
|---|---|---|
| 0.30 | (0.00, 0.45] | WAIT |
| 0.55 | (0.45, 0.60] | PASSIVE |
| 0.85 | (0.60, 1.00] | AGGRESSIVE |

That's how a probability vector becomes a chosen action.

**Regret.** At the end of T days, look back. Compute two things:

1. The total cost you actually paid, with your evolving strategy.
2. The total cost you would have paid if you had picked the *single
   best restaurant in hindsight* and gone there every night.

The difference is your regret:

$$\text{Regret}_T = \underbrace{\sum_{t=1}^{T} L_t(a_t)}_{\text{what you paid}} \;-\; \underbrace{\min_{a^\star} \sum_{t=1}^{T} L_t(a^\star)}_{\text{what the best fixed action would have paid}}$$

Regret says: "if I had been a clairvoyant who knew from day 1 which
restaurant would be best, how much would I have saved?" Lower regret
= smarter strategy.

### The Hedge update rule (the actual math)

After each tick you observed loss $L_t$ for the action $a_t$ you took.
Update only that action's weight:

$$w_{t+1}(a_t) = w_t(a_t) \cdot e^{-\eta L_t}$$

Then renormalize so the weights still sum to 1:

$$w_{t+1}(a) \leftarrow \frac{w_{t+1}(a)}{\sum_{a'} w_{t+1}(a')} \quad \text{for every } a$$

That's it. One line of math.

**$\eta$ ("eta") is the learning rate.** We use $\eta = 0.10$. Higher
$\eta$ → react faster to a single tick's loss (and noisier). Lower
$\eta$ → slower, more conservative.

### Why exp(−ηL)?

Three plain reasons:

1. **It's always positive.** $e^{-\text{anything}} > 0$. Weights stay
   legal probabilities — no risk of going negative or breaking the
   "sums to 1" rule.

2. **It punishes proportionally.** A loss of 2 shrinks the weight by
   a factor of $e^{-0.2}$. A loss of 4 shrinks it by $e^{-0.4}$ — twice
   as much in log-space. That's the right scaling for a quantity that
   should compound over many decisions.

3. **It barely moves on small losses.** A loss of 0.005 (the
   half-cent example below) gives multiplier $e^{-0.0005} \approx
   0.9995$ — nearly 1. Tiny losses don't whipsaw the weights.

A formal proof shows $e^{-\eta L}$ is the multiplier that minimizes
worst-case regret. We'll skip that proof; the intuition above is what
matters.

### What is the "regret bound"?

The famous theorem of Hedge says:

$$\text{Regret}_T \leq \mathcal{O}\!\left(\sqrt{T \log K}\right)$$

In plain words: **after $T$ decisions among $K$ actions, the total
amount you've left on the table compared to the best fixed action is
at most about $\sqrt{T \log K}$ — no matter how the losses happen, no
matter how adversarial the world is.**

For our problem $T = 75{,}000$ (ticks in a session) and $K = 3$
(actions):

$$\sqrt{75{,}000 \times \log 3} \approx \sqrt{75{,}000 \times 1.10} \approx \sqrt{82{,}500} \approx 287$$

So at the end of a 21-hour session, your total regret can be at most
~287 units higher than the best fixed action's total cost would have
been. Per tick that's $287/75{,}000 \approx 0.004$ — about a third of
a cent per decision.

The crucial property: regret grows with $\sqrt{T}$, not with $T$. Per
decision, it goes to *zero* as $T$ grows. **The longer you run, the
closer your average performance gets to the best fixed action's
performance.** That's the formal guarantee that makes Hedge worth
using.

### "But the optimal action depends on the market state."

Right. Hedge alone learns the **best fixed action** — it converges to
"if you had to pick ONE action and use it forever, which would be best
on average?" That's not the right question for execution because the
answer depends on what the market is doing.

The fix is **bucketed Hedge**: split the data into states (using the
Kalman pressure), and run a separate Hedge inside each state. Each
state-bucket has its own per-bucket "best fixed action," and the
overall policy is "do the best fixed action *for the current state*."

That's the entire idea behind 1D bucketed.

---

## Part 8 — A real worked example, end to end

OK, hands-on. Pick ONE real tick from session S7 and walk through
everything: parquet row → Kalman → bucket → probability vector →
sampled action → simulator → loss → Hedge update.

Every number below is what the actual code computes. You can verify
by running:
```
PYTHONPATH=. python scripts/_debug_one_tick.py
```

### The setup

- **Session**: S7 (the longest, 21 h, 76k ticks).
- **Tick number**: 800 (about 13 minutes into the session — past
  warmup, Kalman has converged).
- **What we want to know**: at this exact moment, what does the
  policy decide and why?

### Step 0 — the parquet row

The features pipeline ran offline. Tick 800 of S7 is:

| Field | Value | Plain meaning |
|---|---|---|
| ts_ms | 1,777,590,828,860 | Unix-ms timestamp (May 1 14:33 UTC) |
| best_bid_px | 76,279.16 | best price someone wants to buy at |
| best_ask_px | 76,279.17 | best price someone wants to sell at |
| spread_abs | 0.01 | 1 cent — standard for BTCUSDT |
| mid_price | 76,279.165 | average of bid and ask |
| depth_imbalance | +0.857 | mostly bid-side liquidity (buyers stacked) |
| ofi_l1 | +0.397 | mild buying pressure into top of book |
| vol_30s | 0.0000328 | calm market |

And the next tick's mid (we need it to simulate the fill outcome):

| Field | Value |
|---|---|
| next mid_price | 76,279.165 |

So the snapshot at this moment: **BTC at \$76,279, 1-cent spread,
order book has lots of bids stacked vs asks, recent OFI mildly
positive, calm volatility.**

### Step 1 — standardize the observation

The Kalman filter expects features on similar scales. We rescale only
vol_30s (depth_imbalance and ofi_l1 already live on sensible
magnitudes):

$$
z_{\text{raw}} =
\begin{pmatrix} 0.857 \\ 0.397 \\ 0.0000328 \end{pmatrix},\quad
\mathbf{c} =
\begin{pmatrix} 0 \\ 0 \\ 0.0000317 \end{pmatrix},\quad
\mathbf{s} =
\begin{pmatrix} 1 \\ 1 \\ 36{,}877 \end{pmatrix}
$$

By hand:

```
z_norm[0] = (0.857     − 0)         × 1       = 0.857
z_norm[1] = (0.397     − 0)         × 1       = 0.397
z_norm[2] = (0.0000328 − 0.0000317) × 36,877  = 0.0413
```

vol_30s went from 0.0000328 to 0.0413 — same information, useful
scale.

### Step 2 — Kalman prior

The filter has already processed ticks 0 through 799. The prior state
it carries forward is:

$$
\mathbf{x}_{\text{prior}} =
\begin{pmatrix} +0.0526 \\ -0.0965 \end{pmatrix}, \qquad
\mathbf{P}_{\text{prior}} =
\begin{pmatrix} 0.0344 & 0 \\ 0 & 0.0368 \end{pmatrix}
$$

Plain reading: "From the past 13 minutes, I believe pressure is
slightly positive (+0.05) and regime is slightly negative (calm).
My uncertainty in those numbers is small."

### Step 3 — Kalman predict

Project forward in time using F:

$$
\mathbf{F} =
\begin{pmatrix} 0.90 & 0 \\ 0 & 0.95 \end{pmatrix}
$$

By hand:

```
x_pred[0] = 0.90 × 0.0526 + 0    × (−0.0965)  =  +0.0474
x_pred[1] = 0    × 0.0526 + 0.95 × (−0.0965)  =  −0.0916
```

So $\mathbf{x}_{\text{pred}} = (+0.0474, -0.0916)^\top$. Both numbers
shrunk slightly toward zero (mean reversion), as F instructs.

For the predicted covariance:

```
F · P · F^T  = diag(0.81 × 0.0344,  0.9025 × 0.0368)
             = diag(0.02786,         0.03324)

P_pred       = F · P · F^T + Q
             = diag(0.02786 + 0.010,  0.03324 + 0.005)
             = diag(0.03786,           0.03824)
```

The predicted state has slightly *more* uncertainty than the prior,
because Q models random shocks that could have hit the state during
this tick.

### Step 4 — Kalman update

Now compare prediction with observation, weighted by uncertainty.

**Innovation** $\mathbf{y}$ = "how wrong was the prediction?":

$$
\mathbf{H}\,\mathbf{x}_{\text{pred}} =
\begin{pmatrix} 1 & 0 \\ 1 & 0 \\ 0 & 1 \end{pmatrix}
\begin{pmatrix} +0.0474 \\ -0.0916 \end{pmatrix} =
\begin{pmatrix} +0.0474 \\ +0.0474 \\ -0.0916 \end{pmatrix}
$$

```
y[0] = z_norm[0] − H·x_pred[0] = 0.857  − (+0.0474) = +0.8092
y[1] = z_norm[1] − H·x_pred[1] = 0.397  − (+0.0474) = +0.3491
y[2] = z_norm[2] − H·x_pred[2] = 0.0413 − (−0.0916) = +0.1330
```

Big positive innovations on rows 0 and 1: "the data says pressure is
much higher than we predicted." Small positive innovation on row 2:
"regime is slightly higher than predicted, too."

**Innovation covariance** $\mathbf{S} = \mathbf{H}\mathbf{P}_{\text{pred}}\mathbf{H}^\top + \mathbf{R}$:

$$
\mathbf{S} \approx
\begin{pmatrix}
0.486 & 0.038 & 0 \\
0.038 & 2.326 & 0 \\
0 & 0 & 1.038
\end{pmatrix}
$$

Diagonal entries are predicted noise levels for each observation
channel. Bigger entry = noisier observation = trust this one less.

**Kalman gain** $\mathbf{K} = \mathbf{P}_{\text{pred}}\mathbf{H}^\top\mathbf{S}^{-1}$ (a 2×3 matrix):

$$
\mathbf{K} \approx
\begin{pmatrix}
0.077 & 0.015 & 0 \\
0     & 0     & 0.037
\end{pmatrix}
$$

How to read this:
- Row 0 (pressure): apply 7.7% of depth_imbalance's surprise + 1.5%
  of ofi_l1's surprise to the pressure estimate. Don't touch pressure
  based on vol_30s.
- Row 1 (regime): apply 3.7% of vol_30s's surprise to regime. Don't
  touch regime based on the other two.

The cross-zero entries reflect that H has no cross-talk between
channels.

**Posterior state** $\mathbf{x}_{\text{post}} = \mathbf{x}_{\text{pred}} + \mathbf{K}\mathbf{y}$:

```
K·y[0] = 0.077 × 0.8092 + 0.015 × 0.3491 + 0    × 0.1330  =  +0.0673
K·y[1] = 0    × 0.8092 + 0    × 0.3491 + 0.037 × 0.1330  =  +0.0049

x_post[0] = +0.0474 + 0.0673 = +0.1147
x_post[1] = −0.0916 + 0.0049 = −0.0867
```

**After this tick the filter believes:**
- pressure = **+0.115** (now mild-buyer territory, up from prior)
- regime = **−0.087** (still calm)

The depth and OFI signals pulled pressure UP from where it was. The
filter combined "what I thought" with "what I just saw" using the
Kalman gain as the weighting recipe.

### Step 5 — bucket lookup

The 1D bucketed policy uses pressure to pick a bucket. Edges are
$[-0.5, -0.2, 0, +0.2, +0.5]$, giving 6 buckets:

```
bucket 0:  pressure ≤ -0.5
bucket 1:  -0.5 < pressure ≤ -0.2
bucket 2:  -0.2 < pressure ≤  0
bucket 3:   0   < pressure ≤ +0.2     ← pressure +0.115 lands HERE
bucket 4:  +0.2 < pressure ≤ +0.5
bucket 5:  +0.5 < pressure
```

We're in **bucket 3**: "slight buyer pressure." (Code uses binary
search via `bisect_right` to do this lookup in $O(\log N)$ time.)

### Step 6 — load the frozen weights for bucket 3

These were trained offline and saved to JSON. Bucket 3 was visited
19,322 times during S7 training, so its weights are well-converged:

```
WAIT       : 0.3235     ████████████████ 32.35 %
PASSIVE    : 0.2059     ██████████        20.59 %
AGGRESSIVE : 0.4706     ████████████████████████ 47.06 %
                                          ── total = 100.00 %
```

The policy learned that in mild-buyer territory, AGGRESSIVE is the
modal action (~47%). Makes sense: when buyers are pressing, "grab
inventory now before it gets more expensive."

### Step 7 — sample an action

The RNG (seeded with 0 for reproducibility) draws a uniform random
number in $[0, 1)$:

$$u = 0.6370$$

Walk along the cumulative distribution:
- WAIT covers [0, 0.3235)
- PASSIVE covers [0.3235, 0.5294)
- AGGRESSIVE covers [0.5294, 1.0000)

$u = 0.6370$ falls in AGGRESSIVE's slice → **chosen action = AGGRESSIVE**.

### Step 8 — simulate the fill, compute the loss

AGGRESSIVE = market buy at the current best ask:

```
fill_price  = 76,279.17     (= best_ask)
mid_price   = 76,279.165
next_mid    = 76,279.165    (no change in next tick)
```

**Slippage:**
```
slippage = fill_price − mid = 76,279.17 − 76,279.165 = +$0.005
```
We paid half a cent above mid. Cost of being aggressive.

**Adverse:**
```
adverse = max(0, fill_price − next_mid)
        = max(0, 76,279.17 − 76,279.165)
        = +$0.005
```
The price didn't move down, but we're still half a cent above where
the mid ended up. The adverse term captures this "we paid more than
we needed to" cost.

**Loss:**
```
L = slippage + λ × adverse
  = 0.005 + 0.10 × 0.005
  = +$0.0055
```

This single AGGRESSIVE decision cost us ~0.55 cents in combined loss.

### Step 9 — Hedge update

(The GUI runs in inference-only mode and doesn't do this update —
but in the offline training that *produced* the frozen weights, this
step happened on every tick.)

We just observed loss $L = +0.0055$ for action AGGRESSIVE.

```
multiplier = exp(−η × L) = exp(−0.10 × 0.0055) = exp(−0.00055) = 0.999450
```

Multiplier is *slightly less than 1* because the loss was positive
(small cost). We multiply only the AGGRESSIVE weight:

```
old AGGRESSIVE weight       : 0.4706
new (unnormalized)          : 0.4706 × 0.999450 = 0.470316
```

WAIT and PASSIVE are unchanged at 0.323502 and 0.205923 (untouched
by the update — only the chosen action's weight moves).

Now renormalize so the three weights still sum to 1:

```
sum = 0.323502 + 0.205923 + 0.470316 = 0.999741

WAIT       : 0.323502 / 0.999741 = 0.3236  (was 0.3235; +0.000084)
PASSIVE    : 0.205923 / 0.999741 = 0.2060  (was 0.2059; +0.000053)
AGGRESSIVE : 0.470316 / 0.999741 = 0.4704  (was 0.4706; −0.000137)
```

Tiny changes. AGGRESSIVE went down by 0.000137; the other two went up
slightly because of renormalization (the total had to stay at 1).

A single tick barely moves anything. But after **19,322 visits** to
bucket 3 alone, all those tiny pushes accumulated into the converged
distribution we saw at Step 6 (32% / 21% / 47%). That's how Hedge
learns: not from any single dramatic update, but from tens of
thousands of imperceptibly small updates compounding.

### Mental snapshot of the whole pipeline

| Step | Input | Operation | Output |
|---|---|---|---|
| 0 | (Binance) | feature pipeline | parquet row |
| 1 | parquet row | standardize | $\tilde{\mathbf{z}}$ |
| 2 | (memory) | carry-over | prior $\mathbf{x}, \mathbf{P}$ |
| 3 | prior | Kalman predict | $\mathbf{x}_{\text{pred}}, \mathbf{P}_{\text{pred}}$ |
| 4 | prior + obs | Kalman update | posterior $\mathbf{x}, \mathbf{P}$ |
| 5 | pressure | binary search on edges | bucket index |
| 6 | bucket idx | JSON lookup | weight vector |
| 7 | weights + RNG | inverse-CDF sample | chosen action |
| 8 | action + book | simulator + loss | scalar loss |
| 9 | weights + loss | $w \leftarrow w \cdot e^{-\eta L}$ + renorm | updated weights |

Each row is a tiny operation. Composed, they are the entire decision
loop. Run it 38,000 times → one session. Run it across 4 sessions →
the experiments. That's all the system does.
