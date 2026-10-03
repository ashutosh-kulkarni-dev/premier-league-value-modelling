# Path C — Transfer-Fee Prediction Model

**Deferred extension to the value-wage project.** Pick up after Path A ships.

## Thesis

Predict **what buyers actually pay** in observed Premier League-connected
transfer transactions, as a function of player attributes, performance,
contract state, and buyer/seller context.

The target is `log(transfer_fee_eur)` on *paid* permanent transfers (plus
loans-with-obligation-to-buy). This is a **behavioural** quantity — it
reflects not only player quality but buyer-specific premiums, auction
dynamics, release clauses, contract leverage, and market inflation.

## Why fees, not values

| | Market value (Path A) | Fee (Path C) |
|---|---|---|
| What it is | Crowd-sourced estimate | Observed transaction |
| Causal cleanliness | Guess about a hypothetical | Decision somebody made |
| Overpayment visibility | Hidden (anchored to consensus) | Exposed in residuals |
| Business question answered | "What's the going rate?" | "What will Chelsea pay?" |

Path A gives a stable, averaged read on market consensus. Path C gives a
noisier but far more interesting signal: the **gap between model-predicted fee
and the fee actually paid** is itself the deliverable — a per-club scorecard
of who routinely overpays and who routinely finds bargains.

## The overpayment concern (and why it isn't fatal)

If Chelsea pays €100M for Mudryk, the ground truth is €100M even though his
on-pitch output warrants ~€25M. Three mechanisms stop this from corrupting
the model:

1. **Averaging across ~2,000 transactions.** No single overpayment dominates
   the fit for the "young winger, decent xG, long contract" bucket; dozens
   of saner transfers in the same bucket pull the prediction toward the
   median.
2. **Buyer-club features isolate the premium.** With `buyer_wage_tier`,
   `buyer_recent_european_spend`, `buyer_league`, the model learns a
   "Chelsea tax" or "Saudi tax" as a buyer effect, not a player attribute.
3. **MAE / Huber loss is median-anchored.** A single €100M outlier two σ
   above the model's prediction barely moves the fit.

Honest limits (documented up front, not buried):
- Can't predict **true worth** — nobody can; it isn't observable.
- Can't predict fees in **off-manifold** regimes (first-ever Saudi Pro
  League mega-signings, 2023).
- Can't correct for a **systematic market bias** (if transfer fees are
  universally 30% inflated by agent incentives, the model inherits it).

## Deliverables

Three complementary outputs per hypothetical transfer:

```
Player: Alejandro Garnacho   Buyer: Chelsea   Date: 2025-08

Predicted fee:                €55M   [€38M, €74M]
  = base fee for his profile:  €35M
  + Chelsea-buyer premium:     +€15M
  + contract-length factor:    +€5M

Market-value estimate (Path A): €42M
Post-transfer realised value:   €18M   (next-season xG+xA, if available)
```

Plus two leaderboards generated from residuals across all historical
transfers in scope:

- **Overpayer leaderboard** (fee > model, averaged over 10y of signings).
- **Bargain leaderboard** (fee < model, same).

These are what a scouting department would actually pin to a wall.

## Scope guard-rails

- Last **10 years** of transfers (older fees need inflation adjustment
  that becomes guesswork).
- **Top-7 leagues** on either buyer or seller side.
- Permanent transfers + loans-with-obligation only. Pure loans excluded.
- `fee > 0 AND fee IS NOT NULL` — the ~30% "undisclosed" fees are
  dropped, not imputed.

## Code reuse from Path A

~90% of the Path A codebase is target-agnostic and reusable:

- `src/value_wage/preprocess.py` — unchanged
- `src/value_wage/models/` — unchanged
- `src/value_wage/tuning.py` — unchanged
- `src/value_wage/evaluate.py` — unchanged
- `src/value_wage/mispricing.py` — rename to `residual_analysis.py`,
  core logic unchanged

Rewrites needed:
- `src/value_wage/data.py` — new source (Kaggle transfermarkt dump)
- `src/value_wage/features.py` — buyer/seller context features added
- `src/value_wage/splits.py` — time-based by transfer window, not season
- A new `src/value_wage/buyer_profile.py` — club wage/league/UEFA features

Target of the project: fork the Path A repo into `fee-model/`, keep the
diff small, publish both as sibling capstones.
