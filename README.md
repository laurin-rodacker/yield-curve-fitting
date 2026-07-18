# Yield Curve Fitting — Nelson-Siegel & Svensson

Calibration of **Nelson-Siegel** (1987) and **Svensson** (1994) yield curve models on real Bundesbank spot rate data. Pure `scipy`/`numpy` — no QuantLib, no sklearn.

See [`EXPLAINER.md`](EXPLAINER.md) for the full mathematical derivation.

---

## Models

### Nelson-Siegel (4 parameters)

$$y(\tau) = \beta_0 + \beta_1 \cdot \frac{1 - e^{-\tau/\lambda}}{\tau/\lambda} + \beta_2 \cdot \left(\frac{1 - e^{-\tau/\lambda}}{\tau/\lambda} - e^{-\tau/\lambda}\right)$$

| Parameter | Meaning |
|-----------|---------|
| β₀ | Long-run level: `lim(τ→∞) y(τ) = β₀` |
| β₁ | Short-end slope: `lim(τ→0) y(τ) = β₀ + β₁` |
| β₂ | Medium-term hump/trough |
| λ  | Decay speed (hump peaks near τ ≈ λ) |

Forward rates are derived analytically as `f(τ) = d/dτ [τ · y(τ)]`.

### Svensson (6 parameters)

Extends Nelson-Siegel with a second independent hump term:

$$y(\tau) = \beta_0 + \beta_1 \cdot L(\tau, \lambda_1) + \beta_2 \cdot H(\tau, \lambda_1) + \beta_3 \cdot H(\tau, \lambda_2)$$

where `L(τ,λ) = (1 − e^{−τ/λ}) / (τ/λ)` and `H(τ,λ) = L(τ,λ) − e^{−τ/λ}`.

---

## Calibration

Non-convex problem (due to λ) solved in two stages:

1. **Grid search** over a (λ₁, λ₂) grid — for each pair, solve for (β₀…β₃) via OLS → collect top-k starting points
2. **L-BFGS-B** (scipy) from those starting points → fine optimization over all 6 parameters

The Bundesbank itself publishes its calibrated Svensson parameters, enabling direct validation against the official values.

---

## Setup

```bash
pip install numpy scipy pandas matplotlib
```

---

## Data

Download the spot rate time series from the Bundesbank statistics portal:

> Zeitreihendatenbank → Geld- und Kapitalmärkte → Zinssätze und Renditen →
> **Zinsstruktur am Rentenmarkt – Schätzwerte** →
> Zinsstrukturkurve für börsennotierte Bundeswertpapiere (Tageswerte)

Save the CSV as `data/bundesbank.csv`.

---

## Usage

```bash
python calibration.py
```

Prints the calibrated parameters and displays a plot of the fitted curve vs. observed spot rates.
