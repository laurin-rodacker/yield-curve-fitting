# Yield Curve Fitting — Code Erklärung

---

## 1. `nelson_siegel.py`

### Die Mathematik

Nelson-Siegel (1987) parametrisiert die **instantane Forward-Rate** als Summe aus Niveau, abklingendem Slope-Term und einem Hump:

$$f(\tau) = \beta_0 + \beta_1\, e^{-\tau/\lambda} + \beta_2\, \frac{\tau}{\lambda}\, e^{-\tau/\lambda}$$

Die **Spot Rate** (Zero Rate) ist der Durchschnitt der Forward-Rate von $0$ bis $\tau$:

$$y(\tau) = \frac{1}{\tau}\int_0^\tau f(s)\, ds = \beta_0 + \beta_1\, L(\tau,\lambda) + \beta_2\, H(\tau,\lambda)$$

mit

$$L(\tau,\lambda) = \frac{1 - e^{-\tau/\lambda}}{\tau/\lambda}, \qquad H(\tau,\lambda) = L(\tau,\lambda) - e^{-\tau/\lambda}$$

**Intuition der Parameter:**

| Parameter | Bedeutung |
|-----------|-----------|
| $\beta_0$ | Long-run Level: $\lim_{\tau\to\infty} y(\tau) = \beta_0$ |
| $\beta_1$ | Short-End Slope: $\lim_{\tau\to0} y(\tau) = \beta_0+\beta_1$ |
| $\beta_2$ | Hump/Trough in der Mitte der Kurve, Maximum bei $\tau \approx \lambda$ |
| $\lambda$ | Skaliert, wo der Hump auf der Laufzeitachse sitzt |

### Der Code

```python
def spot_rate(tau, beta0, beta1, beta2, lam):
    y = beta0 + beta1*lam*(1-np.exp(-tau/lam))/tau + beta2*(lam*(1-np.exp(-tau/lam))/tau - np.exp(-tau/lam))
    return y
```

Direkte Umsetzung von $L$ und $H$, ausgeschrieben statt als Hilfsfunktion — vermeidet Funktionsaufruf-Overhead bei Kalibrierung mit vielen Auswertungen.

### Forward Rate durch Differentiation

`forward_rate` berechnet $f(\tau)$ nicht durch numerisches Differenzieren, sondern analytisch über die Identität $f(\tau) = \frac{d}{d\tau}\left[\tau \cdot y(\tau)\right]$ — Ableitung der Definition $y(\tau) = \frac{1}{\tau}\int_0^\tau f(s)\,ds$ nach $\tau$.

Für den $\beta_2$-Term ergibt das (Produktregel + Kettenregel):

$$\frac{d}{d\tau}\Big[\lambda(1-e^{-\tau/\lambda}) - \tau e^{-\tau/\lambda}\Big] = e^{-\tau/\lambda} - \Big(e^{-\tau/\lambda} - \tfrac{\tau}{\lambda}e^{-\tau/\lambda}\Big) = \frac{\tau}{\lambda}e^{-\tau/\lambda}$$

— genau der dritte Term in `forward_rate`. Analytisch statt numerisch heißt: exakt, kein Diskretisierungsfehler, keine Wahl einer Schrittweite $h$.

```python
def forward_rate(tau, beta0, beta1, beta2, lam):
    f = beta0 + beta1*np.exp(-tau/lam) + beta2 * tau/lam * np.exp(-tau/lam)
    return f
```

---

## 2. `svensson.py`

### Warum eine Erweiterung?

Nelson-Siegel hat nur einen Hump — reicht nicht immer, um Kurven mit zwei Wendepunkten (z.B. Buckel im kurzen *und* mittleren Bereich) zu fitten. Svensson (1994) fügt einen zweiten, unabhängigen Hump-Term hinzu:

$$y(\tau) = \beta_0 + \beta_1\, L(\tau,\lambda_1) + \beta_2\, H(\tau,\lambda_1) + \beta_3\, H(\tau,\lambda_2)$$

Gleiche $L$/$H$-Bausteine wie bei NS, aber $\beta_3 H(\tau,\lambda_2)$ mit eigenem Decay $\lambda_2$ erlaubt einen zweiten Buckel an anderer Stelle der Kurve.

### Degenerationsfall

Wenn $\lambda_1 = \lambda_2$, kollabieren die beiden Hump-Terme zu einem — das Modell hat dann effektiv nur noch die NS-Freiheitsgrade, aber 6 statt 4 Parameter, was die Optimierung instabil macht (unendlich viele $(\beta_2,\beta_3)$-Kombinationen liefern denselben Fit). Deshalb erzwingt `calibration.py` explizit $\lambda_1 \neq \lambda_2$.

```python
def spot_rate(tau, beta0, beta1, beta2, beta3, lam1, lam2):
    y = beta0 + beta1*lam1*(1-np.exp(-tau/lam1))/tau + beta2*(lam1*(1-np.exp(-tau/lam1))/tau - np.exp(-tau/lam1)) + beta3*(lam2*(1-np.exp(-tau/lam2))/tau - np.exp(-tau/lam2))
    return y
```

---

## 3. `calibration.py`

### Das Problem: Non-Konvexität

Für feste $(\lambda_1,\lambda_2)$ ist $y(\tau)$ **linear** in $(\beta_0,\beta_1,\beta_2,\beta_3)$ — ein reines Least-Squares-Problem, das exakt lösbar ist. Die Non-Konvexität kommt ausschließlich von $\lambda_1,\lambda_2$, die *innerhalb* der Exponentialfunktionen stehen. Ein Optimierer, der alle 6 Parameter gleichzeitig aus einem zufälligen Startpunkt sucht, bleibt leicht in einem lokalen Minimum hängen.

### Die Lösung: Variable Projection (Grid Search + OLS, dann L-BFGS-B)

**Schritt 1 — Grid Search über $(\lambda_1,\lambda_2)$:**
Für jedes Paar im Raster wird das *lineare* Teilproblem exakt gelöst:

```python
def _design_matrix(taus, lam1, lam2):
    A = np.empty((len(taus), 4))
    A[:, 0] = 1
    A[:, 1] = L(taus, lam1)
    A[:, 2] = H(taus, lam1)
    A[:, 3] = H(taus, lam2)
    return A
```

$$\hat\beta(\lambda_1,\lambda_2) = \arg\min_\beta \|y_{\text{market}} - A(\lambda_1,\lambda_2)\,\beta\|^2 \quad\text{via } \texttt{scipy.linalg.lstsq}$$

Die $k$ besten $(\lambda_1,\lambda_2,\hat\beta)$-Kombinationen nach SSE werden als Startpunkte behalten — das deckt den $\lambda$-Raum grob ab, ohne dass eine einzige nichtlineare Optimierung über alle 6 Parameter laufen muss.

**Schritt 2 — Politur mit L-BFGS-B:**
Von jedem der $k$ Startpunkte aus läuft eine volle nichtlineare Optimierung über alle 6 Parameter gemeinsam:

```python
result = minimize(
    fun=lambda x: np.sum((yields - spot_rate(taus, *x)) ** 2),
    x0=x0, method='L-BFGS-B',
    bounds=((None,None),(None,None),(None,None),(None,None),(1e-4,None),(1e-4,None))
)
```

Das beste Ergebnis über alle $k$ Läufe gewinnt. Die Grid-Search-Startpunkte sind der Grund, warum das robuster ist als ein einzelner L-BFGS-B-Lauf ab einem geratenen Startwert — non-konvexe Probleme brauchen mehrere Startpunkte, um das globale Minimum wahrscheinlich zu treffen.

**Bounds:** $\lambda_1,\lambda_2 > 0$ (Zerfallsraten müssen positiv sein, sonst divergiert $e^{-\tau/\lambda}$ für $\tau>0$).

### Validierung

Die Bundesbank veröffentlicht ihre eigenen kalibrierten Svensson-Parameter für dieselben Handelstage. Das erlaubt einen direkten Abgleich der hier berechneten $(\beta_0,\ldots,\beta_3,\lambda_1,\lambda_2)$ gegen die offiziellen Werte — ein Vorteil gegenüber Bootstrapping-Ansätzen, wo es keine offizielle Referenz zum Gegenchecken gibt.
