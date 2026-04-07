"""
Kalibrierung des Svensson-Modells auf Bundesbank-Spot-Rates.

Ablauf:
  1. load_bundesbank(path)  — CSV einlesen, letzten Tagesquerschnitt extrahieren
  2. grid_search(...)       — (λ1, λ2)-Raster, je OLS für (β0..β3) → Top-k Startpunkte
  3. calibrate(...)         — L-BFGS-B von jedem Startpunkt, bestes Ergebnis zurückgeben

Zielfunktion: min Σ (y_market(τ_i) − y_model(τ_i; θ))²
Constraints:  λ1, λ2 > 0  und  λ1 ≠ λ2  (sonst Degeneration)
"""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import lstsq
from svensson import spot_rate
import pandas as pd
import matplotlib.pyplot as plt


def L(tau,lam):
    return lam*(1-np.exp(-tau/lam))/tau

def H(tau,lam):
    return lam*(1-np.exp(-tau/lam))/tau - np.exp(-tau/lam)

def _design_matrix(taus, lam1, lam2):
    A = np.empty((len(taus), 4))
    A[:, 0] = 1
    A[:, 1] = L(taus, lam1)
    A[:, 2] = H(taus, lam1)
    A[:, 3] = H(taus, lam2)
    return A


def grid_search(taus, yields, lam1_grid, lam2_grid, k=5):
    results = []
    for lam1 in lam1_grid:
        for lam2 in lam2_grid:
            if lam1 == lam2:
                continue
            A = _design_matrix(taus, lam1, lam2)
            betas, _, _, _ = lstsq(A, yields) # least square
            sse = np.sum((yields - A @ betas) ** 2)
            results.append((sse, betas, lam1, lam2))
    results.sort(key=lambda x: x[0])
    return results[:k]
#end grid_search


def calibrate(taus, yields):
    lam1_grid = np.linspace(1,10,10)
    lam2_grid = np.linspace(1,10,10)
    
    k=5
    results = grid_search(taus, yields, lam1_grid, lam2_grid, k=k)
    best = None
    for i in range(k):
        sse, betas, lam1, lam2 = results[i]
        x0 = [betas[0], betas[1], betas[2], betas[3], lam1, lam2]
        
        result = minimize(fun=lambda x: np.sum((yields - spot_rate(taus, *x)) ** 2), x0=x0, method='L-BFGS-B', bounds=((None,None),(None,None),(None,None),(None,None),(1e-4,None),(1e-4,None)))
        if best is None or result.fun < best.fun:
            best = result
        #end if
    #end for
    return best
#end calibrate


def load_bundesbank(path):
    # 5 Metadaten-Zeilen überspringen, kein Header (Spaltennamen sind Seriennummern)
    # Spalten alternieren: Wert ; FLAG ; Wert ; FLAG → jede zweite behalten
    # Datum: DD.MM.YY, Dezimaltrennzeichen: Komma, fehlende Werte: '.'
    df = pd.read_csv(
        path, sep=';', skiprows=5, header=None,
        index_col=0, na_values=['.', ''], dtype=str
    )
    df = df.iloc[:, 0::2]  # nur Wert-Spalten, FLAG-Spalten verwerfen
    df = df.apply(lambda col: col.str.replace(',', '.', regex=False))
    df = df.apply(pd.to_numeric, errors='coerce')

    df.index = pd.to_datetime(df.index, format='%d.%m.%y')
    taus = np.array([0.5] + list(range(1, 31)))  # 0.5, 1, 2, ..., 30 Jahre

    row = df.dropna(how='any').iloc[-1]  # aktuellste vollständige Zeile
    yields = row.values.astype(float) / 100  # Prozent → Dezimal
    return taus, yields
#end load_bundesbank


if __name__ == "__main__":
    taus, yields = load_bundesbank("data/bundesbank.csv")

    print(f"Geladene Laufzeiten: {taus}")
    print(f"Spot Rates:          {np.round(yields * 100, 4)} %")

    result = calibrate(taus, yields)
    b0, b1, b2, b3, lam1, lam2 = result.x

    print("\n--- Kalibrierte Svensson-Parameter ---")
    print(f"  beta0 = {b0:.6f}   (Long-run level)")
    print(f"  beta1 = {b1:.6f}   (Short-end slope)")
    print(f"  beta2 = {b2:.6f}   (Erster Hump)")
    print(f"  beta3 = {b3:.6f}   (Zweiter Hump)")
    print(f"  lam1  = {lam1:.6f}")
    print(f"  lam2  = {lam2:.6f}")
    print(f"  SSE   = {result.fun:.8f}")

    tau_fine = np.linspace(taus.min(), taus.max(), 300)
    y_model  = spot_rate(tau_fine, b0, b1, b2, b3, lam1, lam2)

    plt.figure(figsize=(9, 5))
    plt.scatter(taus, yields * 100, color='black', zorder=5, label='Bundesbank (beobachtet)')
    plt.plot(tau_fine, y_model * 100, color='steelblue', linewidth=2, label='Svensson (kalibriert)')
    plt.xlabel('Laufzeit (Jahre)')
    plt.ylabel('Spot Rate (%)')
    plt.title('Svensson Yield Curve Fit — Bundesbank')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()