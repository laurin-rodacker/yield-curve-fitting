"""
Svensson (1994) yield curve model — 6 parameters: beta0, beta1, beta2, beta3, lam1, lam2.
Extends Nelson-Siegel with a second hump term (beta3, lam2).

Spot rate:    y(tau) = beta0 + beta1*L(tau,lam1) + beta2*H(tau,lam1) + beta3*H(tau,lam2)
Forward rate: f(tau) = d/dtau [tau * y(tau)]
"""

import numpy as np

def spot_rate(tau, beta0, beta1, beta2, beta3, lam1, lam2):
    y = beta0 + beta1*lam1*(1-np.exp(-tau/lam1))/tau + beta2*(lam1*(1-np.exp(-tau/lam1))/tau - np.exp(-tau/lam1)) + beta3*(lam2*(1-np.exp(-tau/lam2))/tau - np.exp(-tau/lam2))
    return y
#end spot_rate

def forward_rate(tau, beta0, beta1, beta2, beta3, lam1, lam2):
    f = beta0 + beta1*np.exp(-tau/lam1) + beta2 * tau/lam1 * np.exp(-tau/lam1) + beta3 * tau/lam2 * np.exp(-tau/lam2)
    return f
#end forward_rate
