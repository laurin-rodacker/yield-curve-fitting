"""                                                                                                        
Nelson-Siegel (1987) yield curve model — 4 parameters: beta0, beta1, beta2, lam.                    
                                                                                                           
Spot rate:   y(tau)  = beta0 + beta1*L(tau) + beta2*H(tau)                                                 
Forward rate: f(tau) = d/dtau [tau * y(tau)]                                                               
"""      
import numpy as np

def spot_rate(tau, beta0, beta1, beta2, lam):
    y = beta0 + beta1*lam*(1-np.exp(-tau/lam))/tau + beta2*(lam*(1-np.exp(-tau/lam))/tau - np.exp(-tau/lam))
    return y
#end spot_rate

def forward_rate(tau, beta0, beta1, beta2, lam):
    f = beta0 + beta1*np.exp(-tau/lam) + beta2 * tau/lam * np.exp(-tau/lam)
    return f
#end forward_rate