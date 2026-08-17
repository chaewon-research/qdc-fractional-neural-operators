from __future__ import annotations
import numpy as np
from scipy.stats import t


def mean_std(x):
    a=np.asarray(x,dtype=float); return float(a.mean()), float(a.std(ddof=1))


def paired_t_interval(differences, confidence=0.95):
    d=np.asarray(differences,dtype=float); n=d.size; mean=float(d.mean()); sd=float(d.std(ddof=1))
    q=float(t.ppf(0.5+confidence/2.0,n-1)); hw=q*sd/np.sqrt(n)
    return mean, mean-hw, mean+hw
