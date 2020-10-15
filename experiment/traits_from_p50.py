#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def slope(P50, which='Christoffersen'):
    """
    Finds the slope of a plant's vulnerability curve, using the water potential
    associated with a 50% decrease in hydraulic conductance as well as a
    statistical relationship, either updated from Christoffersen et al. (2016),
    which was specifically developped for tropical forests, or from
    Martin-StPaul et al. (2017), which was not developped with an eye to
    tropical forests.

    Arguments:
    ----------
    P50: float
        water potential at 50% decrease in hydraulic conductance [-MPa]

    Returns:
    --------
    sl: float
        the slope of the plant's vulnerability curve [% MPa-1]
    """
    if which == 'Christoffersen':
        slope = 65.15 * (P50) ** (-1.25) # % MPa-1 (Christoffersen, pers. com.)
    if which == 'Martin-StPaul':
        slope = 16. + np.exp(-P50) * 1092 # % MPa-1 (Martin-StPaul et al. 2017)

    return -slope

def get_P88(Px1, Px2, x1, x2):
    """
    Finds the leaf water potential associated with a specific x% decrease in
    hydraulic conductance, using the plant vulnerability curve.
    Arguments:
    ----------
    Px: float
        leaf water potential [MPa] at which x% decrease in hydraulic conductance
        is observed
    x: float
        percentage loss in hydraulic conductance
    Returns:
    --------
    P88: float
        leaf water potential [MPa] at which 88% decrease in hydraulic
        conductance is observed
    """
    Px1 = np.abs(Px1)
    Px2 = np.abs(Px2)
    x1 /= 100. # normalise between 0-1
    x2 /= 100.
    # c is derived from both expressions of b
    try:
        c = np.log(np.log(1. - x1) / np.log(1. - x2)) / (np.log(Px1) -
                                                         np.log(Px2))
    except ValueError:
        c = np.log(np.log(1. - x2) / np.log(1. - x1)) / (np.log(Px2) -
                                                         np.log(Px1))
    b = Px1 / ((- np.log(1 - x1)) ** (1. / c))
    P88 = -b * ((- np.log(0.12)) ** (1. / c)) # MPa
    return P88

def get_weibull_params(p12=None, p50=None, p88=None):
    """
    Calculate the Weibull sensitivity (b) and shape (c) parameters
    """
    print(p12, p50, p88)
    if p12 is not None and p50 is not None:
        px1 = p12
        x1 = 12. / 100.
        px2 = p50
        x2 = 50. / 100.
    elif p12 is not None and p88 is not None:
        px1 = p12
        x1 = 12. / 100.
        px2 = p88
        x2 = 88. / 100.
    elif p50 is not None and p88 is not None:
        px1 = p50
        x1 = 50. / 100.
        px2 = p88
        x2 = 88. / 100.

    num = np.log(np.log(1. - x1) / np.log(1. - x2))
    den = np.log(px1) - np.log(px2)
    c = num / den

    b = px1 / ((-np.log(1 - x1))**(1. / c))

    return b, c

def get_weibull_params_cavit(p50, p98):
    """
    Calculate the Weibull sensitivity (b) and shape (c) parameters
    """

    px1 = p50
    x1 = 50. / 100.
    px2 = p98
    x2 = 98. / 100.

    num = np.log(np.log(1. - x1) / np.log(1. - x2))
    den = np.log(px1) - np.log(px2)
    c = num / den

    b = px1 / ((-np.log(1 - x1))**(1. / c))

    return b, c


p12 = -3.08
p50 = -3.63
p98 = -4.5
Kmax = 2.0

(b, c) = get_weibull_params(p12=np.abs(p12), p50=np.abs(p50))
print(b, c)

(b, c) = get_weibull_params_cavit(np.abs(p50), np.abs(p98))
print(b, c)


p = np.linspace(-10, 0.0)
weibull = np.exp(-(-p / b)**c)
Kplant = Kmax * weibull
plc = 100.0 * (1.0 - Kplant / Kmax)
plt.plot(p, plc)
plt.xlim(-5, 0)

plt.show()
