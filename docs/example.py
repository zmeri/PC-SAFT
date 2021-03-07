import numpy as np
from pcsaft import pcsaft_den

# Toluene
x = np.asarray([1.])
m = np.asarray([2.8149])
s = np.asarray([3.7169])
e = np.asarray([285.69])
pyargs = {'m':m, 's':s, 'e':e}

t = 320 # K
p = 101325 # Pa
den = pcsaft_den(t, p, x, pyargs, phase='liq')
print('Density of toluene at {} K:'.format(t), den, 'mol m^-3')

# Water using default 2B association scheme
x = np.asarray([1.])
m = np.asarray([1.2047])
e = np.asarray([353.95])
volAB = np.asarray([0.0451])
eAB = np.asarray([2425.67])

t = 274
p = 101325
s = np.asarray([2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t)]) # temperature dependent sigma is used for better accuracy
pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB}
den = pcsaft_den(t, p, x, pyargs, phase='liq')
print('Density of water at {} K:'.format(t), den, 'mol m^-3')

# Water using 4C association scheme
x = np.asarray([1.])
m = np.asarray([1.2047])
e = np.asarray([353.95])
volAB = np.asarray([0.0451])
eAB = np.asarray([2425.67])
assoc_schemes = ['4c']

t = 274
p = 101325
s = np.asarray([2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t)]) # temperature dependent sigma is used for better accuracy
pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'assoc_scheme':assoc_schemes}
den = pcsaft_den(t, p, x, pyargs, phase='liq')
print('Density of water at {} K:'.format(t), den, 'mol m^-3')
