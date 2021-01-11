# -*- coding: utf-8 -*-
"""
Tests for checking that the PC-SAFT functions are working correctly.

@author: Zach Baird
"""
import numpy as np
from pcsaft import pcsaft_den, pcsaft_hres, pcsaft_gres, pcsaft_sres
from pcsaft import flashTQ, flashPQ, pcsaft_Hvap
from pcsaft import dielc_water, pcsaft_osmoticC, pcsaft_fugcoef
from pcsaft import pcsaft_cp, pcsaft_ares, pcsaft_dadt, pcsaft_p

def test_hres(print_result=False):
    """Test the residual enthalpy function to see if it is working correctly."""
    if print_result:
        print('------ 325 K ------')
        print('\t\t\t PC-SAFT\t Reference \tRelative error')
    t = 325 # K
    p = 101325 # Pa
    # all reference values are from PC-SAFT implemented in Aspen Plus

    # Toluene ----------
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    pyargs = {'m':m, 's':s, 'e':e}

    den = pcsaft_den(t, p, x, pyargs, phase='liq')
    ref = -36809.39 # J mol^-1
    calc = pcsaft_hres(t, den, x, pyargs)
    if print_result:
        print('Toluene, liquid:\t\t', calc, ref, (calc-ref)/ref*100, 'J/mol')
    assert abs((calc-ref)/ref*100) < 1e-2
    den = pcsaft_den(t, p, x, pyargs, phase='vap')
    ref = -362.6777 # J mol^-1
    calc = pcsaft_hres(t, den, x, pyargs)
    if print_result:
        print('Toluene, vapor:\t\t', calc, ref, (calc-ref)/ref*100, 'J/mol')
    assert abs((calc-ref)/ref*100) < 1e-2

    # Acetic acid ---------
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB}

    den = pcsaft_den(t, p, x, pyargs, phase='liq')
    ref = -38924.64 # J mol^-1
    calc = pcsaft_hres(t, den, x, pyargs)
    if print_result:
        print('Acetic acid, liquid:\t\t', calc, ref, (calc-ref)/ref*100, 'J/mol')
    assert abs((calc-ref)/ref*100) < 1e-2
    den = pcsaft_den(t, p, x, pyargs, phase='vap')
    ref = -15393.63 # J mol^-1
    calc = pcsaft_hres(t, den, x, pyargs)
    if print_result:
        print('Acetic acid, vapor:\t\t', calc, ref, (calc-ref)/ref*100, 'J/mol')
    assert abs((calc-ref)/ref*100) < 1e-2


def test_sres(print_result=False):
    """Test the residual entropy function to see if it is working correctly."""
    if print_result:
        print('------ 325 K ------')
        print('\t\t\t PC-SAFT\t Reference \tRelative error')
    t = 325 # K
    p = 101325 # Pa
    # all reference values are from PC-SAFT implemented in Aspen Plus

    # Toluene ----------
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    pyargs = {'m':m, 's':s, 'e':e}

    den = pcsaft_den(t, p, x, pyargs, phase='liq')
    ref = -96.3692 # J mol^-1 K^-1
    calc = pcsaft_sres(t, den, x, pyargs)
    if print_result:
        print('Toluene, liquid:\t\t', calc, ref, (calc-ref)/ref*100, 'J/mol/K')
    assert abs((calc-ref)/ref*100) < 1e-2
    den = pcsaft_den(t, p, x, pyargs, phase='vap')
    ref = -0.71398 # J mol^-1 K^-1
    calc = pcsaft_sres(t, den, x, pyargs)
    if print_result:
        print('Toluene, vapor:\t\t', calc, ref, (calc-ref)/ref*100, 'J/mol/K')
    assert abs((calc-ref)/ref*100) < 1e-2

    # Acetic acid ---------
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB}

    den = pcsaft_den(t, p, x, pyargs, phase='liq')
    ref = -98.1127 # J mol^-1 K^-1
    calc = pcsaft_sres(t, den, x, pyargs)
    if print_result:
        print('Acetic acid, liquid:\t\t', calc, ref, (calc-ref)/ref*100, 'J/mol/K')
    assert abs((calc-ref)/ref*100) < 1e-2
    den = pcsaft_den(t, p, x, pyargs, phase='vap')
    ref = -40.8743 # J mol^-1 K^-1
    calc = pcsaft_sres(t, den, x, pyargs)
    if print_result:
        print('Acetic acid, vapor:\t\t', calc, ref, (calc-ref)/ref*100, 'J/mol/K')
    assert abs((calc-ref)/ref*100) < 1e-2


def test_gres(print_result=False):
    """Test the residual Gibbs energy function to see if it is working correctly."""
    if print_result:
        print('------ 325 K ------')
        print('\t\t\t PC-SAFT\t Reference \tRelative error')
    t = 325 # K
    p = 101325 # Pa
    # all reference values are from PC-SAFT implemented in Aspen Plus

    # Toluene ----------
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    pyargs = {'m':m, 's':s, 'e':e}

    den = pcsaft_den(t, p, x, pyargs, phase='liq')
    ref = -5489.384 # J mol^-1
    calc = pcsaft_gres(t, den, x, pyargs)
    if print_result:
        print('Toluene, liquid:\t\t', calc, ref, (calc-ref)/ref*100, 'J/mol')
    assert abs((calc-ref)/ref*100) < 1e-2
    den = pcsaft_den(t, p, x, pyargs, phase='vap')
    ref = -130.6339 # J mol^-1
    calc = pcsaft_gres(t, den, x, pyargs)
    if print_result:
        print('Toluene, vapor:\t\t', calc, ref, (calc-ref)/ref*100, 'J/mol')
    assert abs((calc-ref)/ref*100) < 1e-2

    # Acetic acid ---------
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB}

    den = pcsaft_den(t, p, x, pyargs, phase='liq')
    ref = -7038.004 # J mol^-1
    calc = pcsaft_gres(t, den, x, pyargs)
    if print_result:
        print('Acetic acid, liquid:\t\t', calc, ref, (calc-ref)/ref*100, 'J/mol')
    assert abs((calc-ref)/ref*100) < 1e-2
    den = pcsaft_den(t, p, x, pyargs, phase='vap')
    ref = -2109.459 # J mol^-1
    calc = pcsaft_gres(t, den, x, pyargs)
    if print_result:
        print('Acetic acid, vapor:\t\t', calc, ref, (calc-ref)/ref*100, 'J/mol')
    assert abs((calc-ref)/ref*100) < 1e-2


def test_density(print_result=False):
    """Test the density function to see if it is working correctly."""
    # Toluene
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    pyargs = {'m':m, 's':s, 'e':e}

    ref = 9135.590853014008 # source: reference EOS in CoolProp
    calc = pcsaft_den(320, 101325, x, pyargs, phase='liq')
    if print_result:
        print('##########  Test with toluene  ##########')
        print('----- Density at 320 K and 101325 Pa -----')
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 2

    # Water
    m = np.asarray([1.2047])
    e = np.asarray([353.95])
    volAB = np.asarray([0.0451])
    eAB = np.asarray([2425.67])

    ref = 55502.5970532902 # source: IAWPS95 EOS
    t = 274
    s = np.asarray([2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t)])
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB}
    calc = pcsaft_den(t, 101325, x, pyargs, phase='liq')
    if print_result:
        print('\n##########  Test with water  ##########')
        print('----- Density at 274 K and 101325 Pa -----')
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 2

    # Acetic acid
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB}

    ref = 17240. # source: DIPPR correlation
    calc = pcsaft_den(305, 101325, x, pyargs, phase='liq')
    if print_result:
        print('\n##########  Test with acetic acid  ##########')
        print('----- Density at 305 K and 101325 Pa -----')
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 2

    # Dimethyl ether
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])
    pyargs = {'m':m, 's':s, 'e':e, 'dipm':dpm, 'dip_num':dip_num}

    ref = 16110. # source: DIPPR correlation
    calc = pcsaft_den(240, 101325, x, pyargs, phase='liq')
    if print_result:
        print('\n##########  Test with dimethyl ether  ##########')
        print('----- Density at 240 K and 101325 Pa -----')
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 2

    # Binary mixture: methanol-cyclohexane
    #0 = methanol, 1 = cyclohexane
    x = np.asarray([0.0550, 0.945])
    m = np.asarray([1.5255, 2.5303])
    s = np.asarray([3.2300, 3.8499])
    e = np.asarray([188.90, 278.11])
    volAB = np.asarray([0.035176, 0.])
    eAB = np.asarray([2899.5, 0.])
    k_ij = np.asarray([[0, 0.051],
                       [0.051, 0]])
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij}

    ref = 9506.1 # source: J. Canosa, A. Rodríguez, and J. Tojo, “Liquid−Liquid Equilibrium and Physical Properties of the Ternary Mixture (Dimethyl Carbonate + Methanol + Cyclohexane) at 298.15 K,” J. Chem. Eng. Data, vol. 46, no. 4, pp. 846–850, Jul. 2001.
    calc = pcsaft_den(298.15, 101325, x, pyargs, phase='liq')
    if print_result:
        print('\n##########  Test with methanol-cyclohexane mixture  ##########')
        print('----- Density at 298.15 K and 101325 Pa -----')
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 2

    # NaCl in water
    # 0 = Na+, 1 = Cl-, 2 = H2O
    x = np.asarray([0.010579869455908, 0.010579869455908, 0.978840261088184])
    m = np.asarray([1, 1, 1.2047])
    s = np.asarray([2.8232, 2.7599589, 0.])
    e = np.asarray([230.00, 170.00, 353.9449])
    volAB = np.asarray([0, 0, 0.0451])
    eAB = np.asarray([0, 0, 2425.67])
    k_ij = np.asarray([[0, 0.317, 0],
                       [0.317, 0, -0.25],
                        [0, -0.25, 0]])
    z = np.asarray([1., -1., 0.])

    ref = 55507.23 # source: Rodriguez H.; Soto A.; Arce A.; Khoshkbarchi M.K.: Apparent Molar Volume, Isentropic Compressibility, Refractive Index, and Viscosity of DL-Alanine in Aqueous NaCl Solutions. J.Solution Chem. 32 (2003) 53-63
    t = 298.15 # K
    s[2] = 2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t) # temperature dependent segment diameter for water
    k_ij[0,2] = -0.007981*t + 2.37999
    k_ij[2,0] = -0.007981*t + 2.37999
    dielc = dielc_water(t)

    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij, 'z':z, 'dielc':dielc}

    calc = pcsaft_den(t, 101325, x, pyargs, phase='liq')
    if print_result:
        print('\n##########  Test with aqueous NaCl  ##########')
        print('----- Density at 298.15 K and 101325 Pa -----')
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 2

    # Propane
    x = np.asarray([1.])
    m = np.asarray([2.0020])
    s = np.asarray([3.6184])
    e = np.asarray([208.11])
    pyargs = {'m':m, 's':s, 'e':e}

    t = 85.525 # K
    p = 1.7551e-4 # Pa
    ref = 16621.0 # source: reference EOS in CoolProp
    calc = pcsaft_den(t, p, x, pyargs, phase='liq')
    if print_result:
        print('##########  Test with propane  ##########')
        print('----- Density at {} K and {} Pa -----'.format(t, p))
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 2

    t = 85.525 # K
    p = 1.39e-4 # Pa
    ref = 1.9547e-7 # source: reference EOS in CoolProp
    calc = pcsaft_den(t, p, x, pyargs, phase='vap')
    if print_result:
        print('----- Density at {} K and {} Pa -----'.format(t, p))
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 2

    t = 293 # K
    p = 833240 # Pa
    ref = 11346.0 # source: reference EOS in CoolProp
    calc = pcsaft_den(t, p, x, pyargs, phase='liq')
    if print_result:
        print('----- Density at {} K and {} Pa -----'.format(t, p))
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 2

    t = 430 # K
    p = 2000000 # Pa
    ref = 623.59 # source: reference EOS in CoolProp
    calc = pcsaft_den(t, p, x, pyargs, phase='liq')
    if print_result:
        print('----- Density at {} K and {} Pa -----'.format(t, p))
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 2


def test_indexes(print_result=False):
    '''
    Check that the properties of a pure compound are the same regardless of
    whether parameters for additional compounds are included.
    '''
    # Binary mixture: water-acetic acid
    # only parameters for acetic acid
    x = np.asarray([1.])
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB}

    ref = 193261.515187248 # source: DIPPR correlation
    t = 413.5385
    rho = 15107.481234283325
    fugcoef1 = pcsaft_fugcoef(t, rho, x, pyargs)

    # same composition, but with mixture parameters
    #0 = water, 1 = acetic acid
    m = np.asarray([1.2047, 1.3403])
    s = np.asarray([0, 3.8582])
    e = np.asarray([353.95, 211.59])
    volAB = np.asarray([0.0451, 0.075550])
    eAB = np.asarray([2425.67, 3044.4])
    k_ij = np.asarray([[0, -0.127],
                       [-0.127, 0]])

    x = np.asarray([0, 1])
    s[0] = 3.8395 + 1.2828*np.exp(-0.0074944*t) - 1.3939*np.exp(-0.00056029*t)
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij}
    fugcoef_mix = pcsaft_fugcoef(t, rho, x, pyargs)
    if print_result:
        print('\n##########  Test with acetic acid  ##########')
        print('pure fugcoef:', fugcoef1[0])
        print('mix fugcoef:', fugcoef_mix[1])
        print('deviation', (fugcoef_mix[1] - fugcoef1[0])/ fugcoef1[0] * 100, '%')
    assert abs((fugcoef_mix[1] - fugcoef1)/ fugcoef1 * 100) < 1e-1

    # Binary mixture: water-furfural
    # only parameters for furfural
    x = np.asarray([1.])
    m = np.asarray([3.9731])
    s = np.asarray([3.0551])
    e = np.asarray([259.15])
    dipm = np.asarray([3.6])
    dip_num = np.asarray([1])
    pyargs = {'m':m, 's':s, 'e':e, 'dipm':dipm, 'dip_num':dip_num}

    t = 400 # K
    p = 34914.37778265716 # Pa
    rho = 10657.129498214763
    fugcoef1 = pcsaft_fugcoef(t, rho, x, pyargs)

    # same composition, but with mixture parameters
    #0 = water, 1 = acetic acid
    m = np.asarray([1.2047, 3.9731])
    s = np.asarray([0, 3.0551])
    e = np.asarray([353.95, 259.15])
    volAB = np.asarray([0.0451, 0.0451])
    eAB = np.asarray([2425.67, 0])
    dipm = np.asarray([0, 3.6])
    dip_num = np.asarray([0, 1])
    k_ij = np.asarray([[0, -0.027],
                       [-0.027, 0]])

    x = np.asarray([0, 1])
    s[0] = 3.8395 + 1.2828*np.exp(-0.0074944*t) - 1.3939*np.exp(-0.00056029*t)
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'dipm':dipm, 'dip_num':dip_num, 'k_ij':k_ij}
    fugcoef_mix = pcsaft_fugcoef(t, rho, x, pyargs)
    if print_result:
        print('\n##########  Test with furfural  ##########')
        print('pure fugcoef:', fugcoef1[0])
        print('mix fugcoef:', fugcoef_mix[1])
        print('deviation', (fugcoef_mix[1] - fugcoef1[0])/ fugcoef1[0] * 100, '%')
    assert abs((fugcoef_mix[1] - fugcoef1)/ fugcoef1 * 100) < 1e-6

    # Mixture: NaCl in water with random 4th component
    # only parameters for water
    x = np.asarray([1.])
    m = np.asarray([1.2047])
    s = np.asarray([0.])
    e = np.asarray([353.9449])
    volAB = np.asarray([0.0451])
    eAB = np.asarray([2425.67])

    t = 298.15 # K
    s[0] = 3.8395 + 1.2828*np.exp(-0.0074944*t) - 1.3939*np.exp(-0.00056029*t)
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB}

    p = 3153.417688548272 # Pa
    rho = 55320.89616248148
    fugcoef1 = pcsaft_fugcoef(t, rho, x, pyargs)

    # same composition, but with mixture parameters
    # Mixture: NaCl in water with random 4th component
    # 0 = Na+, 1 = Cl-, 2 = H2O, 3 = DIMETHOXYMETHANE
    x = np.asarray([0, 0, 1, 0])
    m = np.asarray([1, 1, 1.2047, 2.8454])
    s = np.asarray([2.8232, 2.7599589, 0., 3.4017])
    e = np.asarray([230.00, 170.00, 353.9449, 234.02])
    volAB = np.asarray([0, 0, 0.0451, 0.0451])
    eAB = np.asarray([0, 0, 2425.67, 0])
    dipm = np.asarray([0, 0, 0, 1.2])
    dip_num = np.asarray([0, 0, 0, 1])
    k_ij = np.asarray([[0, 0.317, 0, 0],
                       [0.317, 0, -0.25, 0],
                        [0, -0.25, 0, 0],
                        [0, 0, 0, 0]])
    k_ij[0,2] = -0.007981*t + 2.37999
    k_ij[2,0] = -0.007981*t + 2.37999
    z = np.asarray([1., -1., 0., 0])
    dielc = dielc_water(t)

    s[2] = 3.8395 + 1.2828*np.exp(-0.0074944*t) - 1.3939*np.exp(-0.00056029*t)
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'dipm':dipm, 'dip_num':dip_num, 'k_ij':k_ij, 'z':z, 'dielc':dielc}
    # pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij, 'z':z, 'dielc':dielc}
    fugcoef_mix = pcsaft_fugcoef(t, rho, x, pyargs)
    if print_result:
        print('\n##########  Test with water  ##########')
        print('pure fugcoef:', fugcoef1[0])
        print('mix fugcoef:', fugcoef_mix[2])
        print('deviation', (fugcoef_mix[2] - fugcoef1[0])/ fugcoef1[0] * 100, '%')
    assert abs((fugcoef_mix[2] - fugcoef1)/ fugcoef1 * 100) < 1e-1


def test_flashTQ(print_result=False):
    """Test the flashTQ function to see if it is working correctly."""
    # Toluene
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    pyargs = {'m':m, 's':s, 'e':e}

    ref = 3255792.76201971 # source: reference EOS in CoolProp
    t = 572.6667
    calc, xl, xv = flashTQ(t, 0, x, pyargs)
    if print_result:
        print('##########  Test with toluene  ##########')
        print('----- Vapor pressure at 572.7 K -----')
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 3

    # Water
    m = np.asarray([1.2047])
    e = np.asarray([353.95])
    volAB = np.asarray([0.0451])
    eAB = np.asarray([2425.67])

    ref = 67171.754576141 # source: IAWPS95 EOS
    t = 362
    s = np.asarray([2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t)])
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB}
    calc, xl, xv = flashTQ(t, 0, x, pyargs)
    if print_result:
        print('\n##########  Test with water  ##########')
        print('----- Vapor pressure at 362 K -----')
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 3

    # Acetic acid
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB}

    ref = 193261.515187248 # source: DIPPR correlation
    t = 413.5385
    calc, xl, xv = flashTQ(t, 0, x, pyargs)
    if print_result:
        print('\n##########  Test with acetic acid  ##########')
        print('----- Vapor pressure at 413.5 K -----')
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 3

    # Dimethyl ether
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])
    pyargs = {'m':m, 's':s, 'e':e, 'dipm':dpm, 'dip_num':dip_num}

    ref = 625100. # source: DIPPR correlation
    t = 300
    calc, xl, xv = flashTQ(t, 0, x, pyargs)
    if print_result:
        print('\n##########  Test with dimethyl ether  ##########')
        print('----- Vapor pressure at 300 K -----')
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 3

    # Propane
    x = np.asarray([1.])
    m = np.asarray([2.0020])
    s = np.asarray([3.6184])
    e = np.asarray([208.11])
    pyargs = {'m':m, 's':s, 'e':e}

    ref = 1.7551e-4 # source: reference EOS in CoolProp
    t = 85.525
    calc, xl, xv = flashTQ(t, 0, x, pyargs)
    if print_result:
        print('##########  Test with propane  ##########')
        print('----- Vapor pressure at {} K -----'.format(t))
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 5

    ref = 8.3324e5 # source: reference EOS in CoolProp
    t = 293
    calc, xl, xv = flashTQ(t, 0, x, pyargs)
    if print_result:
        print('----- Vapor pressure at {} K -----'.format(t))
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 3

    ref = 42.477e5 # source: reference EOS in CoolProp
    t = 369.82
    calc, xl, xv = flashTQ(t, 0, x, pyargs)
    if print_result:
        print('----- Vapor pressure at {} K -----'.format(t))
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 3

    # Binary mixture: methane-benzene
    #0 = methane, 1 = benzene
    x = np.asarray([0.0252,0.9748])
    m = np.asarray([1.0000, 2.4653])
    s = np.asarray([3.7039, 3.6478])
    e = np.asarray([150.03, 287.35])
    k_ij = np.asarray([[0, 0.037],
                       [0.037, 0]])
    pyargs = {'m':m, 's':s, 'e':e, 'k_ij':k_ij}

    t = 421.05
    ref = 1986983.25 # source: H.-M. Lin, H. M. Sebastian, J. J. Simnick, and K.-C. Chao, “Gas-liquid equilibrium in binary mixtures of methane with N-decane, benzene, and toluene,” J. Chem. Eng. Data, vol. 24, no. 2, pp. 146–149, Apr. 1979.
    xv_ref = np.asarray([0.6516,0.3484])
    calc, xl, xv = flashTQ(t, 0, x, pyargs)
    if print_result:
        print('\n##########  Test with methane-benzene mixture  ##########')
        print('----- Bubble point pressure at %s K -----' % t)
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
        print('    Vapor composition (reference):', xv_ref)
        print('    Vapor composition (PC-SAFT):', xv)
        print('    Vapor composition relative deviation:', (xv-xv_ref)/xv_ref*100)
    assert abs((calc-ref)/ref*100) < 10
    assert np.all(abs((xv-xv_ref)/xv_ref*100) < 10)

    # # Binary mixture: hydrogen-hexadecane
    # # This does not pass yet. The flash functions still need to be improved.
    # #0 = hydrogen, 1 = hexadecane
    # x = np.asarray([0.0407, 0.9593])
    # m = np.asarray([1.0000, 6.6485])
    # s = np.asarray([2.9860, 3.9552])
    # e = np.asarray([19.2775, 254.70])
    # k_ij = np.asarray([[0, 0],
    #                    [0, 0]])
    # pyargs = {'m':m, 's':s, 'e':e, 'k_ij':k_ij}
    #
    # t = 542.25 # K
    # ref = 2.009 * 1000000 # Pa, source: Lin, H.-M.; Sebastian,H.M.; Chao,K.-C.; J. Chem. Engng. Data. 1980, 25, 252-257.
    # xv_ref = np.asarray([0.9648, 0.0352])
    # calc, xl, xv = flashTQ(t, 0, x, pyargs, p_guess=470472.)
    # if print_result:
    #     print('\n##########  Test with hydrogen-hexadecane mixture  ##########')
    #     print('----- Bubble point pressure at %s K -----' % t)
    #     print('    Reference:', ref, 'Pa')
    #     print('    PC-SAFT:', calc, 'Pa')
    #     print('    Relative deviation:', (calc-ref)/ref*100, '%')
    #     print('    Vapor composition (reference):', xv_ref)
    #     print('    Vapor composition (PC-SAFT):', xv)
    #     print('    Vapor composition relative deviation:', (xv-xv_ref)/xv_ref*100)
    # assert abs((calc-ref)/ref*100) < 10
    # assert np.all(abs((xv-xv_ref)/xv_ref*100) < 10)

    # Binary mixture: methanol-cyclohexane
    #0 = methanol, 1 = cyclohexane
    x = np.asarray([0.3,0.7])
    m = np.asarray([1.5255, 2.5303])
    s = np.asarray([3.2300, 3.8499])
    e = np.asarray([188.90, 278.11])
    volAB = np.asarray([0.035176, 0.])
    eAB = np.asarray([2899.5, 0.])
    k_ij = np.asarray([[0, 0.051],
                       [0.051, 0]])
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij}

    # bubble point
    ref = 101330 # source: Marinichev A.N.; Susarev M.P.: Investigation of Liquid-Vapor Equilibrium in the System Methanol-Cyclohexane at 35, 45 and 55°C and 760 mm Hg. J.Appl.Chem.USSR 38 (1965) 1582-1584
    t = 327.48
    xv_ref = np.asarray([0.59400,0.40600])
    calc, xl, xv = flashTQ(t, 0, x, pyargs)
    if print_result:
        print('\n##########  Test with methanol-cyclohexane mixture  ##########')
        print('----- Bubble point pressure at 327.48 K -----')
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
        print('    Vapor composition (reference):', xv_ref)
        print('    Vapor composition (PC-SAFT):', xv)
        print('    Vapor composition relative deviation:', (xv-xv_ref)/xv_ref*100)
    assert abs((calc-ref)/ref*100) < 10
    assert np.all(abs((xv-xv_ref)/xv_ref*100) < 10)

    # dew point
    x = np.asarray([0.59400,0.40600])
    ref = 101330 # source: Marinichev A.N.; Susarev M.P.: Investigation of Liquid-Vapor Equilibrium in the System Methanol-Cyclohexane at 35, 45 and 55°C and 760 mm Hg. J.Appl.Chem.USSR 38 (1965) 1582-1584
    t = 327.48
    xl_ref = np.asarray([0.3,0.7])
    calc, xl, xv = flashTQ(t, 1, x, pyargs)
    if print_result:
        print('\n##########  Test with methanol-cyclohexane mixture  ##########')
        print('----- Dew point pressure at 327.48 K -----')
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
        print('    Liquid composition (reference):', xl_ref)
        print('    Liquid composition (PC-SAFT):', xl)
        print('    Liquid composition relative deviation:', (xl-xl_ref)/xl_ref*100)
    assert abs((calc-ref)/ref*100) < 10
    assert np.all(abs((xl-xl_ref)/xl_ref*100) < 10)

    # Binary mixture: chloroform-ethanol
    #0 = chloroform, 1 = ethanol
    x = np.asarray([0.3607, 0.6393])
    m = np.asarray([2.5313, 2.3827])
    s = np.asarray([3.4608, 3.1771])
    e = np.asarray([269.47, 198.24])
    volAB = np.asarray([0.032384, 0.032384])
    eAB = np.asarray([0, 2653.4])
    dipm = np.asarray([1.04, 0.])
    dipnum = np.asarray([1, 0.])
    k_ij = np.asarray([[0, 0],
                       [0, 0]])
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'dipm':dipm, 'dip_num':dipnum, 'k_ij':k_ij}

    ref = 101330 # source: Chen GH, Wang Q, Ma ZM, Yan XH, Han SJ. Phase equilibria at superatmospheric pressures for systems containing halohydrocarbon, aromatic hydrocarbon, and alcohol. Journal of Chemical and Engineering Data. 1995 Mar;40(2):361-6.
    t = 337.03
    xv_ref = np.asarray([0.6127, 0.3873])
    calc, xl, xv = flashTQ(t, 0, x, pyargs, p_guess=ref)
    if print_result:
        print('\n##########  Test with chloroform-ethanol mixture  ##########')
        print('----- Bubble point pressure at 327.48 K -----')
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
        print('    Vapor composition (reference):', xv_ref)
        print('    Vapor composition (PC-SAFT):', xv)
        print('    Vapor composition relative deviation:', (xv-xv_ref)/xv_ref*100)
    assert abs((calc-ref)/ref*100) < 10
    assert np.all(abs((xv-xv_ref)/xv_ref*100) < 10)

    # Binary mixture: water-acetic acid
    #0 = water, 1 = acetic acid
    m = np.asarray([1.2047, 1.3403])
    s = np.asarray([0, 3.8582])
    e = np.asarray([353.95, 211.59])
    volAB = np.asarray([0.0451, 0.075550])
    eAB = np.asarray([2425.67, 3044.4])
    k_ij = np.asarray([[0, -0.127],
                       [-0.127, 0]])

    xl_ref = np.asarray([0.9898662364, 0.0101337636])
    t = 403.574
    s[0] = 3.8395 + 1.2828*np.exp(-0.0074944*t) - 1.3939*np.exp(-0.00056029*t)
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij}
    ref = 273722. # source: Othmer, D. F.; Silvis, S. J.; Spiel, A. Ind. Eng. Chem., 1952, 44, 1864-72 Composition of vapors from boiling binary solutions pressure equilibrium still for studying water - acetic acid system
    xv_ref = np.asarray([0.9923666645, 0.0076333355])
    calc, xl, xv = flashTQ(t, 0, xl_ref, pyargs)
    if print_result:
        print('\n##########  Test with water-acetic acid mixture  ##########')
        print('----- Bubble point pressure at %s K -----' % t)
        print('    Liquid composition:', xl_ref)
        print('    Reference pressure:', ref, 'Pa')
        print('    PC-SAFT pressure:', calc, 'Pa')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
        print('    Vapor composition (reference):', xv_ref)
        print('    Vapor composition (PC-SAFT):', xv)
        print('    Vapor composition relative deviation:', (xv-xv_ref)/xv_ref*100)
    assert abs((calc-ref)/ref*100) < 10
    assert np.all(abs((xv-xv_ref)/xv_ref*100) < 15)

    xl_ref = np.asarray([0.2691800943, 0.7308199057])
    t = 372.774
    s[0] = 3.8395 + 1.2828*np.exp(-0.0074944*t) - 1.3939*np.exp(-0.00056029*t)
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij}
    ref = 74463. # source: Freeman, J. R.; Wilson, G. M. AIChE Symp. Ser., 1985, 81, 14-25 High temperature vapor-liquid equilibrium measurements on acetic acid/water mixtures
    xv_ref = np.asarray([0.3878269411, 0.6121730589])
    calc, xl, xv = flashTQ(t, 0, xl_ref, pyargs)
    if print_result:
        print('----- Bubble point pressure at %s K -----' % t)
        print('    Liquid composition:', xl_ref)
        print('    Reference pressure:', ref, 'Pa')
        print('    PC-SAFT pressure:', calc, 'Pa')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
        print('    Vapor composition (reference):', xv_ref)
        print('    Vapor composition (PC-SAFT):', xv)
        print('    Vapor composition relative deviation:', (xv-xv_ref)/xv_ref*100)
    assert abs((calc-ref)/ref*100) < 10
    assert np.all(abs((xv-xv_ref)/xv_ref*100) < 15)

    # NaCl in water
    # 0 = Na+, 1 = Cl-, 2 = H2O
    x = np.asarray([0.0907304774758426, 0.0907304774758426, 0.818539045048315])
    m = np.asarray([1, 1, 1.2047])
    s = np.asarray([2.8232, 2.7599589, 0.])
    e = np.asarray([230.00, 170.00, 353.9449])
    volAB = np.asarray([0, 0, 0.0451])
    eAB = np.asarray([0, 0, 2425.67])
    k_ij = np.asarray([[0, 0.317, 0],
                       [0.317, 0, -0.25],
                        [0, -0.25, 0]])
    z = np.asarray([1., -1., 0.])

    ref = 2393.8 # average of repeat data points from source: A. Apelblat and E. Korin, “The vapour pressures of saturated aqueous solutions of sodium chloride, sodium bromide, sodium nitrate, sodium nitrite, potassium iodate, and rubidium chloride at temperatures from 227 K to 323 K,” J. Chem. Thermodyn., vol. 30, no. 1, pp. 59–71, Jan. 1998. (Solubility calculated using equation from Yaws, Carl L.. (2008). Yaws' Handbook of Properties for Environmental and Green Engineering.)
    t = 298.15 # K
    s[2] = 2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t) # temperature dependent segment diameter for water
    k_ij[0,2] = -0.007981*t + 2.37999
    k_ij[2,0] = -0.007981*t + 2.37999
    dielc = dielc_water(t)

    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij, 'z':z, 'dielc':dielc}

    xv_guess = np.asarray([0., 0., 1.])
    calc, xl, xv = flashTQ(t, 0, x, pyargs)
    if print_result:
        print('\n##########  Test with aqueous NaCl  ##########')
        print('----- Bubble point pressure at 298.15 K -----')
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 10


def test_flashPQ(print_result=False):
    """Test the flashPQ function to see if it is working correctly."""

    # Binary mixture: methanol-cyclohexane
    #0 = methanol, 1 = cyclohexane
    x = np.asarray([0.3,0.7])
    m = np.asarray([1.5255, 2.5303])
    s = np.asarray([3.2300, 3.8499])
    e = np.asarray([188.90, 278.11])
    volAB = np.asarray([0.035176, 0.])
    eAB = np.asarray([2899.5, 0.])
    k_ij = np.asarray([[0, 0.051],
                       [0.051, 0]])
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij}

    # bubble point
    p = 101330
    ref = 327.48 # source: Marinichev A.N.; Susarev M.P.: Investigation of Liquid-Vapor Equilibrium in the System Methanol-Cyclohexane at 35, 45 and 55°C and 760 mm Hg. J.Appl.Chem.USSR 38 (1965) 1582-1584
    xv_ref = np.asarray([0.59400,0.40600])
    calc, xl, xv = flashPQ(p, 0, x, pyargs)
    if print_result:
        print('\n##########  Test with methanol-cyclohexane mixture  ##########')
        print('----- Bubble point temperature at 101330 Pa -----')
        print('    Reference:', ref, 'K')
        print('    PC-SAFT:', calc, 'K')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
        print('    Vapor composition (reference):', xv_ref)
        print('    Vapor composition (PC-SAFT):', xv)
        print('    Vapor composition relative deviation:', (xv-xv_ref)/xv_ref*100)
    assert abs((calc-ref)/ref*100) < 1
    assert np.all(abs((xv-xv_ref)/xv_ref*100) < 10)

    # dew point
    x = np.asarray([0.59400,0.40600])
    p = 101330
    ref = 327.48 # source: Marinichev A.N.; Susarev M.P.: Investigation of Liquid-Vapor Equilibrium in the System Methanol-Cyclohexane at 35, 45 and 55°C and 760 mm Hg. J.Appl.Chem.USSR 38 (1965) 1582-1584
    xl_ref = np.asarray([0.3,0.7])
    calc, xl, xv = flashPQ(p, 1, x, pyargs, t_guess=328.)
    if print_result:
        print('\n##########  Test with methanol-cyclohexane mixture  ##########')
        print('----- Dew point temperature at 101330 Pa -----')
        print('    Reference:', ref, 'K')
        print('    PC-SAFT:', calc, 'K')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
        print('    Liquid composition (reference):', xl_ref)
        print('    Liquid composition (PC-SAFT):', xl)
        print('    Liquid composition relative deviation:', (xl-xl_ref)/xl_ref*100)
    assert abs((calc-ref)/ref*100) < 1
    assert np.all(abs((xl-xl_ref)/xl_ref*100) < 20)

    # NaCl in water
    # 0 = Na+, 1 = Cl-, 2 = H2O
    x = np.asarray([0.0907304774758426, 0.0907304774758426, 0.818539045048315])
    m = np.asarray([1, 1, 1.2047])
    s = np.asarray([2.8232, 2.7599589, 0.])
    e = np.asarray([230.00, 170.00, 353.9449])
    volAB = np.asarray([0, 0, 0.0451])
    eAB = np.asarray([0, 0, 2425.67])
    k_ij = np.asarray([[0, 0.317, 0],
                       [0.317, 0, -0.25],
                        [0, -0.25, 0]])
    z = np.asarray([1., -1., 0.])

    p = 2393.8 # Pa
    ref = 298.15 # K, average of repeat data points from source: A. Apelblat and E. Korin, “The vapour pressures of saturated aqueous solutions of sodium chloride, sodium bromide, sodium nitrate, sodium nitrite, potassium iodate, and rubidium chloride at temperatures from 227 K to 323 K,” J. Chem. Thermodyn., vol. 30, no. 1, pp. 59–71, Jan. 1998. (Solubility calculated using equation from Yaws, Carl L.. (2008). Yaws' Handbook of Properties for Environmental and Green Engineering.)
    s[2] = 2.7927 + 10.11*np.exp(-0.01775*ref) - 1.417*np.exp(-0.01146*ref) # temperature dependent segment diameter for water
    k_ij[0,2] = -0.007981*ref + 2.37999
    k_ij[2,0] = -0.007981*ref + 2.37999
    dielc = dielc_water(ref)

    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij, 'z':z, 'dielc':dielc}

    xv_guess = np.asarray([0., 0., 1.])
    calc, xl, xv = flashPQ(p, 0, x, pyargs, ref)
    if print_result:
        print('\n##########  Test with aqueous NaCl  ##########')
        print('----- Bubble point temperature at 2393.8 Pa -----')
        print('    Reference:', ref, 'K')
        print('    PC-SAFT:', calc, 'K')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 1


def test_osmoticC(print_result=False):
    """Test the function for calculating osmotic coefficients to see if it is working correctly."""
    # NaCl in water
    # 0 = Na+, 1 = Cl-, 2 = H2O
    x = np.asarray([0.0629838206, 0.0629838206, 0.8740323588])
    m = np.asarray([1, 1, 1.2047])
    s = np.asarray([2.8232, 2.7599589, 0.])
    e = np.asarray([230.00, 170.00, 353.9449])
    volAB = np.asarray([0, 0, 0.0451])
    eAB = np.asarray([0, 0, 2425.67])
    k_ij = np.asarray([[0, 0.317, 0],
                       [0.317, 0, -0.25],
                        [0, -0.25, 0]])
    z = np.asarray([1., -1., 0.])

    ref = 1.116 # source: R. A. Robinson and R. H. Stokes, Electrolyte Solutions: Second Revised Edition. Dover Publications, 1959.
    t = 293.15 # K
    s[2] = 2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t) # temperature dependent segment diameter for water
    k_ij[0,2] = -0.007981*t + 2.37999
    k_ij[2,0] = -0.007981*t + 2.37999
    dielc = dielc_water(t)

    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij, 'z':z, 'dielc':dielc}

    rho = pcsaft_den(t, 2339.3, x, pyargs, phase='liq')
    result = pcsaft_osmoticC(t, rho, x, pyargs)
    calc = result[0]
    if print_result:
        print('\n##########  Test with aqueous NaCl  ##########')
        print('----- Osmotic coefficient at 293.15 K -----')
        print('    Reference:', ref)
        print('    PC-SAFT:', calc)
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 2


def test_Hvap(print_result=False):
    """Test the enthalpy of vaporization function to see if it is working correctly."""
    # Toluene
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    pyargs = {'m':m, 's':s, 'e':e}

    ref = 33500. # source: DIPPR correlation
    p = 90998. # source: reference equation of state from Polt, A.; Platzer, B.; Maurer, G., Parameter der thermischen Zustandsgleichung von Bender fuer 14 mehratomige reine Stoffe, Chem. Tech. (Leipzig), 1992, 44, 6, 216-224.
    calc = pcsaft_Hvap(380., x, pyargs)[0]
    if print_result:
        print('##########  Test with toluene  ##########')
        print('----- Enthalpy of vaporization at 380 K -----')
        print('    Reference:', ref, 'J mol^-1')
        print('    PC-SAFT:', calc, 'J mol^-1')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 3

    # Water
    m = np.asarray([1.2047])
    e = np.asarray([353.95])
    volAB = np.asarray([0.0451])
    eAB = np.asarray([2425.67])

    ref = 44761.23 # source: IAWPS95 EOS
    p = 991.82 # source: IAWPS95 EOS
    t = 280
    s = np.asarray([2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t)])
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB}
    calc = pcsaft_Hvap(t, x, pyargs)[0]
    if print_result:
        print('\n##########  Test with water  ##########')
        print('----- Enthalpy of vaporization at 280 K -----')
        print('    Reference:', ref, 'J mol^-1')
        print('    PC-SAFT:', calc, 'J mol^-1')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 3

    # Dimethyl ether
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])
    pyargs = {'m':m, 's':s, 'e':e, 'dipm':dpm, 'dip_num':dip_num}

    ref = 17410. # source: DIPPR correlation
    p = 937300. # source: DIPPR correlation
    calc = pcsaft_Hvap(315., x, pyargs)[0]
    if print_result:
        print('\n##########  Test with dimethyl ether  ##########')
        print('----- Enthalpy of vaporization at 315 K -----')
        print('    Reference:', ref, 'J mol^-1')
        print('    PC-SAFT:', calc, 'J mol^-1')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 3


def test_dadt(print_result=False):
    """Test the function for the temperature derivative of the Helmholtz energy."""
    # Toluene
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    pyargs = {'m':m, 's':s, 'e':e}

    p = 100000.
    t = 330.

    rho = pcsaft_den(t, p, x, pyargs, phase='liq')
    dadt_eos = pcsaft_dadt(t, rho, x, pyargs)

    # calculating numerical derivative
    der1 = pcsaft_ares(t-1, rho, x, pyargs)
    der2 = pcsaft_ares(t+1, rho, x, pyargs)
    dadt_num = (der2-der1)/2.
    if print_result:
        print('\n##########  Test with toluene  ##########')
        print('    Numerical derivative:', dadt_num)
        print('    PC-SAFT derivative:', dadt_eos)
        print('    Relative deviation:', (dadt_eos-dadt_num)/dadt_num*100, '%')
    assert abs((dadt_eos-dadt_num)/dadt_num*100) < 2e-2

    # Acetic acid
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB}

    p = 100000.
    t = 310.

    rho = pcsaft_den(t, p, x, pyargs, phase='liq')
    dadt_eos = pcsaft_dadt(t, rho, x, pyargs)

    # calculating numerical derivative
    der1 = pcsaft_ares(t-1, rho, x, pyargs)
    der2 = pcsaft_ares(t+1, rho, x, pyargs)
    dadt_num = (der2-der1)/2.
    if print_result:
        print('\n##########  Test with acetic acid  ##########')
        print('    Numerical derivative:', dadt_num)
        print('    PC-SAFT derivative:', dadt_eos)
        print('    Relative deviation:', (dadt_eos-dadt_num)/dadt_num*100, '%')
    assert abs((dadt_eos-dadt_num)/dadt_num*100) < 2e-2

    # Water
    m = np.asarray([1.2047])
    e = np.asarray([353.95])
    volAB = np.asarray([0.0451])
    eAB = np.asarray([2425.67])

    p = 100000.
    t = 290.

    s = np.asarray([2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t)])
    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB}

    rho = pcsaft_den(t, p, x, pyargs, phase='liq')
    dadt_eos = pcsaft_dadt(t, rho, x, pyargs)

    # calculating numerical derivative
    der1 = pcsaft_ares(t-1, rho, x, pyargs)
    der2 = pcsaft_ares(t+1, rho, x, pyargs)
    dadt_num = (der2-der1)/2.
    if print_result:
        print('\n##########  Test with water  ##########')
        print('    Numerical derivative:', dadt_num)
        print('    PC-SAFT derivative:', dadt_eos)
        print('    Relative deviation:', (dadt_eos-dadt_num)/dadt_num*100, '%')
    assert abs((dadt_eos-dadt_num)/dadt_num*100) < 2e-2

    # Dimethyl ether
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])
    pyargs = {'m':m, 's':s, 'e':e, 'dipm':dpm, 'dip_num':dip_num}

    p = 100000.
    t = 370.

    rho = pcsaft_den(t, p, x, pyargs, phase='liq')
    dadt_eos = pcsaft_dadt(t, rho, x, pyargs)

    # calculating numerical derivative
    der1 = pcsaft_ares(t-1, rho, x, pyargs)
    der2 = pcsaft_ares(t+1, rho, x, pyargs)
    dadt_num = (der2-der1)/2.
    if print_result:
        print('\n##########  Test with dimethyl ether  ##########')
        print('    Numerical derivative:', dadt_num)
        print('    PC-SAFT derivative:', dadt_eos)
        print('    Relative deviation:', (dadt_eos-dadt_num)/dadt_num*100, '%')
    assert abs((dadt_eos-dadt_num)/dadt_num*100) < 2e-2

    # Aqueous NaCl
    # 0 = Na+, 1 = Cl-, 2 = H2O
    x = np.asarray([0.0907304774758426, 0.0907304774758426, 0.818539045048315])
    m = np.asarray([1, 1, 1.2047])
    s = np.asarray([2.8232, 2.7599589, 0.])
    e = np.asarray([230.00, 170.00, 353.9449])
    volAB = np.asarray([0, 0, 0.0451])
    eAB = np.asarray([0, 0, 2425.67])
    k_ij = np.asarray([[0, 0.317, 0],
                       [0.317, 0, -0.25],
                        [0, -0.25, 0]])
    z = np.asarray([1., -1., 0.])

    t = 298.15 # K
    p = 100000. # Pa
    s[2] = 2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t) # temperature dependent segment diameter for water
    k_ij[0,2] = -0.007981*t + 2.37999
    k_ij[2,0] = -0.007981*t + 2.37999
    dielc = dielc_water(t)

    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij, 'z':z, 'dielc':dielc}

    rho = pcsaft_den(t, p, x, pyargs, phase='liq')
    dadt_eos = pcsaft_dadt(t, rho, x, pyargs)

    # calculating numerical derivative
    der1 = pcsaft_ares(t-1, rho, x, pyargs)
    der2 = pcsaft_ares(t+1, rho, x, pyargs)
    dadt_num = (der2-der1)/2.
    if print_result:
        print('\n##########  Test with aqueous NaCl  ##########')
        print('    Numerical derivative:', dadt_num)
        print('    PC-SAFT derivative:', dadt_eos)
        print('    Relative deviation:', (dadt_eos-dadt_num)/dadt_num*100, '%')
    assert abs((dadt_eos-dadt_num)/dadt_num*100) < 2e-2


def test_cp(print_result=False):
    """Test the heat capacity function to see if it is working correctly."""
    # Benzene
    x = np.asarray([1.])
    m = np.asarray([2.4653])
    s = np.asarray([3.6478])
    e = np.asarray([287.35])
    cnsts = np.asarray([55238., 173380, 764.25, 72545, 2445.7]) # constants for Aly-Lee equation (obtained from DIPPR)
    pyargs = {'m':m, 's':s, 'e':e}

    ref = 140.78 # source: Equation of state from Polt et al. (1992) (available at https://webbook.nist.gov/chemistry/fluid/)
    p = 100000.
    t = 330.
    rho = pcsaft_den(t, p, x, pyargs, phase='liq')
    calc = pcsaft_cp(t, rho, cnsts, x, pyargs)
    if print_result:
        print('\n##########  Test with benzene  ##########')
        print('----- Heat capacity at 330 K -----')
        print('    Reference:', ref, 'J mol^-1 K^-1')
        print('    PC-SAFT:', calc, 'J mol^-1 K^-1')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 3

    # Toluene
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    cnsts = np.asarray([58140., 286300, 1440.6, 189800, 650.43]) # constants for Aly-Lee equation (obtained from DIPPR)
    pyargs = {'m':m, 's':s, 'e':e}

    ref = 179.79 # source: Equation of state from Polt et al. (1992) (available at https://webbook.nist.gov/chemistry/fluid/)
    p = 100000.
    t = 370.
    rho = pcsaft_den(t, p, x, pyargs, phase='liq')
    calc = pcsaft_cp(t, rho, cnsts, x, pyargs)
    if print_result:
        print('\n##########  Test with toluene  ##########')
        print('----- Heat capacity at 370 K -----')
        print('    Reference:', ref, 'J mol^-1 K^-1')
        print('    PC-SAFT:', calc, 'J mol^-1 K^-1')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 3

    # # Acetic acid
    # # print('\n##########  Test with acetic acid  ##########')
    # m = np.asarray([1.3403])
    # s = np.asarray([3.8582])
    # e = np.asarray([211.59])
    # volAB = np.asarray([0.075550])
    # eAB = np.asarray([3044.4])
    # cnsts = np.asarray([40200., 136750, 1262, 70030, 569.7]) # constants for Aly-Lee equation (obtained from DIPPR)
    # pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB}
    #
    # ref = 130.3 # source: DIPPR
    # p = 100000.
    # t = 325.
    # rho = pcsaft_den(t, p, x, pyargs, phase='liq')
    # calc = pcsaft_cp(t, rho, cnsts, x, pyargs)
    # """ Note: Large deviations occur with acetic acid and water. This behavior
    # has been observed before and was described in R. T. C. S. Ribeiro, A. L.
    # Alberton, M. L. L. Paredes, G. M. Kontogeorgis, and X. Liang, “Extensive
    # Study of the Capabilities and Limitations of the CPA and sPC-SAFT Equations
    # of State in Modeling a Wide Range of Acetic Acid Properties,” Ind. Eng.
    # Chem. Res., vol. 57, no. 16, pp. 5690–5704, Apr. 2018. """
    # # print('----- Heat capacity at 325 K -----')
    # # print('    Reference:', ref, 'J mol^-1 K^-1')
    # # print('    PC-SAFT:', calc, 'J mol^-1 K^-1')
    # # print('    Relative deviation:', (calc-ref)/ref*100, '%')

    # Dimethyl ether
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])
    cnsts = np.asarray([57431., 94494, 895.51, 65065, 2467.4]) # constants for Aly-Lee equation (obtained from DIPPR)
    pyargs = {'m':m, 's':s, 'e':e, 'dipm':dpm, 'dip_num':dip_num}

    ref = 102.2 # source: DIPPR correlation
    p = 100000.
    t = 240.
    rho = pcsaft_den(t, p, x, pyargs, phase='liq')
    calc = pcsaft_cp(t, rho, cnsts, x, pyargs)
    if print_result:
        print('\n##########  Test with dimethyl ether  ##########')
        print('----- Heat capacity at 240 K -----')
        print('    Reference:', ref, 'J mol^-1 K^-1')
        print('    PC-SAFT:', calc, 'J mol^-1 K^-1')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 3


def test_pressure(print_result=False):
    """Test the pressure function to see if it is working correctly."""
# #     Toluene
#     x = np.asarray([1.])
#     m = np.asarray([2.8149])
#     s = np.asarray([3.7169])
#     e = np.asarray([285.69])
#     pyargs = {'m':m, 's':s, 'e':e}
#
#     ref = 101325 # Pa
#     t = 320 # K
#     rho = 9033.11420899 # mol m^-3 From density calculation with working PC-SAFT density function
#     calc = pcsaft_p(t, rho, x, pyargs)
#     if print_result:
#         print('\n##########  Test with toluene  ##########')
#         print('----- Pressure at {} K -----'.format(t))
#         print('    Reference:', ref, 'Pa')
#         print('    PC-SAFT:', calc, 'Pa')
#         print('    Relative deviation:', (calc-ref)/ref*100, '%')
#     assert abs((calc-ref)/ref*100) < 1e-6
#
#     # Water
#     m = np.asarray([1.2047])
#     e = np.asarray([353.95])
#     volAB = np.asarray([0.0451])
#     eAB = np.asarray([2425.67])
#
#     ref = 101325 # Pa
#     t = 274 # K
#     s = np.asarray([2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t)])
#     pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB}
#     rho = 55476.44269210048 # mol m^-3 From density calculation with working PC-SAFT density function
#     calc = pcsaft_p(t, rho, x, pyargs)
#     if print_result:
#         print('\n##########  Test with water  ##########')
#         print('----- Pressure at {} K -----'.format(t))
#         print('    Reference:', ref, 'Pa')
#         print('    PC-SAFT:', calc, 'Pa')
#         print('    Relative deviation:', (calc-ref)/ref*100, '%')
#     assert abs((calc-ref)/ref*100) < 1e-6
#
#     # Acetic acid
#     m = np.asarray([1.3403])
#     s = np.asarray([3.8582])
#     e = np.asarray([211.59])
#     volAB = np.asarray([0.075550])
#     eAB = np.asarray([3044.4])
#     pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB}
#
#     ref = 101325 # Pa
#     t = 305 # K
#     rho = 16965.66943969988 # mol m^-3 From density calculation with working PC-SAFT density function
#     calc = pcsaft_p(t, rho, x, pyargs)
#     if print_result:
#         print('\n##########  Test with acetic acid  ##########')
#         print('----- Pressure at {} K -----'.format(t))
#         print('    Reference:', ref, 'Pa')
#         print('    PC-SAFT:', calc, 'Pa')
#         print('    Relative deviation:', (calc-ref)/ref*100, '%')
#     assert abs((calc-ref)/ref*100) < 1e-6
#
#     # dimethyl ether
#     m = np.asarray([2.2634])
#     s = np.asarray([3.2723])
#     e = np.asarray([210.29])
#     dpm = np.asarray([1.3])
#     dip_num = np.asarray([1.0])
#     pyargs = {'m':m, 's':s, 'e':e, 'dipm':dpm, 'dip_num':dip_num}
#
#     ref = 101325 # Pa
#     t = 240 # K
#     rho = 15865.69021378 # mol m^-3 From density calculation with working PC-SAFT density function
#     calc = pcsaft_p(t, rho, x, pyargs)
#     if print_result:
#         print('\n##########  Test with dimethyl ether  ##########')
#         print('----- Pressure at {} K -----'.format(t))
#         print('    Reference:', ref, 'Pa')
#         print('    PC-SAFT:', calc, 'Pa')
#         print('    Relative deviation:', (calc-ref)/ref*100, '%')
#     assert abs((calc-ref)/ref*100) < 1e-6
#
#     # Binary mixture: methanol-cyclohexane
#     #0 = methanol, 1 = cyclohexane
#     x = np.asarray([0.0550, 0.945])
#     m = np.asarray([1.5255, 2.5303])
#     s = np.asarray([3.2300, 3.8499])
#     e = np.asarray([188.90, 278.11])
#     volAB = np.asarray([0.035176, 0.])
#     eAB = np.asarray([2899.5, 0.])
#     k_ij = np.asarray([[0, 0.051],
#                        [0.051, 0]])
#     pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij}
#
#     ref = 101325 # Pa
#     t = 298.15 # K
#     rho = 9368.903682262355 # mol m^-3 From density calculation with working PC-SAFT density function
#     calc = pcsaft_p(t, rho, x, pyargs)
#     if print_result:
#         print('\n##########  Test with methanol-cyclohexane mixture  ##########')
#         print('----- Pressure at {} K -----'.format(t))
#         print('    Reference:', ref, 'Pa')
#         print('    PC-SAFT:', calc, 'Pa')
#         print('    Relative deviation:', (calc-ref)/ref*100, '%')
#     assert abs((calc-ref)/ref*100) < 1e-6

    # # Binary mixture: water-acetic acid
    # #0 = water, 1 = acetic acid
    # x = np.asarray([0.9898662364, 0.0101337636])
    # m = np.asarray([1.2047, 1.3403])
    # s = np.asarray([0, 3.8582])
    # e = np.asarray([353.95, 211.59])
    # volAB = np.asarray([0.0451, 0.075550])
    # eAB = np.asarray([2425.67, 3044.4])
    # k_ij = np.asarray([[0, -0.127],
    #                    [-0.127, 0]])
    #
    # t = 403.574
    # s[0] = 3.8395 + 1.2828*np.exp(-0.0074944*t) - 1.3939*np.exp(-0.00056029*t)
    # pyargs = {'x':x, 'm':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij}
    # ref = 275000. # experimental bubble point pressure is 273722 Pa. source: Othmer, D. F.; Silvis, S. J.; Spiel, A. Ind. Eng. Chem., 1952, 44, 1864-72 Composition of vapors from boiling binary solutions pressure equilibrium still for studying water - acetic acid system
    # xv_ref = np.asarray([0.9923666645, 0.0076333355])
    # rho = 50902.74840603031
    # calc = pcsaft_p(t, rho, x, pyargs)
    # if print_result:
    #     print('\n##########  Test with water-acetic acid mixture  ##########')
    #     print('----- Bubble point pressure at %s K -----' % t)
    #     print('    Liquid composition:', x)
    #     print('    Reference pressure:', ref, 'Pa')
    #     print('    PC-SAFT pressure:', calc, 'Pa')
    #     print('    Relative deviation:', (calc-ref)/ref*100, '%')
    # assert abs((calc-ref)/ref*100) < 1e-6

    # NaCl in water
    # 0 = Na+, 1 = Cl-, 2 = H2O
    x = np.asarray([0.010579869455908, 0.010579869455908, 0.978840261088184])
    m = np.asarray([1, 1, 1.2047])
    s = np.asarray([2.8232, 2.7599589, 0.])
    e = np.asarray([230.00, 170.00, 353.9449])
    volAB = np.asarray([0, 0, 0.0451])
    eAB = np.asarray([0, 0, 2425.67])
    k_ij = np.asarray([[0, 0.317, 0],
                       [0.317, 0, -0.25],
                        [0, -0.25, 0]])
    z = np.asarray([1., -1., 0.])

    ref = 101325 # Pa
    t = 298.15 # K
    s[2] = 2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t) # temperature dependent segment diameter for water
    k_ij[0,2] = -0.007981*t + 2.37999
    k_ij[2,0] = -0.007981*t + 2.37999
    dielc = dielc_water(t)
    rho = 55756.6727267386 # mol m^-3 From density calculation with working PC-SAFT density function

    pyargs = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij, 'z':z, 'dielc':dielc}

    calc = pcsaft_p(t, rho, x, pyargs)
    if print_result:
        print('\n##########  Test with aqueous NaCl  ##########')
        print('----- Pressure at {} K -----'.format(t))
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc-ref)/ref*100, '%')
    assert abs((calc-ref)/ref*100) < 1e-6
