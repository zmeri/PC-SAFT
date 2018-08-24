# -*- coding: utf-8 -*-
"""
Tests for checking that the PC-SAFT functions are working correctly.

@author: Zach Baird
"""
import numpy as np
from pcsaft_electrolyte import pcsaft_den, pcsaft_hres, pcsaft_gres, pcsaft_sres, pcsaft_Hvap
from pcsaft_electrolyte import pcsaft_vaporP, pcsaft_bubbleP, dielc_water, pcsaft_PTz, pcsaft_osmoticC

def test_hres():
    """Test the residual enthalpy function to see if it is working correctly."""
    print('------ 325 K ------')
    print('\t\t\t PC-SAFT\t Reference')
    t = 325 # K
    p = 101325 # Pa
    # all reference values are from PC-SAFT implemented in Aspen Plus

    # Toluene ----------
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    pyargs = {}

    den = pcsaft_den(x, m, s, e, t, p, pyargs, phase='liq')
    calc = pcsaft_hres(x, m, s, e, t, den, pyargs)
    print('Toluene, liquid:\t\t', calc, -36809.39, 'J/mol')
    den = pcsaft_den(x, m, s, e, t, p, pyargs, phase='vap')
    calc = pcsaft_hres(x, m, s, e, t, den, pyargs)
    print('Toluene, vapor:\t\t', calc, -362.6777, 'J/mol')
    
    # Acetic acid ---------
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    pyargs = {'e_assoc':eAB, 'vol_a':volAB}
    
    den = pcsaft_den(x, m, s, e, t, p, pyargs, phase='liq')
    calc = pcsaft_hres(x, m, s, e, t, den, pyargs)
    print('Acetic acid, liquid:\t\t', calc, -38924.64, 'J/mol')
    den = pcsaft_den(x, m, s, e, t, p, pyargs, phase='vap')
    calc = pcsaft_hres(x, m, s, e, t, den, pyargs)
    print('Acetic acid, vapor:\t\t', calc, -15393.63, 'J/mol')
    
    # Butyl acetate ---------
    m = np.asarray([2.76462805])
    s = np.asarray([4.02244938])
    e = np.asarray([263.69902915])
    dpm = np.asarray([1.84])
    dip_num = np.asarray([4.99688339])
    pyargs = {'dipm':dpm, 'dip_num':dip_num}
    
    den = pcsaft_den(x, m, s, e, t, p, pyargs, phase='liq')
    calc = pcsaft_hres(x, m, s, e, t, den, pyargs)
    print('Butyl acetate, liquid:\t', calc, -43443.19, 'J/mol')
    den = pcsaft_den(x, m, s, e, t, p, pyargs, phase='vap')
    calc = pcsaft_hres(x, m, s, e, t, den, pyargs)
    print('Butyl acetate, vapor:\t\t', calc, -516.4779, 'J/mol')
    
    return None
    
def test_sres():
    """Test the residual entropy function to see if it is working correctly."""
    print('------ 325 K ------')
    print('\t\t\t PC-SAFT\t Reference')
    t = 325 # K
    p = 101325 # Pa
    # all reference values are from PC-SAFT implemented in Aspen Plus

    # Toluene ----------
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    pyargs = {}    
    
    den = pcsaft_den(x, m, s, e, t, p, pyargs, phase='liq')
    calc = pcsaft_sres(x, m, s, e, t, den, pyargs)
    print('Toluene, liquid:\t\t', calc, -96.3692, 'J/mol/K')
    den = pcsaft_den(x, m, s, e, t, p, pyargs, phase='vap')
    calc = pcsaft_sres(x, m, s, e, t, den, pyargs)
    print('Toluene, vapor:\t\t', calc, -0.71398, 'J/mol/K')
    
    # Acetic acid ---------
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    pyargs = {'e_assoc':eAB, 'vol_a':volAB}
    
    den = pcsaft_den(x, m, s, e, t, p, pyargs, phase='liq')
    calc = pcsaft_sres(x, m, s, e, t, den, pyargs)
    print('Acetic acid, liquid:\t\t', calc, -98.1127, 'J/mol/K')
    den = pcsaft_den(x, m, s, e, t, p, pyargs, phase='vap')
    calc = pcsaft_sres(x, m, s, e, t, den, pyargs)
    print('Acetic acid, vapor:\t\t', calc, -40.8743, 'J/mol/K')
    
    # Butyl acetate ---------
    m = np.asarray([2.76462805])
    s = np.asarray([4.02244938])
    e = np.asarray([263.69902915])
    dpm = np.asarray([1.84])
    dip_num = np.asarray([4.99688339])
    pyargs = {'dipm':dpm, 'dip_num':dip_num}
    
    den = pcsaft_den(x, m, s, e, t, p, pyargs, phase='liq')
    calc = pcsaft_sres(x, m, s, e, t, den, pyargs)
    print('Butyl acetate, liquid:\t', calc, -108.9615, 'J/mol/K')
    den = pcsaft_den(x, m, s, e, t, p, pyargs, phase='vap')
    calc = pcsaft_sres(x, m, s, e, t, den, pyargs)
    print('Butyl acetate, vapor:\t\t', calc, -1.0361, 'J/mol/K')
    
    return None
    
def test_gres():
    """Test the residual Gibbs energy function to see if it is working correctly."""
    print('------ 325 K ------')
    print('\t\t\t PC-SAFT\t Reference')
    t = 325 # K
    p = 101325 # Pa
    # all reference values are from PC-SAFT implemented in Aspen Plus

    # Toluene ----------
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    pyargs = {}
    
    den = pcsaft_den(x, m, s, e, t, p, pyargs, phase='liq')
    calc = pcsaft_gres(x, m, s, e, t, den, pyargs)
    print('Toluene, liquid:\t\t', calc, -5489.384, 'J/mol')
    den = pcsaft_den(x, m, s, e, t, p, pyargs, phase='vap')
    calc = pcsaft_gres(x, m, s, e, t, den, pyargs)
    print('Toluene, vapor:\t\t', calc, -130.6339, 'J/mol')
    
    # Acetic acid ---------
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    pyargs = {'e_assoc':eAB, 'vol_a':volAB}
    
    den = pcsaft_den(x, m, s, e, t, p, pyargs, phase='liq')
    calc = pcsaft_gres(x, m, s, e, t, den, pyargs)
    print('Acetic acid, liquid:\t\t', calc, -7038.004, 'J/mol')
    den = pcsaft_den(x, m, s, e, t, p, pyargs, phase='vap')
    calc = pcsaft_gres(x, m, s, e, t, den, pyargs)
    print('Acetic acid, vapor:\t\t', calc, -2109.459, 'J/mol')
    
    # Butyl acetate ---------
    m = np.asarray([2.76462805])
    s = np.asarray([4.02244938])
    e = np.asarray([263.69902915])
    dpm = np.asarray([1.84])
    dip_num = np.asarray([4.99688339])
    pyargs = {'dipm':dpm, 'dip_num':dip_num}
    
    den = pcsaft_den(x, m, s, e, t, p, pyargs, phase='liq')
    calc = pcsaft_gres(x, m, s, e, t, den, pyargs)
    print('Butyl acetate, liquid:\t', calc, -8030.709, 'J/mol')
    den = pcsaft_den(x, m, s, e, t, p, pyargs, phase='vap')
    calc = pcsaft_gres(x, m, s, e, t, den, pyargs)
    print('Butyl acetate, vapor:\t\t', calc, -179.7519, 'J/mol')
    
    return None

def test_density():
    """Test the density function to see if it is working correctly."""
#    # Toluene
    print('##########  Test with toluene  ##########')
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    pyargs = {}

    ref = 9135.590853014008 # source: reference EOS in CoolProp
    calc = pcsaft_den(x, m, s, e, 320, 101325, pyargs, phase='liq')
    print('----- Density at 320 K and 101325 Pa -----')
    print('    Reference:', ref, 'mol m^-3')
    print('    PC-SAFT:', calc, 'mol m^-3')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')
    
    # Water
    print('\n##########  Test with water  ##########')
    m = np.asarray([1.2047])
    e = np.asarray([353.95])
    volAB = np.asarray([0.0451])
    eAB = np.asarray([2425.67])
    pyargs = {'e_assoc':eAB, 'vol_a':volAB}
    
    ref = 55502.5970532902 # source: IAWPS95 EOS
    t = 274
    s = np.asarray([2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t)])
    calc = pcsaft_den(x, m, s, e, t, 101325, pyargs, phase='liq')
    print('----- Density at 274 K and 101325 Pa -----')
    print('    Reference:', ref, 'mol m^-3')
    print('    PC-SAFT:', calc, 'mol m^-3')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')
    
    # Acetic acid
    print('\n##########  Test with acetic acid  ##########')
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    pyargs = {'e_assoc':eAB, 'vol_a':volAB}
    
    ref = 17240. # source: DIPPR correlation
    calc = pcsaft_den(x, m, s, e, 305, 101325, pyargs, phase='liq')
    print('----- Density at 305 K and 101325 Pa -----')
    print('    Reference:', ref, 'mol m^-3')
    print('    PC-SAFT:', calc, 'mol m^-3')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')  
    
    # Butyl acetate
    print('\n##########  Test with butyl acetate  ##########')
    m = np.asarray([3.9706])
    s = np.asarray([3.5440])
    e = np.asarray([241.93])
    dpm = np.asarray([1.86])
    dip_num = np.asarray([1.0])
    pyargs = {'dipm':dpm, 'dip_num':dip_num}
    
    ref = 8021. # source: DIPPR correlation
    calc = pcsaft_den(x, m, s, e, 240, 101325, pyargs, phase='liq')
    print('----- Density at 240 K and 101325 Pa -----')
    print('    Reference:', ref, 'mol m^-3')
    print('    PC-SAFT:', calc, 'mol m^-3')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')

    # Binary mixture: methanol-cyclohexane
    print('\n##########  Test with methanol-cyclohexane mixture  ##########')
    #0 = methanol, 1 = cyclohexane
    x = np.asarray([0.0550, 0.945])
    m = np.asarray([1.5255, 2.5303])
    s = np.asarray([3.2300, 3.8499])
    e = np.asarray([188.90, 278.11])
    volAB = np.asarray([0.035176, 0.])
    eAB = np.asarray([2899.5, 0.])
    k_ij = np.asarray([[0, 0.051],
                       [0.051, 0]])
    pyargs = {'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij}
    
    ref = 9506.1 # source: J. Canosa, A. Rodríguez, and J. Tojo, “Liquid−Liquid Equilibrium and Physical Properties of the Ternary Mixture (Dimethyl Carbonate + Methanol + Cyclohexane) at 298.15 K,” J. Chem. Eng. Data, vol. 46, no. 4, pp. 846–850, Jul. 2001.  
    calc = pcsaft_den(x, m, s, e, 298.15, 101325, pyargs, phase='liq')
    print('----- Density at 298.15 K and 101325 Pa -----')
    print('    Reference:', ref, 'mol m^-3')
    print('    PC-SAFT:', calc, 'mol m^-3')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')   
    
    # NaCl in water
    print('\n##########  Test with aqueous NaCl  ##########')
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
    
    pyargs = {'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij, 'z':z, 'dielc':dielc}

    calc = pcsaft_den(x, m, s, e, t, 101325, pyargs, phase='liq')
    print('----- Density at 298.15 K and 101325 Pa -----')
    print('    Reference:', ref, 'mol m^-3')
    print('    PC-SAFT:', calc, 'mol m^-3')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')   
    
    return None


def test_vaporP():
    """Test the vapor pressure function to see if it is working correctly."""
    # Toluene
    print('##########  Test with toluene  ##########')
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    pyargs = {}
    
    ref = 3255792.76201971 # source: reference EOS in CoolProp
    calc = pcsaft_vaporP(ref, x, m, s, e, 572.6667, pyargs)
    print('----- Vapor pressure at 572.7 K -----')
    print('    Reference:', ref, 'Pa')
    print('    PC-SAFT:', calc, 'Pa')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')
    
    # Water
    print('\n##########  Test with water  ##########')
    m = np.asarray([1.2047])
    e = np.asarray([353.95])
    volAB = np.asarray([0.0451])
    eAB = np.asarray([2425.67])
    pyargs = {'e_assoc':eAB, 'vol_a':volAB}
    
    ref = 67171.754576141 # source: IAWPS95 EOS
    t = 362
    s = np.asarray([2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t)])
    calc = pcsaft_vaporP(ref, x, m, s, e, t, pyargs)
    print('----- Vapor pressure at 362 K -----')
    print('    Reference:', ref, 'Pa')
    print('    PC-SAFT:', calc, 'Pa')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')
    
    # Acetic acid
    print('\n##########  Test with acetic acid  ##########')
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    pyargs = {'e_assoc':eAB, 'vol_a':volAB}
    
    ref = 193261.515187248 # source: DIPPR correlation
    calc = pcsaft_vaporP(ref, x, m, s, e, 413.5385, pyargs)
    print('----- Vapor pressure at 413.5 K -----')
    print('    Reference:', ref, 'Pa')
    print('    PC-SAFT:', calc, 'Pa')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')    
    
    # Butyl acetate
    print('\n##########  Test with butyl acetate  ##########')
    m = np.asarray([3.9706])
    s = np.asarray([3.5440])
    e = np.asarray([241.93])
    dpm = np.asarray([1.86])
    dip_num = np.asarray([1.0])
    pyargs = {'dipm':dpm, 'dip_num':dip_num}
    
    ref = 219.6 # source: DIPPR correlation
    calc = pcsaft_vaporP(ref, x, m, s, e, 270, pyargs)
    print('----- Vapor pressure at 270 K -----')
    print('    Reference:', ref, 'Pa')
    print('    PC-SAFT:', calc, 'Pa')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')     
    
    return None

    
def test_bubbleP():
    """Test the bubble point pressure function to see if it is working correctly."""
    # Binary mixture: methanol-cyclohexane
    print('\n##########  Test with methanol-cyclohexane mixture  ##########')
    #0 = methanol, 1 = cyclohexane
    x = np.asarray([0.3,0.7])
    m = np.asarray([1.5255, 2.5303])
    s = np.asarray([3.2300, 3.8499])
    e = np.asarray([188.90, 278.11])
    volAB = np.asarray([0.035176, 0.])
    eAB = np.asarray([2899.5, 0.])
    k_ij = np.asarray([[0, 0.051],
                       [0.051, 0]])
    pyargs = {'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij}
    
    ref = 101330 # source: Marinichev A.N.; Susarev M.P.: Investigation of Liquid-Vapor Equilibrium in the System Methanol-Cyclohexane at 35, 45 and 55°C and 760 mm Hg. J.Appl.Chem.USSR 38 (1965) 1582-1584
    xv_ref = np.asarray([0.59400,0.40600])
    result = pcsaft_bubbleP(ref, xv_ref, x, m, s, e, 327.48, pyargs)
    calc = result[0]
    xv = result[1]
    print('----- Bubble point pressure at 327.48 K -----')
    print('    Reference:', ref, 'Pa')
    print('    PC-SAFT:', calc, 'Pa')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')
    print('    Vapor composition (reference):', xv_ref)
    print('    Vapor composition (PC-SAFT):', xv)     
    
    # NaCl in water
    print('\n##########  Test with aqueous NaCl  ##########')
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
    
    pyargs = {'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij, 'z':z, 'dielc':dielc}

    xv_guess = np.asarray([0., 0., 1.])
    result = pcsaft_bubbleP(ref, xv_guess, x, m, s, e, t, pyargs)
    calc = result[0]
    print('----- Bubble point pressure at 298.15 K -----')
    print('    Reference:', ref, 'Pa')
    print('    PC-SAFT:', calc, 'Pa')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')    
    
    return None
    
    
def test_PTz():
    """Test the function for PTz data to see if it is working correctly."""
    # Binary mixture: methanol-cyclohexane
    print('\n##########  Test with methanol-cyclohexane mixture  ##########')
    #0 = methanol, 1 = cyclohexane
    m = np.asarray([1.5255, 2.5303])
    s = np.asarray([3.2300, 3.8499])
    e = np.asarray([188.90, 278.11])
    volAB = np.asarray([0.035176, 0.])
    eAB = np.asarray([2899.5, 0.])
    k_ij = np.asarray([[0, 0.051],
                       [0.051, 0]])
    pyargs = {'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij}
    
    mol = 1.
    t = 327.48
    p_ref = 101330 # source: Marinichev A.N.; Susarev M.P.: Investigation of Liquid-Vapor Equilibrium in the System Methanol-Cyclohexane at 35, 45 and 55°C and 760 mm Hg. J.Appl.Chem.USSR 38 (1965) 1582-1584
    xl_ref = np.asarray([0.3,0.7])    
    xv_ref = np.asarray([0.59400,0.40600])
    beta_ref = 0.6
    xtot = (beta_ref*mol*xv_ref + (1-beta_ref)*mol*xl_ref)/mol
    rho_l = pcsaft_den(xl_ref, m, s, e, t, p_ref, pyargs, phase='liq')
    rho_v = pcsaft_den(xv_ref, m, s, e, t, p_ref, pyargs, phase='vap')
    vol = rho_v*beta_ref*mol + rho_l*(1-beta_ref)*mol
    result = pcsaft_PTz(p_ref, xl_ref, beta_ref, mol, vol, xtot, m, s, e, t, pyargs)
    p_calc = result[0]
    xl_calc = result[1]
    xv_calc = result[2]
    beta_calc = result[3]
    print('----- Pressure at 327.48 K -----')
    print('    Reference:', p_ref, 'Pa')
    print('    PC-SAFT:', p_calc, 'Pa')
    print('    Relative deviation:', (p_calc-p_ref)/p_ref*100, '%')
    print('----- Liquid phase composition -----')
    print('    Reference:', xl_ref, 'Pa')
    print('    PC-SAFT:', xl_calc, 'Pa')
    print('    Relative deviation:', (xl_calc-xl_ref)/xl_ref*100, '%')  
    print('----- Vapor phase composition -----')
    print('    Reference:', xv_ref, 'Pa')
    print('    PC-SAFT:', xv_calc, 'Pa')
    print('    Relative deviation:', (xv_calc-xv_ref)/xv_ref*100, '%')
    print('----- Beta -----')
    print('    Reference:', beta_ref, 'Pa')
    print('    PC-SAFT:', beta_calc, 'Pa')
    print('    Relative deviation:', (beta_calc-beta_ref)/beta_ref*100, '%')    
    
    return None
    

def test_osmoticC():
    """Test the function for calculating osmotic coefficients to see if it is working correctly."""
    # NaCl in water
    print('\n##########  Test with aqueous NaCl  ##########')
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
    
    pyargs = {'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij, 'z':z, 'dielc':dielc}
 
    rho = pcsaft_den(x, m, s, e, t, 2339.3, pyargs, phase='liq')      
    result = pcsaft_osmoticC(x, m, s, e, t, rho, pyargs)
    calc = result[0]
    print('----- Osmotic coefficient at 293.15 K -----')
    print('    Reference:', ref)
    print('    PC-SAFT:', calc)
    print('    Relative deviation:', (calc-ref)/ref*100, '%')      
    
    return None


def test_Hvap():
    """Test the enthalpy of vaporization function to see if it is working correctly."""
    # Toluene
    print('##########  Test with toluene  ##########')
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    pyargs = {}
    
    ref = 33500. # source: DIPPR correlation
    p = 90998. # source: reference equation of state from Polt, A.; Platzer, B.; Maurer, G., Parameter der thermischen Zustandsgleichung von Bender fuer 14 mehratomige reine Stoffe, Chem. Tech. (Leipzig), 1992, 44, 6, 216-224.
    calc = pcsaft_Hvap(p, x, m, s, e, 380., pyargs)[0]
    print('----- Enthalpy of vaporization at 380 K -----')
    print('    Reference:', ref, 'J mol^-1')
    print('    PC-SAFT:', calc, 'J mol^-1')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')
    
    # Water
    print('\n##########  Test with water  ##########')
    m = np.asarray([1.2047])
    e = np.asarray([353.95])
    volAB = np.asarray([0.0451])
    eAB = np.asarray([2425.67])
    pyargs = {'e_assoc':eAB, 'vol_a':volAB}
    
    ref = 44761.23 # source: IAWPS95 EOS
    p = 991.82 # source: IAWPS95 EOS
    t = 280
    s = np.asarray([2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t)])
    calc = pcsaft_Hvap(p, x, m, s, e, t, pyargs)[0]
    print('----- Enthalpy of vaporization at 280 K -----')
    print('    Reference:', ref, 'J mol^-1')
    print('    PC-SAFT:', calc, 'J mol^-1')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')    
    
    # Butyl acetate
    print('\n##########  Test with butyl acetate  ##########')
    m = np.asarray([2.76462805])
    s = np.asarray([4.02244938])
    e = np.asarray([263.69902915])
    dpm = np.asarray([1.84])
    dip_num = np.asarray([4.99688339])
    pyargs = {'dipm':dpm, 'dip_num':dip_num}
    
    ref = 32280. # source: DIPPR correlation
    p = 362200. # source: DIPPR correlation
    calc = pcsaft_Hvap(p, x, m, s, e, 450., pyargs)[0]
    print('----- Enthalpy of vaporization at 450 K -----')
    print('    Reference:', ref, 'J mol^-1')
    print('    PC-SAFT:', calc, 'J mol^-1')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')     
    
    return None

test_density()
#test_vaporP()
#test_bubbleP()
#test_osmoticC()
#test_Hvap()
#test_hres()
#test_sres()
#test_gres()
#test_PTz()
