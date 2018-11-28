# -*- coding: utf-8 -*-
"""
Tests for checking that the PC-SAFT functions are working correctly.

@author: Zach Baird
"""
import numpy as np
import timeit
from pcsaft_electrolyte import pcsaft_den, pcsaft_hres, pcsaft_gres, pcsaft_sres, pcsaft_Hvap
from pcsaft_electrolyte import pcsaft_vaporP, pcsaft_bubbleP, dielc_water, pcsaft_PTz, pcsaft_osmoticC
from pcsaft_electrolyte import pcsaft_dadt, pcsaft_d2adt, pcsaft_cp, pcsaft_ares, pcsaft_p, pcsaft_fugcoef
import pylab as pl

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
    
    den = pcsaft_den(x, m, s, e, t, p, phase='liq')
    calc = pcsaft_hres(x, m, s, e, t, den)
    print('Toluene, liquid:\t\t', calc, -36809.39, 'J/mol')
    den = pcsaft_den(x, m, s, e, t, p, phase='vap')
    calc = pcsaft_hres(x, m, s, e, t, den)
    print('Toluene, vapor:\t\t', calc, -362.6777, 'J/mol')
    
    # Acetic acid ---------
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])    
    
    den = pcsaft_den(x, m, s, e, t, p, phase='liq', e_assoc=eAB, vol_a=volAB)
    calc = pcsaft_hres(x, m, s, e, t, den, e_assoc=eAB, vol_a=volAB)
    print('Acetic acid, liquid:\t\t', calc, -38924.64, 'J/mol')
    den = pcsaft_den(x, m, s, e, t, p, phase='vap', e_assoc=eAB, vol_a=volAB)
    calc = pcsaft_hres(x, m, s, e, t, den, e_assoc=eAB, vol_a=volAB)
    print('Acetic acid, vapor:\t\t', calc, -15393.63, 'J/mol')
    
    # dimethyl ether ---------
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])
    
    den = pcsaft_den(x, m, s, e, t, p, phase='liq', dipm=dpm, dip_num=dip_num)
    calc = pcsaft_hres(x, m, s, e, t, den, dipm=dpm, dip_num=dip_num)
    print('dimethyl ether, liquid:\t', calc, -18242.5, 'J/mol')
    den = pcsaft_den(x, m, s, e, t, p, phase='vap', dipm=dpm, dip_num=dip_num)
    calc = pcsaft_hres(x, m, s, e, t, den, dipm=dpm, dip_num=dip_num)
    print('dimethyl ether, vapor:\t\t', calc, -89.6574, 'J/mol')
    
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
    
    den = pcsaft_den(x, m, s, e, t, p, phase='liq')
    calc = pcsaft_sres(x, m, s, e, t, den)
    print('Toluene, liquid:\t\t', calc, -96.3692, 'J/mol/K')
    den = pcsaft_den(x, m, s, e, t, p, phase='vap')
    calc = pcsaft_sres(x, m, s, e, t, den)
    print('Toluene, vapor:\t\t', calc, -0.71398, 'J/mol/K')
    
    # Acetic acid ---------
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])    
    
    den = pcsaft_den(x, m, s, e, t, p, phase='liq', e_assoc=eAB, vol_a=volAB)
    calc = pcsaft_sres(x, m, s, e, t, den, e_assoc=eAB, vol_a=volAB)
    print('Acetic acid, liquid:\t\t', calc, -98.1127, 'J/mol/K')
    den = pcsaft_den(x, m, s, e, t, p, phase='vap', e_assoc=eAB, vol_a=volAB)
    calc = pcsaft_sres(x, m, s, e, t, den, e_assoc=eAB, vol_a=volAB)
    print('Acetic acid, vapor:\t\t', calc, -40.8743, 'J/mol/K')
    
    # dimethyl ether ---------
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])
    
    den = pcsaft_den(x, m, s, e, t, p, phase='liq', dipm=dpm, dip_num=dip_num)
    calc = pcsaft_sres(x, m, s, e, t, den, dipm=dpm, dip_num=dip_num)
    print('dimethyl ether, liquid:\t', calc, -75.2232, 'J/mol/K')
    den = pcsaft_den(x, m, s, e, t, p, phase='vap', dipm=dpm, dip_num=dip_num)
    calc = pcsaft_sres(x, m, s, e, t, den, dipm=dpm, dip_num=dip_num)
    print('dimethyl ether, vapor:\t\t', calc, -0.17854, 'J/mol/K')
    
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
    
    den = pcsaft_den(x, m, s, e, t, p, phase='liq')
    calc = pcsaft_gres(x, m, s, e, t, den)
    print('Toluene, liquid:\t\t', calc, -5489.384, 'J/mol')
    den = pcsaft_den(x, m, s, e, t, p, phase='vap')
    calc = pcsaft_gres(x, m, s, e, t, den)
    print('Toluene, vapor:\t\t', calc, -130.6339, 'J/mol')
    
    # Acetic acid ---------
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])    
    
    den = pcsaft_den(x, m, s, e, t, p, phase='liq', e_assoc=eAB, vol_a=volAB)
    calc = pcsaft_gres(x, m, s, e, t, den, e_assoc=eAB, vol_a=volAB)
    print('Acetic acid, liquid:\t\t', calc, -7038.004, 'J/mol')
    den = pcsaft_den(x, m, s, e, t, p, phase='vap', e_assoc=eAB, vol_a=volAB)
    calc = pcsaft_gres(x, m, s, e, t, den, e_assoc=eAB, vol_a=volAB)
    print('Acetic acid, vapor:\t\t', calc, -2109.459, 'J/mol')
    
    # dimethyl ether ---------
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])
    
    den = pcsaft_den(x, m, s, e, t, p, phase='liq', dipm=dpm, dip_num=dip_num)
    calc = pcsaft_gres(x, m, s, e, t, den, dipm=dpm, dip_num=dip_num)
    print('dimethyl ether, liquid:\t', calc, -6205.02, 'J/mol')
    den = pcsaft_den(x, m, s, e, t, p, phase='vap', dipm=dpm, dip_num=dip_num)
    calc = pcsaft_gres(x, m, s, e, t, den, dipm=dpm, dip_num=dip_num)
    print('dimethyl ether, vapor:\t\t', calc, -31.6322, 'J/mol')
    
    return None

def test_density():
    """Test the density function to see if it is working correctly."""
#     Toluene
    print('##########  Test with toluene  ##########')
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    
    ref = 9135.590853014008 # source: reference EOS in CoolProp
    calc = pcsaft_den(x, m, s, e, 320, 101325, phase='liq')
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
    
    ref = 55502.5970532902 # source: IAWPS95 EOS
    t = 274
    s = np.asarray([2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t)])
    calc = pcsaft_den(x, m, s, e, t, 101325, phase='liq', e_assoc=eAB, vol_a=volAB)
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
    
    ref = 17240. # source: DIPPR correlation
    calc = pcsaft_den(x, m, s, e, 305, 101325, phase='liq', e_assoc=eAB, vol_a=volAB)
    print('----- Density at 305 K and 101325 Pa -----')
    print('    Reference:', ref, 'mol m^-3')
    print('    PC-SAFT:', calc, 'mol m^-3')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')  
    
    # dimethyl ether
    print('\n##########  Test with dimethyl ether  ##########')
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])
    
    ref = 16110. # source: DIPPR correlation
    calc = pcsaft_den(x, m, s, e, 240, 101325, phase='liq', dipm=dpm, dip_num=dip_num)
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
    
    ref = 9506.1 # source: J. Canosa, A. Rodríguez, and J. Tojo, “Liquid−Liquid Equilibrium and Physical Properties of the Ternary Mixture (Dimethyl Carbonate + Methanol + Cyclohexane) at 298.15 K,” J. Chem. Eng. Data, vol. 46, no. 4, pp. 846–850, Jul. 2001.  
    calc = pcsaft_den(x, m, s, e, 298.15, 101325, phase='liq', k_ij=k_ij, e_assoc=eAB, vol_a=volAB)
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

    calc = pcsaft_den(x, m, s, e, t, 101325, phase='liq', k_ij=k_ij, e_assoc=eAB, vol_a=volAB, z=z, dielc=dielc)
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
    
    ref = 3255792.76201971 # source: reference EOS in CoolProp
    calc = pcsaft_vaporP(ref, x, m, s, e, 572.6667)
    print('----- Vapor pressure at 572.7 K -----')
    print('    Reference:', ref, 'Pa')
    print('    PC-SAFT:', calc, 'Pa')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')
    
#     Water
    print('\n##########  Test with water  ##########')
    m = np.asarray([1.2047])
    e = np.asarray([353.95])
    volAB = np.asarray([0.0451])
    eAB = np.asarray([2425.67])
    
    ref = 67171.754576141 # source: IAWPS95 EOS
    t = 362
    s = np.asarray([2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t)])
    calc = pcsaft_vaporP(ref, x, m, s, e, t, e_assoc=eAB, vol_a=volAB)
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
    
    ref = 193261.515187248 # source: DIPPR correlation
    calc = pcsaft_vaporP(ref, x, m, s, e, 413.5385, e_assoc=eAB, vol_a=volAB)
    print('----- Vapor pressure at 413.5 K -----')
    print('    Reference:', ref, 'Pa')
    print('    PC-SAFT:', calc, 'Pa')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')    
    
#    # dimethyl ether
    print('\n##########  Test with dimethyl ether  ##########')
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])
    
    ref = 625100. # source: DIPPR correlation
    calc = pcsaft_vaporP(ref, x, m, s, e, 300, dipm=dpm, dip_num=dip_num)
    print('----- Vapor pressure at 270 K -----')
    print('    Reference:', ref, 'Pa')
    print('    PC-SAFT:', calc, 'Pa')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')     
    
    return None

    
def test_bubbleP():
    """Test the bubble point pressure function to see if it is working correctly."""
#     Binary mixture: methanol-cyclohexane
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
    
    ref = 101330 # source: Marinichev A.N.; Susarev M.P.: Investigation of Liquid-Vapor Equilibrium in the System Methanol-Cyclohexane at 35, 45 and 55°C and 760 mm Hg. J.Appl.Chem.USSR 38 (1965) 1582-1584
    xv_ref = np.asarray([0.59400,0.40600])
    result = pcsaft_bubbleP(ref, xv_ref, x, m, s, e, 327.48, k_ij=k_ij, e_assoc=eAB, vol_a=volAB)
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

    xv_guess = np.asarray([0., 0., 1.])
    result = pcsaft_bubbleP(ref, xv_guess, x, m, s, e, t, k_ij=k_ij, e_assoc=eAB, vol_a=volAB, z=z, dielc=dielc)
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
    
    mol = 1.
    t = 327.48
    p_ref = 101330 # source: Marinichev A.N.; Susarev M.P.: Investigation of Liquid-Vapor Equilibrium in the System Methanol-Cyclohexane at 35, 45 and 55°C and 760 mm Hg. J.Appl.Chem.USSR 38 (1965) 1582-1584
    xl_ref = np.asarray([0.3,0.7])    
    xv_ref = np.asarray([0.59400,0.40600])
    beta_ref = 0.6
    xtot = (beta_ref*mol*xv_ref + (1-beta_ref)*mol*xl_ref)/mol
    rho_l = pcsaft_den(xl_ref, m, s, e, t, p_ref, phase='liq', k_ij=k_ij, e_assoc=eAB, vol_a=volAB)
    rho_v = pcsaft_den(xv_ref, m, s, e, t, p_ref, phase='vap', k_ij=k_ij, e_assoc=eAB, vol_a=volAB)
    vol = beta_ref*mol/rho_v + (1-beta_ref)*mol/rho_l
    result = pcsaft_PTz(p_ref, xl_ref, beta_ref, mol, vol, xtot, m, s, e, t, k_ij=k_ij, e_assoc=eAB, vol_a=volAB)
    p_calc = result[0]
    xl_calc = result[1]
    xv_calc = result[2]
    beta_calc = result[3]
    print('----- Pressure at 327.48 K -----')
    print('    Reference:', p_ref, 'Pa')
    print('    PC-SAFT:', p_calc, 'Pa')
    print('    Relative deviation:', (p_calc-p_ref)/p_ref*100, '%')
    print('----- Liquid phase composition -----')
    print('    Reference:', xl_ref)
    print('    PC-SAFT:', xl_calc)
    print('    Relative deviation:', (xl_calc-xl_ref)/xl_ref*100, '%')  
    print('----- Vapor phase composition -----')
    print('    Reference:', xv_ref)
    print('    PC-SAFT:', xv_calc)
    print('    Relative deviation:', (xv_calc-xv_ref)/xv_ref*100, '%')
    print('----- Beta -----')
    print('    Reference:', beta_ref)
    print('    PC-SAFT:', beta_calc)
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
 
    rho = pcsaft_den(x, m, s, e, t, 2339.3, phase='liq', k_ij=k_ij, e_assoc=eAB, vol_a=volAB, z=z, dielc=dielc)      
    result = pcsaft_osmoticC(x, m, s, e, t, rho, k_ij=k_ij, e_assoc=eAB, vol_a=volAB, z=z, dielc=dielc)
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
    
    ref = 33500. # source: DIPPR correlation
    p = 90998. # source: reference equation of state from Polt, A.; Platzer, B.; Maurer, G., Parameter der thermischen Zustandsgleichung von Bender fuer 14 mehratomige reine Stoffe, Chem. Tech. (Leipzig), 1992, 44, 6, 216-224.
    calc = pcsaft_Hvap(p, x, m, s, e, 380.)[0]
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
    
    ref = 44761.23 # source: IAWPS95 EOS
    p = 991.82 # source: IAWPS95 EOS
    t = 280
    s = np.asarray([2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t)])
    calc = pcsaft_Hvap(p, x, m, s, e, t, e_assoc=eAB, vol_a=volAB)[0]
    print('----- Enthalpy of vaporization at 280 K -----')
    print('    Reference:', ref, 'J mol^-1')
    print('    PC-SAFT:', calc, 'J mol^-1')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')    
    
    # dimethyl ether
    print('\n##########  Test with dimethyl ether  ##########')
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])
    
    ref = 17410. # source: DIPPR correlation
    p = 937300. # source: DIPPR correlation
    calc = pcsaft_Hvap(p, x, m, s, e, 315., dipm=dpm, dip_num=dip_num)[0]
    print('----- Enthalpy of vaporization at 315 K -----')
    print('    Reference:', ref, 'J mol^-1')
    print('    PC-SAFT:', calc, 'J mol^-1')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')     
    
    return None


def test_dadt():
    """Test the function for the temperature derivative of the Helmholtz energy."""
    print('##########  Testing pcsaft_dadt  ##########')

    # Toluene    
    print('##########  Test with toluene  ##########')
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    
    p = 100000.
    t = 330.
    
    rho = pcsaft_den(x, m, s, e, t, p, phase='liq')
    dadt_eos = pcsaft_dadt(x, m, s, e, t, rho)
    
    # calculating numerical derivative
    der1 = pcsaft_ares(x, m, s, e, t-1, rho)
    der2 = pcsaft_ares(x, m, s, e, t+1, rho)
    dadt_num = (der2-der1)/2.
    print('    Numerical derivative:', dadt_num)
    print('    PC-SAFT derivative:', dadt_eos)
    print('    Relative deviation:', (dadt_eos-dadt_num)/dadt_num*100, '%')
    
    # Acetic acid   
    print('##########  Test with acetic acid  ##########')
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])

    p = 100000.
    t = 310.
    
    rho = pcsaft_den(x, m, s, e, t, p, phase='liq', e_assoc=eAB, vol_a=volAB)
    dadt_eos = pcsaft_dadt(x, m, s, e, t, rho, e_assoc=eAB, vol_a=volAB)
    
    # calculating numerical derivative
    der1 = pcsaft_ares(x, m, s, e, t-1, rho, e_assoc=eAB, vol_a=volAB)
    der2 = pcsaft_ares(x, m, s, e, t+1, rho, e_assoc=eAB, vol_a=volAB)
    dadt_num = (der2-der1)/2.
    print('    Numerical derivative:', dadt_num)
    print('    PC-SAFT derivative:', dadt_eos)
    print('    Relative deviation:', (dadt_eos-dadt_num)/dadt_num*100, '%')      
    
    # Water    
    print('##########  Test with water  ##########')
    m = np.asarray([1.2047])
    e = np.asarray([353.95])
    volAB = np.asarray([0.0451])
    eAB = np.asarray([2425.67])

    p = 100000.
    t = 290.
    
    s = np.asarray([2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t)])

    rho = pcsaft_den(x, m, s, e, t, p, phase='liq', e_assoc=eAB, vol_a=volAB)
    dadt_eos = pcsaft_dadt(x, m, s, e, t, rho, e_assoc=eAB, vol_a=volAB)
    
    # calculating numerical derivative
    der1 = pcsaft_ares(x, m, s, e, t-1, rho, e_assoc=eAB, vol_a=volAB)
    der2 = pcsaft_ares(x, m, s, e, t+1, rho, e_assoc=eAB, vol_a=volAB)
    dadt_num = (der2-der1)/2.
    print('    Numerical derivative:', dadt_num)
    print('    PC-SAFT derivative:', dadt_eos)
    print('    Relative deviation:', (dadt_eos-dadt_num)/dadt_num*100, '%')   
    
    # dimethyl ether    
    print('##########  Test with dimethyl ether  ##########')
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])

    p = 100000.
    t = 370.

    rho = pcsaft_den(x, m, s, e, t, p, phase='liq', dipm=dpm, dip_num=dip_num)
    dadt_eos = pcsaft_dadt(x, m, s, e, t, rho, dipm=dpm, dip_num=dip_num)
    
    # calculating numerical derivative
    der1 = pcsaft_ares(x, m, s, e, t-1, rho, dipm=dpm, dip_num=dip_num)
    der2 = pcsaft_ares(x, m, s, e, t+1, rho, dipm=dpm, dip_num=dip_num)
    dadt_num = (der2-der1)/2.
    print('    Numerical derivative:', dadt_num)
    print('    PC-SAFT derivative:', dadt_eos)
    print('    Relative deviation:', (dadt_eos-dadt_num)/dadt_num*100, '%')  
    
    # Aqueous NaCl    
    print('##########  Test with aqueous NaCl  ##########')
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

    rho = pcsaft_den(x, m, s, e, t, p, phase='liq', k_ij=k_ij, e_assoc=eAB, vol_a=volAB, z=z, dielc=dielc)
    dadt_eos = pcsaft_dadt(x, m, s, e, t, rho, k_ij=k_ij, e_assoc=eAB, vol_a=volAB, z=z, dielc=dielc)
    
    # calculating numerical derivative
    der1 = pcsaft_ares(x, m, s, e, t-1, rho, k_ij=k_ij, e_assoc=eAB, vol_a=volAB, z=z, dielc=dielc)
    der2 = pcsaft_ares(x, m, s, e, t+1, rho, k_ij=k_ij, e_assoc=eAB, vol_a=volAB, z=z, dielc=dielc)
    dadt_num = (der2-der1)/2.
    print('    Numerical derivative:', dadt_num)
    print('    PC-SAFT derivative:', dadt_eos)
    print('    Relative deviation:', (dadt_eos-dadt_num)/dadt_num*100, '%')      
    
    return None


def test_d2adt():
    """Test the function for the second temperature derivative of the Helmholtz energy."""
    print('##########  Testing pcsaft_d2adt  ##########')

    # Toluene    
    print('##########  Test with toluene  ##########')
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    
    p = 100000.
    t = 330.
    
    rho = pcsaft_den(x, m, s, e, t, p, phase='liq')
    d2adt_eos = pcsaft_d2adt(x, m, s, e, t, rho)
    
    # calculating numerical derivative
    rho = pcsaft_den(x, m, s, e, t-1, p, phase='liq')
    der1 = pcsaft_dadt(x, m, s, e, t-1, rho)
    rho = pcsaft_den(x, m, s, e, t+1, p, phase='liq')
    der2 = pcsaft_dadt(x, m, s, e, t+1, rho)
    d2adt_num = (der2-der1)/2.
    print('    Numerical derivative:', d2adt_num)
    print('    PC-SAFT derivative:', d2adt_eos)
    print('    Relative deviation:', (d2adt_eos-d2adt_num)/d2adt_num*100, '%')
    
    # Water    
    print('##########  Test with water  ##########')
    m = np.asarray([1.2047])
    e = np.asarray([353.95])
    volAB = np.asarray([0.0451])
    eAB = np.asarray([2425.67])

    p = 100000.
    t = 290.
    
    s = np.asarray([2.7927 + 10.11*np.exp(-0.01775*t) - 1.417*np.exp(-0.01146*t)])

    rho = pcsaft_den(x, m, s, e, t, p, phase='liq', e_assoc=eAB, vol_a=volAB)
    d2adt_eos = pcsaft_d2adt(x, m, s, e, t, rho, e_assoc=eAB, vol_a=volAB)
    
    # calculating numerical derivative
    der1 = pcsaft_dadt(x, m, s, e, t-1, rho, e_assoc=eAB, vol_a=volAB)
    der2 = pcsaft_dadt(x, m, s, e, t+1, rho, e_assoc=eAB, vol_a=volAB)
    d2adt_num = (der2-der1)/2.
    print('    Numerical derivative:', d2adt_num)
    print('    PC-SAFT derivative:', d2adt_eos)
    print('    Relative deviation:', (d2adt_eos-d2adt_num)/d2adt_num*100, '%')   
    
    # dimethyl ether    
    print('##########  Test with dimethyl ether  ##########')
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])

    p = 100000.
    t = 370.

    rho = pcsaft_den(x, m, s, e, t, p, phase='liq', dipm=dpm, dip_num=dip_num)
    d2adt_eos = pcsaft_d2adt(x, m, s, e, t, rho, dipm=dpm, dip_num=dip_num)
    
    # calculating numerical derivative
    der1 = pcsaft_dadt(x, m, s, e, t-1, rho, dipm=dpm, dip_num=dip_num)
    der2 = pcsaft_dadt(x, m, s, e, t+1, rho, dipm=dpm, dip_num=dip_num)
    d2adt_num = (der2-der1)/2.
    print('    Numerical derivative:', d2adt_num)
    print('    PC-SAFT derivative:', d2adt_eos)
    print('    Relative deviation:', (d2adt_eos-d2adt_num)/d2adt_num*100, '%')  
    
    # Aqueous NaCl    
    print('##########  Test with aqueous NaCl  ##########')
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

    rho = pcsaft_den(x, m, s, e, t, p, phase='liq', k_ij=k_ij, e_assoc=eAB, vol_a=volAB, z=z, dielc=dielc)
    d2adt_eos = pcsaft_d2adt(x, m, s, e, t, rho, k_ij=k_ij, e_assoc=eAB, vol_a=volAB, z=z, dielc=dielc)
    
    # calculating numerical derivative
    der1 = pcsaft_dadt(x, m, s, e, t-1, rho, k_ij=k_ij, e_assoc=eAB, vol_a=volAB, z=z, dielc=dielc)
    der2 = pcsaft_dadt(x, m, s, e, t+1, rho, k_ij=k_ij, e_assoc=eAB, vol_a=volAB, z=z, dielc=dielc)
    d2adt_num = (der2-der1)/2.
    print('    Numerical derivative:', d2adt_num)
    print('    PC-SAFT derivative:', d2adt_eos)
    print('    Relative deviation:', (d2adt_eos-d2adt_num)/d2adt_num*100, '%')      
    
    return None


def test_cp():
    """Test the heat capacity function to see if it is working correctly."""
    # Benzene
    print('##########  Test with benzene  ##########')
    x = np.asarray([1.])
    m = np.asarray([2.4653])
    s = np.asarray([3.6478])
    e = np.asarray([287.35])
    cnsts = np.asarray([55238., 173380, 764.25, 72545, 2445.7]) # constants for Aly-Lee equation (obtained from DIPPR)
    
    ref = 140.78 # source: Equation of state from Polt et al. (1992) (available at https://webbook.nist.gov/chemistry/fluid/)
    p = 100000.
    t = 330.
    rho = pcsaft_den(x, m, s, e, t, p, phase='liq')
    calc = pcsaft_cp(x, m, s, e, t, rho, cnsts)
    print('----- Heat capacity at 330 K -----')
    print('    Reference:', ref, 'J mol^-1 K^-1')
    print('    PC-SAFT:', calc, 'J mol^-1 K^-1')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')

    # Toluene
    print('##########  Test with toluene  ##########')
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    cnsts = np.asarray([58140., 286300, 1440.6, 189800, 650.43]) # constants for Aly-Lee equation (obtained from DIPPR)
    
    ref = 179.79 # source: Equation of state from Polt et al. (1992) (available at https://webbook.nist.gov/chemistry/fluid/)
    p = 100000.
    t = 370.
    rho = pcsaft_den(x, m, s, e, t, p, phase='liq')
    calc = pcsaft_cp(x, m, s, e, t, rho, cnsts)
    print('----- Heat capacity at 370 K -----')
    print('    Reference:', ref, 'J mol^-1 K^-1')
    print('    PC-SAFT:', calc, 'J mol^-1 K^-1')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')
    
    # Water
    print('\n##########  Test with acetic acid  ##########')
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    cnsts = np.asarray([40200., 136750, 1262, 70030, 569.7]) # constants for Aly-Lee equation (obtained from DIPPR)

    ref = 130.3 # source: DIPPR
    p = 100000.
    t = 325.
    rho = pcsaft_den(x, m, s, e, t, p, phase='liq', e_assoc=eAB, vol_a=volAB)
    calc = pcsaft_cp(x, m, s, e, t, rho, cnsts, e_assoc=eAB, vol_a=volAB)
    """ Note: Large deviations occur with acetic acid and water. This behavior 
    has been observed before and was described in R. T. C. S. Ribeiro, A. L. 
    Alberton, M. L. L. Paredes, G. M. Kontogeorgis, and X. Liang, “Extensive 
    Study of the Capabilities and Limitations of the CPA and sPC-SAFT Equations 
    of State in Modeling a Wide Range of Acetic Acid Properties,” Ind. Eng. 
    Chem. Res., vol. 57, no. 16, pp. 5690–5704, Apr. 2018. """
    print('----- Heat capacity at 325 K -----')
    print('    Reference:', ref, 'J mol^-1 K^-1')
    print('    PC-SAFT:', calc, 'J mol^-1 K^-1')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')    
    
    # dimethyl ether
    print('\n##########  Test with dimethyl ether  ##########')
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])
    cnsts = np.asarray([57431., 94494, 895.51, 65065, 2467.4]) # constants for Aly-Lee equation (obtained from DIPPR)

    ref = 102.2 # source: DIPPR correlation
    p = 100000.
    t = 240.
    rho = pcsaft_den(x, m, s, e, t, p, phase='liq', dipm=dpm, dip_num=dip_num)
    calc = pcsaft_cp(x, m, s, e, t, rho, cnsts, dipm=dpm, dip_num=dip_num)
    print('----- Heat capacity at 240 K -----')
    print('    Reference:', ref, 'J mol^-1 K^-1')
    print('    PC-SAFT:', calc, 'J mol^-1 K^-1')
    print('    Relative deviation:', (calc-ref)/ref*100, '%')     
    
    return None
