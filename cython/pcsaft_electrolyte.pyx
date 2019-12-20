# -*- coding: utf-8 -*-
# distutils: language=c++
"""
PC-SAFT with electrolyte term

These functions implement the PC-SAFT equation of state. In addition to the
hard chain and dispersion terms, these functions also include dipole,
association and ion terms for use with these types of compounds.

@author: Zach Baird

Functions
---------
- pcsaft_vaporP : calculate the vapor pressure
- pcsaft_bubbleT : calculate the bubble point temperature of a mixture
- pcsaft_bubbleP : calculate the bubble point pressure of a mixture
- pcsaft_Hvap : calculate the enthalpy of vaporization
- pcsaft_osmoticC : calculate the osmotic coefficient for the mixture
- pcsaft_cp : calculate the heat capacity
- pcsaft_PTz : allows PTz data to be used for parameter fitting
- pcsaft_PTzfit : used for fitting using PTz data
- pcsaft_den : calculate the molar density
- pcsaft_p : calculate the pressure
- pcsaft_hres : calculate the residual enthalpy
- pcsaft_sres : calculate the residual entropy (as a function of temperature and pressure)
- pcsaft_gres : calculate the residual Gibbs free energy
- pcsaft_fugcoef : calculate the fugacity coefficients
- pcsaft_Z : calculate the compressibility factor
- pcsaft_ares : calculate the residual Helmholtz energy
- pcsaft_dadt : calculate the temperature derivative of the residual Helmholtz energy
- XA_find : used internally to solve for XA
- dXA_find : used internally to solve for the derivative of XA wrt density
- dXAdt_find : used internally to solve for the derivative of XA wrt temperature
- vaporPfit : used internally to solve for the vapor pressure
- bubblePfit : used internally to solve for the bubble point pressure
- bubbleTfit : used internally to solve for the bubble point temperature
- PTzfit : used internally to solve for pressure and compositions
- aly_lee : returns the ideal gas heat capacity
- dielc_water : returns the dielectric constant of water
- np_to_vector : converts a numpy array to a C++ vector
- create_struct : converts additional arguments to a C++ struct

References
----------
* J. Gross and G. Sadowski, “Perturbed-Chain SAFT:  An Equation of State
Based on a Perturbation Theory for Chain Molecules,” Ind. Eng. Chem.
Res., vol. 40, no. 4, pp. 1244–1260, Feb. 2001.
* M. Kleiner and G. Sadowski, “Modeling of Polar Systems Using PCP-SAFT: 
An Approach to Account for Induced-Association Interactions,” J. Phys.
Chem. C, vol. 111, no. 43, pp. 15544–15553, Nov. 2007.
* Gross Joachim and Vrabec Jadran, “An equation‐of‐state contribution
for polar components: Dipolar molecules,” AIChE J., vol. 52, no. 3,
pp. 1194–1204, Feb. 2006.
* A. J. de Villiers, C. E. Schwarz, and A. J. Burger, “Improving
vapour–liquid-equilibria predictions for mixtures with non-associating polar
components using sPC-SAFT extended with two dipolar terms,” Fluid Phase
Equilibria, vol. 305, no. 2, pp. 174–184, Jun. 2011.
* S. H. Huang and M. Radosz, “Equation of state for small, large,
polydisperse, and associating molecules,” Ind. Eng. Chem. Res., vol. 29,
no. 11, pp. 2284–2294, Nov. 1990.
* S. H. Huang and M. Radosz, “Equation of state for small, large,
polydisperse, and associating molecules: extension to fluid mixtures,”
Ind. Eng. Chem. Res., vol. 30, no. 8, pp. 1994–2005, Aug. 1991.
* S. H. Huang and M. Radosz, “Equation of state for small, large,
polydisperse, and associating molecules: extension to fluid mixtures.
[Erratum to document cited in CA115(8):79950j],” Ind. Eng. Chem. Res.,
vol. 32, no. 4, pp. 762–762, Apr. 1993.
* J. Gross and G. Sadowski, “Application of the Perturbed-Chain SAFT
Equation of State to Associating Systems,” Ind. Eng. Chem. Res., vol.
41, no. 22, pp. 5510–5515, Oct. 2002.
* L. F. Cameretti, G. Sadowski, and J. M. Mollerup, “Modeling of Aqueous
Electrolyte Solutions with Perturbed-Chain Statistical Associated Fluid
Theory,” Ind. Eng. Chem. Res., vol. 44, no. 9, pp. 3355–3362, Apr. 2005.
* L. F. Cameretti, G. Sadowski, and J. M. Mollerup, “Modeling of Aqueous
Electrolyte Solutions with Perturbed-Chain Statistical Association Fluid
Theory,” Ind. Eng. Chem. Res., vol. 44, no. 23, pp. 8944–8944, Nov. 2005.
* C. Held, L. F. Cameretti, and G. Sadowski, “Modeling aqueous
electrolyte solutions: Part 1. Fully dissociated electrolytes,” Fluid
Phase Equilibria, vol. 270, no. 1, pp. 87–96, Aug. 2008.
* C. Held, T. Reschke, S. Mohammad, A. Luza, and G. Sadowski, “ePC-SAFT
revised,” Chem. Eng. Res. Des., vol. 92, no. 12, pp. 2884–2897, Dec. 2014.
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize
from libcpp.vector cimport vector
cimport pcsaft_electrolyte

class InputError(Exception):
    """Exception raised for errors in the input.
    """
    def __init__(self, message):
        self.message = message

def check_input(x, name1, var1, name2, var2):
    ''' Perform a few basic checks to make sure the input is reasonable. '''
    if abs(np.sum(x) - 1) > 1e-7:
        raise InputError('The mole fractions do not sum to 1. x = {}'.format(x))
    if var1 <= 0:
        raise InputError('The {} must be a positive number. {} = {}'.format(name1, name1, var1))
    if var2 <= 0:
        raise InputError('The {} must be a positive number. {} = {}'.format(name2, name2, var2))


def pcsaft_p(x, m, s, e, t, rho, pyargs):
    """
    Calculate pressure.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : ndarray, shape (n,)
        Segment number for each component.
    s : ndarray, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : ndarray, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    pyargs : dict
        A dictionary containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    P : float
        Pressure (Pa)
    """
    check_input(x, 'density', rho, 'temperature', t)
    cppargs = create_struct(pyargs)
    if type(m) == np.float_:
        m = np.asarray([m])
    if type(s) == np.float_:
        s = np.asarray([s])
    if type(e) == np.float_:
        e = np.asarray([e])
    return pcsaft_p_cpp(x, m, s, e, t, rho, cppargs)


def pcsaft_fugcoef(x, m, s, e, t, rho, pyargs):
    """
    Calculate the fugacity coefficients for one phase of the system.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : ndarray, shape (n,)
        Segment number for each component.
    s : ndarray, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : ndarray, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    pyargs : dict
        A dictionary containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    fugcoef : ndarray, shape (n,)
        Fugacity coefficients of each component.
    """
    check_input(x, 'density', rho, 'temperature', t)
    cppargs = create_struct(pyargs)
    if type(m) == np.float_:
        m = np.asarray([m])
    if type(s) == np.float_:
        s = np.asarray([s])
    if type(e) == np.float_:
        e = np.asarray([e])
    return pcsaft_fugcoef_cpp(x, m, s, e, t, rho, cppargs)


def pcsaft_Z(x, m, s, e, t, rho, pyargs):
    """
    Calculate the compressibility factor.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : ndarray, shape (n,)
        Segment number for each component.
    s : ndarray, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : ndarray, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    pyargs : dict
        A dictionary containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    Z : float
        Compressibility factor
    """
    check_input(x, 'density', rho, 'temperature', t)
    cppargs = create_struct(pyargs)
    if type(m) == np.float_:
        m = np.asarray([m])
    if type(s) == np.float_:
        s = np.asarray([s])
    if type(e) == np.float_:
        e = np.asarray([e])
    return pcsaft_Z_cpp(x, m, s, e, t, rho, cppargs)


def pcsaft_vaporP(p_guess, x, m, s, e, t, pyargs):
    """
    Wrapper around solver that determines the vapor pressure.

    Parameters
    ----------
    p_guess : float
        Guess for the vapor pressure (Pa)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : ndarray, shape (n,)
        Segment number for each component.
    s : ndarray, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : ndarray, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : float
        Temperature (K)
    pyargs : dict
        A dictionary containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    Pvap : float
        Vapor pressure (Pa)
    """
    check_input(x, 'guess pressure', p_guess, 'temperature', t)
    cppargs = create_struct(pyargs)
    if type(m) == np.float_:
        m = np.asarray([m])
    if type(s) == np.float_:
        s = np.asarray([s])
    if type(e) == np.float_:
        e = np.asarray([e])

    result = minimize(vaporPfit, p_guess, args=(x, m, s, e, t, cppargs), tol=1e-10, method='Nelder-Mead', options={'maxiter': 100})

    p_guess = 1e-2
    while (result.fun > 1e-3) and (p_guess < 1e10): # Sometimes optimization doesn't work if p_guess is not close
        result = minimize(vaporPfit, p_guess, args=(x, m, s, e, t, cppargs), tol=1e-10, method='Nelder-Mead', options={'maxiter': 100})
        p_guess = p_guess * 10

    Pvap = result.x
    return Pvap


def pcsaft_bubbleT(t_guess, xv_guess, x, m, s, e, p, pyargs):
    """
    Calculate the bubble point temperature of a mixture and the vapor composition.

    Parameters
    ----------
    t_guess : float
        Guess for the bubble point temperature (K)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : ndarray, shape (n,)
        Segment number for each component.
    s : ndarray, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : ndarray, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    p : float
        Pressure (Pa)
    pyargs : dict
        A dictionary containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    results : list
        A list containing the following results:
            0 : Bubble point temperature (K)
            1 : Composition of the liquid phase
    """
    check_input(x, 'pressure', p, 'guess temperature', t_guess)
    cppargs = create_struct(pyargs)
    if type(m) == np.float_:
        m = np.asarray([m])
    if type(s) == np.float_:
        s = np.asarray([s])
    if type(e) == np.float_:
        e = np.asarray([e])

    result = minimize(bubbleTfit, t_guess, args=(xv_guess, x, m, s, e, p, cppargs), tol=1e-10, method='Nelder-Mead', options={'maxiter': 100})
    bubT = result.x

#   Determine vapor phase composition at bubble pressure
    if cppargs['z'] == []: # Check that the mixture does not contain electrolytes. For electrolytes, a different equilibrium criterion should be used.
        rho = pcsaft_den_cpp(x, m, s, e, bubT, p, 0, cppargs)
        fugcoef_l = np.asarray(pcsaft_fugcoef_cpp(x, m, s, e, bubT, rho, cppargs))

        itr = 0
        dif = 10000.
        xv = np.copy(xv_guess)
        xv_old = np.zeros_like(xv)
        while (dif>1e-9) and (itr<100):
            xv_old[:] = xv
            rho = pcsaft_den_cpp(xv, m, s, e, bubT, p, 1, cppargs)
            fugcoef_v = np.asarray(pcsaft_fugcoef_cpp(xv, m, s, e, bubT, rho, cppargs))
            xv = fugcoef_l*x/fugcoef_v
            xv = xv/np.sum(xv)
            dif = np.sum(abs(xv - xv_old))
            itr += 1
    else:
        z = np.asarray(cppargs['z'])
        rho = pcsaft_den_cpp(x, m, s, e, bubT, p, 0, cppargs)
        fugcoef_l = np.asarray(pcsaft_fugcoef_cpp(x, m, s, e, bubT, rho, cppargs))

        itr = 0
        dif = 10000.
        xv = np.copy(xv_guess)
        xv_old = np.zeros_like(xv)
        while (dif>1e-9) and (itr<100):
            xv_old[:] = xv
            rho = pcsaft_den_cpp(xv, m, s, e, bubT, p, 1, cppargs)
            fugcoef_v = np.asarray(pcsaft_fugcoef_cpp(xv, m, s, e, bubT, rho, cppargs))

            xv[np.where(z == 0)[0]] = (fugcoef_l*x/fugcoef_v)[np.where(z == 0)[0]] # here it is assumed that the ionic compounds are nonvolatile
            xv = xv/np.sum(xv)
            dif = np.sum(abs(xv - xv_old))
            itr += 1

    results = [bubT, xv]
    return results


def pcsaft_bubbleP(p_guess, xv_guess, x, m, s, e, t, pyargs):
    """
    Calculate the bubble point pressure of a mixture and the vapor composition.

    Parameters
    ----------
    p_guess : float
        Guess for the vapor pressure (Pa)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : ndarray, shape (n,)
        Segment number for each component.
    s : ndarray, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : ndarray, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : float
        Temperature (K)
    pyargs : dict
        A dictionary containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    results : list
        A list containing the following results:
            0 : Bubble point pressure (Pa)
            1 : Composition of the liquid phase
    """
    check_input(x, 'guess pressure', p_guess, 'temperature', t)
    cppargs = create_struct(pyargs)
    if type(m) == np.float_:
        m = np.asarray([m])
    if type(s) == np.float_:
        s = np.asarray([s])
    if type(e) == np.float_:
        e = np.asarray([e])

    result = minimize(bubblePfit, p_guess, args=(xv_guess, x, m, s, e, t, cppargs), tol=1e-10, method='Nelder-Mead', options={'maxiter': 100})

    p_guess = 1e-2
    while (result.fun > 1e-3) and (p_guess < 1e10): # Sometimes optimization doesn't work if p_guess is not close
        result = minimize(bubblePfit, p_guess, args=(xv_guess, x, m, s, e, t, cppargs), tol=1e-10, method='Nelder-Mead', options={'maxiter': 100})
        p_guess = p_guess * 10

    bubP = result.x

#     Determine vapor phase composition at bubble pressure
    if cppargs['z'] == []: # Check that the mixture does not contain electrolytes. For electrolytes, a different equilibrium criterion should be used.
        rho = pcsaft_den_cpp(x, m, s, e, t, p_guess, 0, cppargs)
        fugcoef_l = np.asarray(pcsaft_fugcoef_cpp(x, m, s, e, t, rho, cppargs))

        itr = 0
        dif = 10000.
        xv = np.copy(xv_guess)
        xv_old = np.zeros_like(xv)
        while (dif>1e-9) and (itr<100):
            xv_old[:] = xv
            rho = pcsaft_den_cpp(xv, m, s, e, t, p_guess, 1, cppargs)
            fugcoef_v = np.asarray(pcsaft_fugcoef_cpp(xv, m, s, e, t, rho, cppargs))
            xv = fugcoef_l*x/fugcoef_v
            xv = xv/np.sum(xv)
            dif = np.sum(abs(xv - xv_old))
            itr += 1
    else:
        z = np.asarray(cppargs['z'])
        rho = pcsaft_den_cpp(x, m, s, e, t, p_guess, 0, cppargs)
        fugcoef_l = np.asarray(pcsaft_fugcoef_cpp(x, m, s, e, t, rho, cppargs))

        itr = 0
        dif = 10000.
        xv = np.copy(xv_guess)
        xv_old = np.zeros_like(xv)
        while (dif>1e-9) and (itr<100):
            xv_old[:] = xv
            rho = pcsaft_den_cpp(xv, m, s, e, t, p_guess, 1, cppargs)
            fugcoef_v = np.asarray(pcsaft_fugcoef_cpp(xv, m, s, e, t, rho, cppargs))

            xv[np.where(z == 0)[0]] = (fugcoef_l*x/fugcoef_v)[np.where(z == 0)[0]] # here it is assumed that the ionic compounds are nonvolatile
            xv = xv/np.sum(xv)
            dif = np.sum(abs(xv - xv_old))
            itr += 1

    results = [bubP, xv]
    return results


def pcsaft_Hvap(p_guess, x, m, s, e, t, pyargs):
    """
    Calculate the enthalpy of vaporization.

    Parameters
    ----------
    p_guess : float
        Guess for the vapor pressure (Pa)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : ndarray, shape (n,)
        Segment number for each component.
    s : ndarray, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : ndarray, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : float
        Temperature (K)
    pyargs : dict
        A dictionary containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    output : list
        A list containing the following results:
            0 : enthalpy of vaporization (J/mol), float
            1 : vapor pressure (Pa), float
    """
    check_input(x, 'guess pressure', p_guess, 'temperature', t)
    cppargs = create_struct(pyargs)
    if type(m) == np.float_:
        m = np.asarray([m])
    if type(s) == np.float_:
        s = np.asarray([s])
    if type(e) == np.float_:
        e = np.asarray([e])

    Pvap = minimize(vaporPfit, p_guess, args=(x, m, s, e, t, cppargs), tol=1e-10, method='Nelder-Mead', options={'maxiter': 100}).x

    rho = pcsaft_den_cpp(x, m, s, e, t, Pvap, 0, cppargs)
    hres_l = pcsaft_hres_cpp(x, m, s, e, t, rho, cppargs)
    rho = pcsaft_den_cpp(x, m, s, e, t, Pvap, 1, cppargs)
    hres_v = pcsaft_hres_cpp(x, m, s, e, t, rho, cppargs)
    Hvap = hres_v - hres_l

    output = [Hvap, Pvap]
    return output


def pcsaft_osmoticC(x, m, s, e, t, rho, pyargs):
    """
    Calculate the osmotic coefficient.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : ndarray, shape (n,)
        Segment number for each component.
    s : ndarray, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : ndarray, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    pyargs : dict
        A dictionary containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    osmC : float
        Molal osmotic coefficient
    """
    check_input(x, 'density', rho, 'temperature', t)
    cppargs = create_struct(pyargs)
    if type(m) == np.float_:
        m = np.asarray([m])
    if type(s) == np.float_:
        s = np.asarray([s])
    if type(e) == np.float_:
        e = np.asarray([e])

    indx_water = np.where(e == 353.9449)[0] # to find index for water
    molality = x/(x[indx_water]*18.0153/1000.)
    molality[indx_water] = 0
    x0 = np.zeros_like(x)
    x0[indx_water] = 1.

    fugcoef = np.asarray(pcsaft_fugcoef_cpp(x, m, s, e, t, rho, cppargs))
    p = pcsaft_p_cpp(x, m, s, e, t, rho, cppargs)
    if rho < 900:
        ph = 1
    else:
        ph = 0
    rho0 = pcsaft_den_cpp(x0, m, s, e, t, p, ph, cppargs)
    fugcoef0 = np.asarray(pcsaft_fugcoef_cpp(x0, m, s, e, t, rho0, cppargs))
    gamma = fugcoef[indx_water]/fugcoef0[indx_water]

    osmC = -1000*np.log(x[indx_water]*gamma)/18.0153/np.sum(molality)
    return osmC

def pcsaft_cp(x, m, s, e, t, rho, params, pyargs):
    """
    Calculate the specific molar isobaric heat capacity.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : ndarray, shape (n,)
        Segment number for each component.
    s : ndarray, shape (n,)
        Segment diameter for each compopynent. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : ndarray, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    params : ndarray, shape (5,)
        Constants for the Aly-Lee equation. Can be substituted with parameters for
        another equation if the ideal gas heat capacity is given using a different
        equation.
    pyargs : dict
        A dictionary containing additional arguments that can be passed for
        use in PC-SAFT:

            k_ij : ndarray, shape (n,n)
                Binary interaction parameters between components in the mixture.
                (dimensions: ncomp x ncomp)
            e_assoc : ndarray, shape (n,)
                Association energy of the associating components. For non associating
                compounds this is set to 0. Units of K.
            vol_a : ndarray, shape (n,)
                Effective association volume of the associating components. For non
                associating compounds this is set to 0.
            dipm : ndarray, shape (n,)
                Dipole moment of the polar components. For components where the dipole
                term is not used this is set to 0. Units of Debye.
            dip_num : ndarray, shape (n,)
                The effective number of dipole functional groups on each component
                molecule. Some implementations use this as an adjustable parameter
                that is fit to data.
            z : ndarray, shape (n,)
                Charge number of the ions
            dielc : float
                Dielectric constant of the medium to be used for electrolyte
                calculations.

    Returns
    -------
    cp : float
        Specific molar isobaric heat capacity (J mol^-1 K^-1)
    """
    check_input(x, 'density', rho, 'temperature', t)

    if rho > 900:
        ph = 0
    else:
        ph = 1

    cppargs = create_struct(pyargs)
    if type(m) == np.float_:
        m = np.asarray([m])
    if type(s) == np.float_:
        s = np.asarray([s])
    if type(e) == np.float_:
        e = np.asarray([e])

    cp_ideal = aly_lee(t, params)
    p = pcsaft_p_cpp(x, m, s, e, t, rho, cppargs)
    rho0 = pcsaft_den_cpp(x, m, s, e, t-0.001, p, ph, cppargs)
    hres0 = pcsaft_hres_cpp(x, m, s, e, t-0.001, rho0, cppargs)
    rho1 = pcsaft_den_cpp(x, m, s, e, t+0.001, p, ph, cppargs)
    hres1 = pcsaft_hres_cpp(x, m, s, e, t+0.001, rho1, cppargs)
    dhdt = (hres1-hres0)/0.002 # a numerical derivative is used for now until analytical derivatives are ready
    return cp_ideal + dhdt


def pcsaft_PTz(p_guess, x_guess, beta_guess, mol, vol, x_total, m, s, e, t, pyargs):
    """
    Calculate the pressure and compositions of each phase when given the overall
    composition and the total volume and number of moles. This allows PTz data
    to be used in fitting PC-SAFT parameters.

    Parameters
    ----------
    p_guess : float
        Guess for the pressure of the system (Pa)
    x_guess : ndarray, shape (n,)
        Guess for the liquid phase composition
    beta_guess : float
        Guess for the mole fraction of the system in the vapor phase
    mol : float
        Total number of moles in the system (mol)
    vol : float
        Total volume of the system (m^{3})
    x_total : ndarray, shape (n,)
        Overall mole fraction of each component in the system as a whole. It
        has a length of n, where n is the number of components in the system.
    m : ndarray, shape (n,)
        Segment number for each component.
    s : ndarray, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : ndarray, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : float
        Temperature (K)
    pyargs : dict
        A dictionary containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    output : list
        A list containing the following results:
            0 : pressure in the system (Pa), float
            1 : composition of the liquid phase, ndarray, shape (n,)
            2 : composition of the vapor phase, ndarray, shape (n,)
            3 : mole fraction of the mixture vaporized
    """
    check_input(x_total, 'guess pressure', p_guess, 'temperature', t)
    cppargs = create_struct(pyargs)
    if type(m) == np.float_:
        m = np.asarray([m])
    if type(s) == np.float_:
        s = np.asarray([s])
    if type(e) == np.float_:
        e = np.asarray([e])

    result = minimize(PTzfit, p_guess, args=(x_guess, beta_guess, mol, vol, x_total, m, s, e, t, cppargs), tol=1e-10, method='Nelder-Mead', options={'maxiter': 100})
    p = result.x

    if cppargs['z'] == []: # Check that the mixture does not contain electrolytes. For electrolytes, a different equilibrium criterion should be used.
        itr = 0
        dif = 10000.
        xl = np.copy(x_guess)
        beta = beta_guess
        xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol
        while (dif>1e-9) and (itr<100):
            beta_old = beta
            rhol = pcsaft_den_cpp(xl, m, s, e, t, p, 0, cppargs)
            fugcoef_l = np.asarray(pcsaft_fugcoef_cpp(xl, m, s, e, t, rhol, cppargs))
            rhov = pcsaft_den_cpp(xv, m, s, e, t, p, 1, cppargs)
            fugcoef_v = np.asarray(pcsaft_fugcoef_cpp(xv, m, s, e, t, rhov, cppargs))
            if beta > 0.5:
                xl = fugcoef_v*xv/fugcoef_l
                xl = xl/np.sum(xl)
                xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol # if beta is close to zero then this equation behaves poorly, and that is why we use this if statement to switch the equation around
            else:
                xv = fugcoef_l*xl/fugcoef_v
                xv = xv/np.sum(xv)
                xl = (mol*x_total - (beta)*mol*xv)/(1-beta)/mol
            beta = (vol/mol*rhov*rhol-rhov)/(rhol-rhov)
            dif = np.sum(abs(beta - beta_old))
            itr += 1
    else:
        z = np.asarray(cppargs['z'])
        # internal iteration loop to solve for compositions
        itr = 0
        dif = 10000.
        xl = np.copy(x_guess)
        beta = beta_guess
        xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol
        xv[np.where(z != 0)[0]] = 0.
        xv = xv/np.sum(xv)
        while (dif>1e-9) and (itr<100):
#            xl = chem_equil_cpp(xl, m, s, e, t, p, cppargs)
            x_total = xl + xv
            beta_old = beta
            rhol = pcsaft_den_cpp(xl, m, s, e, t, p, 0, cppargs)
            fugcoef_l = np.asarray(pcsaft_fugcoef_cpp(xl, m, s, e, t, rhol, cppargs))
            rhov = pcsaft_den_cpp(xv, m, s, e, t, p, 1, cppargs)
            fugcoef_v = np.asarray(pcsaft_fugcoef_cpp(xv, m, s, e, t, rhov, cppargs))
            if beta > 0.5:
                xl = fugcoef_v*xv/fugcoef_l
                xl = xl/np.sum(xl)*(((1-beta) - np.sum(x_total[np.where(z != 0)[0]]))/(1-beta)) # ensures that mole fractions add up to 1
                xl[np.where(z != 0)[0]] = x_total[np.where(z != 0)[0]]/(1-beta)
                xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol
                xv[np.where(xv < 0)[0]] = 0.
            else:
                xv = fugcoef_l*xl/fugcoef_v
                xv[np.where(z != 0)[0]] = 0. # here it is assumed that the ionic compounds are nonvolatile
                xv = xv/np.sum(xv)
                xl = (mol*x_total - (beta)*mol*xv)/(1-beta)/mol
            beta = (vol/mol*rhov*rhol-rhov)/(rhol-rhov)
            dif = np.sum(abs(beta - beta_old))
            itr += 1

    output = [p, xl, xv, beta]
    return output

def pcsaft_den(x, m, s, e, t, p, pyargs, phase='liq'):
    """
    Wrapper for C++ pcsaft_den_cpp function because a C++ struct is needed for
    that function.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : ndarray, shape (n,)
        Segment number for each component.
    s : ndarray, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : ndarray, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : float
        Temperature (K)
    p : float
        Pressure (Pa)
    phase : string
        The phase for which the calculation is performed. Options: "liq" (liquid),
        "vap" (vapor).
    pyargs : dict
        A dictionary containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    rho : float
        Molar density (mol m^{-3})
    """
    check_input(x, 'pressure', p, 'temperature', t)
    cppargs = create_struct(pyargs)
    if phase == 'liq':
        phase_num = 0
    else:
        phase_num = 1

    if type(m) == np.float_:
        m = np.asarray([m])
    if type(s) == np.float_:
        s = np.asarray([s])
    if type(e) == np.float_:
        e = np.asarray([e])
    return pcsaft_den_cpp(x, m, s, e, t, p, phase_num, cppargs)


def pcsaft_hres(x, m, s, e, t, rho, pyargs):
    """
    Calculate the residual enthalpy for one phase of the system.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : ndarray, shape (n,)
        Segment number for each component.
    s : ndarray, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : ndarray, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    pyargs : dict
        A dictionary containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    hres : float
        Residual enthalpy (J mol^{-1})
    """
    check_input(x, 'density', rho, 'temperature', t)
    cppargs = create_struct(pyargs)
    if type(m) == np.float_:
        m = np.asarray([m])
    if type(s) == np.float_:
        s = np.asarray([s])
    if type(e) == np.float_:
        e = np.asarray([e])
    return pcsaft_hres_cpp(x, m, s, e, t, rho, cppargs)

def pcsaft_sres(x, m, s, e, t, rho, pyargs):
    """
    Calculate the residual entropy (as a function of temperature and pressure) for one phase of the system.
    Note that the residual entropy in terms of the variables temperature and density is different.
    See Equation A.47 in Gross and Sadowski 2001.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : ndarray, shape (n,)
        Segment number for each component.
    s : ndarray, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : ndarray, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    pyargs : dict
        A dictionary containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    sres : float
        Residual entropy (J mol^{-1} K^{-1})
    """
    check_input(x, 'density', rho, 'temperature', t)
    cppargs = create_struct(pyargs)
    if type(m) == np.float_:
        m = np.asarray([m])
    if type(s) == np.float_:
        s = np.asarray([s])
    if type(e) == np.float_:
        e = np.asarray([e])
    return pcsaft_sres_cpp(x, m, s, e, t, rho, cppargs)

def pcsaft_gres(x, m, s, e, t, rho, pyargs):
    """
    Calculate the residual Gibbs energy for one phase of the system.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : ndarray, shape (n,)
        Segment number for each component.
    s : ndarray, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : ndarray, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    pyargs : dict
        A dictionary containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    gres : float
        Residual Gibbs energy (J mol^{-1})
    """
    check_input(x, 'density', rho, 'temperature', t)
    cppargs = create_struct(pyargs)
    if type(m) == np.float_:
        m = np.asarray([m])
    if type(s) == np.float_:
        s = np.asarray([s])
    if type(e) == np.float_:
        e = np.asarray([e])
    return pcsaft_gres_cpp(x, m, s, e, t, rho, cppargs)


def pcsaft_ares(x, m, s, e, t, rho, pyargs):
    """
    Calculates the residual Helmholtz energy.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : ndarray, shape (n,)
        Segment number for each component.
    s : ndarray, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : ndarray, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    pyargs : dict
        A dictionary containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    ares : float
        Residual Helmholtz energy (J mol^{-1})
    """
    check_input(x, 'density', rho, 'temperature', t)
    cppargs = create_struct(pyargs)
    if type(m) == np.float_:
        m = np.asarray([m])
    if type(s) == np.float_:
        s = np.asarray([s])
    if type(e) == np.float_:
        e = np.asarray([e])
    return pcsaft_ares_cpp(x, m, s, e, t, rho, cppargs)


def pcsaft_dadt(x, m, s, e, t, rho, pyargs):
    """
    Calculates the temperature derivative of the residual Helmholtz energy.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : ndarray, shape (n,)
        Segment number for each component.
    s : ndarray, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : ndarray, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    pyargs : dict
        A dictionary containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    dadt : float
        Temperature derivative of the residual Helmholtz energy (J mol^{-1})
    """
    check_input(x, 'density', rho, 'temperature', t)
    cppargs = create_struct(pyargs)
    if type(m) == np.float_:
        m = np.asarray([m])
    if type(s) == np.float_:
        s = np.asarray([s])
    if type(e) == np.float_:
        e = np.asarray([e])
    return pcsaft_dadt_cpp(x, m, s, e, t, rho, cppargs)


def dXAdt_find(ncA, ncomp, delta_ij, den, XA, ddelta_dt, x, n_sites):
    """Solve for the derivative of XA with respect to temperature."""
    B = np.zeros((n_sites*ncA,), dtype='float_')
    A = np.zeros((n_sites*ncA,n_sites*ncA), dtype='float_')

    i_out = -1 # index of outer iteration loop (follows row of matrices)
    for i in range(ncA):
        for ai in range(n_sites):
            i_out += 1
            i_in = -1 # index for summation loops
            summ = 0
            for j in range(ncA):
                for bj in range(n_sites):
                    i_in += 1
                    B[i_out] -= x[j]*XA[i_in]*ddelta_dt[i,j]*((i_in+i_out)%2)
                    A[i_out,i_in] = x[j]*delta_ij[i,j]*((i_in+i_out)%2)
                    summ += x[j]*XA[i_in]*delta_ij[i,j]*((i_in+i_out)%2)
            A[i_out,i_out] = A[i_out,i_out] + (1+den*summ)**2/den

    dXAdt_dd = np.linalg.solve(A, B) #Solves linear system of equations
    return dXAdt_dd


def bubblePfit(p_guess, xv_guess, x, m, s, e, t, cppargs):
    """Minimize this function to calculate the bubble point pressure."""
    error = bubblePfit_cpp(p_guess, xv_guess, x, m, s, e, t, cppargs)
    return error

def bubbleTfit(t_guess, xv_guess, x, m, s, e, p, cppargs):
    """Minimize this function to calculate the bubble point temperature."""
    error = bubbleTfit_cpp(t_guess, xv_guess, x, m, s, e, p, cppargs)
    return error

def vaporPfit(p_guess, x, m, s, e, t, cppargs):
    """Minimize this function to calculate the vapor pressure."""
    if p_guess <= 0:
        error = 10000000000.
    else:
        rho = pcsaft_den_cpp(x, m, s, e, t, p_guess, 0, cppargs)
        fugcoef_l = np.asarray(pcsaft_fugcoef_cpp(x, m, s, e, t, rho, cppargs))
        rho = pcsaft_den_cpp(x, m, s, e, t, p_guess, 1, cppargs)
        fugcoef_v = np.asarray(pcsaft_fugcoef_cpp(x, m, s, e, t, rho, cppargs))
        error = 100000*np.sum((fugcoef_l-fugcoef_v)**2)
        if not np.isfinite(error):
            error = 1e20
    return error

def PTzfit(p_guess, x_guess, beta_guess, mol, vol, x_total, m, s, e, t, cppargs):
    """Minimize this function to solve for the pressure to compare with PTz data."""
    error = PTzfit_cpp(p_guess, x_guess, beta_guess, mol, vol, x_total, m, s, e, t, cppargs)

    if cppargs['z'] == []: # Check that the mixture does not contain electrolytes. For electrolytes, a different equilibrium criterion should be used.
        # internal iteration loop to solve for compositions
        itr = 0
        dif = 10000.
        xl = np.copy(x_guess)
        beta = beta_guess
        xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol
        while (dif>1e-9) and (itr<100):
            beta_old = beta
            rhol = pcsaft_den_cpp(xl, m, s, e, t, p_guess, 0, cppargs)
            fugcoef_l = np.asarray(pcsaft_fugcoef_cpp(xl, m, s, e, t, rhol, cppargs))
            rhov = pcsaft_den_cpp(xv, m, s, e, t, p_guess, 1, cppargs)
            fugcoef_v = np.asarray(pcsaft_fugcoef_cpp(xv, m, s, e, t, rhov, cppargs))
            xl = fugcoef_v*xv/fugcoef_l
            xl = xl/np.sum(xl)
            xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol
            beta = (vol/mol-rhol)/(rhov-rhol)
            dif = np.sum(abs(beta - beta_old))
            itr += 1
        error = np.sum((vol - rhov*beta*mol - rhol*(1-beta)*mol)**2)
        error += np.sum((xl*fugcoef_l - xv*fugcoef_v)**2)
        error += np.sum((mol*x_total - beta*mol*xv - (1-beta)*mol*xl)**2)
    else:
        z = np.asarray(cppargs['z'])
        # internal iteration loop to solve for compositions
        itr = 0
        dif = 10000.
        xl = np.copy(x_guess)
        beta = beta_guess
        xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol
        xv[np.where(z == 0)[0]] = 0.
        xv = xv/np.sum(xv)
        while (dif>1e-9) and (itr<100):
            beta_old = beta
            rhol = pcsaft_den_cpp(xl, m, s, e, t, p_guess, 0, cppargs)
            fugcoef_l = np.asarray(pcsaft_fugcoef_cpp(xl, m, s, e, t, rhol, cppargs))
            rhov = pcsaft_den_cpp(xv, m, s, e, t, p_guess, 1, cppargs)
            fugcoef_v = np.asarray(pcsaft_fugcoef_cpp(xv, m, s, e, t, rhov, cppargs))
            xl = fugcoef_v*xv/fugcoef_l
            xl = xl/np.sum(xl)
            xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol
            xv[np.where(z == 0)[0]] = 0. # here it is assumed that the ionic compounds are nonvolatile
            xv = xv/np.sum(xv)
            beta = (vol/mol-rhol)/(rhov-rhol)
            dif = np.sum(abs(beta - beta_old))
            itr += 1
        error = (vol - rhov*beta*mol - rhol*(1-beta)*mol)**2
        error += np.sum((xl*fugcoef_l - xv*fugcoef_v)[np.where(z == 0)[0]]**2)
        error += np.sum((mol*x_total - beta*mol*xv - (1-beta)*mol*xl)**2)

    if np.isnan(error):
        error = 100000000.
    return error

def aly_lee(t, c):
    """
    Calculate the ideal gas isobaric heat capacity using the Aly-Lee equation.

    Parameters
    ----------
    t : float
        Temperature (K)
    c : ndarray, shape (5,)
        Constants for the Aly-Lee equation

    Returns
    -------
    cp_ideal : float
        Ideal gas isobaric heat capacity (J mol^-1 K^-1)

    Reference:
    F. A. Aly and L. L. Lee, “Self-consistent equations for calculating the ideal
    gas heat capacity, enthalpy, and entropy,” Fluid Phase Equilibria, vol. 6,
    no. 3–4, pp. 169–179, 1981.

    """
    cp_ideal = (c[0] + c[1]*(c[2]/t/np.sinh(c[2]/t))**2 + c[3]*(c[4]/t/np.cosh(c[4]/t))**2)/1000.
    return cp_ideal

def dielc_water(t):
    """
    Return the dielectric constant of water at the given temperature.

    t : float
        Temperature (K)

    This equation was fit to values given in the reference. For temperatures
    from 263.15 to 368.15 K values at 1 bar were used. For temperatures from
    368.15 to 443.15 K values at 10 bar were used.

    Reference:
    D. G. Archer and P. Wang, “The Dielectric Constant of Water and Debye‐Hückel
    Limiting Law Slopes,” J. Phys. Chem. Ref. Data, vol. 19, no. 2, pp. 371–411,
    Mar. 1990.
    """
    if t <= 368.15:
        dielc = 7.6555618295E-04*t**2 - 8.1783881423E-01*t + 2.5419616803E+02
    else:
        dielc = 0.0005003272124*t**2 - 0.6285556029*t + 220.4467027
    return dielc


def np_to_vector(np_array):
    """Take a numpy array and return a C++ vector."""
    cdef vector[double] cpp_vector

    try:
        np_array = np_array.flatten()
        N = np_array.shape[0]
        for i in range(N):
            cpp_vector.push_back(np_array[i])
    except TypeError:
        cpp_vector.push_back(np_array)

    return cpp_vector

def create_struct(pyargs):
    """Convert additional arguments to a C++ struct."""
    cdef add_args cppargs

    if 'k_ij' in pyargs:
        cppargs.k_ij = np_to_vector(pyargs['k_ij'])
    if ('e_assoc' in pyargs) and np.any(pyargs['e_assoc']):
        cppargs.e_assoc = np_to_vector(pyargs['e_assoc'])
    if ('vol_a' in pyargs) and np.any(pyargs['vol_a']):
        cppargs.vol_a = np_to_vector(pyargs['vol_a'])
    if ('dipm' in pyargs) and np.any(pyargs['dip_num']) and np.any(pyargs['dipm']):
        cppargs.dipm = np_to_vector(pyargs['dipm'])
    if ('dip_num' in pyargs) and np.any(pyargs['dip_num']):
        cppargs.dip_num = np_to_vector(pyargs['dip_num'])
    if 'z' in pyargs:
        cppargs.z = np_to_vector(pyargs['z'])
    if 'dielc' in pyargs:
        cppargs.dielc = pyargs['dielc']
    if 'k_hb' in pyargs:
        cppargs.k_hb = np_to_vector(pyargs['k_hb'])
    if 'l_ij' in pyargs:
        cppargs.l_ij = np_to_vector(pyargs['l_ij'])

    return cppargs
