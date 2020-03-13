# -*- coding: utf-8 -*-
# setuptools: language=c++
"""
PC-SAFT with electrolyte term

These functions implement the PC-SAFT equation of state. In addition to the
hard chain and dispersion terms, these functions also include dipole,
association and ion terms for use with these types of compounds.

@author: Zach Baird

Functions
---------
- flashTQ : calculate the equilibrium pressure when there are two phases
- flashPQ : calculate the equilibrium temperature when there are two phases
- pcsaft_Hvap : calculate the enthalpy of vaporization
- pcsaft_osmoticC : calculate the osmotic coefficient for the mixture
- pcsaft_cp : calculate the heat capacity
- pcsaft_den : calculate the molar density
- pcsaft_p : calculate the pressure
- pcsaft_hres : calculate the residual enthalpy
- pcsaft_sres : calculate the residual entropy
- pcsaft_gres : calculate the residual Gibbs free energy
- pcsaft_fugcoef : calculate the fugacity coefficients
- pcsaft_Z : calculate the compressibility factor
- pcsaft_ares : calculate the residual Helmholtz energy
- pcsaft_dadt : calculate the temperature derivative of the residual Helmholtz energy
- XA_find : used internally to solve for XA
- dXA_find : used internally to solve for the derivative of XA wrt density
- dXAdt_find : used internally to solve for the derivative of XA wrt temperature
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
from libcpp.vector cimport vector
from copy import deepcopy
cimport pcsaft_electrolyte

class InputError(Exception):
    """Exception raised for errors in the input.
    """
    def __init__(self, message):
        self.message = message

class SolutionError(Exception):
    """Exception raised when a solver does not return a value.
    """
    def __init__(self, message):
        self.message = message

def check_input(x, vars):
    ''' Perform a few basic checks to make sure the input is reasonable. '''
    if abs(np.sum(x) - 1) > 1e-7:
        raise InputError('The mole fractions do not sum to 1. x = {}'.format(x))
    if 'temperature' in vars:
        if vars['temperature'] <= 0:
            raise InputError('The {} must be a positive number. {} = {}'.format('temperature', 'temperature', vars['temperature']))
    if 'density' in vars:
        if vars['density'] <= 0:
            raise InputError('The {} must be a positive number. {} = {}'.format('density', 'density', vars['density']))
    if 'pressure' in vars:
        if vars['pressure'] <= 0:
            raise InputError('The {} must be a positive number. {} = {}'.format('pressure', 'pressure', vars['pressure']))
    if 'Q' in vars:
        if (vars['Q'] < 0) or (vars['Q'] > 1):
            raise InputError('{} must be <= 1 and >= 0. {} = {}'.format('Q', 'Q', vars['Q']))

def ensure_numpy_input(pyargs):
    if type(pyargs['x']) == np.float_:
        pyargs['x'] = np.asarray([pyargs['x']])
    if type(pyargs['m']) == np.float_:
        pyargs['m'] = np.asarray([pyargs['m']])
    if type(pyargs['s']) == np.float_:
        pyargs['s'] = np.asarray([pyargs['s']])
    if type(pyargs['e']) == np.float_:
        pyargs['e'] = np.asarray([pyargs['e']])
    return pyargs

def pcsaft_p(t, rho, pyargs):
    """
    Calculate pressure.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
    pyargs = ensure_numpy_input(pyargs)
    check_input(pyargs['x'], {'density':rho, 'temperature':t})
    cppargs = create_struct(pyargs)
    return pcsaft_p_cpp(t, rho, cppargs)


def pcsaft_fugcoef(t, rho, pyargs):
    """
    Calculate the fugacity coefficients for one phase of the system.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
    pyargs = ensure_numpy_input(pyargs)
    check_input(pyargs['x'], {'density':rho, 'temperature':t})
    cppargs = create_struct(pyargs)
    return pcsaft_fugcoef_cpp(t, rho, cppargs)


def pcsaft_Z(t, rho, pyargs):
    """
    Calculate the compressibility factor.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
    pyargs = ensure_numpy_input(pyargs)
    check_input(pyargs['x'], {'density':rho, 'temperature':t})
    cppargs = create_struct(pyargs)
    return pcsaft_Z_cpp(t, rho, cppargs)


def flashPQ(p, q, pyargs, t_guess=None):
    """
    Calculate the temperature of the system where vapor and liquid phases are in equilibrium.

    Parameters
    ----------
    p : float
        Pressure (Pa)
    q : float
        Mole fraction of the fluid in the vapor phase
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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

    t_guess : float
        Initial guess for the temperature (K) (optional)

    Returns
    -------
    t : float
        Temperature (K)
    """
    pyargs = ensure_numpy_input(pyargs)
    check_input(pyargs['x'], {'pressure':p, 'Q':q})
    cppargs = create_struct(pyargs)
    try:
        if t_guess is not None:
            t = flashPQ_cpp(p, q, cppargs, t_guess)
        else:
            t = flashPQ_cpp(p, q, cppargs)
    except:
        raise SolutionError('A solution was not found for flashPQ. P={}'.format(p))

    return t


def flashTQ(t, q, pyargs, p_guess=None):
    """
    Calculate the pressure of the system where vapor and liquid phases are in equilibrium.

    Parameters
    ----------
    t : float
        Temperature (K)
    q : float
        Mole fraction of the fluid in the vapor phase
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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

    p_guess : float
        Initial guess for the pressure (Pa) (optional)

    Returns
    -------
    p : float
        Pressure (Pa)
    """
    pyargs = ensure_numpy_input(pyargs)
    check_input(pyargs['x'], {'temperature':t, 'Q':q})
    cppargs = create_struct(pyargs)
    try:
        if p_guess is not None:
            p = flashTQ_cpp(t, q, cppargs, p_guess)
        else:
            p = flashTQ_cpp(t, q, cppargs)
    except:
        raise SolutionError('A solution was not found for flashTQ. T={}'.format(t))

    return p

def pcsaft_Hvap(t, pyargs, p_guess=None):
    """
    Calculate the enthalpy of vaporization.

    Parameters
    ----------
    t : float
        Temperature (K)
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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

    p_guess : float
        Guess for the vapor pressure (Pa) (optional)

    Returns
    -------
    output : list
        A list containing the following results:
            0 : enthalpy of vaporization (J/mol), float
            1 : vapor pressure (Pa), float
    """
    pyargs = ensure_numpy_input(pyargs)
    check_input(pyargs['x'], {'temperature': t})
    cppargs = create_struct(pyargs)

    q = 0
    try:
        if p_guess is not None:
            Pvap = flashTQ_cpp(t, q, cppargs, p_guess)
        else:
            Pvap = flashTQ_cpp(t, q, cppargs)
    except:
        raise SolutionError('A solution was not found for flashTQ. T={}'.format(t))

    rho = pcsaft_den_cpp(t, Pvap, 0, cppargs)
    hres_l = pcsaft_hres_cpp(t, rho, cppargs)
    rho = pcsaft_den_cpp(t, Pvap, 1, cppargs)
    hres_v = pcsaft_hres_cpp(t, rho, cppargs)
    Hvap = hres_v - hres_l

    output = [Hvap, Pvap]
    return output


def pcsaft_osmoticC(t, rho, pyargs):
    """
    Calculate the osmotic coefficient.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
    pyargs = ensure_numpy_input(pyargs)
    check_input(pyargs['x'], {'density':rho, 'temperature':t})
    cppargs = create_struct(pyargs)

    indx_water = np.where(pyargs['e'] == 353.9449)[0] # to find index for water
    molality = pyargs['x']/(pyargs['x'][indx_water]*18.0153/1000.)
    molality[indx_water] = 0
    x0 = np.zeros_like(pyargs['x'])
    x0[indx_water] = 1.
    pyargs0 = deepcopy(pyargs)
    pyargs0['x'] = x0
    cppargs0 = create_struct(pyargs0)

    fugcoef = np.asarray(pcsaft_fugcoef_cpp(t, rho, cppargs))
    p = pcsaft_p_cpp(t, rho, cppargs)
    if rho < 900:
        ph = 1
    else:
        ph = 0
    rho0 = pcsaft_den_cpp(t, p, ph, cppargs0)
    fugcoef0 = np.asarray(pcsaft_fugcoef_cpp(t, rho0, cppargs0))
    gamma = fugcoef[indx_water]/fugcoef0[indx_water]

    osmC = -1000*np.log(pyargs['x'][indx_water]*gamma)/18.0153/np.sum(molality)
    return osmC

def pcsaft_cp(t, rho, params, pyargs):
    """
    Calculate the specific molar isobaric heat capacity.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    params : ndarray, shape (5,)
        Constants for the Aly-Lee equation. Can be substituted with parameters for
        another equation if the ideal gas heat capacity is given using a different
        equation.
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
    pyargs = ensure_numpy_input(pyargs)
    check_input(pyargs['x'], {'density':rho, 'temperature':t})

    if rho > 900:
        ph = 0
    else:
        ph = 1

    cppargs = create_struct(pyargs)
    # if type(m) == np.float_:
    #     m = np.asarray([m])
    # if type(s) == np.float_:
    #     s = np.asarray([s])
    # if type(e) == np.float_:
    #     e = np.asarray([e])

    cp_ideal = aly_lee(t, params)
    p = pcsaft_p_cpp(t, rho, cppargs)
    rho0 = pcsaft_den_cpp(t-0.001, p, ph, cppargs)
    hres0 = pcsaft_hres_cpp(t-0.001, rho0, cppargs)
    rho1 = pcsaft_den_cpp(t+0.001, p, ph, cppargs)
    hres1 = pcsaft_hres_cpp(t+0.001, rho1, cppargs)
    dhdt = (hres1-hres0)/0.002 # a numerical derivative is used for now until analytical derivatives are ready
    return cp_ideal + dhdt


# def pcsaft_PTz(p_guess, x_guess, beta_guess, mol, vol, t, pyargs):
#     """
#     Calculate the pressure and compositions of each phase when given the overall
#     composition and the total volume and number of moles. This allows PTz data
#     to be used in fitting PC-SAFT parameters.
#
#     Parameters
#     ----------
#     p_guess : float
#         Guess for the pressure of the system (Pa)
#     x_guess : ndarray, shape (n,)
#         Guess for the liquid phase composition
#     beta_guess : float
#         Guess for the mole fraction of the system in the vapor phase
#     mol : float
#         Total number of moles in the system (mol)
#     vol : float
#         Total volume of the system (m^{3})
#     t : float
#         Temperature (K)
#     pyargs : dict
#         A dictionary containing PC-SAFT parameters that can be passed for
#         use in PC-SAFT:
#
#         x : ndarray, shape (n,)
#             Overall mole fraction of each component in the system as a whole. It
#             has a length of n, where n is the number of components in the system.
#         m : ndarray, shape (n,)
#             Segment number for each component.
#         s : ndarray, shape (n,)
#             Segment diameter for each compopynent. For ions this is the diameter of
#             the hydrated ion. Units of Angstrom.
#         e : ndarray, shape (n,)
#             Dispersion energy of each component. For ions this is the dispersion
#             energy of the hydrated ion. Units of K.
#         k_ij : ndarray, shape (n,n)
#             Binary interaction parameters between components in the mixture.
#             (dimensions: ncomp x ncomp)
#         e_assoc : ndarray, shape (n,)
#             Association energy of the associating components. For non associating
#             compounds this is set to 0. Units of K.
#         vol_a : ndarray, shape (n,)
#             Effective association volume of the associating components. For non
#             associating compounds this is set to 0.
#         dipm : ndarray, shape (n,)
#             Dipole moment of the polar components. For components where the dipole
#             term is not used this is set to 0. Units of Debye.
#         dip_num : ndarray, shape (n,)
#             The effective number of dipole functional groups on each component
#             molecule. Some implementations use this as an adjustable parameter
#             that is fit to data.
#         z : ndarray, shape (n,)
#             Charge number of the ions
#         dielc : float
#             Dielectric constant of the medium to be used for electrolyte
#             calculations.
#
#     Returns
#     -------
#     output : list
#         A list containing the following results:
#             0 : pressure in the system (Pa), float
#             1 : composition of the liquid phase, ndarray, shape (n,)
#             2 : composition of the vapor phase, ndarray, shape (n,)
#             3 : mole fraction of the mixture vaporized
#     """
#     check_input(x_total, 'guess pressure', p_guess, 'temperature', t)
#     cppargs = create_struct(pyargs)
#     if type(m) == np.float_:
#         m = np.asarray([m])
#     if type(s) == np.float_:
#         s = np.asarray([s])
#     if type(e) == np.float_:
#         e = np.asarray([e])
#
#     result = minimize(PTzfit, p_guess, args=(x_guess, beta_guess, mol, vol, x_total, m, s, e, t, cppargs), tol=1e-10, method='Nelder-Mead', options={'maxiter': 100})
#     p = result.x
#
#     if cppargs['z'] == []: # Check that the mixture does not contain electrolytes. For electrolytes, a different equilibrium criterion should be used.
#         itr = 0
#         dif = 10000.
#         xl = np.copy(x_guess)
#         beta = beta_guess
#         xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol
#         while (dif>1e-9) and (itr<100):
#             beta_old = beta
#             rhol = pcsaft_den_cpp(xl, m, s, e, t, p, 0, cppargs)
#             fugcoef_l = np.asarray(pcsaft_fugcoef_cpp(xl, m, s, e, t, rhol, cppargs))
#             rhov = pcsaft_den_cpp(xv, m, s, e, t, p, 1, cppargs)
#             fugcoef_v = np.asarray(pcsaft_fugcoef_cpp(xv, m, s, e, t, rhov, cppargs))
#             if beta > 0.5:
#                 xl = fugcoef_v*xv/fugcoef_l
#                 xl = xl/np.sum(xl)
#                 xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol # if beta is close to zero then this equation behaves poorly, and that is why we use this if statement to switch the equation around
#             else:
#                 xv = fugcoef_l*xl/fugcoef_v
#                 xv = xv/np.sum(xv)
#                 xl = (mol*x_total - (beta)*mol*xv)/(1-beta)/mol
#             beta = (vol/mol*rhov*rhol-rhov)/(rhol-rhov)
#             dif = np.sum(abs(beta - beta_old))
#             itr += 1
#     else:
#         z = np.asarray(cppargs['z'])
#         # internal iteration loop to solve for compositions
#         itr = 0
#         dif = 10000.
#         xl = np.copy(x_guess)
#         beta = beta_guess
#         xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol
#         xv[np.where(z != 0)[0]] = 0.
#         xv = xv/np.sum(xv)
#         while (dif>1e-9) and (itr<100):
# #            xl = chem_equil_cpp(xl, m, s, e, t, p, cppargs)
#             x_total = xl + xv
#             beta_old = beta
#             rhol = pcsaft_den_cpp(xl, m, s, e, t, p, 0, cppargs)
#             fugcoef_l = np.asarray(pcsaft_fugcoef_cpp(xl, m, s, e, t, rhol, cppargs))
#             rhov = pcsaft_den_cpp(xv, m, s, e, t, p, 1, cppargs)
#             fugcoef_v = np.asarray(pcsaft_fugcoef_cpp(xv, m, s, e, t, rhov, cppargs))
#             if beta > 0.5:
#                 xl = fugcoef_v*xv/fugcoef_l
#                 xl = xl/np.sum(xl)*(((1-beta) - np.sum(x_total[np.where(z != 0)[0]]))/(1-beta)) # ensures that mole fractions add up to 1
#                 xl[np.where(z != 0)[0]] = x_total[np.where(z != 0)[0]]/(1-beta)
#                 xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol
#                 xv[np.where(xv < 0)[0]] = 0.
#             else:
#                 xv = fugcoef_l*xl/fugcoef_v
#                 xv[np.where(z != 0)[0]] = 0. # here it is assumed that the ionic compounds are nonvolatile
#                 xv = xv/np.sum(xv)
#                 xl = (mol*x_total - (beta)*mol*xv)/(1-beta)/mol
#             beta = (vol/mol*rhov*rhol-rhov)/(rhol-rhov)
#             dif = np.sum(abs(beta - beta_old))
#             itr += 1
#
#     output = [p, xl, xv, beta]
#     return output

def pcsaft_den(t, p, pyargs, phase='liq'):
    """
    Wrapper for C++ pcsaft_den_cpp function because a C++ struct is needed for
    that function.

    Parameters
    ----------
    t : float
        Temperature (K)
    p : float
        Pressure (Pa)
    phase : string
        The phase for which the calculation is performed. Options: "liq" (liquid),
        "vap" (vapor).
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
    pyargs = ensure_numpy_input(pyargs)
    check_input(pyargs['x'], {'pressure':p, 'temperature':t})
    cppargs = create_struct(pyargs)
    if phase == 'liq':
        phase_num = 0
    else:
        phase_num = 1

    return pcsaft_den_cpp(t, p, phase_num, cppargs)


def pcsaft_hres(t, rho, pyargs):
    """
    Calculate the residual enthalpy for one phase of the system.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
    pyargs = ensure_numpy_input(pyargs)
    check_input(pyargs['x'], {'density':rho, 'temperature':t})
    cppargs = create_struct(pyargs)
    return pcsaft_hres_cpp(t, rho, cppargs)

def pcsaft_sres(t, rho, pyargs):
    """
    Calculate the residual entropy (constant volume) for one phase of the system.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
    pyargs = ensure_numpy_input(pyargs)
    check_input(pyargs['x'], {'density':rho, 'temperature':t})
    cppargs = create_struct(pyargs)
    return pcsaft_sres_cpp(t, rho, cppargs)

def pcsaft_gres(t, rho, pyargs):
    """
    Calculate the residual Gibbs energy for one phase of the system.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
    pyargs = ensure_numpy_input(pyargs)
    check_input(pyargs['x'], {'density':rho, 'temperature':t})
    cppargs = create_struct(pyargs)
    return pcsaft_gres_cpp(t, rho, cppargs)


def pcsaft_ares(t, rho, pyargs):
    """
    Calculates the residual Helmholtz energy.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
    pyargs = ensure_numpy_input(pyargs)
    check_input(pyargs['x'], {'density':rho, 'temperature':t})
    cppargs = create_struct(pyargs)
    return pcsaft_ares_cpp(t, rho, cppargs)


def pcsaft_dadt(t, rho, pyargs):
    """
    Calculates the temperature derivative of the residual Helmholtz energy.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m^{-3})
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
    pyargs = ensure_numpy_input(pyargs)
    check_input(pyargs['x'], {'density':rho, 'temperature':t})
    cppargs = create_struct(pyargs)
    return pcsaft_dadt_cpp(t, rho, cppargs)


# def bubblePfit(p_guess, xv_guess, x, m, s, e, t, cppargs):
#     """Minimize this function to calculate the bubble point pressure."""
#     error = bubblePfit_cpp(p_guess, xv_guess, x, m, s, e, t, cppargs)
#     return error
#
# def bubbleTfit(t_guess, xv_guess, x, m, s, e, p, cppargs):
#     """Minimize this function to calculate the bubble point temperature."""
#     error = bubbleTfit_cpp(t_guess, xv_guess, x, m, s, e, p, cppargs)
#     return error
#
# def vaporPfit(p_guess, x, m, s, e, t, cppargs):
#     """Minimize this function to calculate the vapor pressure."""
#     if p_guess <= 0:
#         error = 10000000000.
#     else:
#         rho = pcsaft_den_cpp(x, m, s, e, t, p_guess, 0, cppargs)
#         fugcoef_l = np.asarray(pcsaft_fugcoef_cpp(x, m, s, e, t, rho, cppargs))
#         rho = pcsaft_den_cpp(x, m, s, e, t, p_guess, 1, cppargs)
#         fugcoef_v = np.asarray(pcsaft_fugcoef_cpp(x, m, s, e, t, rho, cppargs))
#         error = 100000*np.sum((fugcoef_l-fugcoef_v)**2)
#         if not np.isfinite(error):
#             error = 1e20
#     return error
#
# def PTzfit(p_guess, x_guess, beta_guess, mol, vol, x_total, m, s, e, t, cppargs):
#     """Minimize this function to solve for the pressure to compare with PTz data."""
#     error = PTzfit_cpp(p_guess, x_guess, beta_guess, mol, vol, x_total, m, s, e, t, cppargs)
#
#     if cppargs['z'] == []: # Check that the mixture does not contain electrolytes. For electrolytes, a different equilibrium criterion should be used.
#         # internal iteration loop to solve for compositions
#         itr = 0
#         dif = 10000.
#         xl = np.copy(x_guess)
#         beta = beta_guess
#         xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol
#         while (dif>1e-9) and (itr<100):
#             beta_old = beta
#             rhol = pcsaft_den_cpp(xl, m, s, e, t, p_guess, 0, cppargs)
#             fugcoef_l = np.asarray(pcsaft_fugcoef_cpp(xl, m, s, e, t, rhol, cppargs))
#             rhov = pcsaft_den_cpp(xv, m, s, e, t, p_guess, 1, cppargs)
#             fugcoef_v = np.asarray(pcsaft_fugcoef_cpp(xv, m, s, e, t, rhov, cppargs))
#             xl = fugcoef_v*xv/fugcoef_l
#             xl = xl/np.sum(xl)
#             xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol
#             beta = (vol/mol-rhol)/(rhov-rhol)
#             dif = np.sum(abs(beta - beta_old))
#             itr += 1
#         error = np.sum((vol - rhov*beta*mol - rhol*(1-beta)*mol)**2)
#         error += np.sum((xl*fugcoef_l - xv*fugcoef_v)**2)
#         error += np.sum((mol*x_total - beta*mol*xv - (1-beta)*mol*xl)**2)
#     else:
#         z = np.asarray(cppargs['z'])
#         # internal iteration loop to solve for compositions
#         itr = 0
#         dif = 10000.
#         xl = np.copy(x_guess)
#         beta = beta_guess
#         xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol
#         xv[np.where(z == 0)[0]] = 0.
#         xv = xv/np.sum(xv)
#         while (dif>1e-9) and (itr<100):
#             beta_old = beta
#             rhol = pcsaft_den_cpp(xl, m, s, e, t, p_guess, 0, cppargs)
#             fugcoef_l = np.asarray(pcsaft_fugcoef_cpp(xl, m, s, e, t, rhol, cppargs))
#             rhov = pcsaft_den_cpp(xv, m, s, e, t, p_guess, 1, cppargs)
#             fugcoef_v = np.asarray(pcsaft_fugcoef_cpp(xv, m, s, e, t, rhov, cppargs))
#             xl = fugcoef_v*xv/fugcoef_l
#             xl = xl/np.sum(xl)
#             xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol
#             xv[np.where(z == 0)[0]] = 0. # here it is assumed that the ionic compounds are nonvolatile
#             xv = xv/np.sum(xv)
#             beta = (vol/mol-rhol)/(rhov-rhol)
#             dif = np.sum(abs(beta - beta_old))
#             itr += 1
#         error = (vol - rhov*beta*mol - rhol*(1-beta)*mol)**2
#         error += np.sum((xl*fugcoef_l - xv*fugcoef_v)[np.where(z == 0)[0]]**2)
#         error += np.sum((mol*x_total - beta*mol*xv - (1-beta)*mol*xl)**2)
#
#     if np.isnan(error):
#         error = 100000000.
#     return error

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
    """Convert PC-SAFT parameters to a C++ struct."""
    cdef add_args cppargs

    cppargs.x = np_to_vector(pyargs['x'])
    cppargs.m = np_to_vector(pyargs['m'])
    cppargs.s = np_to_vector(pyargs['s'])
    cppargs.e = np_to_vector(pyargs['e'])
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
