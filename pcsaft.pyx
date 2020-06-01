# -*- coding: utf-8 -*-
# setuptools: language=c++

import numpy as np
from libcpp.vector cimport vector
from copy import deepcopy
cimport pcsaft

class InputError(Exception):
    # Exception raised for errors in the input.
    def __init__(self, message):
        self.message = message

class SolutionError(Exception):
    # Exception raised when a solver does not return a value.
    def __init__(self, message):
        self.message = message

def check_input(x, vars):
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

def check_association(pyargs):
    if ('e_assoc' in pyargs) and ('vol_a' not in pyargs):
        raise InputError('e_assoc was given, but not vol_a.')
    elif ('vol_a' in pyargs) and ('e_assoc' not in pyargs):
        raise InputError('vol_a was given, but not e_assoc.')
    elif ('e_assoc' in pyargs) and ('assoc_num' in pyargs) and ('assoc_scheme' not in pyargs):
        raise InputError('When assoc_num is given, assoc_scheme must also be specified.')

    if ('vol_a' in pyargs) and ('assoc_num' not in pyargs):
        pyargs['assoc_num'] = np.zeros_like(pyargs['vol_a'])
        pyargs['assoc_num'][pyargs['vol_a'] != 0] = 2

        pyargs['assoc_scheme'] = np.ones((int(np.sum(pyargs['assoc_num'])),int(np.sum(pyargs['assoc_num']))))
        for i in range(pyargs['assoc_scheme'].shape[0]):
            for j in range(pyargs['assoc_scheme'].shape[1]):
                if ((i + j) % 2 == 0):
                    pyargs['assoc_scheme'][i,j] = 0

    return pyargs

def ensure_numpy_input(x, pyargs):
    if type(x) == np.float_:
        x = np.asarray([x])
    if type(pyargs['m']) == np.float_:
        pyargs['m'] = np.asarray([pyargs['m']])
    if type(pyargs['s']) == np.float_:
        pyargs['s'] = np.asarray([pyargs['s']])
    if type(pyargs['e']) == np.float_:
        pyargs['e'] = np.asarray([pyargs['e']])
    return x, pyargs

def pcsaft_p(t, rho, x, pyargs):
    """
    Calculate pressure.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_num : ndarray, shape (n,)
            Number of association sites for each component. Note that each associating
            functional group generally has multiple association sites (see Huang and
            Radosz 1990). If `e_assoc` and `vol_a` are given but this is not, it is
            assumed there are 2 association sites (which would, for example, correspond
            to one hydroxyl functional group).
        assoc_scheme : ndarray, shape (m,m)
            Array showing whether or not association occurs between the various
            association sites in the system. 0 indicates that association does not
            occur and 1 (or any other nonzero number) indicates that association
            does occur between those two sites. The order for the sites must be the same
            as used for assoc_num. If this array and assoc_num are
            omitted, then a 2B association scheme is assumed. If `assoc_num` is given
            and `assoc_scheme` is not, then an error is thrown.

    Returns
    -------
    P : float
        Pressure (Pa)
    """
    x, pyargs = ensure_numpy_input(x, pyargs)
    check_input(x, {'density':rho, 'temperature':t})
    pyargs = check_association(pyargs)
    cppargs = create_struct(pyargs)
    return pcsaft_p_cpp(t, rho, x, cppargs)


def pcsaft_fugcoef(t, rho, x, pyargs):
    """
    Calculate the fugacity coefficients for one phase of the system.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_num : ndarray, shape (n,)
            Number of association sites for each component. Note that each associating
            functional group generally has multiple association sites (see Huang and
            Radosz 1990). If `e_assoc` and `vol_a` are given but this is not, it is
            assumed there are 2 association sites (which would, for example, correspond
            to one hydroxyl functional group).
        assoc_scheme : ndarray, shape (m,m)
            Array showing whether or not association occurs between the various
            association sites in the system. 0 indicates that association does not
            occur and 1 (or any other nonzero number) indicates that association
            does occur between those two sites. The order for the sites must be the same
            as used for assoc_num. If this array and assoc_num are
            omitted, then a 2B association scheme is assumed. If `assoc_num` is given
            and `assoc_scheme` is not, then an error is thrown.

    Returns
    -------
    fugcoef : ndarray, shape (n,)
        Fugacity coefficients of each component.
    """
    x, pyargs = ensure_numpy_input(x, pyargs)
    check_input(x, {'density':rho, 'temperature':t})
    pyargs = check_association(pyargs)
    cppargs = create_struct(pyargs)
    return np.asarray(pcsaft_fugcoef_cpp(t, rho, x, cppargs))


def pcsaft_Z(t, rho, x, pyargs):
    """
    Calculate the compressibility factor.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_num : ndarray, shape (n,)
            Number of association sites for each component. Note that each associating
            functional group generally has multiple association sites (see Huang and
            Radosz 1990). If `e_assoc` and `vol_a` are given but this is not, it is
            assumed there are 2 association sites (which would, for example, correspond
            to one hydroxyl functional group).
        assoc_scheme : ndarray, shape (m,m)
            Array showing whether or not association occurs between the various
            association sites in the system. 0 indicates that association does not
            occur and 1 (or any other nonzero number) indicates that association
            does occur between those two sites. The order for the sites must be the same
            as used for assoc_num. If this array and assoc_num are
            omitted, then a 2B association scheme is assumed. If `assoc_num` is given
            and `assoc_scheme` is not, then an error is thrown.

    Returns
    -------
    Z : float
        Compressibility factor
    """
    x, pyargs = ensure_numpy_input(x, pyargs)
    check_input(x, {'density':rho, 'temperature':t})
    pyargs = check_association(pyargs)
    cppargs = create_struct(pyargs)
    return pcsaft_Z_cpp(t, rho, x, cppargs)


def flashPQ(p, q, x, pyargs, t_guess=None):
    """
    Calculate the temperature of the system where vapor and liquid phases are in equilibrium.

    Parameters
    ----------
    p : float
        Pressure (Pa)
    q : float
        Mole fraction of the fluid in the vapor phase
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_num : ndarray, shape (n,)
            Number of association sites for each component. Note that each associating
            functional group generally has multiple association sites (see Huang and
            Radosz 1990). If `e_assoc` and `vol_a` are given but this is not, it is
            assumed there are 2 association sites (which would, for example, correspond
            to one hydroxyl functional group).
        assoc_scheme : ndarray, shape (m,m)
            Array showing whether or not association occurs between the various
            association sites in the system. 0 indicates that association does not
            occur and 1 (or any other nonzero number) indicates that association
            does occur between those two sites. The order for the sites must be the same
            as used for assoc_num. If this array and assoc_num are
            omitted, then a 2B association scheme is assumed. If `assoc_num` is given
            and `assoc_scheme` is not, then an error is thrown.

    t_guess : float
        Initial guess for the temperature (K) (optional)

    Returns
    -------
    t : float
        Temperature (K)
    xl : ndarray, shape (n,)
        Liquid mole fractions after flash
    xv : ndarray, shape (n,)
        Vapor mole fractions after flash
    """
    x, pyargs = ensure_numpy_input(x, pyargs)
    check_input(x, {'pressure':p, 'Q':q})
    pyargs = check_association(pyargs)
    cppargs = create_struct(pyargs)
    try:
        if t_guess is not None:
            result = flashPQ_cpp(p, q, x, cppargs, t_guess)
        else:
            result = flashPQ_cpp(p, q, x, cppargs)
    except:
        raise SolutionError('A solution was not found for flashPQ. P={}'.format(p))

    t = result[0]
    xl = np.asarray(result[1:])
    xl, xv = np.split(xl, 2)
    return t, xl, xv


def flashTQ(t, q, x, pyargs, p_guess=None):
    """
    Calculate the pressure of the system where vapor and liquid phases are in equilibrium.

    Parameters
    ----------
    t : float
        Temperature (K)
    q : float
        Mole fraction of the fluid in the vapor phase
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_num : ndarray, shape (n,)
            Number of association sites for each component. Note that each associating
            functional group generally has multiple association sites (see Huang and
            Radosz 1990). If `e_assoc` and `vol_a` are given but this is not, it is
            assumed there are 2 association sites (which would, for example, correspond
            to one hydroxyl functional group).
        assoc_scheme : ndarray, shape (m,m)
            Array showing whether or not association occurs between the various
            association sites in the system. 0 indicates that association does not
            occur and 1 (or any other nonzero number) indicates that association
            does occur between those two sites. The order for the sites must be the same
            as used for assoc_num. If this array and assoc_num are
            omitted, then a 2B association scheme is assumed. If `assoc_num` is given
            and `assoc_scheme` is not, then an error is thrown.

    p_guess : float
        Initial guess for the pressure (Pa) (optional)

    Returns
    -------
    p : float
        Pressure (Pa)
    xl : ndarray, shape (n,)
        Liquid mole fractions after flash
    xv : ndarray, shape (n,)
        Vapor mole fractions after flash
    """
    x, pyargs = ensure_numpy_input(x, pyargs)
    check_input(x, {'temperature':t, 'Q':q})
    pyargs = check_association(pyargs)
    cppargs = create_struct(pyargs)
    try:
        if p_guess is not None:
            result = flashTQ_cpp(t, q, x, cppargs, p_guess)
        else:
            result = flashTQ_cpp(t, q, x, cppargs)
    except:
        raise SolutionError('A solution was not found for flashTQ. T={}'.format(t))

    p = result[0]
    xl = np.asarray(result[1:])
    xl, xv = np.split(xl, 2)
    return p, xl, xv

def pcsaft_Hvap(t, x, pyargs, p_guess=None):
    """
    Calculate the enthalpy of vaporization.

    Parameters
    ----------
    t : float
        Temperature (K)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_num : ndarray, shape (n,)
            Number of association sites for each component. Note that each associating
            functional group generally has multiple association sites (see Huang and
            Radosz 1990). If `e_assoc` and `vol_a` are given but this is not, it is
            assumed there are 2 association sites (which would, for example, correspond
            to one hydroxyl functional group).
        assoc_scheme : ndarray, shape (m,m)
            Array showing whether or not association occurs between the various
            association sites in the system. 0 indicates that association does not
            occur and 1 (or any other nonzero number) indicates that association
            does occur between those two sites. The order for the sites must be the same
            as used for assoc_num. If this array and assoc_num are
            omitted, then a 2B association scheme is assumed. If `assoc_num` is given
            and `assoc_scheme` is not, then an error is thrown.

    p_guess : float
        Guess for the vapor pressure (Pa) (optional)

    Returns
    -------
    output : list
        A list containing the following results:
            0 : enthalpy of vaporization (J/mol), float
            1 : vapor pressure (Pa), float
    """
    x, pyargs = ensure_numpy_input(x, pyargs)
    check_input(x, {'temperature': t})
    pyargs = check_association(pyargs)
    cppargs = create_struct(pyargs)

    q = 0
    try:
        if p_guess is not None:
            result = np.asarray(flashTQ_cpp(t, q, x, cppargs, p_guess))
            Pvap = result[0]
        else:
            result = np.asarray(flashTQ_cpp(t, q, x, cppargs))
            Pvap = result[0]
    except:
        raise SolutionError('A solution was not found for flashTQ. T={}'.format(t))

    rho = pcsaft_den_cpp(t, Pvap, x, 0, cppargs)
    hres_l = pcsaft_hres_cpp(t, rho, x, cppargs)
    rho = pcsaft_den_cpp(t, Pvap, x, 1, cppargs)
    hres_v = pcsaft_hres_cpp(t, rho, x, cppargs)
    Hvap = hres_v - hres_l

    output = [Hvap, Pvap]
    return output


def pcsaft_osmoticC(t, rho, x, pyargs):
    """
    Calculate the osmotic coefficient.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_num : ndarray, shape (n,)
            Number of association sites for each component. Note that each associating
            functional group generally has multiple association sites (see Huang and
            Radosz 1990). If `e_assoc` and `vol_a` are given but this is not, it is
            assumed there are 2 association sites (which would, for example, correspond
            to one hydroxyl functional group).
        assoc_scheme : ndarray, shape (m,m)
            Array showing whether or not association occurs between the various
            association sites in the system. 0 indicates that association does not
            occur and 1 (or any other nonzero number) indicates that association
            does occur between those two sites. The order for the sites must be the same
            as used for assoc_num. If this array and assoc_num are
            omitted, then a 2B association scheme is assumed. If `assoc_num` is given
            and `assoc_scheme` is not, then an error is thrown.

    Returns
    -------
    osmC : float
        Molal osmotic coefficient
    """
    x, pyargs = ensure_numpy_input(x, pyargs)
    check_input(x, {'density':rho, 'temperature':t})
    pyargs = check_association(pyargs)
    cppargs = create_struct(pyargs)

    indx_water = np.where(pyargs['e'] == 353.9449)[0] # to find index for water
    molality = x/(x[indx_water]*18.0153/1000.)
    molality[indx_water] = 0
    x0 = np.zeros_like(x)
    x0[indx_water] = 1.

    fugcoef = np.asarray(pcsaft_fugcoef_cpp(t, rho, x, cppargs))
    p = pcsaft_p_cpp(t, rho, x, cppargs)
    if rho < 900:
        ph = 1
    else:
        ph = 0
    rho0 = pcsaft_den_cpp(t, p, x0, ph, cppargs)
    fugcoef0 = np.asarray(pcsaft_fugcoef_cpp(t, rho0, x0, cppargs))
    gamma = fugcoef[indx_water]/fugcoef0[indx_water]

    osmC = -1000*np.log(x[indx_water]*gamma)/18.0153/np.sum(molality)
    return osmC

def pcsaft_cp(t, rho, params, x, pyargs):
    """
    Calculate the specific molar isobaric heat capacity.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    params : ndarray, shape (5,)
        Constants for the Aly-Lee equation. Can be substituted with parameters for
        another equation if the ideal gas heat capacity is given using a different
        equation.
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_num : ndarray, shape (n,)
            Number of association sites for each component. Note that each associating
            functional group generally has multiple association sites (see Huang and
            Radosz 1990). If `e_assoc` and `vol_a` are given but this is not, it is
            assumed there are 2 association sites (which would, for example, correspond
            to one hydroxyl functional group).
        assoc_scheme : ndarray, shape (m,m)
            Array showing whether or not association occurs between the various
            association sites in the system. 0 indicates that association does not
            occur and 1 (or any other nonzero number) indicates that association
            does occur between those two sites. The order for the sites must be the same
            as used for assoc_num. If this array and assoc_num are
            omitted, then a 2B association scheme is assumed. If `assoc_num` is given
            and `assoc_scheme` is not, then an error is thrown.

    Returns
    -------
    cp : float
        Specific molar isobaric heat capacity (J mol\ :sup:`-1` K\ :sup:`-1`)
    """
    x, pyargs = ensure_numpy_input(x, pyargs)
    check_input(x, {'density':rho, 'temperature':t})
    pyargs = check_association(pyargs)

    if rho > 900:
        ph = 0
    else:
        ph = 1

    cppargs = create_struct(pyargs)

    cp_ideal = aly_lee(t, params)
    p = pcsaft_p_cpp(t, rho, x, cppargs)
    rho0 = pcsaft_den_cpp(t-0.001, p, x, ph, cppargs)
    hres0 = pcsaft_hres_cpp(t-0.001, rho0, x, cppargs)
    rho1 = pcsaft_den_cpp(t+0.001, p, x, ph, cppargs)
    hres1 = pcsaft_hres_cpp(t+0.001, rho1, x, cppargs)
    dhdt = (hres1-hres0)/0.002 # a numerical derivative is used for now until analytical derivatives are ready
    return cp_ideal + dhdt


def pcsaft_den(t, p, x, pyargs, phase='liq'):
    """
    Calculate the molar density.

    Parameters
    ----------
    t : float
        Temperature (K)
    p : float
        Pressure (Pa)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_num : ndarray, shape (n,)
            Number of association sites for each component. Note that each associating
            functional group generally has multiple association sites (see Huang and
            Radosz 1990). If `e_assoc` and `vol_a` are given but this is not, it is
            assumed there are 2 association sites (which would, for example, correspond
            to one hydroxyl functional group).
        assoc_scheme : ndarray, shape (m,m)
            Array showing whether or not association occurs between the various
            association sites in the system. 0 indicates that association does not
            occur and 1 (or any other nonzero number) indicates that association
            does occur between those two sites. The order for the sites must be the same
            as used for assoc_num. If this array and assoc_num are
            omitted, then a 2B association scheme is assumed. If `assoc_num` is given
            and `assoc_scheme` is not, then an error is thrown.

    phase : string
        The phase for which the calculation is performed. Options: "liq" (liquid),
        "vap" (vapor).

    Returns
    -------
    rho : float
        Molar density (mol m\ :sup:`-3`)
    """
    x, pyargs = ensure_numpy_input(x, pyargs)
    check_input(x, {'pressure':p, 'temperature':t})
    pyargs = check_association(pyargs)
    cppargs = create_struct(pyargs)
    if phase == 'liq':
        phase_num = 0
    else:
        phase_num = 1

    return pcsaft_den_cpp(t, p, x, phase_num, cppargs)


def pcsaft_hres(t, rho, x, pyargs):
    """
    Calculate the residual enthalpy for one phase of the system.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_num : ndarray, shape (n,)
            Number of association sites for each component. Note that each associating
            functional group generally has multiple association sites (see Huang and
            Radosz 1990). If `e_assoc` and `vol_a` are given but this is not, it is
            assumed there are 2 association sites (which would, for example, correspond
            to one hydroxyl functional group).
        assoc_scheme : ndarray, shape (m,m)
            Array showing whether or not association occurs between the various
            association sites in the system. 0 indicates that association does not
            occur and 1 (or any other nonzero number) indicates that association
            does occur between those two sites. The order for the sites must be the same
            as used for assoc_num. If this array and assoc_num are
            omitted, then a 2B association scheme is assumed. If `assoc_num` is given
            and `assoc_scheme` is not, then an error is thrown.

    Returns
    -------
    hres : float
        Residual enthalpy (J mol\ :sup:`-1`)
    """
    x, pyargs = ensure_numpy_input(x, pyargs)
    check_input(x, {'density':rho, 'temperature':t})
    pyargs = check_association(pyargs)
    cppargs = create_struct(pyargs)
    return pcsaft_hres_cpp(t, rho, x, cppargs)

def pcsaft_sres(t, rho, x, pyargs):
    """
    Calculate the residual entropy (constant volume) for one phase of the system.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_num : ndarray, shape (n,)
            Number of association sites for each component. Note that each associating
            functional group generally has multiple association sites (see Huang and
            Radosz 1990). If `e_assoc` and `vol_a` are given but this is not, it is
            assumed there are 2 association sites (which would, for example, correspond
            to one hydroxyl functional group).
        assoc_scheme : ndarray, shape (m,m)
            Array showing whether or not association occurs between the various
            association sites in the system. 0 indicates that association does not
            occur and 1 (or any other nonzero number) indicates that association
            does occur between those two sites. The order for the sites must be the same
            as used for assoc_num. If this array and assoc_num are
            omitted, then a 2B association scheme is assumed. If `assoc_num` is given
            and `assoc_scheme` is not, then an error is thrown.

    Returns
    -------
    sres : float
        Residual entropy (J mol\ :sup:`-1` K\ :sup:`-1`)
    """
    x, pyargs = ensure_numpy_input(x, pyargs)
    check_input(x, {'density':rho, 'temperature':t})
    pyargs = check_association(pyargs)
    cppargs = create_struct(pyargs)
    return pcsaft_sres_cpp(t, rho, x, cppargs)

def pcsaft_gres(t, rho, x, pyargs):
    """
    Calculate the residual Gibbs energy for one phase of the system.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_num : ndarray, shape (n,)
            Number of association sites for each component. Note that each associating
            functional group generally has multiple association sites (see Huang and
            Radosz 1990). If `e_assoc` and `vol_a` are given but this is not, it is
            assumed there are 2 association sites (which would, for example, correspond
            to one hydroxyl functional group).
        assoc_scheme : ndarray, shape (m,m)
            Array showing whether or not association occurs between the various
            association sites in the system. 0 indicates that association does not
            occur and 1 (or any other nonzero number) indicates that association
            does occur between those two sites. The order for the sites must be the same
            as used for assoc_num. If this array and assoc_num are
            omitted, then a 2B association scheme is assumed. If `assoc_num` is given
            and `assoc_scheme` is not, then an error is thrown.

    Returns
    -------
    gres : float
        Residual Gibbs energy (J mol\ :sup:`-1`)
    """
    x, pyargs = ensure_numpy_input(x, pyargs)
    check_input(x, {'density':rho, 'temperature':t})
    pyargs = check_association(pyargs)
    cppargs = create_struct(pyargs)
    return pcsaft_gres_cpp(t, rho, x, cppargs)


def pcsaft_ares(t, rho, x, pyargs):
    """
    Calculate the residual Helmholtz energy.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_num : ndarray, shape (n,)
            Number of association sites for each component. Note that each associating
            functional group generally has multiple association sites (see Huang and
            Radosz 1990). If `e_assoc` and `vol_a` are given but this is not, it is
            assumed there are 2 association sites (which would, for example, correspond
            to one hydroxyl functional group).
        assoc_scheme : ndarray, shape (m,m)
            Array showing whether or not association occurs between the various
            association sites in the system. 0 indicates that association does not
            occur and 1 (or any other nonzero number) indicates that association
            does occur between those two sites. The order for the sites must be the same
            as used for assoc_num. If this array and assoc_num are
            omitted, then a 2B association scheme is assumed. If `assoc_num` is given
            and `assoc_scheme` is not, then an error is thrown.

    Returns
    -------
    ares : float
        Residual Helmholtz energy (J mol\ :sup:`-1`)
    """
    x, pyargs = ensure_numpy_input(x, pyargs)
    check_input(x, {'density':rho, 'temperature':t})
    pyargs = check_association(pyargs)
    cppargs = create_struct(pyargs)
    return pcsaft_ares_cpp(t, rho, x, cppargs)


def pcsaft_dadt(t, rho, x, pyargs):
    """
    Calculate the temperature derivative of the residual Helmholtz energy.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    pyargs : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

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
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_num : ndarray, shape (n,)
            Number of association sites for each component. Note that each associating
            functional group generally has multiple association sites (see Huang and
            Radosz 1990). If `e_assoc` and `vol_a` are given but this is not, it is
            assumed there are 2 association sites (which would, for example, correspond
            to one hydroxyl functional group).
        assoc_scheme : ndarray, shape (m,m)
            Array showing whether or not association occurs between the various
            association sites in the system. 0 indicates that association does not
            occur and 1 (or any other nonzero number) indicates that association
            does occur between those two sites. The order for the sites must be the same
            as used for assoc_num. If this array and assoc_num are
            omitted, then a 2B association scheme is assumed. If `assoc_num` is given
            and `assoc_scheme` is not, then an error is thrown.

    Returns
    -------
    dadt : float
        Temperature derivative of the residual Helmholtz energy (J mol\ :sup:`-1`)
    """
    x, pyargs = ensure_numpy_input(x, pyargs)
    check_input(x, {'density':rho, 'temperature':t})
    pyargs = check_association(pyargs)
    cppargs = create_struct(pyargs)
    return pcsaft_dadt_cpp(t, rho, x, cppargs)


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
        Ideal gas isobaric heat capacity (J mol\ :sup:`-1` K\ :sup:`-1`)

    References
    ----------
    - F. A. Aly and L. L. Lee, “Self-consistent equations for calculating the ideal gas heat capacity, enthalpy, and entropy,” Fluid Phase Equilibria, vol. 6, no. 3–4, pp. 169–179, 1981.
    """
    cp_ideal = (c[0] + c[1]*(c[2]/t/np.sinh(c[2]/t))**2 + c[3]*(c[4]/t/np.cosh(c[4]/t))**2)/1000.
    return cp_ideal

def dielc_water(t):
    """
    Return the dielectric constant of water at the given temperature.

    This equation was fit to values given in the reference. For temperatures from
    263.15 to 368.15 K values at 1 bar were used. For temperatures from 368.15 to
    443.15 K values at 10 bar were used. Below 263.15 K and above 443.15 K an
    error is raised.

    Parameters
    ----------
    t : float
        Temperature (K)

    Returns
    -------
    dielc : float
        Dielectric constant of water

    References
    ----------
    - D. G. Archer and P. Wang, “The Dielectric Constant of Water and Debye‐Hückel Limiting Law Slopes,” J. Phys. Chem. Ref. Data, vol. 19, no. 2, pp. 371–411, Mar. 1990.
    """
    if t < 263.15:
        raise ValueError('For dielc_water t must be greater than 263.15 K.')
    elif t > 443.15:
        raise ValueError('For dielc_water t must be less than 443.15 K.')

    if t <= 368.15:
        dielc = 7.6555618295E-04*t**2 - 8.1783881423E-01*t + 2.5419616803E+02
    else:
        dielc = 0.0005003272124*t**2 - 0.6285556029*t + 220.4467027
    return dielc


def np_to_vector_double(np_array):
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

def np_to_vector_int(np_array):
    """Take a numpy array and return a C++ vector."""
    cdef vector[int] cpp_vector

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

    cppargs.m = np_to_vector_double(pyargs['m'])
    cppargs.s = np_to_vector_double(pyargs['s'])
    cppargs.e = np_to_vector_double(pyargs['e'])
    if 'k_ij' in pyargs:
        cppargs.k_ij = np_to_vector_double(pyargs['k_ij'])
    if ('e_assoc' in pyargs) and np.any(pyargs['e_assoc']):
        cppargs.e_assoc = np_to_vector_double(pyargs['e_assoc'])
    if ('vol_a' in pyargs) and np.any(pyargs['vol_a']):
        cppargs.vol_a = np_to_vector_double(pyargs['vol_a'])
    if ('dipm' in pyargs) and np.any(pyargs['dip_num']) and np.any(pyargs['dipm']):
        cppargs.dipm = np_to_vector_double(pyargs['dipm'])
    if ('dip_num' in pyargs) and np.any(pyargs['dip_num']):
        cppargs.dip_num = np_to_vector_double(pyargs['dip_num'])
    if 'z' in pyargs:
        cppargs.z = np_to_vector_double(pyargs['z'])
    if 'dielc' in pyargs:
        cppargs.dielc = pyargs['dielc']
    if 'assoc_num' in pyargs:
        cppargs.assoc_num = np_to_vector_int(pyargs['assoc_num'])
    if 'assoc_scheme' in pyargs:
        cppargs.assoc_scheme = np_to_vector_int(pyargs['assoc_scheme'])
    if 'k_hb' in pyargs:
        cppargs.k_hb = np_to_vector_double(pyargs['k_hb'])
    if 'l_ij' in pyargs:
        cppargs.l_ij = np_to_vector_double(pyargs['l_ij'])

    return cppargs
