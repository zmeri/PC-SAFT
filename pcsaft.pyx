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

    if ('e_assoc' in pyargs) and ('assoc_scheme' not in pyargs):
        pyargs['assoc_scheme'] = []
        for a in pyargs['vol_a']:
            if a != 0:
                pyargs['assoc_scheme'].append('2b')
            else:
                pyargs['assoc_scheme'].append(None)

    if ('e_assoc' in pyargs):
        pyargs = create_assoc_matrix(pyargs)

    return pyargs

def create_assoc_matrix(pyargs):
    charge = [] # whether the association site has a partial positive charge (i.e. hydrogen), negative charge, or elements of both (e.g. for acids modelled as type 1)

    scheme_charges = {
        '1': [0],
        '2a': [0, 0],
        '2b': [-1, 1],
        '3a': [0, 0, 0],
        '3b': [-1, -1, 1],
        '4a': [0, 0, 0, 0],
        '4b': [1, 1, 1, -1],
        '4c': [-1, -1, 1, 1]
    }

    assoc_num = []
    for comp in pyargs['assoc_scheme']:
        if comp is None:
            assoc_num.append(0)
            pass
        elif type(comp) is list:
            num = 0
            for site in comp:
                if site.lower() not in scheme_charges:
                    raise InputError('{} is not a valid association type.'.format(site))
                charge.extend(scheme_charges[site.lower()])
                num += len(scheme_charges[site.lower()])
            assoc_num.append(num)
        else:
            if comp.lower() not in scheme_charges:
                raise InputError('{} is not a valid association type.'.format(comp))
            charge.extend(scheme_charges[comp.lower()])
            assoc_num.append(len(scheme_charges[comp.lower()]))
    pyargs['assoc_num'] = np.asarray(assoc_num)

    pyargs['assoc_matrix'] = np.zeros((len(charge)*len(charge)))
    ctr = 0
    for c1 in charge:
        for c2 in charge:
            if (c1 == 0 or c2 == 0):
                pyargs['assoc_matrix'][ctr] = 1;
            elif (c1 == 1 and c2 == -1):
                pyargs['assoc_matrix'][ctr] = 1;
            elif (c1 == -1 and c2 == 1):
                pyargs['assoc_matrix'][ctr] = 1;
            else:
                pyargs['assoc_matrix'][ctr] = 0;
            ctr += 1

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
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

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
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

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
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

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
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

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

    Notes
    -----
    To solve the PQ flash the temperature must be varied. This adds additional complexity
    for water and electrolyte mixtures. For water, a temperature dependent sigma is often
    used. However, there does not appear to be a way to pass a Python function to the C++
    code without requiring the user to compile it using Cython. To avoid this, the `flashPQ`
    function uses the following relationship internally to calculate sigma for water as a
    function of temperature: ::

        3.8395 + 1.2828 * exp(-0.0074944 * t) - 1.3939 * exp(-0.00056029 * t);

    For electrolyte solutions the dielectric constant is calculated using the `dielc_water`
    function. This means that the sigma value for water and the dielectric constant given by
    the user are not used by the `flashPQ` function.

    The code identifies which component is water by the epsilon/k value. Therefore, when
    using `flashPQ` with water `e` must be exactly 353.9449, if you want the temperature
    dependence of sigma to be accounted for.

    If you want to use different functions for temperature dependent parameters with `flashPQ`
    then you will need to modify the source code and recompile it.
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
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

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
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

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
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

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
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

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
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

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
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

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
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

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
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

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
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

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
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

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
    if 'assoc_matrix' in pyargs:
        cppargs.assoc_matrix = np_to_vector_int(pyargs['assoc_matrix'])
    if 'k_hb' in pyargs:
        cppargs.k_hb = np_to_vector_double(pyargs['k_hb'])
    if 'l_ij' in pyargs:
        cppargs.l_ij = np_to_vector_double(pyargs['l_ij'])

    return cppargs
