# -*- coding: utf-8 -*-
"""
PC-SAFT with electrolyte term

These functions implement the PC-SAFT equation of state. In addition to the 
hard chain and dispersion terms, these functions also include dipole, 
association and ion terms for use with these types of compounds.

@author: Zach Baird

Functions
---------
- pcsaft_vaporP : calculate the vapor pressure
- pcsaft_bubbleP : calculate the bubble point pressure of a mixture
- pcsaft_Hvap : calculate the enthalpy of vaporization
- pcsaft_PTz : allows PTz data to be used for parameter fitting
- pcsaft_den : calculate the molar density
- pcsaft_p : calculate the pressure
- pcsaft_hres : calculate the residual enthalpy
- pcsaft_sres : calculate the residual entropy
- pcsaft_gres : calculate the residual Gibbs free energy
- pcsaft_fugcoef : calculate the fugacity coefficients
- pcsaft_Z : calculate the compressibility factor
- pcsaft_ares : calculate the residual Helmholtz energy
- XA_find : used internally to solve for XA
- dXA_find : used internally to solve for the derivative of XA wrt density
- dXAdt_find : used internally to solve for the derivative of XA wrt temperature
- bubblePfit : used internally to solve for the bubble point pressure
- vaporPfit : used internally to solve for the vapor pressure
- denfit : used internally to solve for the density
- PTzfit : used internally to solve for pressure and compositions
- dielc_water : returns the dielectric constant of water
- pcsaft_fit_pure : can be used when fitting PC-SAFT parameters to data
    
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
   
def pcsaft_vaporP(p_guess, x, m, s, e, t, **kwargs):
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
    kwargs : dict
        Additional parameters that can be passed through to the PC-SAFT functions.
        The PC-SAFT functions can take the following additional parameters:
        
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
    Pvap = minimize(vaporPfit, p_guess, args=(x, m, s, e, t, kwargs), tol=1e-10, method='Nelder-Mead', options={'maxiter': 100}).x
    return Pvap
    
def pcsaft_bubbleP(p_guess, xv_guess, x, m, s, e, t, **kwargs):
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
    kwargs : dict
        Additional parameters that can be passed through to the PC-SAFT functions.
        The PC-SAFT functions can take the following additional parameters:
        
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
    result = minimize(bubblePfit, p_guess, args=(xv_guess, x, m, s, e, t, kwargs), tol=1e-10, method='Nelder-Mead', options={'maxiter': 100})
    bubP = result.x

    # Determine vapor phase composition at bubble pressure    
    if not ('z' in kwargs): # Check that the mixture does not contain electrolytes. For electrolytes, a different equilibrium criterion should be used. 
        rho = pcsaft_den(x, m, s, e, t, p_guess, phase='liq', **kwargs)        
        fugcoef_l = pcsaft_fugcoef(x, m, s, e, t, rho, **kwargs)
        
        itr = 0
        dif = 10000.
        xv = np.copy(xv_guess)
        xv_old = np.zeros_like(xv)
        while (dif>1e-9) and (itr<100):
            xv_old[:] = xv
            rho = pcsaft_den(xv, m, s, e, t, p_guess, phase='vap', **kwargs)        
            fugcoef_v = pcsaft_fugcoef(xv, m, s, e, t, rho, **kwargs)
            xv = fugcoef_l*x/fugcoef_v
            xv = xv/np.sum(xv)
            dif = np.sum(abs(xv - xv_old))
            itr += 1
    else:
        z = kwargs['z']
        rho = pcsaft_den(x, m, s, e, t, p_guess, phase='liq', **kwargs)        
        fugcoef_l = pcsaft_fugcoef(x, m, s, e, t, rho, **kwargs)        
        
        itr = 0
        dif = 10000.
        xv = np.copy(xv_guess)
        xv_old = np.zeros_like(xv)
        while (dif>1e-9) and (itr<100):
            xv_old[:] = xv
            rho = pcsaft_den(xv_guess, m, s, e, t, p_guess, phase='vap', **kwargs)        
            fugcoef_v = pcsaft_fugcoef(xv_guess, m, s, e, t, rho, **kwargs)        
        
            xv[np.where(z == 0)[0]] = (fugcoef_l*x/fugcoef_v)[np.where(z == 0)[0]] # here it is assumed that the ionic compounds are nonvolatile
            xv = xv/np.sum(xv)
            dif = np.sum(abs(xv - xv_old))
            itr += 1
    
    results = [bubP, xv]
    return results


def pcsaft_Hvap(p_guess, x, m, s, e, t, **kwargs):
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
    kwargs : dict
        Additional parameters that can be passed through to the PC-SAFT functions.
        The PC-SAFT functions can take the following additional parameters:
        
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
    Pvap = minimize(vaporPfit, p_guess, args=(x, m, s, e, t, kwargs), tol=1e-10, method='Nelder-Mead', options={'maxiter': 100}).x

    rho = pcsaft_den(x, m, s, e, t, Pvap, phase='liq', **kwargs)        
    hres_l = pcsaft_hres(x, m, s, e, t, rho, **kwargs)
    rho = pcsaft_den(x, m, s, e, t, Pvap, phase='vap', **kwargs)
    hres_v = pcsaft_hres(x, m, s, e, t, rho, **kwargs)
    Hvap = hres_v - hres_l
    
    output = [Hvap, Pvap]    
    return output


def pcsaft_PTz(p_guess, x_guess, beta_guess, mol, vol, x_total, m, s, e, t, **kwargs):
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
    kwargs : dict
        Additional parameters that can be passed through to the PC-SAFT functions.
        The PC-SAFT functions can take the following additional parameters:
        
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
    result = minimize(PTzfit, p_guess, args=(x_guess, beta_guess, mol, vol, x_total, m, s, e, t, kwargs), tol=1e-10, method='Nelder-Mead', options={'maxiter': 100})
    p = result.x

    if not ('z' in kwargs): # Check that the mixture does not contain electrolytes. For electrolytes, a different equilibrium criterion should be used.
        itr = 0
        dif = 10000.
        xl = np.copy(x_guess)
        beta = beta_guess
        xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol
        while (dif>1e-9) and (itr<100):
            beta_old = beta
            rhol = pcsaft_den(xl, m, s, e, t, p, phase='liq', **kwargs)        
            fugcoef_l = pcsaft_fugcoef(xl, m, s, e, t, rhol, **kwargs)
            rhov = pcsaft_den(xv, m, s, e, t, p, phase='vap', **kwargs)        
            fugcoef_v = pcsaft_fugcoef(xv, m, s, e, t, rhov, **kwargs)
            xl = fugcoef_v*xv/fugcoef_l
            xl = xl/np.sum(xl)
            xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol
            beta = (vol/mol-rhol)/(rhov-rhol)
            dif = np.sum(abs(beta - beta_old))
            itr += 1
    else:
        z = kwargs['z']
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
            rhol = pcsaft_den(xl, m, s, e, t, p, phase='liq', **kwargs)        
            fugcoef_l = pcsaft_fugcoef(xl, m, s, e, t, rhol, **kwargs)
            rhov = pcsaft_den(xv, m, s, e, t, p, phase='vap', **kwargs)        
            fugcoef_v = pcsaft_fugcoef(xv, m, s, e, t, rhov, **kwargs)
            xl = fugcoef_v*xv/fugcoef_l
            xl = xl/np.sum(xl)
            xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol
            xv[np.where(z == 0)[0]] = 0. # here it is assumed that the ionic compounds are nonvolatile
            xv = xv/np.sum(xv)
            beta = (vol/mol-rhol)/(rhov-rhol)
            dif = np.sum(abs(beta - beta_old))
            itr += 1

    output = [p, xl, xv, beta]
    return output

def pcsaft_den(x, m, s, e, t, p, phase='liq', **kwargs): 
    """
    Solve for the molar density when temperature and pressure are given.
    
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
    kwargs : dict
        Additional parameters that can be passed through to the PC-SAFT functions.
        The PC-SAFT functions can take the following additional parameters:
        
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
    if phase == 'liq':
        eta_guess = 0.5
    elif phase == 'vap':
        eta_guess = 1e-9
    else:
        ValueError('The phase must be specified as either "liq" or "vap".')

    N_AV = 6.022140857e23 # Avagadro's number
    
    d = s*(1-0.12*np.exp(-3*e/t))
    if ('z' in kwargs):
        z = kwargs['z']
        d[np.where(z != 0)[0]] = s[np.where(z != 0)[0]]*(1-0.12) # for ions the diameter is assumed to be temperature independent (see Held et al. 2014)

    den_guess = 6/np.pi*eta_guess/np.sum(x*m*d**3)
    rho_guess = den_guess*1.0e30/N_AV

    rho = fsolve(denfit, rho_guess, args=(x, m, s, e, t, p, kwargs), full_output=True)[0]
    return rho
    

def pcsaft_p(x, m, s, e, t, rho, k_ij=None, e_assoc=None, vol_a=None, dipm=None, \
            dip_num=None, z=None, dielc=None):
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
    kb = 1.380648465952442093e-23 # Boltzmann constant, J K^-1
    N_AV = 6.022140857e23 # Avagadro's number
    den = rho*N_AV/1.0e30 # number density, units of Angstrom^-3
   
    Z = pcsaft_Z(x, m, s, e, t, rho, k_ij, e_assoc, vol_a, dipm, dip_num, z, dielc)   
    P = Z*kb*t*den*1.0e30 # Pa
    return P
    

def pcsaft_hres(x, m, s, e, t, rho, k_ij=None, e_assoc=None, vol_a=None, dipm=None, \
            dip_num=None, z=None, dielc=None):
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
    ncomp = x.shape[0] # number of components
    kb = 1.380648465952442093e-23 # Boltzmann constant, J K^-1
    N_AV = 6.022140857e23 # Avagadro's number
    
    d = s*(1-0.12*np.exp(-3*e/t))
    dd_dt = s*-3*e/t/t*0.12*np.exp(-3*e/t)
    if type(z) == np.ndarray:
        d[np.where(z != 0)[0]] = s[np.where(z != 0)[0]]*(1-0.12) # for ions the diameter is assumed to be temperature independent (see Held et al. 2014)
        dd_dt[np.where(z != 0)[0]] = 0.

    den = rho*N_AV/1.0e30

    if type(k_ij) != np.ndarray:
        k_ij = np.zeros((ncomp,ncomp), dtype='float_')

    zeta = np.zeros((4,), dtype='float_')
    dzeta_dt = np.zeros_like(zeta)
    ghs = np.zeros((ncomp,ncomp), dtype='float_')
    dghs_dt = np.zeros_like(ghs)
    e_ij = np.zeros_like(ghs)
    s_ij = np.zeros_like(ghs)
    m2es3_avg = 0.
    m2e2s3_avg = 0.
    a0 = np.asarray([0.910563145, 0.636128145, 2.686134789, -26.54736249, 97.75920878, -159.5915409, 91.29777408])
    a1 = np.asarray([-0.308401692, 0.186053116, -2.503004726, 21.41979363, -65.25588533, 83.31868048, -33.74692293])
    a2 = np.asarray([-0.090614835, 0.452784281, 0.596270073, -1.724182913, -4.130211253, 13.77663187, -8.672847037])
    b0 = np.asarray([0.724094694, 2.238279186, -4.002584949, -21.00357682, 26.85564136, 206.5513384, -355.6023561])
    b1 = np.asarray([-0.575549808, 0.699509552, 3.892567339, -17.21547165, 192.6722645, -161.8264617, -165.2076935])
    b2 = np.asarray([0.097688312, -0.255757498, -9.155856153, 20.64207597, -38.80443005, 93.62677408, -29.66690559])

    for i in range(4):
        zeta[i] = np.pi/6.*den*np.sum(x*m*d**i)
    
    for i in range(1,4):
        dzeta_dt[i] = np.pi/6.*den*np.sum(x*m*i*dd_dt*d**(i-1))
        
    eta = zeta[3]
    m_avg = np.sum(x*m)  

    for i in range(ncomp):
        for j in range(ncomp):
            s_ij[i,j] = (s[i] + s[j])/2.
            if type(z) == np.ndarray:
                if z[i]*z[j] <= 0: # for two cations or two anions e_ij is kept at zero to avoid dispersion between like ions (see Held et al. 2014)
                    e_ij[i,j] = np.sqrt(e[i]*e[j])*(1-k_ij[i,j])
            else:
                e_ij[i,j] = np.sqrt(e[i]*e[j])*(1-k_ij[i,j])
            m2es3_avg = m2es3_avg + x[i]*x[j]*m[i]*m[j]*e_ij[i,j]/t*s_ij[i,j]**3
            m2e2s3_avg = m2e2s3_avg + x[i]*x[j]*m[i]*m[j]*(e_ij[i,j]/t)**2*s_ij[i,j]**3  
    
            ghs[i,j] = 1/(1-zeta[3]) + (d[i]*d[j]/(d[i]+d[j]))*3*zeta[2]/(1-zeta[3])**2 + \
                (d[i]*d[j]/(d[i]+d[j]))**2*2*zeta[2]**2/(1-zeta[3])**3
            ddij_dt = (d[i]*d[j]/(d[i]+d[j]))*(dd_dt[i]/d[i]+dd_dt[j]/d[j]-(dd_dt[i]+dd_dt[j])/(d[i]+d[j]))
            dghs_dt[i,j] = dzeta_dt[3]/(1-zeta[3])**2 \
                + 3*(ddij_dt*zeta[2]+(d[i]*d[j]/(d[i]+d[j]))*dzeta_dt[2])/(1-zeta[3])**2 \
                + 4*(d[i]*d[j]/(d[i]+d[j]))*zeta[2]*(1.5*dzeta_dt[3]+ddij_dt*zeta[2] \
                +(d[i]*d[j]/(d[i]+d[j]))*dzeta_dt[2])/(1-zeta[3])**3 \
                + 6*((d[i]*d[j]/(d[i]+d[j]))*zeta[2])**2*dzeta_dt[3]/(1-zeta[3])**4
            
    dadt_hs = 1/zeta[0]*(3*(dzeta_dt[1]*zeta[2] + zeta[1]*dzeta_dt[2])/(1-zeta[3]) \
        + 3*zeta[1]*zeta[2]*dzeta_dt[3]/(1-zeta[3])**2 \
        + 3*zeta[2]**2*dzeta_dt[2]/zeta[3]/(1-zeta[3])**2 \
        + zeta[2]**3*dzeta_dt[3]*(3*zeta[3]-1)/zeta[3]**2/(1-zeta[3])**3 \
        + (3*zeta[2]**2*dzeta_dt[2]*zeta[3] - 2*zeta[2]**3*dzeta_dt[3])/zeta[3]**3 \
        * np.log(1-zeta[3]) \
        + (zeta[0]-zeta[2]**3/zeta[3]**2)*dzeta_dt[3]/(1-zeta[3]))

    a = a0 + (m_avg-1)/m_avg*a1 + (m_avg-1)/m_avg*(m_avg-2)/m_avg*a2
    b = b0 + (m_avg-1)/m_avg*b1 + (m_avg-1)/m_avg*(m_avg-2)/m_avg*b2
    
    idx = np.arange(7)
    I1 = np.sum(a*eta**idx)
    I2 = np.sum(b*eta**idx)
    C1 = 1/(1 + m_avg*(8*eta-2*eta**2)/(1-eta)**4 + (1-m_avg)*(20*eta-27*eta**2+12*eta**3-2*eta**4)/((1-eta)*(2-eta))**2)
    C2 = -1*C1**2*(m_avg*(-4*eta**2+20*eta+8)/(1-eta)**5 + (1-m_avg)*(2*eta**3+12*eta**2-48*eta+40)/((1-eta)*(2-eta))**3)
    dI1_dt = np.sum(a*dzeta_dt[3]*idx*eta**(idx-1))
    dI2_dt =np.sum(b*dzeta_dt[3]*idx*eta**(idx-1))
    dC1_dt = C2*dzeta_dt[3]
    
    summ = 0.    
    
    for i in range(ncomp):
        summ += x[i]*(m[i]-1)*dghs_dt[i,i]/ghs[i,i]

    dadt_hc = m_avg*dadt_hs - summ
    dadt_disp = -2*np.pi*den*(dI1_dt-I1/t)*m2es3_avg - np.pi*den*m_avg*(dC1_dt*I2+C1*dI2_dt-2*C1*I2/t)*m2e2s3_avg

    # Dipole term (Gross and Vrabec term) --------------------------------------
    if type(dipm) != np.ndarray:
        dadt_polar = 0
    else:
        a0dip = np.asarray([0.3043504, -0.1358588, 1.4493329, 0.3556977, -2.0653308])
        a1dip = np.asarray([0.9534641, -1.8396383, 2.0131180, -7.3724958, 8.2374135])
        a2dip = np.asarray([-1.1610080, 4.5258607, 0.9751222, -12.281038, 5.9397575])
        b0dip = np.asarray([0.2187939, -1.1896431, 1.1626889, 0, 0])
        b1dip = np.asarray([-0.5873164, 1.2489132, -0.5085280, 0, 0])
        b2dip = np.asarray([3.4869576, -14.915974, 15.372022, 0, 0])
        c0dip = np.asarray([-0.0646774, 0.1975882, -0.8087562, 0.6902849, 0])
        c1dip = np.asarray([-0.9520876, 2.9924258, -2.3802636, -0.2701261, 0])
        c2dip = np.asarray([-0.6260979, 1.2924686, 1.6542783, -3.4396744, 0])
        
        A2 = 0.
        A3 = 0.
        dA2_dt = 0.
        dA3_dt = 0.
        idxd = np.arange(5)
        if type(dip_num) != np.ndarray:
            dip_num = np.ones_like(x)
        
        conv = 7242.702976750923 # conversion factor, see the note below Table 2 in Gross and Vrabec 2006

        dipmSQ = dipm**2/(m*e*s**3)*conv

        for i in range(ncomp):
            for j in range(ncomp):
                m_ij = np.sqrt(m[i]*m[j])
                if m_ij > 2:
                    m_ij = 2
                adip = a0dip + (m_ij-1)/m_ij*a1dip + (m_ij-1)/m_ij*(m_ij-2)/m_ij*a2dip
                bdip = b0dip + (m_ij-1)/m_ij*b1dip + (m_ij-1)/m_ij*(m_ij-2)/m_ij*b2dip                
                J2 = np.sum(adip + bdip*e_ij[j,j]/t*eta**idxd)
                dJ2_dt = np.sum(-bdip*e_ij[j,j]/t**2*eta**idxd)
                A2 += x[i]*x[j]*e_ij[i,i]/t*e_ij[j,j]/t*s_ij[i,i]**3*s_ij[j,j]**3 \
                    /s_ij[i,j]**3*dip_num[i]*dip_num[j]*dipmSQ[i]*dipmSQ[j]*J2
                dA2_dt += x[i]*x[j]*e_ij[i,i]*e_ij[j,j]*s_ij[i,i]**3*s_ij[j,j]**3 \
                    /s_ij[i,j]**3*dip_num[i]*dip_num[j]*dipmSQ[i]*dipmSQ[j]* \
                    (dJ2_dt/t**2-2*J2/t**3)
                
        for i in range(ncomp):
            for j in range(ncomp):
                for k in range(ncomp):
                    m_ijk = (m[i]*m[j]*m[k])**(1/3.)
                    if m_ijk > 2:
                        m_ijk = 2
                    cdip = c0dip + (m_ij-1)/m_ij*c1dip + (m_ij-1)/m_ij*(m_ij-2)/m_ij*c2dip
                    J3 = np.sum(cdip*eta**idxd)
                    A3 += x[i]*x[j]*x[k]*e_ij[i,i]/t*e_ij[j,j]/t*e_ij[k,k]/t* \
                        s_ij[i,i]**3*s_ij[j,j]**3*s_ij[k,k]**3/s_ij[i,j]/s_ij[i,k] \
                        /s_ij[j,k]*dip_num[i]*dip_num[j]*dip_num[k]*dipmSQ[i] \
                        *dipmSQ[j]*dipmSQ[k]*J3
                    dA3_dt += -3*x[i]*x[j]*x[k]*e_ij[i,i]*e_ij[j,j]*e_ij[k,k]* \
                        s_ij[i,i]**3*s_ij[j,j]**3*s_ij[k,k]**3/s_ij[i,j]/s_ij[i,k] \
                        /s_ij[j,k]*dip_num[i]*dip_num[j]*dip_num[k]*dipmSQ[i] \
                        *dipmSQ[j]*dipmSQ[k]*J3/t**4
        
        A2 = -np.pi*den*A2 
        A3 = -4/3.*np.pi**2*den**2*A3
        dA2_dt = -np.pi*den*dA2_dt
        dA3_dt = -4/3.*np.pi**2*den**2*dA3_dt
        
        dadt_polar = (dA2_dt*(1-A3/A2) + (dA3_dt*A2 - A3*dA2_dt)/A2)/(1-A3/A2)**2
    
    # Association term -------------------------------------------------------
    # only the 2B association type is currently implemented
    if type(e_assoc) != np.ndarray:
        dadt_assoc = 0
    else:
        a_sites = 2
        iA = np.nonzero(e_assoc)[0] #indicies of associating compounds
        ncA = iA.shape[0] # number of associating compounds in the fluid
        XA = np.zeros((ncA,a_sites), dtype='float_')

        eABij = np.zeros((ncA,ncA), dtype='float_')
        volABij = np.zeros_like(eABij)
        delta_ij = np.zeros_like(eABij)
        ddelta_dt = np.zeros_like(eABij)
       
        for i in range(ncA):
            for j in range(ncA):
                eABij[i,j] = (e_assoc[iA[i]]+e_assoc[iA[j]])/2.
                volABij[i,j] = np.sqrt(vol_a[iA[i]]*vol_a[iA[j]])*(np.sqrt(s_ij[iA[i],iA[i]] \
                    *s_ij[iA[j],iA[j]])/(0.5*(s_ij[iA[i],iA[i]]+s_ij[iA[j],iA[j]])))**3
                delta_ij[i,j] = ghs[iA[j],iA[j]]*(np.exp(eABij[i,j]/t)-1)*s_ij[iA[i],iA[j]]**3*volABij[i,j]
                XA[i,:] = (-1 + np.sqrt(1+8*den*delta_ij[i,i]))/(4*den*delta_ij[i,i])
                ddelta_dt[i,j] = s_ij[iA[j],iA[j]]**3*volABij[i,j]*(-eABij[i,j]/t**2 \
                    *np.exp(eABij[i,j]/t)*ghs[iA[j],iA[j]] + dghs_dt[iA[j],iA[j]] \
                    *(np.exp(eABij[i,j]/t)-1))

        ctr = 0
        dif = 1000.
        XA_old = np.copy(XA)
        while (ctr < 500) and (dif > 1e-9):
            ctr += 1
            XA = XA_find(XA, ncA, delta_ij, den, x[iA])
            dif = np.sum(abs(XA - XA_old))
            XA_old[:] = XA
        XA = XA.flatten('F')
        
        dXA_dt = dXAdt_find(ncA, ncomp, delta_ij, den, XA, ddelta_dt, x[iA], a_sites)

        dadt_assoc = 0.
        idx = -1
        for i in range(ncA):
            for j in range(a_sites):
                idx += 1
                dadt_assoc += x[iA[i]]*(1/XA[idx]-0.5)*dXA_dt[idx]

    # Ion term ---------------------------------------------------------------
    if type(z) != np.ndarray:
        dadt_ion = 0
    else:
        if dielc == None:
            ValueError('A value for the dielectric constant of the medium must be given when including the electrolyte term.')

        E_CHRG = 1.6022e-19 # elementary charge, units of coulomb
        perm_vac = 8.85416e-22 #permittivity in vacuum, C V^-1 Angstrom^-1        
        
        q = z*E_CHRG
        
        kappa = np.sqrt(den*E_CHRG**2/kb/t/(dielc*perm_vac)*np.sum(z**2*x)) # the inverse Debye screening length. Equation 4 in Held et al. 2008.
        
        if kappa == 0:
            dadt_ion = 0
        else:
            dkappa_dt = -0.5*den*E_CHRG**2/kb/t**2/(dielc*perm_vac)*np.sum(z**2*x)/kappa           
            chi = 3/(kappa*s)**3*(1.5 + np.log(1+kappa*s) - 2*(1+kappa*s) + \
                0.5*(1+kappa*s)**2)
            dchikap_dk = (s*kappa*(6+3*s*kappa)-(6+6*s*kappa)*np.log(1+s*kappa)) \
                /(s**3*kappa**3 (1+s*kappa))
            
            dadt_ion = -1/12./np.pi/kb/(dielc*perm_vac)*np.sum(x*q**2* \
                (dchikap_dk*dkappa_dt/t-kappa*chi/t**2))
    
    dares_dt = dadt_hc + dadt_disp + dadt_assoc + dadt_polar + dadt_ion
    Z = pcsaft_Z(x, m, s, e, t, rho, k_ij, e_assoc, vol_a, dipm, dip_num, z, dielc)

    hres = (-t*dares_dt + (Z-1))*kb*N_AV*t # Equation A.46 from Gross and Sadowski 2001
    return hres

def pcsaft_sres(x, m, s, e, t, rho, k_ij=None, e_assoc=None, vol_a=None, dipm=None, \
            dip_num=None, z=None, dielc=None):
    """
    Calculate the residual entropy (constant volume) for one phase of the system.
    
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
    gres = pcsaft_gres(x, m, s, e, t, rho, k_ij, e_assoc, vol_a, dipm, dip_num, z, dielc)
    hres = pcsaft_hres(x, m, s, e, t, rho, k_ij, e_assoc, vol_a, dipm, dip_num, z, dielc)

    sres = (hres - gres)/t
    return sres

def pcsaft_gres(x, m, s, e, t, rho, k_ij=None, e_assoc=None, vol_a=None, dipm=None, \
            dip_num=None, z=None, dielc=None):
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
    kb = 1.380648465952442093e-23 # Boltzmann constant, J K^-1
    N_AV = 6.022140857e23 # Avagadro's number
    
    ares = pcsaft_ares(x, m, s, e, t, rho, k_ij, e_assoc, vol_a, dipm, dip_num, z, dielc)
    Z = pcsaft_Z(x, m, s, e, t, rho, k_ij, e_assoc, vol_a, dipm, dip_num, z, dielc)

    gres = (ares + (Z - 1) - np.log(Z))*kb*N_AV*t # Equation A.50 from Gross and Sadowski 2001
    return gres
    

def pcsaft_fugcoef(x, m, s, e, t, rho, k_ij=None, e_assoc=None, vol_a=None, dipm=None, \
            dip_num=None, z=None, dielc=None):
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
    ncomp = x.shape[0] # number of components
    kb = 1.380648465952442093e-23 # Boltzmann constant, J K^-1
    N_AV = 6.022140857e23 # Avagadro's number

    d = s*(1-0.12*np.exp(-3*e/t))
    if type(z) == np.ndarray:
        d[np.where(z != 0)[0]] = s[np.where(z != 0)[0]]*(1-0.12) # for ions the diameter is assumed to be temperature independent (see Held et al. 2014)

    den = rho*N_AV/1.0e30

    if type(k_ij) != np.ndarray:
        k_ij = np.zeros((ncomp,ncomp), dtype='float_')
        
    zeta = np.zeros((4,), dtype='float_')
    ghs = np.zeros((ncomp,ncomp), dtype='float_')
    e_ij = np.zeros_like(ghs)
    s_ij = np.zeros_like(ghs)
    denghs = np.zeros_like(ghs)
    dghs_dx = np.zeros((ncomp,ncomp), dtype='float_')
    dahs_dx = np.zeros_like(x)
    mu = np.zeros_like(x)
    mu_hc = np.zeros_like(x)
    mu_disp = np.zeros_like(x)
    dadisp_dx = np.zeros_like(x)
    dahc_dx = np.zeros_like(x)
    m2es3_avg = 0.
    m2e2s3_avg = 0.
    a0 = np.asarray([0.910563145, 0.636128145, 2.686134789, -26.54736249, 97.75920878, -159.5915409, 91.29777408])
    a1 = np.asarray([-0.308401692, 0.186053116, -2.503004726, 21.41979363, -65.25588533, 83.31868048, -33.74692293])
    a2 = np.asarray([-0.090614835, 0.452784281, 0.596270073, -1.724182913, -4.130211253, 13.77663187, -8.672847037])
    b0 = np.asarray([0.724094694, 2.238279186, -4.002584949, -21.00357682, 26.85564136, 206.5513384, -355.6023561])
    b1 = np.asarray([-0.575549808, 0.699509552, 3.892567339, -17.21547165, 192.6722645, -161.8264617, -165.2076935])
    b2 = np.asarray([0.097688312, -0.255757498, -9.155856153, 20.64207597, -38.80443005, 93.62677408, -29.66690559])

    for i in range(4):
        zeta[i] = np.pi/6.*den*np.sum(x*m*d**i)
        
    eta = zeta[3]
    m_avg = np.sum(x*m)
    
    for i in range(ncomp):
        for j in range(ncomp):
            s_ij[i,j] = (s[i] + s[j])/2.
            if type(z) == np.ndarray:
                if z[i]*z[j] <= 0: # for two cations or two anions e_ij is kept at zero to avoid dispersion between like ions (see Held et al. 2014)
                    e_ij[i,j] = np.sqrt(e[i]*e[j])*(1-k_ij[i,j])
            else:
                e_ij[i,j] = np.sqrt(e[i]*e[j])*(1-k_ij[i,j])
            m2es3_avg = m2es3_avg + x[i]*x[j]*m[i]*m[j]*e_ij[i,j]/t*s_ij[i,j]**3
            m2e2s3_avg = m2e2s3_avg + x[i]*x[j]*m[i]*m[j]*(e_ij[i,j]/t)**2*s_ij[i,j]**3  
    
            ghs[i,j] = 1/(1-zeta[3]) + (d[i]*d[j]/(d[i]+d[j]))*3*zeta[2]/(1-zeta[3])**2 + \
                (d[i]*d[j]/(d[i]+d[j]))**2*2*zeta[2]**2/(1-zeta[3])**3
            denghs[i,j] = zeta[3]/(1-zeta[3])**2 + (d[i]*d[j]/(d[i]+d[j]))*(3*zeta[2]/(1-zeta[3])**2 + \
                6*zeta[2]*zeta[3]/(1-zeta[3])**3) + (d[i]*d[j]/(d[i]+d[j]))**2*(4*zeta[2]**2/(1-zeta[3])**3 + \
                6*zeta[2]**2*zeta[3]/(1-zeta[3])**4)

    ares_hs = 1/zeta[0]*(3*zeta[1]*zeta[2]/(1-zeta[3]) + zeta[2]**3/(zeta[3]*(1-zeta[3])**2) \
            + (zeta[2]**3/zeta[3]**2 - zeta[0])*np.log(1-zeta[3]))

    a = a0 + (m_avg-1)/m_avg*a1 + (m_avg-1)/m_avg*(m_avg-2)/m_avg*a2
    b = b0 + (m_avg-1)/m_avg*b1 + (m_avg-1)/m_avg*(m_avg-2)/m_avg*b2
    
    idx = np.arange(7)
    I1 = np.sum(a*eta**idx)
    I2 = np.sum(b*eta**idx)
    detI1_det = np.sum(a*(idx+1)*eta**idx)
    detI2_det = np.sum(b*(idx+1)*eta**idx)
    C1 = 1/(1 + m_avg*(8*eta-2*eta**2)/(1-eta)**4 + (1-m_avg)*(20*eta-27*eta**2+12*eta**3-2*eta**4)/((1-eta)*(2-eta))**2)
    C2 = -1*C1**2*(m_avg*(-4*eta**2+20*eta+8)/(1-eta)**5 + (1-m_avg)*(2*eta**3+12*eta**2-48*eta+40)/((1-eta)*(2-eta))**3)

    summ = 0.    
    for i in range(ncomp):
        summ += x[i]*(m[i]-1)*np.log(ghs[i,i])

    ares_hc = m_avg*ares_hs - summ
    ares_disp = -2*np.pi*den*I1*m2es3_avg - np.pi*den*m_avg*C1*I2*m2e2s3_avg

    Zhs = zeta[3]/(1-zeta[3]) + 3*zeta[1]*zeta[2]/zeta[0]/(1-zeta[3])**2 + \
        (3*zeta[2]**3 - zeta[3]*zeta[2]**3)/zeta[0]/(1-zeta[3])**3
    
    summ = 0.    
    for i in range(ncomp):
        summ += x[i]*(m[i]-1)/ghs[i,i]*denghs[i,i]

    Zhc = m_avg*Zhs - summ
    Zdisp = -2*np.pi*den*detI1_det*m2es3_avg - np.pi*den*m_avg*(C1*detI2_det + C2*eta*I2)*m2e2s3_avg

    idx2 = np.arange(4)
    for i in range(ncomp):
        dzeta_dx = np.pi/6.*den*m[i]*d[i]**idx2
        for j in range(ncomp):
            dghs_dx[j,i] = dzeta_dx[3]/(1-zeta[3])**2 + (d[j]*d[j]/(d[j]+d[j]))* \
                    (3*dzeta_dx[2]/(1-zeta[3])**2 + 6*zeta[2]*dzeta_dx[3]/(1-zeta[3])**3) \
                    + (d[j]*d[j]/(d[j]+d[j]))**2*(4*zeta[2]*dzeta_dx[2]/(1-zeta[3])**3 \
                    + 6*zeta[2]**2*dzeta_dx[3]/(1-zeta[3])**4)
        dahs_dx[i] = -dzeta_dx[0]/zeta[0]*ares_hs + 1/zeta[0]*(3*(dzeta_dx[1]*zeta[2] \
                + zeta[1]*dzeta_dx[2])/(1-zeta[3]) + 3*zeta[1]*zeta[2]*dzeta_dx[3] \
                /(1-zeta[3])**2 + 3*zeta[2]**2*dzeta_dx[2]/zeta[3]/(1-zeta[3])**2 \
                + zeta[2]**3*dzeta_dx[3]*(3*zeta[3]-1)/zeta[3]**2/(1-zeta[3])**3 \
                + np.log(1-zeta[3])*((3*zeta[2]**2*dzeta_dx[2]*zeta[3] - \
                2*zeta[2]**3*dzeta_dx[3])/zeta[3]**3 - dzeta_dx[0]) + \
                (zeta[0]-zeta[2]**3/zeta[3]**2)*dzeta_dx[3]/(1-zeta[3]))
   
    for i in range(ncomp):
        dzeta3_dx = np.pi/6.*den*m[i]*d[i]**3
        daa_dx = m[i]/m_avg**2*a1 + m[i]/m_avg**2*(3-4/m_avg)*a2        
        db_dx = m[i]/m_avg**2*b1 + m[i]/m_avg**2*(3-4/m_avg)*b2 
        dI1_dx = np.sum(a*idx*dzeta3_dx*eta**(idx-1) + daa_dx*eta**idx)
        dI2_dx = np.sum(b*idx*dzeta3_dx*eta**(idx-1) + db_dx*eta**idx)  
        dm2es3_dx = 2*m[i]*np.sum(x*m*(e_ij[i,:]/t)*s_ij[i,:]**3)
        dm2e2s3_dx = 2*m[i]*np.sum(x*m*(e_ij[i,:]/t)**2*s_ij[i,:]**3)
        dC1_dx = C2*dzeta3_dx - C1**2*(m[i]*(8*eta-2*eta**2)/(1-eta)**4 - \
            m[i]*(20*eta-27*eta**2+12*eta**3-2*eta**4)/((1-eta)*(2-eta))**2)
        
        dahc_dx[i] = m[i]*ares_hs + m_avg*dahs_dx[i] - np.sum(x*(m-1)/np.diagonal(ghs)*dghs_dx[:,i]) \
            - (m[i]-1)*np.log(ghs[i,i])
        dadisp_dx[i] = -2*np.pi*den*(dI1_dx*m2es3_avg + I1*dm2es3_dx) - np.pi*den \
            *((m[i]*C1*I2 + m_avg*dC1_dx*I2 + m_avg*C1*dI2_dx)*m2e2s3_avg \
            + m_avg*C1*I2*dm2e2s3_dx)
   
    for i in range(ncomp):            
        mu_hc[i] = ares_hc + Zhc + dahc_dx[i] - np.sum(x*dahc_dx)
        mu_disp[i] = ares_disp + Zdisp + dadisp_dx[i] - np.sum(x*dadisp_dx)

    # Dipole term (Gross and Vrabec term) --------------------------------------
    if type(dipm) != np.ndarray:
        mu_polar = 0
    else:
        mu_polar = np.zeros_like(x)   
        dapolar_dx = np.zeros_like(x)
        
        A2 = 0.
        A3 = 0.
        dA2_det = 0.
        dA3_det = 0. 
        dA2_dx = np.zeros_like(x)
        dA3_dx = np.zeros_like(x) 
        
        a0dip = np.asarray([0.3043504, -0.1358588, 1.4493329, 0.3556977, -2.0653308])
        a1dip = np.asarray([0.9534641, -1.8396383, 2.0131180, -7.3724958, 8.2374135])
        a2dip = np.asarray([-1.1610080, 4.5258607, 0.9751222, -12.281038, 5.9397575])
        b0dip = np.asarray([0.2187939, -1.1896431, 1.1626889, 0, 0])
        b1dip = np.asarray([-0.5873164, 1.2489132, -0.5085280, 0, 0])
        b2dip = np.asarray([3.4869576, -14.915974, 15.372022, 0, 0])
        c0dip = np.asarray([-0.0646774, 0.1975882, -0.8087562, 0.6902849, 0])
        c1dip = np.asarray([-0.9520876, 2.9924258, -2.3802636, -0.2701261, 0])
        c2dip = np.asarray([-0.6260979, 1.2924686, 1.6542783, -3.4396744, 0])
        
        idxd = np.arange(5)
        if type(dip_num) != np.ndarray:
            dip_num = np.ones_like(x)
        
        conv = 7242.702976750923 # conversion factor, see the note below Table 2 in Gross and Vrabec 2006
       
        dipmSQ = dipm**2/(m*e*s**3)*conv

        for i in range(ncomp):
            for j in range(ncomp):
                m_ij = np.sqrt(m[i]*m[j])
                if m_ij > 2:
                    m_ij = 2
                adip = a0dip + (m_ij-1)/m_ij*a1dip + (m_ij-1)/m_ij*(m_ij-2)/m_ij*a2dip
                bdip = b0dip + (m_ij-1)/m_ij*b1dip + (m_ij-1)/m_ij*(m_ij-2)/m_ij*b2dip                
                J2 = np.sum(adip + bdip*e_ij[j,j]/t*eta**idxd)
                detJ2_det = np.sum(adip + bdip*e_ij[j,j]/t*(idxd+1)*eta**idxd)
                A2 += x[i]*x[j]*e_ij[i,i]/t*e_ij[j,j]/t*s_ij[i,i]**3*s_ij[j,j]**3 \
                    /s_ij[i,j]**3*dip_num[i]*dip_num[j]*dipmSQ[i]*dipmSQ[j]*J2
                dA2_det += x[i]*x[j]*e_ij[i,i]/t*e_ij[j,j]/t*s_ij[i,i]**3 \
                    *s_ij[j,j]**3/s_ij[i,j]**3*dip_num[i]*dip_num[j]*dipmSQ[i]*dipmSQ[j]*detJ2_det
                if i == j:
                    dA2_dx[i] += e_ij[i,i]/t*e_ij[j,j]/t*s_ij[i,i]**3*s_ij[j,j]**3 \
                        /s_ij[i,j]**3*dip_num[i]*dip_num[j]*dipmSQ[i]*dipmSQ[j]* \
                        (x[i]*x[j]*detJ2_det*np.pi/6.*den*m[i]*d[i]**3 + 2*x[j]*J2)                   
                else:
                    dA2_dx[i] += e_ij[i,i]/t*e_ij[j,j]/t*s_ij[i,i]**3*s_ij[j,j]**3 \
                        /s_ij[i,j]**3*dip_num[i]*dip_num[j]*dipmSQ[i]*dipmSQ[j]* \
                        (x[i]*x[j]*detJ2_det*np.pi/6.*den*m[i]*d[i]**3 + x[j]*J2)
                
                for k in range(ncomp):
                    m_ijk = (m[i]*m[j]*m[k])**(1/3.)
                    if m_ijk > 2:
                        m_ijk = 2
                    cdip = c0dip + (m_ij-1)/m_ij*c1dip + (m_ij-1)/m_ij*(m_ij-2)/m_ij*c2dip
                    J3 = np.sum(cdip*eta**idxd)
                    detJ3_det = np.sum(cdip*(idxd+2)*eta**(idxd+1))
                    A3 += x[i]*x[j]*x[k]*e_ij[i,i]/t*e_ij[j,j]/t*e_ij[k,k]/t* \
                        s_ij[i,i]**3*s_ij[j,j]**3*s_ij[k,k]**3/s_ij[i,j]/s_ij[i,k] \
                        /s_ij[j,k]*dip_num[i]*dip_num[j]*dip_num[k]*dipmSQ[i] \
                        *dipmSQ[j]*dipmSQ[k]*J3
                    dA3_det += x[i]*x[j]*x[k]*e_ij[i,i]/t*e_ij[j,j]/t*e_ij[k,k]/t* \
                        s_ij[i,i]**3*s_ij[j,j]**3*s_ij[k,k]**3/s_ij[i,j]/s_ij[i,k] \
                        /s_ij[j,k]*dip_num[i]*dip_num[j]*dip_num[k]*dipmSQ[i] \
                        *dipmSQ[j]*dipmSQ[k]*detJ3_det
                    if (i == j) and (i == k):
                        dA3_dx[i] += e_ij[i,i]/t*e_ij[j,j]/t*e_ij[k,k]/t*s_ij[i,i]**3 \
                            *s_ij[j,j]**3*s_ij[k,k]**3/s_ij[i,j]/s_ij[i,k]/s_ij[j,k] \
                            *dip_num[i]*dip_num[j]*dip_num[k]*dipmSQ[i]*dipmSQ[j] \
                            *dipmSQ[k]*(x[i]*x[j]*x[k]*detJ3_det*np.pi/6.*den*m[i]*d[i]**3 \
                            + 3*x[j]*x[k]*J3)
                    elif (i == j) or (i == k):
                        dA3_dx[i] += e_ij[i,i]/t*e_ij[j,j]/t*e_ij[k,k]/t*s_ij[i,i]**3 \
                            *s_ij[j,j]**3*s_ij[k,k]**3/s_ij[i,j]/s_ij[i,k]/s_ij[j,k] \
                            *dip_num[i]*dip_num[j]*dip_num[k]*dipmSQ[i]*dipmSQ[j] \
                            *dipmSQ[k]*(x[i]*x[j]*x[k]*detJ3_det*np.pi/6.*den*m[i]*d[i]**3 \
                            + 2*x[j]*x[k]*J3)
                    else:
                        dA3_dx[i] += e_ij[i,i]/t*e_ij[j,j]/t*e_ij[k,k]/t*s_ij[i,i]**3 \
                            *s_ij[j,j]**3*s_ij[k,k]**3/s_ij[i,j]/s_ij[i,k]/s_ij[j,k] \
                            *dip_num[i]*dip_num[j]*dip_num[k]*dipmSQ[i]*dipmSQ[j] \
                            *dipmSQ[k]*(x[i]*x[j]*x[k]*detJ3_det*np.pi/6.*den*m[i]*d[i]**3 \
                            + x[j]*x[k]*J3)
                 
        A2 = -np.pi*den*A2   
        A3 = -4/3.*np.pi**2*den**2*A3
        dA2_det = -6./np.sum(x*m*d**3)*dA2_det
        dA3_det = -48./np.sum(x*m*d**3)**2*dA3_det
        dA2_dx = -np.pi*den*dA2_dx
        dA3_dx = -4/3.*np.pi**2*den**2*dA3_dx

        dapolar_dx = (dA2_dx*(1-A3/A2) + (dA3_dx*A2 - A3*dA2_dx)/A2)/(1-A3/A2)**2
        ares_polar = A2/(1-A3/A2)
        Zpolar = eta*((dA2_det*(1-A3/A2)+(dA3_det*A2-A3*dA2_det)/A2)/(1-A3/A2)**2)

        for i in range(ncomp):
            mu_polar[i] = ares_polar + Zpolar + dapolar_dx[i] - np.sum(x*dapolar_dx)    

    # Association term -------------------------------------------------------
    # only the 2B association type is currently implemented
    if type(e_assoc) != np.ndarray:
        mu_assoc = 0
    else:
        a_sites = 2
        iA = np.nonzero(e_assoc)[0] #indicies of associating compounds
        ncA = iA.shape[0] # number of associating compounds in the fluid
        mu_assoc = np.zeros_like(x)
        XA = np.ones((ncA,a_sites), dtype='float_')

        eABij = np.zeros((ncA,ncA), dtype='float_')
        volABij = np.zeros_like(eABij)
        delta_ij = np.zeros_like(eABij)
        ddelta_dd = np.zeros((ncA,ncA,ncomp), dtype='float_')
        
        for i in range(ncA):
            for j in range(ncA):
                eABij[i,j] = (e_assoc[iA[i]]+e_assoc[iA[j]])/2.
                volABij[i,j] = np.sqrt(vol_a[iA[i]]*vol_a[iA[j]])*(np.sqrt(s_ij[iA[i],iA[i]] \
                    *s_ij[iA[j],iA[j]])/(0.5*(s_ij[iA[i],iA[i]]+s_ij[iA[j],iA[j]])))**3
                delta_ij[i,j] = ghs[iA[j],iA[j]]*(np.exp(eABij[i,j]/t)-1)*s_ij[iA[i],iA[j]]**3*volABij[i,j]
                for k in range(ncomp):
                    dghsd_dd = np.pi/6.*m[k]*(d[k]**3/(1-zeta[3])**2 + 3*d[iA[i]]*d[iA[j]] \
                        /(d[iA[i]]+d[iA[j]])*(d[k]**2/(1-zeta[3])**2+2*d[k]**3* \
                        zeta[2]/(1-zeta[3])**3) + 2*(d[iA[i]]*d[iA[j]]/(d[iA[i]]+d[iA[j]]))**2* \
                        (2*d[k]**2*zeta[2]/(1-zeta[3])**3+3*d[k]**3*zeta[2]**2 \
                        /(1-zeta[3])**4))
                    ddelta_dd[i,j,k] = dghsd_dd*(np.exp(eABij[i,j]/t)-1)*s_ij[iA[i],iA[j]]**3*volABij[i,j]
            XA[i,:] = (-1 + np.sqrt(1+8*den*delta_ij[i,i]))/(4*den*delta_ij[i,i])

        ctr = 0
        dif = 1000.
        XA_old = np.copy(XA)
        while (ctr < 500) and (dif > 1e-9):
            ctr += 1
            XA = XA_find(XA, ncA, delta_ij, den, x[iA])
            dif = np.sum(abs(XA - XA_old))
            XA_old[:] = XA
        XA = XA.flatten('F')
      
        dXA_dd = dXA_find(ncA, ncomp, iA, delta_ij, den, XA, ddelta_dd, x[iA], a_sites)

        for i in range(ncomp):
            for j in range(ncA):
                for k in range(a_sites):
                    mu_assoc[i] += x[iA[j]]*den*dXA_dd[(j+1)*k,i]*(1/XA[(j+1)*k]-0.5)
                    
        for i in range(ncA):
            for l in range(a_sites):
                mu_assoc[iA[i]] += np.log(XA[(i+1)*l])-0.5*XA[(i+1)*l]
            mu_assoc[iA[i]] += 0.5*a_sites

    # Ion term ---------------------------------------------------------------
    if type(z) != np.ndarray:
        mu_ion = 0
    else:
        if dielc == None:
            ValueError('A value for the dielectric constant of the medium must be given when including the electrolyte term.')

        E_CHRG = 1.6022e-19 # elementary charge, units of coulomb
        perm_vac = 8.85416e-22 #permittivity in vacuum, C V^-1 Angstrom^-1

        mu_ion = np.zeros_like(x)
        q = z*E_CHRG
        
        kappa = np.sqrt(den*E_CHRG**2/kb/t/(dielc*perm_vac)*np.sum(z**2*x)) # the inverse Debye screening length. Equation 4 in Held et al. 2008.

        if kappa != 0:       
            chi = 3/(kappa*s)**3*(1.5 + np.log(1+kappa*s) - 2*(1+kappa*s) + \
                0.5*(1+kappa*s)**2)
            
            for i in range(ncomp):
                sigma_k = -2*chi+3/(1+kappa*s)
                mu_ion[i] = -q[i]**2*kappa/24./np.pi/kb/t/(dielc*perm_vac)* \
                    (2*chi[i] + np.sum(x*q**2*sigma_k)/np.sum(x*q**2))

    mu = mu_hc + mu_disp + mu_polar + mu_assoc + mu_ion
    Z = pcsaft_Z(x, m, s, e, t, rho, k_ij, e_assoc, vol_a, dipm, dip_num, z, dielc)

    fugcoef = np.exp(mu - np.log(Z)) # the fugacity coefficients    
    return fugcoef
    

def pcsaft_Z(x, m, s, e, t, rho, k_ij=None, e_assoc=None, vol_a=None, dipm=None, \
            dip_num=None, z=None, dielc=None):
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
    ncomp = x.shape[0] # number of components
    kb = 1.380648465952442093e-23 # Boltzmann constant, J K^-1
    N_AV = 6.022140857e23 # Avagadro's number
    
    d = s*(1-0.12*np.exp(-3*e/t))
    if type(z) == np.ndarray:
        d[np.where(z != 0)[0]] = s[np.where(z != 0)[0]]*(1-0.12) # for ions the diameter is assumed to be temperature independent (see Held et al. 2014)
    
    den = rho*N_AV/1.0e30

    if type(k_ij) != np.ndarray:
        k_ij = np.zeros((ncomp,ncomp), dtype='float_')

    zeta = np.zeros((4,), dtype='float_')
    ghs = np.zeros((ncomp,ncomp), dtype='float_')
    denghs = np.zeros_like(ghs)
    e_ij = np.zeros_like(ghs)
    s_ij = np.zeros_like(ghs)
    m2es3_avg = 0.
    m2e2s3_avg = 0.
    a0 = np.asarray([0.910563145, 0.636128145, 2.686134789, -26.54736249, 97.75920878, -159.5915409, 91.29777408])
    a1 = np.asarray([-0.308401692, 0.186053116, -2.503004726, 21.41979363, -65.25588533, 83.31868048, -33.74692293])
    a2 = np.asarray([-0.090614835, 0.452784281, 0.596270073, -1.724182913, -4.130211253, 13.77663187, -8.672847037])
    b0 = np.asarray([0.724094694, 2.238279186, -4.002584949, -21.00357682, 26.85564136, 206.5513384, -355.6023561])
    b1 = np.asarray([-0.575549808, 0.699509552, 3.892567339, -17.21547165, 192.6722645, -161.8264617, -165.2076935])
    b2 = np.asarray([0.097688312, -0.255757498, -9.155856153, 20.64207597, -38.80443005, 93.62677408, -29.66690559])

    for i in range(4):
        zeta[i] = np.pi/6.*den*np.sum(x*m*d**i)

    eta = zeta[3]
    m_avg = np.sum(x*m)

    for i in range(ncomp):
        for j in range(ncomp):
            s_ij[i,j] = (s[i] + s[j])/2.
            if type(z) == np.ndarray:
                if z[i]*z[j] <= 0: # for two cations or two anions e_ij is kept at zero to avoid dispersion between like ions (see Held et al. 2014)
                    e_ij[i,j] = np.sqrt(e[i]*e[j])*(1-k_ij[i,j])
            else:
                e_ij[i,j] = np.sqrt(e[i]*e[j])*(1-k_ij[i,j])
            m2es3_avg = m2es3_avg + x[i]*x[j]*m[i]*m[j]*e_ij[i,j]/t*s_ij[i,j]**3
            m2e2s3_avg = m2e2s3_avg + x[i]*x[j]*m[i]*m[j]*(e_ij[i,j]/t)**2*s_ij[i,j]**3  
    
            ghs[i,j] = 1/(1-zeta[3]) + (d[i]*d[j]/(d[i]+d[j]))*3*zeta[2]/(1-zeta[3])**2 + \
                (d[i]*d[j]/(d[i]+d[j]))**2*2*zeta[2]**2/(1-zeta[3])**3
            denghs[i,j] = zeta[3]/(1-zeta[3])**2 + (d[i]*d[j]/(d[i]+d[j]))*(3*zeta[2]/(1-zeta[3])**2 + \
                6*zeta[2]*zeta[3]/(1-zeta[3])**3) + (d[i]*d[j]/(d[i]+d[j]))**2*(4*zeta[2]**2/(1-zeta[3])**3 + \
                6*zeta[2]**2*zeta[3]/(1-zeta[3])**4)

    Zhs = zeta[3]/(1-zeta[3]) + 3*zeta[1]*zeta[2]/zeta[0]/(1-zeta[3])**2 + \
        (3*zeta[2]**3 - zeta[3]*zeta[2]**3)/zeta[0]/(1-zeta[3])**3

    a = a0 + (m_avg-1)/m_avg*a1 + (m_avg-1)/m_avg*(m_avg-2)/m_avg*a2
    b = b0 + (m_avg-1)/m_avg*b1 + (m_avg-1)/m_avg*(m_avg-2)/m_avg*b2
    
    idx = np.arange(7)
    detI1_det = np.sum(a*(idx+1)*eta**idx)
    detI2_det = np.sum(b*(idx+1)*eta**idx)
    I2 = np.sum(b*eta**idx)
    C1 = 1/(1 + m_avg*(8*eta-2*eta**2)/(1-eta)**4 + (1-m_avg)*(20*eta-27*eta**2+12*eta**3-2*eta**4)/((1-eta)*(2-eta))**2)
    C2 = -1*C1**2*(m_avg*(-4*eta**2+20*eta+8)/(1-eta)**5 + (1-m_avg)*(2*eta**3+12*eta**2-48*eta+40)/((1-eta)*(2-eta))**3)
    
    summ = 0.    
    
    for i in range(ncomp):
        summ += x[i]*(m[i]-1)/ghs[i,i]*denghs[i,i]

    Zid = 1
    Zhc = m_avg*Zhs - summ
    Zdisp = -2*np.pi*den*detI1_det*m2es3_avg - np.pi*den*m_avg*(C1*detI2_det + C2*eta*I2)*m2e2s3_avg

    # Dipole term (Gross and Vrabec term) --------------------------------------
    if type(dipm) != np.ndarray:
        Zpolar = 0
    else:
        A2 = 0.
        A3 = 0.
        dA2_det = 0.
        dA3_det = 0.
        
        a0dip = np.asarray([0.3043504, -0.1358588, 1.4493329, 0.3556977, -2.0653308])
        a1dip = np.asarray([0.9534641, -1.8396383, 2.0131180, -7.3724958, 8.2374135])
        a2dip = np.asarray([-1.1610080, 4.5258607, 0.9751222, -12.281038, 5.9397575])
        b0dip = np.asarray([0.2187939, -1.1896431, 1.1626889, 0, 0])
        b1dip = np.asarray([-0.5873164, 1.2489132, -0.5085280, 0, 0])
        b2dip = np.asarray([3.4869576, -14.915974, 15.372022, 0, 0])
        c0dip = np.asarray([-0.0646774, 0.1975882, -0.8087562, 0.6902849, 0])
        c1dip = np.asarray([-0.9520876, 2.9924258, -2.3802636, -0.2701261, 0])
        c2dip = np.asarray([-0.6260979, 1.2924686, 1.6542783, -3.4396744, 0])        

        idxd = np.arange(5)
        conv = 7242.702976750923 # conversion factor, see the note below Table 2 in Gross and Vrabec 2006
       
        dipmSQ = dipm**2/(m*e*s**3)*conv

        for i in range(ncomp):
            for j in range(ncomp):
                m_ij = np.sqrt(m[i]*m[j])
                if m_ij > 2:
                    m_ij = 2
                adip = a0dip + (m_ij-1)/m_ij*a1dip + (m_ij-1)/m_ij*(m_ij-2)/m_ij*a2dip
                bdip = b0dip + (m_ij-1)/m_ij*b1dip + (m_ij-1)/m_ij*(m_ij-2)/m_ij*b2dip                
                J2 = np.sum((adip + bdip*e_ij[j,j]/t)*eta**idxd)         
                detJ2_det = np.sum((adip + bdip*e_ij[j,j]/t)*(idxd+1)*eta**idxd)
                A2 += x[i]*x[j]*e_ij[i,i]/t*e_ij[j,j]/t*s_ij[i,i]**3*s_ij[j,j]**3 \
                    /s_ij[i,j]**3*dip_num[i]*dip_num[j]*dipmSQ[i]*dipmSQ[j]*J2
                dA2_det += x[i]*x[j]*e_ij[i,i]/t*e_ij[j,j]/t*s_ij[i,i]**3 \
                    *s_ij[j,j]**3/s_ij[i,j]**3*dip_num[i]*dip_num[j]*dipmSQ[i]*dipmSQ[j]*detJ2_det
                
        for i in range(ncomp):
            for j in range(ncomp):
                for k in range(ncomp):
                    m_ijk = (m[i]*m[j]*m[k])**(1/3.)
                    if m_ijk > 2:
                        m_ijk = 2
                    cdip = c0dip + (m_ij-1)/m_ij*c1dip + (m_ij-1)/m_ij*(m_ij-2)/m_ij*c2dip
                    J3 = np.sum(cdip*eta**idxd)
                    detJ3_det = np.sum(cdip*(idxd+2)*eta**(idxd+1))
                    A3 += x[i]*x[j]*x[k]*e_ij[i,i]/t*e_ij[j,j]/t*e_ij[k,k]/t* \
                        s_ij[i,i]**3*s_ij[j,j]**3*s_ij[k,k]**3/s_ij[i,j]/s_ij[i,k] \
                        /s_ij[j,k]*dip_num[i]*dip_num[j]*dip_num[k]*dipmSQ[i] \
                        *dipmSQ[j]*dipmSQ[k]*J3
                    dA3_det += x[i]*x[j]*x[k]*e_ij[i,i]/t*e_ij[j,j]/t*e_ij[k,k]/t* \
                        s_ij[i,i]**3*s_ij[j,j]**3*s_ij[k,k]**3/s_ij[i,j]/s_ij[i,k] \
                        /s_ij[j,k]*dip_num[i]*dip_num[j]*dip_num[k]*dipmSQ[i] \
                        *dipmSQ[j]*dipmSQ[k]*detJ3_det

        A2 = -np.pi*den*A2   
        A3 = -4/3.*np.pi**2*den**2*A3
        dA2_det = -6./np.sum(x*m*d**3)*dA2_det
        dA3_det = -48./np.sum(x*m*d**3)**2*dA3_det
       
        Zpolar = eta*((dA2_det*(1-A3/A2)+(dA3_det*A2-A3*dA2_det)/A2)/(1-A3/A2)**2)
    
    # Association term -------------------------------------------------------
    # only the 2B association type is currently implemented
    if type(e_assoc) != np.ndarray:
        Zassoc = 0
    else:
        a_sites = 2
        iA = np.nonzero(e_assoc)[0] #indicies of associating compounds
        ncA = iA.shape[0] # number of associating compounds in the fluid
        XA = np.zeros((ncA,a_sites), dtype='float_')

        eABij = np.zeros((ncA,ncA), dtype='float_')
        volABij = np.zeros_like(eABij)
        delta_ij = np.zeros_like(eABij)
        ddelta_dd = np.zeros((ncA,ncA,ncomp), dtype='float_')
       
        for i in range(ncA):
            for j in range(ncA):
                eABij[i,j] = (e_assoc[iA[i]]+e_assoc[iA[j]])/2.
                volABij[i,j] = np.sqrt(vol_a[iA[i]]*vol_a[iA[j]])*(np.sqrt(s_ij[iA[i],iA[i]] \
                    *s_ij[iA[j],iA[j]])/(0.5*(s_ij[iA[i],iA[i]]+s_ij[iA[j],iA[j]])))**3
                delta_ij[i,j] = ghs[iA[j],iA[j]]*(np.exp(eABij[i,j]/t)-1)*s_ij[iA[i],iA[j]]**3*volABij[i,j]              
                for k in range(ncomp):
                    dghsd_dd = np.pi/6.*m[k]*(d[k]**3/(1-zeta[3])**2 + 3*d[iA[i]]*d[iA[j]] \
                        /(d[iA[i]]+d[iA[j]])*(d[k]**2/(1-zeta[3])**2+2*d[k]**3* \
                        zeta[2]/(1-zeta[3])**3) + 2*(d[iA[i]]*d[iA[j]]/(d[iA[i]]+d[iA[j]]))**2* \
                        (2*d[k]**2*zeta[2]/(1-zeta[3])**3+3*d[k]**3*zeta[2]**2 \
                        /(1-zeta[3])**4))
                    ddelta_dd[i,j,k] = dghsd_dd*(np.exp(eABij[i,j]/t)-1)*s_ij[iA[i],iA[j]]**3*volABij[i,j]
            XA[i,:] = (-1 + np.sqrt(1+8*den*delta_ij[i,i]))/(4*den*delta_ij[i,i])
     
        ctr = 0
        dif = 1000.
        XA_old = np.copy(XA)
        while (ctr < 500) and (dif > 1e-9):
            ctr += 1
            XA = XA_find(XA, ncA, delta_ij, den, x[iA])
            dif = np.sum(abs(XA - XA_old))
            XA_old[:] = XA
        XA = XA.flatten('F')
        
        dXA_dd = dXA_find(ncA, ncomp, iA, delta_ij, den, XA, ddelta_dd, x[iA], a_sites)

        summ = 0.
        for i in range(ncomp):
            for j in range(ncA):
                for k in range(a_sites):
                    summ += x[i]*den*x[iA[j]]*(1/XA[(j+1)*k]-0.5)*dXA_dd[(j+1)*k,i]

        Zassoc = summ

    # Ion term ---------------------------------------------------------------
    if type(z) != np.ndarray:
        Zion = 0
    else:
        if dielc == None:
            raise ValueError('A value for the dielectric constant of the medium must be given when including the electrolyte term.')

        E_CHRG = 1.6022e-19 # elementary charge, units of coulomb
        perm_vac = 8.85416e-22 #permittivity in vacuum, C V^-1 Angstrom^-1        
        
        q = z*E_CHRG
        
        kappa = np.sqrt(den*E_CHRG**2/kb/t/(dielc*perm_vac)*np.sum(z**2*x)) # the inverse Debye screening length. Equation 4 in Held et al. 2008.
        if kappa == 0:
            Zion = 0
        else:
            chi = 3/(kappa*s)**3*(1.5 + np.log(1+kappa*s) - 2*(1+kappa*s) + \
                0.5*(1+kappa*s)**2)
            
            sigma_k = -2*chi+3/(1+kappa*s)
            Zion = -1*kappa/24./np.pi/kb/t/(dielc*perm_vac)*np.sum(x*q**2*sigma_k)

    Z = Zid + Zhc + Zdisp + Zpolar + Zassoc + Zion
    return Z


def pcsaft_ares(x, m, s, e, t, rho, k_ij=None, e_assoc=None, vol_a=None, dipm=None, \
            dip_num=None, z=None, dielc=None):
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
    ncomp = x.shape[0] # number of components
    kb = 1.380648465952442093e-23 # Boltzmann constant, J K^-1
    N_AV = 6.022140857e23 # Avagadro's number
    
    d = s*(1-0.12*np.exp(-3*e/t))
    if type(z) == np.ndarray:
        d[np.where(z != 0)[0]] = s[np.where(z != 0)[0]]*(1-0.12) # for ions the diameter is assumed to be temperature independent (see Held et al. 2014)
    
    den = rho*N_AV/1.0e30

    if type(k_ij) != np.ndarray:
        k_ij = np.zeros((ncomp,ncomp), dtype='float_')

    zeta = np.zeros((4,), dtype='float_')
    ghs = np.zeros((ncomp,ncomp), dtype='float_')
    e_ij = np.zeros_like(ghs)
    s_ij = np.zeros_like(ghs)
    m2es3_avg = 0.
    m2e2s3_avg = 0.
    a0 = np.asarray([0.910563145, 0.636128145, 2.686134789, -26.54736249, 97.75920878, -159.5915409, 91.29777408])
    a1 = np.asarray([-0.308401692, 0.186053116, -2.503004726, 21.41979363, -65.25588533, 83.31868048, -33.74692293])
    a2 = np.asarray([-0.090614835, 0.452784281, 0.596270073, -1.724182913, -4.130211253, 13.77663187, -8.672847037])
    b0 = np.asarray([0.724094694, 2.238279186, -4.002584949, -21.00357682, 26.85564136, 206.5513384, -355.6023561])
    b1 = np.asarray([-0.575549808, 0.699509552, 3.892567339, -17.21547165, 192.6722645, -161.8264617, -165.2076935])
    b2 = np.asarray([0.097688312, -0.255757498, -9.155856153, 20.64207597, -38.80443005, 93.62677408, -29.66690559])

    for i in range(4):
        zeta[i] = np.pi/6.*den*np.sum(x*m*d**i)
        
    eta = zeta[3]
    m_avg = np.sum(x*m)  

    for i in range(ncomp):
        for j in range(ncomp):
            s_ij[i,j] = (s[i] + s[j])/2.
            if type(z) == np.ndarray:
                if z[i]*z[j] <= 0: # for two cations or two anions e_ij is kept at zero to avoid dispersion between like ions (see Held et al. 2014)
                    e_ij[i,j] = np.sqrt(e[i]*e[j])*(1-k_ij[i,j])
            else:
                e_ij[i,j] = np.sqrt(e[i]*e[j])*(1-k_ij[i,j])
            m2es3_avg = m2es3_avg + x[i]*x[j]*m[i]*m[j]*e_ij[i,j]/t*s_ij[i,j]**3
            m2e2s3_avg = m2e2s3_avg + x[i]*x[j]*m[i]*m[j]*(e_ij[i,j]/t)**2*s_ij[i,j]**3  
    
            ghs[i,j] = 1/(1-zeta[3]) + (d[i]*d[j]/(d[i]+d[j]))*3*zeta[2]/(1-zeta[3])**2 + \
                (d[i]*d[j]/(d[i]+d[j]))**2*2*zeta[2]**2/(1-zeta[3])**3

    ares_hs = 1/zeta[0]*(3*zeta[1]*zeta[2]/(1-zeta[3]) + zeta[2]**3/(zeta[3]*(1-zeta[3])**2) \
            + (zeta[2]**3/zeta[3]**2 - zeta[0])*np.log(1-zeta[3]))

    a = a0 + (m_avg-1)/m_avg*a1 + (m_avg-1)/m_avg*(m_avg-2)/m_avg*a2
    b = b0 + (m_avg-1)/m_avg*b1 + (m_avg-1)/m_avg*(m_avg-2)/m_avg*b2
    
    idx = np.arange(7)
    I1 = np.sum(a*eta**idx)
    I2 = np.sum(b*eta**idx)
    C1 = 1/(1 + m_avg*(8*eta-2*eta**2)/(1-eta)**4 + (1-m_avg)*(20*eta-27*eta**2+12*eta**3-2*eta**4)/((1-eta)*(2-eta))**2)
    
    summ = 0.    
    
    for i in range(ncomp):
        summ += x[i]*(m[i]-1)*np.log(ghs[i,i])

    ares_hc = m_avg*ares_hs - summ
    ares_disp = -2*np.pi*den*I1*m2es3_avg - np.pi*den*m_avg*C1*I2*m2e2s3_avg

    # Dipole term (Gross and Vrabec term) --------------------------------------
    if type(dipm) != np.ndarray:
        ares_polar = 0
    else:
        a0dip = np.asarray([0.3043504, -0.1358588, 1.4493329, 0.3556977, -2.0653308])
        a1dip = np.asarray([0.9534641, -1.8396383, 2.0131180, -7.3724958, 8.2374135])
        a2dip = np.asarray([-1.1610080, 4.5258607, 0.9751222, -12.281038, 5.9397575])
        b0dip = np.asarray([0.2187939, -1.1896431, 1.1626889, 0, 0])
        b1dip = np.asarray([-0.5873164, 1.2489132, -0.5085280, 0, 0])
        b2dip = np.asarray([3.4869576, -14.915974, 15.372022, 0, 0])
        c0dip = np.asarray([-0.0646774, 0.1975882, -0.8087562, 0.6902849, 0])
        c1dip = np.asarray([-0.9520876, 2.9924258, -2.3802636, -0.2701261, 0])
        c2dip = np.asarray([-0.6260979, 1.2924686, 1.6542783, -3.4396744, 0])
        
        A2 = 0.
        A3 = 0.
        idxd = np.arange(5)
        if type(dip_num) != np.ndarray:
            dip_num = np.ones_like(x)
        
        conv = 7242.702976750923 # conversion factor, see the note below Table 2 in Gross and Vrabec 2006

        dipmSQ = dipm**2/(m*e*s**3)*conv

        for i in range(ncomp):
            for j in range(ncomp):
                m_ij = np.sqrt(m[i]*m[j])
                if m_ij > 2:
                    m_ij = 2
                adip = a0dip + (m_ij-1)/m_ij*a1dip + (m_ij-1)/m_ij*(m_ij-2)/m_ij*a2dip
                bdip = b0dip + (m_ij-1)/m_ij*b1dip + (m_ij-1)/m_ij*(m_ij-2)/m_ij*b2dip                
                J2 = np.sum(adip + bdip*e_ij[j,j]/t*eta**idxd)         
                A2 += x[i]*x[j]*e_ij[i,i]/t*e_ij[j,j]/t*s_ij[i,i]**3*s_ij[j,j]**3 \
                    /s_ij[i,j]**3*dip_num[i]*dip_num[j]*dipmSQ[i]*dipmSQ[j]*J2
                
        for i in range(ncomp):
            for j in range(ncomp):
                for k in range(ncomp):
                    m_ijk = (m[i]*m[j]*m[k])**(1/3.)
                    if m_ijk > 2:
                        m_ijk = 2
                    cdip = c0dip + (m_ij-1)/m_ij*c1dip + (m_ij-1)/m_ij*(m_ij-2)/m_ij*c2dip
                    J3 = np.sum(cdip*eta**idxd)
                    A3 += x[i]*x[j]*x[k]*e_ij[i,i]/t*e_ij[j,j]/t*e_ij[k,k]/t* \
                        s_ij[i,i]**3*s_ij[j,j]**3*s_ij[k,k]**3/s_ij[i,j]/s_ij[i,k] \
                        /s_ij[j,k]*dip_num[i]*dip_num[j]*dip_num[k]*dipmSQ[i] \
                        *dipmSQ[j]*dipmSQ[k]*J3
        
        A2 = -np.pi*den*A2 
        A3 = -4/3.*np.pi**2*den**2*A3
        
        ares_polar = A2/(1-A3/A2)
    
    # Association term -------------------------------------------------------
    # only the 2B association type is currently implemented
    if type(e_assoc) != np.ndarray:
        ares_assoc = 0
    else:
        a_sites = 2
        iA = np.nonzero(e_assoc)[0] #indicies of associating compounds
        ncA = iA.shape[0] # number of associating compounds in the fluid
        XA = np.zeros((ncA,a_sites), dtype='float_')

        eABij = np.zeros((ncA,ncA), dtype='float_')
        volABij = np.zeros_like(eABij)
        delta_ij = np.zeros_like(eABij)
       
        for i in range(ncA):
            for j in range(ncA):
                eABij[i,j] = (e_assoc[iA[i]]+e_assoc[iA[j]])/2.
                volABij[i,j] = np.sqrt(vol_a[iA[i]]*vol_a[iA[j]])*(np.sqrt(s_ij[iA[i],iA[i]] \
                    *s_ij[iA[j],iA[j]])/(0.5*(s_ij[iA[i],iA[i]]+s_ij[iA[j],iA[j]])))**3
                delta_ij[i,j] = ghs[iA[j],iA[j]]*(np.exp(eABij[i,j]/t)-1)*s_ij[iA[i],iA[j]]**3*volABij[i,j]
                XA[i,:] = (-1 + np.sqrt(1+8*den*delta_ij[i,i]))/(4*den*delta_ij[i,i])

        ctr = 0
        dif = 1000.
        XA_old = np.copy(XA)
        while (ctr < 500) and (dif > 1e-9):
            ctr += 1
            XA = XA_find(XA, ncA, delta_ij, den, x[iA])
            dif = np.sum(abs(XA - XA_old))
            XA_old[:] = XA
        XA = XA.flatten('F')
        
        summ = 0.
        for i in range(ncA):
            for j in range(ncA):
                for k in range(a_sites):
                    summ += x[iA[i]]*(np.log(XA[(j+1)*k])-0.5*XA[(j+1)*k])
            summ += 0.5*a_sites
        
        ares_assoc = summ

    # Ion term ---------------------------------------------------------------
    if type(z) != np.ndarray:
        ares_ion = 0
    else:
        if dielc == None:
            ValueError('A value for the dielectric constant of the medium must be given when including the electrolyte term.')

        E_CHRG = 1.6022e-19 # elementary charge, units of coulomb
        perm_vac = 8.85416e-22 #permittivity in vacuum, C V^-1 Angstrom^-1        
        
        q = z*E_CHRG
        
        kappa = np.sqrt(den*E_CHRG**2/kb/t/(dielc*perm_vac)*np.sum(z**2*x)) # the inverse Debye screening length. Equation 4 in Held et al. 2008.
        
        if kappa == 0:
            ares_ion = 0
        else:
            chi = 3/(kappa*s)**3*(1.5 + np.log(1+kappa*s) - 2*(1+kappa*s) + \
                0.5*(1+kappa*s)**2)        
            
            ares_ion = -1/12./np.pi/kb/t/(dielc*perm_vac)*np.sum(x*q**2*kappa*chi)
   
    ares = ares_hc + ares_disp + ares_polar + ares_assoc + ares_ion
    return ares


def XA_find(XA_guess, ncomp, delta_ij, den, x):
    """Iterate over this function in order to solve for XA"""
    n_sites = int(XA_guess.shape[1]/ncomp)    
    ABmatrix = np.asarray([[0.,1.],
                           [1.,0.]]) # using this matrix ensures that A-A and B-B associations are set to zero
    summ2 = np.zeros((n_sites,), dtype='float_')
    XA = np.zeros_like(XA_guess)

    for i in range(ncomp):
        summ2 = 0*summ2
        for j in range(ncomp):
            summ2 += den*x[j]*np.matmul(XA_guess[j,:],delta_ij[i,j]*ABmatrix)
        XA[i,:] = 1/(1+summ2)
    
    return XA
    
    
def dXA_find(ncA, ncomp, iA, delta_ij, den, XA, ddelta_dd, x, n_sites):
    """Solve for the derivative of XA with respect to density."""
    B = np.zeros((n_sites*ncA*ncomp,), dtype='float_')
    A = np.zeros((n_sites*ncA*ncomp,n_sites*ncA*ncomp), dtype='float_')

    indx4 = -1
    indx3 = -1
    for i in range(ncomp):
        indx1 = -1
        if i in iA:
            indx4 += 1        
        for j in range(ncA):
            for m in range(n_sites):
                indx1 = indx1 + 1
                indx3 = indx3 + 1
                indx2 = -1
                sum1 = 0
                for k in range(ncA):
                    for l in range(n_sites):
                        indx2 = indx2 + 1
                        (m+1)*(l+1)-1
                        sum1 = sum1 + den*x[k]*(XA[indx2]*ddelta_dd[j,k,i]*((indx1+indx2)%2))  # (indx1+indx2)%2 ensures that A-A and B-B associations are set to zero
                        A[indx1+(i)*n_sites*ncA,indx2+(i)*n_sites*ncA] = \
                        A[indx1+(i)*n_sites*ncA,indx2+(i)*n_sites*ncA] + \
                        XA[indx1]**2*den*x[k]*delta_ij[j,k]*((indx1+indx2)%2)
                
                sum2 = 0
                if i in iA:
                    for k in range(n_sites):
                        sum2 = sum2 + XA[n_sites*(indx4)+k]*delta_ij[indx4,j]*((indx1+k)%2)

                A[indx3,indx3] = A[indx3,indx3] + 1
                B[indx3] = -1*XA[indx1]**2*(sum1 + sum2)

    dXA_dd = np.linalg.solve(A, B) #Solves linear system of equations
    dXA_dd = np.reshape(dXA_dd, (-1,ncomp), order='F')
    return dXA_dd


def dXAdt_find(ncA, ncomp, delta_ij, den, XA, ddelta_dt, x, n_sites):
    """Solve for the derivative of XA with respect to temperature."""
    B = np.zeros((n_sites*ncA,), dtype='float_')
    A = np.zeros((n_sites*ncA,n_sites*ncA), dtype='float_')

    i_out = -1 # index of outer iteration loop (follows row of matricies)
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


def bubblePfit(p_guess, xv_guess, x, m, s, e, t, kwargs):
    """Minimize this function to calculate the bubble point pressure."""
    if not ('z' in kwargs): # Check that the mixture does not contain electrolytes. For electrolytes, a different equilibrium criterion should be used. 
        rho = pcsaft_den(x, m, s, e, t, p_guess, phase='liq', **kwargs)        
        fugcoef_l = pcsaft_fugcoef(x, m, s, e, t, rho, **kwargs)
        
        # internal iteration loop for vapor phase composition
        itr = 0
        dif = 10000.
        xv = np.copy(xv_guess)
        xv_old = np.zeros_like(xv)
        while (dif>1e-9) and (itr<100):
            xv_old[:] = xv
            rho = pcsaft_den(xv, m, s, e, t, p_guess, phase='vap', **kwargs)        
            fugcoef_v = pcsaft_fugcoef(xv, m, s, e, t, rho, **kwargs)
            xv = fugcoef_l*x/fugcoef_v
            xv = xv/np.sum(xv)
            dif = np.sum(abs(xv - xv_old))
            itr += 1
        error = np.sum((fugcoef_l*x - fugcoef_v*xv)**2)
    else:
        z = kwargs['z']
        rho = pcsaft_den(x, m, s, e, t, p_guess, phase='liq', **kwargs)        
        fugcoef_l = pcsaft_fugcoef(x, m, s, e, t, rho, **kwargs)        
        
        # internal iteration loop for vapor phase composition
        itr = 0
        dif = 10000.
        xv = np.copy(xv_guess)
        xv_old = np.zeros_like(xv)
        while (dif>1e-9) and (itr<100):
            xv_old[:] = xv
            rho = pcsaft_den(xv_guess, m, s, e, t, p_guess, phase='vap', **kwargs)        
            fugcoef_v = pcsaft_fugcoef(xv_guess, m, s, e, t, rho, **kwargs)        
        
            xv[np.where(z == 0)[0]] = (fugcoef_l*x/fugcoef_v)[np.where(z == 0)[0]] # here it is assumed that the ionic compounds are nonvolatile
            xv = xv/np.sum(xv)
            dif = np.sum(abs(xv - xv_old))
            itr += 1

        error = (fugcoef_l*x - fugcoef_v*xv)**2
        error = np.sum(error[np.where(z == 0)[0]]) # again, ionic components are considered to be nonvolatile and are not taken into account

    if np.isnan(error):
        error = 100000000.
    return error
    
def vaporPfit(p_guess, x, m, s, e, t, kwargs):
    """Minimize this function to calculate the vapor pressure."""
    if p_guess <= 0:
        error = 10000000000.
    else:
        rho = pcsaft_den(x, m, s, e, t, p_guess, phase='liq', **kwargs)        
        fugcoef_l = pcsaft_fugcoef(x, m, s, e, t, rho, **kwargs)
        rho = pcsaft_den(x, m, s, e, t, p_guess, phase='vap', **kwargs)
        fugcoef_v = pcsaft_fugcoef(x, m, s, e, t, rho, **kwargs)
        error = 100000*np.sum((fugcoef_l-fugcoef_v)**2)
        if np.isnan(error):
            error = 100000000.
    return error

def denfit(rho_guess, x, m, s, e, t, p, kwargs):
    """Minimize this function to calculate the density."""
    P_fit = pcsaft_p(x, m, s, e, t, rho_guess, **kwargs)
    return np.sum(((P_fit-p)/p*100)**2)
    
def PTzfit(p_guess, x_guess, beta_guess, mol, vol, x_total, m, s, e, t, kwargs):
    """Minimize this function to solve for the pressure to compare with PTz data."""
    if not ('z' in kwargs): # Check that the mixture does not contain electrolytes. For electrolytes, a different equilibrium criterion should be used.
        # internal iteration loop to solve for compositions
        itr = 0
        dif = 10000.
        xl = np.copy(x_guess)
        beta = beta_guess
        xv = (mol*x_total - (1-beta)*mol*xl)/beta/mol
        while (dif>1e-9) and (itr<100):
            beta_old = beta
            rhol = pcsaft_den(xl, m, s, e, t, p_guess, phase='liq', **kwargs)        
            fugcoef_l = pcsaft_fugcoef(xl, m, s, e, t, rhol, **kwargs)
            rhov = pcsaft_den(xv, m, s, e, t, p_guess, phase='vap', **kwargs)        
            fugcoef_v = pcsaft_fugcoef(xv, m, s, e, t, rhov, **kwargs)
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
        z = kwargs['z']
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
            rhol = pcsaft_den(xl, m, s, e, t, p_guess, phase='liq', **kwargs)        
            fugcoef_l = pcsaft_fugcoef(xl, m, s, e, t, rhol, **kwargs)
            rhov = pcsaft_den(xv, m, s, e, t, p_guess, phase='vap', **kwargs)        
            fugcoef_v = pcsaft_fugcoef(xv, m, s, e, t, rhov, **kwargs)
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

def dielc_water(t):
    """
    Return the dielectric constant of water at 1 bar and the given temperature.
    
    t : float
        Temperature (K)
    
    This equation was fit to values given in the reference over the temperature
    range of 263.15 to 368.15 K.
    
    Reference:
    D. G. Archer and P. Wang, “The Dielectric Constant of Water and Debye‐Hückel 
    Limiting Law Slopes,” J. Phys. Chem. Ref. Data, vol. 19, no. 2, pp. 371–411, 
    Mar. 1990.
    """
    dielc = 7.6555618295E-04*t**2 - 8.1783881423E-01*t + 2.5419616803E+02
    return dielc    
    
def pcsaft_fit_pure(params, prop, T, P, molden, **kwargs):
    """
    For fitting PC-SAFT parameters for a single compound to data. The function 
    must be manually adjusted to match the parameters that are chosen to be 
    used for the compound.    
    
    Parameters
    ----------
    params : ndarray
        The parameters to be fit.
    prop : ndarray, shape (m,)
        Property type (0 = density, 1 = vapor pressure). The array has length m,
        which is equal to the number of experimental data points.
    T : ndarray, shape (m,)
        Temperature (K)
    P : ndarray, shape (m,)
        Pressure (Pa)
    molden : ndarray, shape (m,)
        Molar density (mol m^{-3}) (for vapor pressure this is not needed and 
        any number can be used as a placeholder)
    """
    kwargs = {'dipm': np.asarray([1.75]), 'dip_num': np.asarray([params[3]])} 

    error = np.zeros((prop.shape[0],), dtype='float_')
    x = np.ones((1,), dtype='float_')

    for i in range(prop.shape[0]):
        if prop[i] == 0:
            if molden[i] < 900:
                error[i] = (pcsaft_den(x, np.asarray([params[0]]), np.asarray([params[1]]), np.asarray([params[2]]), T[i], P[i], phase='vap', **kwargs) - molden[i])/molden[i]*100
            else:
                error[i] = (pcsaft_den(x, np.asarray([params[0]]), np.asarray([params[1]]), np.asarray([params[2]]), T[i], P[i], **kwargs) - molden[i])/molden[i]*100
        elif prop[i] == 1:
            P_fit = pcsaft_vaporP(P[i], x, np.asarray([params[0]]), np.asarray([params[1]]), np.asarray([params[2]]), T[i], **kwargs)
            error[i] = (P_fit - P[i])/P[i]*100
        else:
            ValueError('Currently only density (0) and vapor pressure (1) data are defined for use in the fitting function.')   
    
    # Can print or output the results for each set of parameters tested        
    print(params, np.sum(error**2))
    outfile = 'solving_data.txt'
    with open(outfile, 'a+') as f:
        f.write(str(params[0]) + '\t' + str(params[1]) + '\t' + str(params[2]) + '\t' + str(params[3]) + '\t' + str(np.sum(error**2)) + '\n')

    return np.sum(error**2)
