# -*- coding: utf-8 -*-
# distutils: language=c++
"""
Created on Thu Jul 19 14:23:00 2018

@author: Zach Baird
"""
from libcpp.vector cimport vector

cdef extern from "pcsaft.cpp":
    double pcsaft_p_cpp(vector[double] x, vector[double] m, vector[double] s, vector[double] e, \
        double t, double rho, add_args &cppargs)
    double pcsaft_Z_cpp(vector[double] x, vector[double] m, vector[double] s, vector[double] e, \
        double t, double rho, add_args &cppargs)
    vector[double] pcsaft_fugcoef_cpp(vector[double] x, vector[double] m, vector[double] s, vector[double] e, \
        double t, double rho, add_args &cppargs)
    double pcsaft_den_cpp(vector[double] x, vector[double] m, vector[double] s, vector[double] e, \
        double t, double p, int phase, add_args &cppargs)
    double bubblePfit_cpp(double p_guess, vector[double] xv_guess, vector[double] x, vector[double] m, \
        vector[double] s, vector[double] e, double t, add_args &cppargs)
    double bubbleTfit_cpp(double t_guess, vector[double] xv_guess, vector[double] x, vector[double] m, \
        vector[double] s, vector[double] e, double p, add_args &cppargs)
    double PTzfit_cpp(double p_guess, vector[double] x_guess, double beta_guess, double mol, \
        double vol, vector[double] x_total, vector[double] m, vector[double] s, vector[double] e, \
        double t, add_args &cppargs)
#    vector[double] chem_equil_cpp(vector[double] x_guess, vector[double] m, vector[double] s, \
#        vector[double] e, double t, double p, add_args &cppargs)
    double pcsaft_ares_cpp(vector[double] x, vector[double] m, vector[double] s, vector[double] e, \
        double t, double rho, add_args &cppargs)
    double pcsaft_dadt_cpp(vector[double] x, vector[double] m, vector[double] s, vector[double] e, \
        double t, double rho, add_args &cppargs)
    double pcsaft_hres_cpp(vector[double] x, vector[double] m, vector[double] s, vector[double] e, \
        double t, double rho, add_args &cppargs)
    double pcsaft_sres_cpp(vector[double] x, vector[double] m, vector[double] s, vector[double] e, \
        double t, double rho, add_args &cppargs)
    double pcsaft_gres_cpp(vector[double] x, vector[double] m, vector[double] s, vector[double] e, \
        double t, double rho, add_args &cppargs)
   
    ctypedef struct add_args: 
        vector[double] k_ij
        vector[double] e_assoc
        vector[double] vol_a
        vector[double] dipm
        vector[double] dip_num
        vector[double] z
        double dielc
        vector[double] k_hb
        vector[double] l_ij
