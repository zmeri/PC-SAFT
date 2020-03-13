# -*- coding: utf-8 -*-
# distutils: language=c++
"""
Created on Thu Jul 19 14:23:00 2018

@author: Zach Baird
"""
from libcpp.vector cimport vector

cdef extern from "pcsaft.cpp":
    double pcsaft_p_cpp(double t, double rho, add_args &cppargs)
    double pcsaft_Z_cpp(double t, double rho, add_args &cppargs)
    vector[double] pcsaft_fugcoef_cpp(double t, double rho, add_args &cppargs)
    double pcsaft_den_cpp(double t, double p, int phase, add_args &cppargs)
    double pcsaft_ares_cpp(double t, double rho, add_args &cppargs)
    double pcsaft_dadt_cpp(double t, double rho, add_args &cppargs)
    double pcsaft_hres_cpp(double t, double rho, add_args &cppargs)
    double pcsaft_sres_cpp(double t, double rho, add_args &cppargs)
    double pcsaft_gres_cpp(double t, double rho, add_args &cppargs)
    double flashTQ_cpp(double t, double Q, add_args &cppargs) except +
    double flashTQ_cpp(double t, double Q, add_args &cppargs, double p_guess) except +
    double flashPQ_cpp(double p, double Q, add_args &cppargs) except +
    double flashPQ_cpp(double p, double Q, add_args &cppargs, double t_guess) except +

    ctypedef struct add_args:
        vector[double] x
        vector[double] m
        vector[double] s
        vector[double] e
        vector[double] k_ij
        vector[double] e_assoc
        vector[double] vol_a
        vector[double] dipm
        vector[double] dip_num
        vector[double] z
        double dielc
        vector[double] k_hb
        vector[double] l_ij
