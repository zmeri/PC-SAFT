#include <vector>
#include <string>
#include <cmath>
#include "math.h"
#include "externals/eigen/Eigen/Dense"

#include "pcsaft_electrolyte.h"

using std::vector;

#if defined(HUGE_VAL) && !defined(_HUGE)
    # define _HUGE HUGE_VAL
#else
    // GCC Version of huge value macro
    #if defined(HUGE) && !defined(_HUGE)
    #  define _HUGE HUGE
    #endif
#endif

vector<double> XA_find(vector<double> XA_guess, int ncA, vector<double> delta_ij, double den,
    vector<double> x) {
    /**Iterate over this function in order to solve for XA*/
    int n_sites = XA_guess.size()/ncA;
    double summ2;
    vector<double> XA = XA_guess;

    for (int i = 0; i < ncA; i++) {
        for (int kout = 0; kout < n_sites; kout++) {
            summ2 = 0.;
            for (int j = 0; j < ncA; j++) {
                for (int kin = 0; kin < n_sites; kin++) {
                    if (kin != kout) {
                        summ2 += den*x[j]*XA_guess[j*n_sites+kin]*delta_ij[i*ncA+j];
                    }
                }
            }
            XA[i*n_sites+kout] = 1./(1.+summ2);
        }
    }

    return XA;
    }

vector<double> dXA_find(int ncA, int ncomp, vector<int> iA, vector<double> delta_ij,
    double den, vector<double> XA, vector<double> ddelta_dd, vector<double> x, int n_sites) {
    /**Solve for the derivative of XA with respect to density.*/
    Eigen::MatrixXd B(n_sites*ncA*ncomp, 1);
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_sites*ncA*ncomp, n_sites*ncA*ncomp);

    double sum1, sum2;
    int indx1, indx2;
    int indx4 = -1;
    int indx3 = -1;
    for (int i = 0; i < ncomp; i++) {
        indx1 = -1;
        if (find(iA.begin(), iA.end(), i) != iA.end()) {
            indx4 += 1;
        }
        for (int j = 0; j < ncA; j++) {
            for (int h = 0; h < n_sites; h++) {
                indx1 += 1;
                indx3 += 1;
                indx2 = -1;
                sum1 = 0;
                for (int k = 0; k < ncA; k++) {
                    for (int l = 0; l < n_sites; l++) {
                        indx2 += 1;
                        sum1 = sum1 + den*x[k]*(XA[indx2]*ddelta_dd[j*(ncA*ncomp)+k*(ncomp)+i]*((indx1+indx2)%2)); // (indx1+indx2)%2 ensures that A-A and B-B associations are set to zero
                        A(indx1+i*n_sites*ncA,indx2+i*n_sites*ncA) =
                        A(indx1+i*n_sites*ncA,indx2+i*n_sites*ncA) +
                        XA[indx1]*XA[indx1]*den*x[k]*delta_ij[j*ncA+k]*((indx1+indx2)%2);
                    }
                }

                sum2 = 0;
                if (find(iA.begin(), iA.end(), i) != iA.end()) {
                    for (int k = 0; k < n_sites; k++) {
                        sum2 = sum2 + XA[n_sites*(indx4)+k]*delta_ij[indx4*ncA+j]*((indx1+k)%2);
                    }
                }

                A(indx3,indx3) = A(indx3,indx3) + 1;
                B(indx3) = -1*XA[indx1]*XA[indx1]*(sum1 + sum2);
            }
        }
    }

    Eigen::MatrixXd solution = A.lu().solve(B); //Solves linear system of equations
    vector<double> dXA_dd(n_sites*ncA*ncomp);
    for (int i = 0; i < n_sites*ncA*ncomp; i++) {
        dXA_dd[i] = solution(i);
    }
    return dXA_dd;
}

vector<double> dXAdt_find(int ncA, vector<double> delta_ij, double den,
    vector<double> XA, vector<double> ddelta_dt, vector<double> x, int n_sites) {
    /**Solve for the derivative of XA with respect to temperature.*/
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n_sites*ncA, 1);
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_sites*ncA, n_sites*ncA);

    double summ;
    int i_in, i_out = -1; // i_out is index of outer iteration loop (follows row of matrices)
    for (int i = 0; i < ncA; i++) {
        for (int ai = 0; ai < n_sites; ai++) {
            i_out += 1;
            i_in = -1; // index for summation loops
            summ = 0;
            for (int j = 0; j < ncA; j++) {
                for (int bj = 0; bj < n_sites; bj++) {
                    i_in += 1;
                    B(i_out) -= x[j]*XA[i_in]*ddelta_dt[i*ncA+j]*((i_in+i_out)%2); // (i_in+i_out)%2 ensures that A-A and B-B associations are set to zero
                    A(i_out,i_in) = x[j]*delta_ij[i*ncA+j]*((i_in+i_out)%2);
                    summ += x[j]*XA[i_in]*delta_ij[i*ncA+j]*((i_in+i_out)%2);
                }
            }
            A(i_out,i_out) = A(i_out,i_out) + pow(1+den*summ, 2.)/den;
        }
    }

    Eigen::MatrixXd solution = A.lu().solve(B); //Solves linear system of equations
    vector<double> dXA_dt(n_sites*ncA);
    for (int i = 0; i < n_sites*ncA; i++) {
        dXA_dt[i] = solution(i);
    }
    return dXA_dt;
}


double pcsaft_Z_cpp(double t, double rho, add_args &cppargs) {
    /**
    Calculate the compressibility factor.

    Parameters
    ----------
    x : vector<double>, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : vector<double>, shape (n,)
        Segment number for each component.
    s : vector<double>, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : vector<double>, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : double
        Temperature (K)
    rho : double
        Molar density (mol m^{-3})
    cppargs : add_args
        A struct containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : vector<double>, shape (n*n,)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : vector<double>, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : vector<double>, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : vector<double>, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : vector<double>, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : vector<double>, shape (n,)
            Charge number of the ions
        dielc : double
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    Z : double
        Compressibility factor
    */
    int ncomp = cppargs.x.size(); // number of components
    vector<double> d (ncomp);
    for (int i = 0; i < ncomp; i++) {
        d[i] = cppargs.s[i]*(1-0.12*exp(-3*cppargs.e[i]/t));
    }
    if (!cppargs.z.empty()) {
        for (int i = 0; i < ncomp; i++) {
            if (cppargs.z[i] != 0) {
                d[i] = cppargs.s[i]*(1-0.12); // for ions the diameter is assumed to be temperature independent (see Held et al. 2014)
            }
        }
    }

    double den = rho*N_AV/1.0e30;

    vector<double> zeta (4, 0);
    double summ;
    for (int i = 0; i < 4; i++) {
        summ = 0;
        for (int j = 0; j < ncomp; j++) {
            summ += cppargs.x[j]*cppargs.m[j]*pow(d[j], i);
        }
        zeta[i] = PI/6*den*summ;
    }

    double eta = zeta[3];
    double m_avg = 0;
    for (int i = 0; i < ncomp; i++) {
        m_avg += cppargs.x[i]*cppargs.m[i];
    }

    vector<double> ghs (ncomp*ncomp, 0);
    vector<double> denghs (ncomp*ncomp, 0);
    vector<double> e_ij (ncomp*ncomp, 0);
    vector<double> s_ij (ncomp*ncomp, 0);
    double m2es3 = 0.;
    double m2e2s3 = 0.;
    int idx = -1;
    for (int i = 0; i < ncomp; i++) {
        for (int j = 0; j < ncomp; j++) {
            idx += 1;
            if (cppargs.l_ij.empty()) {
                s_ij[idx] = (cppargs.s[i] + cppargs.s[j])/2.;
            }
            else {
                s_ij[idx] = (cppargs.s[i] + cppargs.s[j])/2.*(1-cppargs.l_ij[idx]);
            }
            if (!cppargs.z.empty()) {
                if (cppargs.z[i]*cppargs.z[j] <= 0) { // for two cations or two anions e_ij is kept at zero to avoid dispersion between like ions (see Held et al. 2014)
                    if (cppargs.k_ij.empty()) {
                        e_ij[idx] = sqrt(cppargs.e[i]*cppargs.e[j]);
                    }
                    else {
                        e_ij[idx] = sqrt(cppargs.e[i]*cppargs.e[j])*(1-cppargs.k_ij[idx]);
                    }
                }
            } else {
                if (cppargs.k_ij.empty()) {
                    e_ij[idx] = sqrt(cppargs.e[i]*cppargs.e[j]);
                }
                else {
                    e_ij[idx] = sqrt(cppargs.e[i]*cppargs.e[j])*(1-cppargs.k_ij[idx]);
                }
            }
            m2es3 = m2es3 + cppargs.x[i]*cppargs.x[j]*cppargs.m[i]*cppargs.m[j]*e_ij[idx]/t*pow(s_ij[idx], 3);
            m2e2s3 = m2e2s3 + cppargs.x[i]*cppargs.x[j]*cppargs.m[i]*cppargs.m[j]*pow(e_ij[idx]/t,2)*pow(s_ij[idx], 3);
            ghs[idx] = 1/(1-zeta[3]) + (d[i]*d[j]/(d[i]+d[j]))*3*zeta[2]/(1-zeta[3])/(1-zeta[3]) +
                    pow(d[i]*d[j]/(d[i]+d[j]), 2)*2*zeta[2]*zeta[2]/pow(1-zeta[3], 3);
            denghs[idx] = zeta[3]/(1-zeta[3])/(1-zeta[3]) +
                (d[i]*d[j]/(d[i]+d[j]))*(3*zeta[2]/(1-zeta[3])/(1-zeta[3]) +
                6*zeta[2]*zeta[3]/pow(1-zeta[3], 3)) +
                pow(d[i]*d[j]/(d[i]+d[j]), 2)*(4*zeta[2]*zeta[2]/pow(1-zeta[3], 3) +
                6*zeta[2]*zeta[2]*zeta[3]/pow(1-zeta[3], 4));
        }
    }

    double Zhs = zeta[3]/(1-zeta[3]) + 3.*zeta[1]*zeta[2]/zeta[0]/(1.-zeta[3])/(1.-zeta[3]) +
        (3.*pow(zeta[2], 3.) - zeta[3]*pow(zeta[2], 3.))/zeta[0]/pow(1.-zeta[3], 3.);

    static double a0[7] = { 0.910563145, 0.636128145, 2.686134789, -26.54736249, 97.75920878, -159.5915409, 91.29777408 };
    static double a1[7] = { -0.308401692, 0.186053116, -2.503004726, 21.41979363, -65.25588533, 83.31868048, -33.74692293 };
    static double a2[7] = { -0.090614835, 0.452784281, 0.596270073, -1.724182913, -4.130211253, 13.77663187, -8.672847037 };
    static double b0[7] = { 0.724094694, 2.238279186, -4.002584949, -21.00357682, 26.85564136, 206.5513384, -355.6023561 };
    static double b1[7] = { -0.575549808, 0.699509552, 3.892567339, -17.21547165, 192.6722645, -161.8264617, -165.2076935 };
    static double b2[7] = { 0.097688312, -0.255757498, -9.155856153, 20.64207597, -38.80443005, 93.62677408, -29.66690559 };

    vector<double> a (7, 0);
    vector<double> b (7, 0);
    for (int i = 0; i < 7; i++) {
        a[i] = a0[i] + (m_avg-1.)/m_avg*a1[i] + (m_avg-1.)/m_avg*(m_avg-2.)/m_avg*a2[i];
        b[i] = b0[i] + (m_avg-1.)/m_avg*b1[i] + (m_avg-1.)/m_avg*(m_avg-2.)/m_avg*b2[i];
    }

    double detI1_det = 0.0;
    double detI2_det = 0.0;
    double I2 = 0.0;
    for (int i = 0; i < 7; i++) {
        detI1_det += a[i]*(i+1)*pow(eta, i);
        detI2_det += b[i]*(i+1)*pow(eta, i);
        I2 += b[i]*pow(eta, i);
    }
    double C1 = 1./(1. + m_avg*(8*eta-2*eta*eta)/pow(1-eta, 4) + (1-m_avg)*(20*eta-27*eta*eta+12*pow(eta, 3)-2*pow(eta, 4))/pow((1-eta)*(2-eta), 2.0));
    double C2 = -1.*C1*C1*(m_avg*(-4*eta*eta+20*eta+8)/pow(1-eta, 5) + (1-m_avg)*(2*pow(eta, 3)+12*eta*eta-48*eta+40)/pow((1-eta)*(2-eta), 3.0));

    summ = 0.0;
    for (int i = 0; i < ncomp; i++) {
        summ += cppargs.x[i]*(cppargs.m[i]-1)/ghs[i*ncomp+i]*denghs[i*ncomp+i];
    }

    double Zid = 1.0;
    double Zhc = m_avg*Zhs - summ;
    double Zdisp = -2*PI*den*detI1_det*m2es3 - PI*den*m_avg*(C1*detI2_det + C2*eta*I2)*m2e2s3;

    // Dipole term (Gross and Vrabec term) --------------------------------------
    double Zpolar = 0;
    if (!cppargs.dipm.empty()) {
        double A2 = 0.;
        double A3 = 0.;
        double dA2_det = 0.;
        double dA3_det = 0.;
        vector<double> adip (5, 0);
        vector<double> bdip (5, 0);
        vector<double> cdip (5, 0);
        vector<double> dipmSQ (ncomp, 0);
        double J2, dJ2_det, J3, dJ3_det;

        static double a0dip[5] = { 0.3043504, -0.1358588, 1.4493329, 0.3556977, -2.0653308 };
        static double a1dip[5] = { 0.9534641, -1.8396383, 2.0131180, -7.3724958, 8.2374135 };
        static double a2dip[5] = { -1.1610080, 4.5258607, 0.9751222, -12.281038, 5.9397575 };
        static double b0dip[5] = { 0.2187939, -1.1896431, 1.1626889, 0, 0 };
        static double b1dip[5] = { -0.5873164, 1.2489132, -0.5085280, 0, 0 };
        static double b2dip[5] = { 3.4869576, -14.915974, 15.372022, 0, 0 };
        static double c0dip[5] = { -0.0646774, 0.1975882, -0.8087562, 0.6902849, 0 };
        static double c1dip[5] = { -0.9520876, 2.9924258, -2.3802636, -0.2701261, 0 };
        static double c2dip[5] = { -0.6260979, 1.2924686, 1.6542783, -3.4396744, 0 };

        const static double conv = 7242.702976750923; // conversion factor, see the note below Table 2 in Gross and Vrabec 2006

        for (int i = 0; i < ncomp; i++) {
            dipmSQ[i] = pow(cppargs.dipm[i], 2.)/(cppargs.m[i]*cppargs.e[i]*pow(cppargs.s[i],3.))*conv;
        }

        double m_ij;
        for (int i = 0; i < ncomp; i++) {
            for (int j = 0; j < ncomp; j++) {
                m_ij = sqrt(cppargs.m[i]*cppargs.m[j]);
                if (m_ij > 2) {
                    m_ij = 2;
                }
                J2 = 0.;
                dJ2_det = 0.;
                for (int l = 0; l < 5; l++) {
                    adip[l] = a0dip[l] + (m_ij-1)/m_ij*a1dip[l] + (m_ij-1)/m_ij*(m_ij-2)/m_ij*a2dip[l];
                    bdip[l] = b0dip[l] + (m_ij-1)/m_ij*b1dip[l] + (m_ij-1)/m_ij*(m_ij-2)/m_ij*b2dip[l];
                    J2 += (adip[l] + bdip[l]*e_ij[j*ncomp+j]/t)*pow(eta, l); // j*ncomp+j needs to be used for e_ij because it is formatted as a 1D vector
                    dJ2_det += (adip[l] + bdip[l]*e_ij[j*ncomp+j]/t)*l*pow(eta, l-1);
                }
                A2 += cppargs.x[i]*cppargs.x[j]*e_ij[i*ncomp+i]/t*e_ij[j*ncomp+j]/t*pow(s_ij[i*ncomp+i],3)*pow(s_ij[j*ncomp+j],3)/
                    pow(s_ij[i*ncomp+j],3)*cppargs.dip_num[i]*cppargs.dip_num[j]*dipmSQ[i]*dipmSQ[j]*J2;
                dA2_det += cppargs.x[i]*cppargs.x[j]*e_ij[i*ncomp+i]/t*e_ij[j*ncomp+j]/t*pow(s_ij[i*ncomp+i],3)*
                    pow(s_ij[j*ncomp+j],3)/pow(s_ij[i*ncomp+j],3)*cppargs.dip_num[i]*cppargs.dip_num[j]*dipmSQ[i]*dipmSQ[j]*dJ2_det;
            }
        }

        double m_ijk;
        for (int i = 0; i < ncomp; i++) {
            for (int j = 0; j < ncomp; j++) {
                for (int k = 0; k < ncomp; k++) {
                    m_ijk = pow((cppargs.m[i]*cppargs.m[j]*cppargs.m[k]),1/3.);
                    if (m_ijk > 2) {
                        m_ijk = 2;
                    }
                    J3 = 0.;
                    dJ3_det = 0.;
                    for (int l = 0; l < 5; l++) {
                        cdip[l] = c0dip[l] + (m_ijk-1)/m_ijk*c1dip[l] + (m_ijk-1)/m_ijk*(m_ijk-2)/m_ijk*c2dip[l];
                        J3 += cdip[l]*pow(eta, l);
                        dJ3_det += cdip[l]*l*pow(eta, (l-1));
                    }
                    A3 += cppargs.x[i]*cppargs.x[j]*cppargs.x[k]*e_ij[i*ncomp+i]/t*e_ij[j*ncomp+j]/t*e_ij[k*ncomp+k]/t*
                        pow(s_ij[i*ncomp+i],3)*pow(s_ij[j*ncomp+j],3)*pow(s_ij[k*ncomp+k],3)/s_ij[i*ncomp+j]/s_ij[i*ncomp+k]/
                        s_ij[j*ncomp+k]*cppargs.dip_num[i]*cppargs.dip_num[j]*cppargs.dip_num[k]*dipmSQ[i]*
                        dipmSQ[j]*dipmSQ[k]*J3;
                    dA3_det += cppargs.x[i]*cppargs.x[j]*cppargs.x[k]*e_ij[i*ncomp+i]/t*e_ij[j*ncomp+j]/t*e_ij[k*ncomp+k]/t*
                        pow(s_ij[i*ncomp+i],3)*pow(s_ij[j*ncomp+j],3)*pow(s_ij[k*ncomp+k],3)/s_ij[i*ncomp+j]/s_ij[i*ncomp+k]/
                        s_ij[j*ncomp+k]*cppargs.dip_num[i]*cppargs.dip_num[j]*cppargs.dip_num[k]*dipmSQ[i]*
                        dipmSQ[j]*dipmSQ[k]*dJ3_det;
                }
            }
        }

        A2 = -PI*den*A2;
        A3 = -4/3.*PI*PI*den*den*A3;
        dA2_det = -PI*den*dA2_det;
        dA3_det = -4/3.*PI*PI*den*den*dA3_det;

        Zpolar = eta*((dA2_det*(1-A3/A2)+(dA3_det*A2-A3*dA2_det)/A2)/(1-A3/A2)/(1-A3/A2));
    }

    // Association term -------------------------------------------------------
    // only the 2B association type is currently implemented
    double Zassoc = 0;
    if (!cppargs.e_assoc.empty()) {
        int a_sites = 2;
        int ncA = count_if(cppargs.vol_a.begin(), cppargs.vol_a.end(), IsNotZero); // number of associating compounds in the fluid

        vector<int> iA (ncA, 0); //indices of associating compounds
        int ctr = 0;
        for (int i = 0; i < ncomp; i++) {
            if (cppargs.vol_a[i] != 0.0) {
                iA[ctr] = i;
                ctr += 1;
            }
        }

        vector<double> XA (ncA*a_sites, 0);
        vector<double> eABij (ncA*ncA, 0);
        vector<double> volABij (ncA*ncA, 0);
        vector<double> delta_ij (ncA*ncA, 0);
        vector<double> ddelta_dd (ncA*ncA*ncomp, 0);

        // these indices are necessary because we are only using 1D vectors
        int idxa = -1; // index over only associating compounds
        int idxi = 0; // index for the ii-th compound
        int idxj = 0; // index for the jj-th compound
        int idx_ddelta = -1; // index for ddelta_dd vector
        double dghsd_dd;
        for (int i = 0; i < ncA; i++) {
            idxi = iA[i]*ncomp+iA[i];
            for (int j = 0; j < ncA; j++) {
                idxa += 1;
                idxj = iA[j]*ncomp+iA[j];
                eABij[idxa] = (cppargs.e_assoc[iA[i]]+cppargs.e_assoc[iA[j]])/2.;
                if (cppargs.k_hb.empty()) {
                    volABij[idxa] = sqrt(cppargs.vol_a[iA[i]]*cppargs.vol_a[iA[j]])*pow(sqrt(s_ij[idxi]*
                        s_ij[idxj])/(0.5*(s_ij[idxi]+s_ij[idxj])), 3);
                }
                else {
                    volABij[idxa] = sqrt(cppargs.vol_a[iA[i]]*cppargs.vol_a[iA[j]])*pow(sqrt(s_ij[idxi]*
                        s_ij[idxj])/(0.5*(s_ij[idxi]+s_ij[idxj])), 3)*(1-cppargs.k_hb[iA[i]*ncomp+iA[j]]);
                }
                delta_ij[idxa] = ghs[iA[i]*ncomp+iA[j]]*(exp(eABij[idxa]/t)-1)*pow(s_ij[iA[i]*ncomp+iA[j]], 3)*volABij[idxa];
                for (int k = 0; k < ncomp; k++) {
                    idx_ddelta += 1;
                    dghsd_dd = PI/6.*cppargs.m[k]*(pow(d[k], 3)/(1-zeta[3])/(1-zeta[3]) + 3*d[iA[i]]*d[iA[j]]/
                        (d[iA[i]]+d[iA[j]])*(d[k]*d[k]/(1-zeta[3])/(1-zeta[3])+2*pow(d[k], 3)*
                        zeta[2]/pow(1-zeta[3], 3)) + 2*pow((d[iA[i]]*d[iA[j]]/(d[iA[i]]+d[iA[j]])), 2)*
                        (2*d[k]*d[k]*zeta[2]/pow(1-zeta[3], 3)+3*(pow(d[k], 3)*zeta[2]*zeta[2]
                        /pow(1-zeta[3], 4))));
                    ddelta_dd[idx_ddelta] = dghsd_dd*(exp(eABij[idxa]/t)-1)*pow(s_ij[iA[i]*ncomp+iA[j]], 3)*volABij[idxa];
                }
            }
            XA[i*2] = (-1 + sqrt(1+8*den*delta_ij[i*ncA+i]))/(4*den*delta_ij[i*ncA+i]);
            if (!std::isfinite(XA[i*2])) {
                XA[i*2] = 0.02;
            }
            XA[i*2+1] = XA[i*2];
        }

        vector<double> x_assoc(ncA); // mole fractions of only the associating compounds
        for (int i = 0; i < ncA; i++) {
            x_assoc[i] = cppargs.x[iA[i]];
        }

        ctr = 0;
        double dif = 1000.;
        vector<double> XA_old = XA;
        while ((ctr < 500) && (dif > 1e-9)) {
            ctr += 1;
            XA = XA_find(XA, ncA, delta_ij, den, x_assoc);
            dif = 0.;
            for (int i = 0; i < ncA*2; i++) {
                dif += abs(XA[i] - XA_old[i]);
            }
            XA_old = XA;
        }

        vector<double> dXA_dd(ncA*a_sites*ncomp, 0);
        dXA_dd = dXA_find(ncA, ncomp, iA, delta_ij, den, XA, ddelta_dd, x_assoc, a_sites);

        summ = 0.;
        for (int i = 0; i < ncomp; i++) {
            for (int j = 0; j < ncA; j++) {
                for (int k = 0; k < a_sites; k++) {
                    summ += cppargs.x[i]*den*cppargs.x[iA[j]]*(1/XA[j*a_sites+k]-0.5)*dXA_dd[i*(ncA*a_sites)+j*(a_sites)+k];
                }
            }
        }

        Zassoc = summ;
    }

    // Ion term ---------------------------------------------------------------
    double Zion = 0;
    if (!cppargs.z.empty()) {
        vector<double> q(cppargs.z.begin(), cppargs.z.end());
        for (int i = 0; i < ncomp; i++) {
            q[i] = q[i]*E_CHRG;
        }

        summ = 0.;
        for (int i = 0; i < ncomp; i++) {
            summ += cppargs.z[i]*cppargs.z[i]*cppargs.x[i];
        }

        double kappa = sqrt(den*E_CHRG*E_CHRG/kb/t/(cppargs.dielc*perm_vac)*summ); // the inverse Debye screening length. Equation 4 in Held et al. 2008.

        if (kappa != 0) {
            double chi, sigma_k;
            summ = 0.;
            for (int i = 0; i < ncomp; i++) {
                chi = 3/pow(kappa*cppargs.s[i], 3)*(1.5 + log(1+kappa*cppargs.s[i]) - 2*(1+kappa*cppargs.s[i]) +
                    0.5*pow(1+kappa*cppargs.s[i], 2));
                sigma_k = -2*chi+3/(1+kappa*cppargs.s[i]);
                summ += q[i]*q[i]*cppargs.x[i]*sigma_k;
            }
            Zion = -1*kappa/24./PI/kb/t/(cppargs.dielc*perm_vac)*summ;
        }
    }

    double Z = Zid + Zhc + Zdisp + Zpolar + Zassoc + Zion;
    return Z;
}


vector<double> pcsaft_fugcoef_cpp(double t, double rho, add_args &cppargs) {
    /**
    Calculate the fugacity coefficients for one phase of the system.

    Parameters
    ----------
    x : vector, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : vector<double>, shape (n,)
        Segment number for each component.
    s : vector<double>, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : vector<double>, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : double
        Temperature (K)
    rho : double
        Molar density (mol m^{-3})
    cppargs : add_args
        A struct containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : vector<double>, shape (n*n,)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : vector<double>, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : vector<double>, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : vector<double>, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : vector<double>, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : vector<double>, shape (n,)
            Charge number of the ions
        dielc : double
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    fugcoef : vector<double>, shape (n,)
        Fugacity coefficients of each component.
    */
    int ncomp = cppargs.x.size(); // number of components
    vector<double> d (ncomp);
    for (int i = 0; i < ncomp; i++) {
        d[i] = cppargs.s[i]*(1-0.12*exp(-3*cppargs.e[i]/t));
    }
    if (!cppargs.z.empty()) {
        for (int i = 0; i < ncomp; i++) {
            if (cppargs.z[i] != 0) {
                d[i] = cppargs.s[i]*(1-0.12); // for ions the diameter is assumed to be temperature independent (see Held et al. 2014)
            }
        }
    }

    double den = rho*N_AV/1.0e30;

    vector<double> zeta (4, 0);
    double summ;
    for (int i = 0; i < 4; i++) {
        summ = 0;
        for (int j = 0; j < ncomp; j++) {
            summ += cppargs.x[j]*cppargs.m[j]*pow(d[j], i);
        }
        zeta[i] = PI/6*den*summ;
    }

    double eta = zeta[3];
    double m_avg = 0;
    for (int i = 0; i < ncomp; i++) {
        m_avg += cppargs.x[i]*cppargs.m[i];
    }

    vector<double> ghs(ncomp*ncomp, 0);
    vector<double> denghs(ncomp*ncomp, 0);
    vector<double> e_ij(ncomp*ncomp, 0);
    vector<double> s_ij(ncomp*ncomp, 0);
    double m2es3 = 0.;
    double m2e2s3 = 0.;
    int idx = -1;
    for (int i = 0; i < ncomp; i++) {
        for (int j = 0; j < ncomp; j++) {
            idx += 1;
            if (cppargs.l_ij.empty()) {
                s_ij[idx] = (cppargs.s[i] + cppargs.s[j])/2.;
            }
            else {
                s_ij[idx] = (cppargs.s[i] + cppargs.s[j])/2.*(1-cppargs.l_ij[idx]);
            }
            if (!cppargs.z.empty()) {
                if (cppargs.z[i]*cppargs.z[j] <= 0) { // for two cations or two anions e_ij is kept at zero to avoid dispersion between like ions (see Held et al. 2014)
                    if (cppargs.k_ij.empty()) {
                        e_ij[idx] = sqrt(cppargs.e[i]*cppargs.e[j]);
                    }
                    else {
                        e_ij[idx] = sqrt(cppargs.e[i]*cppargs.e[j])*(1-cppargs.k_ij[idx]);
                    }
                }
            } else {
                if (cppargs.k_ij.empty()) {
                    e_ij[idx] = sqrt(cppargs.e[i]*cppargs.e[j]);
                }
                else {
                    e_ij[idx] = sqrt(cppargs.e[i]*cppargs.e[j])*(1-cppargs.k_ij[idx]);
                }
            }
            m2es3 = m2es3 + cppargs.x[i]*cppargs.x[j]*cppargs.m[i]*cppargs.m[j]*e_ij[idx]/t*pow(s_ij[idx], 3);
            m2e2s3 = m2e2s3 + cppargs.x[i]*cppargs.x[j]*cppargs.m[i]*cppargs.m[j]*pow(e_ij[idx]/t,2)*pow(s_ij[idx], 3);
            ghs[idx] = 1/(1-zeta[3]) + (d[i]*d[j]/(d[i]+d[j]))*3*zeta[2]/(1-zeta[3])/(1-zeta[3]) +
                    pow(d[i]*d[j]/(d[i]+d[j]), 2)*2*zeta[2]*zeta[2]/pow(1-zeta[3], 3);
            denghs[idx] = zeta[3]/(1-zeta[3])/(1-zeta[3]) +
                (d[i]*d[j]/(d[i]+d[j]))*(3*zeta[2]/(1-zeta[3])/(1-zeta[3]) +
                6*zeta[2]*zeta[3]/pow(1-zeta[3], 3)) +
                pow(d[i]*d[j]/(d[i]+d[j]), 2)*(4*zeta[2]*zeta[2]/pow(1-zeta[3], 3) +
                6*zeta[2]*zeta[2]*zeta[3]/pow(1-zeta[3], 4));
        }
    }

    double ares_hs = 1/zeta[0]*(3*zeta[1]*zeta[2]/(1-zeta[3]) + pow(zeta[2], 3.)/(zeta[3]*pow(1-zeta[3],2))
            + (pow(zeta[2], 3.)/pow(zeta[3], 2.) - zeta[0])*log(1-zeta[3]));
    double Zhs = zeta[3]/(1-zeta[3]) + 3.*zeta[1]*zeta[2]/zeta[0]/(1.-zeta[3])/(1.-zeta[3]) +
        (3.*pow(zeta[2], 3.) - zeta[3]*pow(zeta[2], 3.))/zeta[0]/pow(1.-zeta[3], 3.);

    static double a0[7] = { 0.910563145, 0.636128145, 2.686134789, -26.54736249, 97.75920878, -159.5915409, 91.29777408 };
    static double a1[7] = { -0.308401692, 0.186053116, -2.503004726, 21.41979363, -65.25588533, 83.31868048, -33.74692293 };
    static double a2[7] = { -0.090614835, 0.452784281, 0.596270073, -1.724182913, -4.130211253, 13.77663187, -8.672847037 };
    static double b0[7] = { 0.724094694, 2.238279186, -4.002584949, -21.00357682, 26.85564136, 206.5513384, -355.6023561 };
    static double b1[7] = { -0.575549808, 0.699509552, 3.892567339, -17.21547165, 192.6722645, -161.8264617, -165.2076935 };
    static double b2[7] = { 0.097688312, -0.255757498, -9.155856153, 20.64207597, -38.80443005, 93.62677408, -29.66690559 };

    vector<double> a (7, 0);
    vector<double> b (7, 0);
    for (int i = 0; i < 7; i++) {
        a[i] = a0[i] + (m_avg-1.)/m_avg*a1[i] + (m_avg-1.)/m_avg*(m_avg-2.)/m_avg*a2[i];
        b[i] = b0[i] + (m_avg-1.)/m_avg*b1[i] + (m_avg-1.)/m_avg*(m_avg-2.)/m_avg*b2[i];
    }

    double detI1_det = 0.0;
    double detI2_det = 0.0;
    double I1 = 0.0;
    double I2 = 0.0;
    for (int i = 0; i < 7; i++) {
        detI1_det += a[i]*(i+1)*pow(eta, i);
        detI2_det += b[i]*(i+1)*pow(eta, i);
        I2 += b[i]*pow(eta, i);
        I1 += a[i]*pow(eta, i);
    }
    double C1 = 1./(1. + m_avg*(8*eta-2*eta*eta)/pow(1-eta, 4) + (1-m_avg)*(20*eta-27*eta*eta+12*pow(eta, 3)-2*pow(eta, 4))/pow((1-eta)*(2-eta), 2.0));
    double C2 = -1.*C1*C1*(m_avg*(-4*eta*eta+20*eta+8)/pow(1-eta, 5) + (1-m_avg)*(2*pow(eta, 3)+12*eta*eta-48*eta+40)/pow((1-eta)*(2-eta), 3.0));

    summ = 0.0;
    for (int i = 0; i < ncomp; i++) {
        summ += cppargs.x[i]*(cppargs.m[i]-1)*log(ghs[i*ncomp+i]);
    }

    double ares_hc = m_avg*ares_hs - summ;
    double ares_disp = -2*PI*den*I1*m2es3 - PI*den*m_avg*C1*I2*m2e2s3;

    summ = 0.0;
    for (int i = 0; i < ncomp; i++) {
        summ += cppargs.x[i]*(cppargs.m[i]-1)/ghs[i*ncomp+i]*denghs[i*ncomp+i];
    }

    double Zhc = m_avg*Zhs - summ;
    double Zdisp = -2*PI*den*detI1_det*m2es3 - PI*den*m_avg*(C1*detI2_det + C2*eta*I2)*m2e2s3;

    vector<double> dghsii_dx(ncomp*ncomp, 0);
    vector<double> dahs_dx(ncomp, 0);
    vector<double> dzeta_dx(4, 0);
    idx = -1;
    for (int i = 0; i < ncomp; i++) {
        for (int l = 0; l < 4; l++) {
            dzeta_dx[l] = PI/6.*den*cppargs.m[i]*pow(d[i],l);
        }
        for (int j = 0; j < ncomp; j++) {
            idx += 1;
            dghsii_dx[idx] = dzeta_dx[3]/(1-zeta[3])/(1-zeta[3]) + (d[j]*d[j]/(d[j]+d[j]))*
                    (3*dzeta_dx[2]/(1-zeta[3])/(1-zeta[3]) + 6*zeta[2]*dzeta_dx[3]/pow(1-zeta[3],3))
                    + pow(d[j]*d[j]/(d[j]+d[j]),2)*(4*zeta[2]*dzeta_dx[2]/pow(1-zeta[3],3)
                    + 6*zeta[2]*zeta[2]*dzeta_dx[3]/pow(1-zeta[3],4));
        }
        dahs_dx[i] = -dzeta_dx[0]/zeta[0]*ares_hs + 1/zeta[0]*(3*(dzeta_dx[1]*zeta[2]
                + zeta[1]*dzeta_dx[2])/(1-zeta[3]) + 3*zeta[1]*zeta[2]*dzeta_dx[3]
                /(1-zeta[3])/(1-zeta[3]) + 3*zeta[2]*zeta[2]*dzeta_dx[2]/zeta[3]/(1-zeta[3])/(1-zeta[3])
                + pow(zeta[2],3)*dzeta_dx[3]*(3*zeta[3]-1)/zeta[3]/zeta[3]/pow(1-zeta[3],3)
                + log(1-zeta[3])*((3*zeta[2]*zeta[2]*dzeta_dx[2]*zeta[3] -
                2*pow(zeta[2],3)*dzeta_dx[3])/pow(zeta[3],3) - dzeta_dx[0]) +
                (zeta[0]-pow(zeta[2],3)/zeta[3]/zeta[3])*dzeta_dx[3]/(1-zeta[3]));
    }

    vector<double> dadisp_dx(ncomp, 0);
    vector<double> dahc_dx(ncomp, 0);
    double dzeta3_dx, daa_dx, db_dx, dI1_dx, dI2_dx, dm2es3_dx, dm2e2s3_dx, dC1_dx;
    for (int i = 0; i < ncomp; i++) {
        dzeta3_dx = PI/6.*den*cppargs.m[i]*pow(d[i],3);
        dI1_dx = 0.0;
        dI2_dx = 0.0;
        dm2es3_dx = 0.0;
        dm2e2s3_dx = 0.0;
        for (int l = 0; l < 7; l++) {
            daa_dx = cppargs.m[i]/m_avg/m_avg*a1[l] + cppargs.m[i]/m_avg/m_avg*(3-4/m_avg)*a2[l];
            db_dx = cppargs.m[i]/m_avg/m_avg*b1[l] + cppargs.m[i]/m_avg/m_avg*(3-4/m_avg)*b2[l];
            dI1_dx += a[l]*l*dzeta3_dx*pow(eta,l-1) + daa_dx*pow(eta,l);
            dI2_dx += b[l]*l*dzeta3_dx*pow(eta,l-1) + db_dx*pow(eta,l);
        }
        for (int j = 0; j < ncomp; j++) {
            dm2es3_dx += cppargs.x[j]*cppargs.m[j]*(e_ij[i*ncomp+j]/t)*pow(s_ij[i*ncomp+j],3);
            dm2e2s3_dx += cppargs.x[j]*cppargs.m[j]*pow(e_ij[i*ncomp+j]/t,2)*pow(s_ij[i*ncomp+j],3);
            dahc_dx[i] += cppargs.x[j]*(cppargs.m[j]-1)/ghs[j*ncomp+j]*dghsii_dx[i*ncomp+j];
        }
        dm2es3_dx = dm2es3_dx*2*cppargs.m[i];
        dm2e2s3_dx = dm2e2s3_dx*2*cppargs.m[i];
        dahc_dx[i] = cppargs.m[i]*ares_hs + m_avg*dahs_dx[i] - dahc_dx[i] - (cppargs.m[i]-1)*log(ghs[i*ncomp+i]);
        dC1_dx = C2*dzeta3_dx - C1*C1*(cppargs.m[i]*(8*eta-2*eta*eta)/pow(1-eta,4) -
            cppargs.m[i]*(20*eta-27*eta*eta+12*pow(eta,3)-2*pow(eta,4))/pow((1-eta)*(2-eta),2));

        dadisp_dx[i] = -2*PI*den*(dI1_dx*m2es3 + I1*dm2es3_dx) - PI*den
            *((cppargs.m[i]*C1*I2 + m_avg*dC1_dx*I2 + m_avg*C1*dI2_dx)*m2e2s3
            + m_avg*C1*I2*dm2e2s3_dx);
    }

    vector<double> mu_hc(ncomp, 0);
    vector<double> mu_disp(ncomp, 0);
    for (int i = 0; i < ncomp; i++) {
        for (int j = 0; j < ncomp; j++) {
            mu_hc[i] += cppargs.x[j]*dahc_dx[j];
            mu_disp[i] += cppargs.x[j]*dadisp_dx[j];
        }
        mu_hc[i] = ares_hc + Zhc + dahc_dx[i] - mu_hc[i];
        mu_disp[i] = ares_disp + Zdisp + dadisp_dx[i] - mu_disp[i];
    }

    // Dipole term (Gross and Vrabec term) --------------------------------------
    vector<double> mu_polar(ncomp, 0);
    if (!cppargs.dipm.empty()) {
        double A2 = 0.;
        double A3 = 0.;
        double dA2_det = 0.;
        double dA3_det = 0.;
        vector<double> dA2_dx(ncomp, 0);
        vector<double> dA3_dx(ncomp, 0);

        static double a0dip[5] = { 0.3043504, -0.1358588, 1.4493329, 0.3556977, -2.0653308 };
        static double a1dip[5] = { 0.9534641, -1.8396383, 2.0131180, -7.3724958, 8.2374135 };
        static double a2dip[5] = { -1.1610080, 4.5258607, 0.9751222, -12.281038, 5.9397575 };
        static double b0dip[5] = { 0.2187939, -1.1896431, 1.1626889, 0, 0 };
        static double b1dip[5] = { -0.5873164, 1.2489132, -0.5085280, 0, 0 };
        static double b2dip[5] = { 3.4869576, -14.915974, 15.372022, 0, 0 };
        static double c0dip[5] = { -0.0646774, 0.1975882, -0.8087562, 0.6902849, 0 };
        static double c1dip[5] = { -0.9520876, 2.9924258, -2.3802636, -0.2701261, 0 };
        static double c2dip[5] = { -0.6260979, 1.2924686, 1.6542783, -3.4396744, 0 };

        const static double conv = 7242.702976750923; // conversion factor, see the note below Table 2 in Gross and Vrabec 2006

        vector<double> dipmSQ (ncomp, 0);
        for (int i = 0; i < ncomp; i++) {
            dipmSQ[i] = pow(cppargs.dipm[i], 2.)/(cppargs.m[i]*cppargs.e[i]*pow(cppargs.s[i],3.))*conv;
        }

        vector<double> adip (5, 0);
        vector<double> bdip (5, 0);
        vector<double> cdip (5, 0);
        double J2, dJ2_det, J3, dJ3_det;
        double m_ij;
        double m_ijk;
        for (int i = 0; i < ncomp; i++) {
            for (int j = 0; j < ncomp; j++) {
                m_ij = sqrt(cppargs.m[i]*cppargs.m[j]);
                if (m_ij > 2) {
                    m_ij = 2;
                }
                J2 = 0.;
                dJ2_det = 0.;
                for (int l = 0; l < 5; l++) {
                    adip[l] = a0dip[l] + (m_ij-1)/m_ij*a1dip[l] + (m_ij-1)/m_ij*(m_ij-2)/m_ij*a2dip[l];
                    bdip[l] = b0dip[l] + (m_ij-1)/m_ij*b1dip[l] + (m_ij-1)/m_ij*(m_ij-2)/m_ij*b2dip[l];
                    J2 += (adip[l] + bdip[l]*e_ij[j*ncomp+j]/t)*pow(eta, l); // j*ncomp+j needs to be used for e_ij because it is formatted as a 1D vector
                    dJ2_det += (adip[l] + bdip[l]*e_ij[j*ncomp+j]/t)*l*pow(eta, l-1);
                }
                A2 += cppargs.x[i]*cppargs.x[j]*e_ij[i*ncomp+i]/t*e_ij[j*ncomp+j]/t*pow(s_ij[i*ncomp+i],3)*pow(s_ij[j*ncomp+j],3)/
                    pow(s_ij[i*ncomp+j],3)*cppargs.dip_num[i]*cppargs.dip_num[j]*dipmSQ[i]*dipmSQ[j]*J2;
                dA2_det += cppargs.x[i]*cppargs.x[j]*e_ij[i*ncomp+i]/t*e_ij[j*ncomp+j]/t*pow(s_ij[i*ncomp+i],3)*
                    pow(s_ij[j*ncomp+j],3)/pow(s_ij[i*ncomp+j],3)*cppargs.dip_num[i]*cppargs.dip_num[j]*dipmSQ[i]*dipmSQ[j]*dJ2_det;
                if (i == j) {
                    dA2_dx[i] += e_ij[i*ncomp+i]/t*e_ij[j*ncomp+j]/t*pow(s_ij[i*ncomp+i],3)*pow(s_ij[j*ncomp+j],3)
                        /pow(s_ij[i*ncomp+j],3)*cppargs.dip_num[i]*cppargs.dip_num[j]*dipmSQ[i]*dipmSQ[j]*
                        (cppargs.x[i]*cppargs.x[j]*dJ2_det*PI/6.*den*cppargs.m[i]*pow(d[i],3) + 2*cppargs.x[j]*J2);
                }
                else {
                    dA2_dx[i] += e_ij[i*ncomp+i]/t*e_ij[j*ncomp+j]/t*pow(s_ij[i*ncomp+i],3)*pow(s_ij[j*ncomp+j],3)
                        /pow(s_ij[i*ncomp+j],3)*cppargs.dip_num[i]*cppargs.dip_num[j]*dipmSQ[i]*dipmSQ[j]*
                        (cppargs.x[i]*cppargs.x[j]*dJ2_det*PI/6.*den*cppargs.m[i]*pow(d[i],3) + cppargs.x[j]*J2);
                }

                for (int k = 0; k < ncomp; k++) {
                    m_ijk = pow((cppargs.m[i]*cppargs.m[j]*cppargs.m[k]),1/3.);
                    if (m_ijk > 2) {
                        m_ijk = 2;
                    }
                    J3 = 0.;
                    dJ3_det = 0.;
                    for (int l = 0; l < 5; l++) {
                        cdip[l] = c0dip[l] + (m_ijk-1)/m_ijk*c1dip[l] + (m_ijk-1)/m_ijk*(m_ijk-2)/m_ijk*c2dip[l];
                        J3 += cdip[l]*pow(eta, l);
                        dJ3_det += cdip[l]*l*pow(eta, (l-1));
                    }
                    A3 += cppargs.x[i]*cppargs.x[j]*cppargs.x[k]*e_ij[i*ncomp+i]/t*e_ij[j*ncomp+j]/t*e_ij[k*ncomp+k]/t*
                        pow(s_ij[i*ncomp+i],3)*pow(s_ij[j*ncomp+j],3)*pow(s_ij[k*ncomp+k],3)/s_ij[i*ncomp+j]/s_ij[i*ncomp+k]/
                        s_ij[j*ncomp+k]*cppargs.dip_num[i]*cppargs.dip_num[j]*cppargs.dip_num[k]*dipmSQ[i]*
                        dipmSQ[j]*dipmSQ[k]*J3;
                    dA3_det += cppargs.x[i]*cppargs.x[j]*cppargs.x[k]*e_ij[i*ncomp+i]/t*e_ij[j*ncomp+j]/t*e_ij[k*ncomp+k]/t*
                        pow(s_ij[i*ncomp+i],3)*pow(s_ij[j*ncomp+j],3)*pow(s_ij[k*ncomp+k],3)/s_ij[i*ncomp+j]/s_ij[i*ncomp+k]/
                        s_ij[j*ncomp+k]*cppargs.dip_num[i]*cppargs.dip_num[j]*cppargs.dip_num[k]*dipmSQ[i]*
                        dipmSQ[j]*dipmSQ[k]*dJ3_det;
                    if ((i == j) && (i == k)) {
                        dA3_dx[i] += e_ij[i*ncomp+i]/t*e_ij[j*ncomp+j]/t*e_ij[k*ncomp+k]/t*pow(s_ij[i*ncomp+i],3)
                            *pow(s_ij[j*ncomp+j],3)*pow(s_ij[k*ncomp+k],3)/s_ij[i*ncomp+j]/s_ij[i*ncomp+k]/s_ij[j*ncomp+k]
                            *cppargs.dip_num[i]*cppargs.dip_num[j]*cppargs.dip_num[k]*dipmSQ[i]*dipmSQ[j]
                            *dipmSQ[k]*(cppargs.x[i]*cppargs.x[j]*cppargs.x[k]*dJ3_det*PI/6.*den*cppargs.m[i]*pow(d[i],3)
                            + 3*cppargs.x[j]*cppargs.x[k]*J3);
                    }
                    else if ((i == j) || (i == k)) {
                        dA3_dx[i] += e_ij[i*ncomp+i]/t*e_ij[j*ncomp+j]/t*e_ij[k*ncomp+k]/t*pow(s_ij[i*ncomp+i],3)
                            *pow(s_ij[j*ncomp+j],3)*pow(s_ij[k*ncomp+k],3)/s_ij[i*ncomp+j]/s_ij[i*ncomp+k]/s_ij[j*ncomp+k]
                            *cppargs.dip_num[i]*cppargs.dip_num[j]*cppargs.dip_num[k]*dipmSQ[i]*dipmSQ[j]
                            *dipmSQ[k]*(cppargs.x[i]*cppargs.x[j]*cppargs.x[k]*dJ3_det*PI/6.*den*cppargs.m[i]*pow(d[i],3)
                            + 2*cppargs.x[j]*cppargs.x[k]*J3);
                    }
                    else {
                        dA3_dx[i] += e_ij[i*ncomp+i]/t*e_ij[j*ncomp+j]/t*e_ij[k*ncomp+k]/t*pow(s_ij[i*ncomp+i],3)
                            *pow(s_ij[j*ncomp+j],3)*pow(s_ij[k*ncomp+k],3)/s_ij[i*ncomp+j]/s_ij[i*ncomp+k]/s_ij[j*ncomp+k]
                            *cppargs.dip_num[i]*cppargs.dip_num[j]*cppargs.dip_num[k]*dipmSQ[i]*dipmSQ[j]
                            *dipmSQ[k]*(cppargs.x[i]*cppargs.x[j]*cppargs.x[k]*dJ3_det*PI/6.*den*cppargs.m[i]*pow(d[i],3)
                            + cppargs.x[j]*cppargs.x[k]*J3);
                    }
                }
            }
        }

        A2 = -PI*den*A2;
        A3 = -4/3.*PI*PI*den*den*A3;
        dA2_det = -PI*den*dA2_det;
        dA3_det = -4/3.*PI*PI*den*den*dA3_det;
        for (int i = 0; i < ncomp; i++) {
            dA2_dx[i] = -PI*den*dA2_dx[i];
            dA3_dx[i] = -4/3.*PI*PI*den*den*dA3_dx[i];
        }

        vector<double> dapolar_dx(ncomp);
        for (int i = 0; i < ncomp; i++) {
            dapolar_dx[i] = (dA2_dx[i]*(1-A3/A2) + (dA3_dx[i]*A2 - A3*dA2_dx[i])/A2)/pow(1-A3/A2,2);
        }

        double ares_polar = A2/(1-A3/A2);
        double Zpolar = eta*((dA2_det*(1-A3/A2)+(dA3_det*A2-A3*dA2_det)/A2)/(1-A3/A2)/(1-A3/A2));
        for (int i = 0; i < ncomp; i++) {
            for (int j = 0; j < ncomp; j++) {
                mu_polar[i] += cppargs.x[j]*dapolar_dx[j];
            }
            mu_polar[i] = ares_polar + Zpolar + dapolar_dx[i] - mu_polar[i];
        }
    }

    // Association term -------------------------------------------------------
    // only the 2B association type is currently implemented
    vector<double> mu_assoc(ncomp, 0);
    if (!cppargs.e_assoc.empty()) {
        int a_sites = 2;
        int ncA = count_if(cppargs.vol_a.begin(), cppargs.vol_a.end(), IsNotZero); // number of associating compounds in the fluid

        vector<int> iA (ncA, 0); //indices of associating compounds
        int ctr = 0;
        for (int i = 0; i < ncomp; i++) {
            if (cppargs.vol_a[i] != 0.0) {
                iA[ctr] = i;
                ctr += 1;
            }
        }

        vector<double> XA (ncA*a_sites, 0);
        vector<double> eABij (ncA*ncA, 0);
        vector<double> volABij (ncA*ncA, 0);
        vector<double> delta_ij (ncA*ncA, 0);
        vector<double> ddelta_dd (ncA*ncA*ncomp, 0);

        // these indices are necessary because we are only using 1D vectors
        int idxa = -1; // index over only associating compounds
        int idxi = 0; // index for the ii-th compound
        int idxj = 0; // index for the jj-th compound
        int idx_ddelta = -1; // index for ddelta_dd vector
        double dghsd_dd;
        for (int i = 0; i < ncA; i++) {
            idxi = iA[i]*ncomp+iA[i];
            for (int j = 0; j < ncA; j++) {
                idxa += 1;
                idxj = iA[j]*ncomp+iA[j];
                eABij[idxa] = (cppargs.e_assoc[iA[i]]+cppargs.e_assoc[iA[j]])/2.;
                if (cppargs.k_hb.empty()) {
                    volABij[idxa] = sqrt(cppargs.vol_a[iA[i]]*cppargs.vol_a[iA[j]])*pow(sqrt(s_ij[idxi]*
                        s_ij[idxj])/(0.5*(s_ij[idxi]+s_ij[idxj])), 3);
                }
                else {
                    volABij[idxa] = sqrt(cppargs.vol_a[iA[i]]*cppargs.vol_a[iA[j]])*pow(sqrt(s_ij[idxi]*
                        s_ij[idxj])/(0.5*(s_ij[idxi]+s_ij[idxj])), 3)*(1-cppargs.k_hb[iA[i]*ncomp+iA[j]]);
                }
                delta_ij[idxa] = ghs[iA[i]*ncomp+iA[j]]*(exp(eABij[idxa]/t)-1)*pow(s_ij[iA[i]*ncomp+iA[j]], 3)*volABij[idxa];
                for (int k = 0; k < ncomp; k++) {
                    idx_ddelta += 1;
                    dghsd_dd = PI/6.*cppargs.m[k]*(pow(d[k], 3)/(1-zeta[3])/(1-zeta[3]) + 3*d[iA[i]]*d[iA[j]]/
                        (d[iA[i]]+d[iA[j]])*(d[k]*d[k]/(1-zeta[3])/(1-zeta[3])+2*pow(d[k], 3)*
                        zeta[2]/pow(1-zeta[3], 3)) + 2*pow((d[iA[i]]*d[iA[j]]/(d[iA[i]]+d[iA[j]])), 2)*
                        (2*d[k]*d[k]*zeta[2]/pow(1-zeta[3], 3)+3*(pow(d[k], 3)*zeta[2]*zeta[2]
                        /pow(1-zeta[3], 4))));
                    ddelta_dd[idx_ddelta] = dghsd_dd*(exp(eABij[idxa]/t)-1)*pow(s_ij[iA[i]*ncomp+iA[j]], 3)*volABij[idxa];
                }
            }
            XA[i*2] = (-1 + sqrt(1+8*den*delta_ij[i*ncA+i]))/(4*den*delta_ij[i*ncA+i]);
            if (!std::isfinite(XA[i*2])) {
                XA[i*2] = 0.02;
            }
            XA[i*2+1] = XA[i*2];
        }

        vector<double> x_assoc(ncA); // mole fractions of only the associating compounds
        for (int i = 0; i < ncA; i++) {
            x_assoc[i] = cppargs.x[iA[i]];
        }

        ctr = 0;
        double dif = 1000.;
        vector<double> XA_old = XA;
        while ((ctr < 500) && (dif > 1e-9)) {
            ctr += 1;
            XA = XA_find(XA, ncA, delta_ij, den, x_assoc);
            dif = 0.;
            for (int i = 0; i < ncA*2; i++) {
                dif += abs(XA[i] - XA_old[i]);
            }
            XA_old = XA;
        }

        vector<double> dXA_dd(ncA*a_sites*ncomp, 0);
        dXA_dd = dXA_find(ncA, ncomp, iA, delta_ij, den, XA, ddelta_dd, x_assoc, a_sites);

        for (int i = 0; i < ncomp; i++) {
            for (int j = 0; j < ncA; j++) {
                for (int k = 0; k < a_sites; k++) {
                    mu_assoc[i] += cppargs.x[iA[j]]*den*dXA_dd[i*(ncA*a_sites)+j*a_sites+k]*(1/XA[j*a_sites+k]-0.5);
                }
            }
        }

        for (int i = 0; i < ncA; i++) {
            for (int l = 0; l < a_sites; l++) {
                mu_assoc[iA[i]] += log(XA[i*a_sites+l])-0.5*XA[i*a_sites+l];
            }
            mu_assoc[iA[i]] += 0.5*a_sites;
        }
    }

    // Ion term ---------------------------------------------------------------
    vector<double> mu_ion(ncomp, 0);
    if (!cppargs.z.empty()) {
        vector<double> q(cppargs.z.begin(), cppargs.z.end());
        for (int i = 0; i < ncomp; i++) {
            q[i] = q[i]*E_CHRG;
        }

        summ = 0.;
        for (int i = 0; i < ncomp; i++) {
            summ += cppargs.z[i]*cppargs.z[i]*cppargs.x[i];
        }
        double kappa = sqrt(den*E_CHRG*E_CHRG/kb/t/(cppargs.dielc*perm_vac)*summ); // the inverse Debye screening length. Equation 4 in Held et al. 2008.

        if (kappa != 0) {
            vector<double> chi(ncomp);
            vector<double> sigma_k(ncomp);
            double summ1 = 0.;
            double summ2 = 0.;
            for (int i = 0; i < ncomp; i++) {
                chi[i] = 3/pow(kappa*cppargs.s[i], 3)*(1.5 + log(1+kappa*cppargs.s[i]) - 2*(1+kappa*cppargs.s[i]) +
                    0.5*pow(1+kappa*cppargs.s[i], 2));
                sigma_k[i] = -2*chi[i]+3/(1+kappa*cppargs.s[i]);
                summ1 += q[i]*q[i]*cppargs.x[i]*sigma_k[i];
                summ2 += cppargs.x[i]*q[i]*q[i];
            }

            for (int i = 0; i < ncomp; i++) {
                mu_ion[i] = -q[i]*q[i]*kappa/24./PI/kb/t/(cppargs.dielc*perm_vac)*
                    (2*chi[i] + summ1/summ2);
            }
        }
    }

    double Z = pcsaft_Z_cpp(t, rho, cppargs);

    vector<double> mu(ncomp, 0);
    vector<double> fugcoef(ncomp, 0);
    for (int i = 0; i < ncomp; i++) {
        mu[i] = mu_hc[i] + mu_disp[i] + mu_polar[i] + mu_assoc[i] + mu_ion[i];
        fugcoef[i] = exp(mu[i] - log(Z)); // the fugacity coefficients
    }

    return fugcoef;
}


double pcsaft_p_cpp(double t, double rho, add_args &cppargs) {
    /**
    Calculate pressure.

    Parameters
    ----------
    x : vector<double>, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : vector<double>, shape (n,)
        Segment number for each component.
    s : vector<double>, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : vector<double>, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : double
        Temperature (K)
    rho : double
        Molar density (mol m^{-3})
    cppargs : add_args
        A struct containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : vector<double>, shape (n*n,)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : vector<double>, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : vector<double>, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : vector<double>, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : vector<double>, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : vector<double>, shape (n,)
            Charge number of the ions
        dielc : double
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    P : double
        Pressure (Pa)
    */
    double den = rho*N_AV/1.0e30;

    double Z = pcsaft_Z_cpp(t, rho, cppargs);
    double P = Z*kb*t*den*1.0e30; // Pa
    return P;
}


double pcsaft_ares_cpp(double t, double rho, add_args &cppargs) {
    /**
    Calculates the residual Helmholtz energy.

    Parameters
    ----------
    x : vector<double>, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : vector<double>, shape (n,)
        Segment number for each component.
    s : vector<double>, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : vector<double>, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : double
        Temperature (K)
    rho : double
        Molar density (mol m^-3)
    cppargs : add_args
        A struct containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : vector<double>, shape (n*n,)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : vector<double>, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : vector<double>, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : vector<double>, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : vector<double>, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : vector<double>, shape (n,)
            Charge number of the ions
        dielc : double
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    ares : double
        Residual Helmholtz energy (J mol^-1)
    */
    int ncomp = cppargs.x.size(); // number of components
    vector<double> d (ncomp);
    for (int i = 0; i < ncomp; i++) {
        d[i] = cppargs.s[i]*(1-0.12*exp(-3*cppargs.e[i]/t));
    }
    if (!cppargs.z.empty()) {
        for (int i = 0; i < ncomp; i++) {
            if (cppargs.z[i] != 0) {
                d[i] = cppargs.s[i]*(1-0.12); // for ions the diameter is assumed to be temperature independent (see Held et al. 2014)
            }
        }
    }

    double den = rho*N_AV/1.0e30;

    vector<double> zeta (4, 0);
    double summ;
    for (int i = 0; i < 4; i++) {
        summ = 0;
        for (int j = 0; j < ncomp; j++) {
            summ += cppargs.x[j]*cppargs.m[j]*pow(d[j], i);
        }
        zeta[i] = PI/6*den*summ;
    }

    double eta = zeta[3];
    double m_avg = 0;
    for (int i = 0; i < ncomp; i++) {
        m_avg += cppargs.x[i]*cppargs.m[i];
    }

    vector<double> ghs (ncomp*ncomp, 0);
    vector<double> e_ij (ncomp*ncomp, 0);
    vector<double> s_ij (ncomp*ncomp, 0);
    double m2es3 = 0.;
    double m2e2s3 = 0.;
    int idx = -1;
    for (int i = 0; i < ncomp; i++) {
        for (int j = 0; j < ncomp; j++) {
            idx += 1;
            if (cppargs.l_ij.empty()) {
                s_ij[idx] = (cppargs.s[i] + cppargs.s[j])/2.;
            }
            else {
                s_ij[idx] = (cppargs.s[i] + cppargs.s[j])/2.*(1-cppargs.l_ij[idx]);
            }
            if (!cppargs.z.empty()) {
                if (cppargs.z[i]*cppargs.z[j] <= 0) { // for two cations or two anions e_ij is kept at zero to avoid dispersion between like ions (see Held et al. 2014)
                    if (cppargs.k_ij.empty()) {
                        e_ij[idx] = sqrt(cppargs.e[i]*cppargs.e[j]);
                    }
                    else {
                        e_ij[idx] = sqrt(cppargs.e[i]*cppargs.e[j])*(1-cppargs.k_ij[idx]);
                    }
                }
            } else {
                if (cppargs.k_ij.empty()) {
                    e_ij[idx] = sqrt(cppargs.e[i]*cppargs.e[j]);
                }
                else {
                    e_ij[idx] = sqrt(cppargs.e[i]*cppargs.e[j])*(1-cppargs.k_ij[idx]);
                }
            }
            m2es3 = m2es3 + cppargs.x[i]*cppargs.x[j]*cppargs.m[i]*cppargs.m[j]*e_ij[idx]/t*pow(s_ij[idx], 3);
            m2e2s3 = m2e2s3 + cppargs.x[i]*cppargs.x[j]*cppargs.m[i]*cppargs.m[j]*pow(e_ij[idx]/t,2)*pow(s_ij[idx], 3);
            ghs[idx] = 1/(1-zeta[3]) + (d[i]*d[j]/(d[i]+d[j]))*3*zeta[2]/(1-zeta[3])/(1-zeta[3]) +
                pow(d[i]*d[j]/(d[i]+d[j]), 2)*2*zeta[2]*zeta[2]/pow(1-zeta[3], 3);
        }
    }

    double ares_hs = 1/zeta[0]*(3*zeta[1]*zeta[2]/(1-zeta[3]) + pow(zeta[2], 3.)/(zeta[3]*pow(1-zeta[3],2))
            + (pow(zeta[2], 3.)/pow(zeta[3], 2.) - zeta[0])*log(1-zeta[3]));

    static double a0[7] = { 0.910563145, 0.636128145, 2.686134789, -26.54736249, 97.75920878, -159.5915409, 91.29777408 };
    static double a1[7] = { -0.308401692, 0.186053116, -2.503004726, 21.41979363, -65.25588533, 83.31868048, -33.74692293 };
    static double a2[7] = { -0.090614835, 0.452784281, 0.596270073, -1.724182913, -4.130211253, 13.77663187, -8.672847037 };
    static double b0[7] = { 0.724094694, 2.238279186, -4.002584949, -21.00357682, 26.85564136, 206.5513384, -355.6023561 };
    static double b1[7] = { -0.575549808, 0.699509552, 3.892567339, -17.21547165, 192.6722645, -161.8264617, -165.2076935 };
    static double b2[7] = { 0.097688312, -0.255757498, -9.155856153, 20.64207597, -38.80443005, 93.62677408, -29.66690559 };

    vector<double> a (7, 0);
    vector<double> b (7, 0);
    for (int i = 0; i < 7; i++) {
        a[i] = a0[i] + (m_avg-1.)/m_avg*a1[i] + (m_avg-1.)/m_avg*(m_avg-2.)/m_avg*a2[i];
        b[i] = b0[i] + (m_avg-1.)/m_avg*b1[i] + (m_avg-1.)/m_avg*(m_avg-2.)/m_avg*b2[i];
    }

    double I1 = 0.0;
    double I2 = 0.0;
    for (int i = 0; i < 7; i++) {
        I1 += a[i]*pow(eta, i);
        I2 += b[i]*pow(eta, i);
    }
    double C1 = 1./(1. + m_avg*(8*eta-2*eta*eta)/pow(1-eta, 4) + (1-m_avg)*(20*eta-27*eta*eta+12*pow(eta, 3)-2*pow(eta, 4))/pow((1-eta)*(2-eta), 2.0));

    summ = 0.0;
    for (int i = 0; i < ncomp; i++) {
        summ += cppargs.x[i]*(cppargs.m[i]-1)*log(ghs[i*ncomp+i]);
    }

    double ares_hc = m_avg*ares_hs - summ;
    double ares_disp = -2*PI*den*I1*m2es3 - PI*den*m_avg*C1*I2*m2e2s3;

    // Dipole term (Gross and Vrabec term) --------------------------------------
    double ares_polar = 0.;
    if (!cppargs.dipm.empty()) {
        double A2 = 0.;
        double A3 = 0.;
        vector<double> dipmSQ (ncomp, 0);

        static double a0dip[5] = { 0.3043504, -0.1358588, 1.4493329, 0.3556977, -2.0653308 };
        static double a1dip[5] = { 0.9534641, -1.8396383, 2.0131180, -7.3724958, 8.2374135 };
        static double a2dip[5] = { -1.1610080, 4.5258607, 0.9751222, -12.281038, 5.9397575 };
        static double b0dip[5] = { 0.2187939, -1.1896431, 1.1626889, 0, 0 };
        static double b1dip[5] = { -0.5873164, 1.2489132, -0.5085280, 0, 0 };
        static double b2dip[5] = { 3.4869576, -14.915974, 15.372022, 0, 0 };
        static double c0dip[5] = { -0.0646774, 0.1975882, -0.8087562, 0.6902849, 0 };
        static double c1dip[5] = { -0.9520876, 2.9924258, -2.3802636, -0.2701261, 0 };
        static double c2dip[5] = { -0.6260979, 1.2924686, 1.6542783, -3.4396744, 0 };

        const static double conv = 7242.702976750923; // conversion factor, see the note below Table 2 in Gross and Vrabec 2006

        for (int i = 0; i < ncomp; i++) {
            dipmSQ[i] = pow(cppargs.dipm[i], 2.)/(cppargs.m[i]*cppargs.e[i]*pow(cppargs.s[i],3.))*conv;
        }

        vector<double> adip (5, 0);
        vector<double> bdip (5, 0);
        vector<double> cdip (5, 0);
        double J2, J3;
        double m_ij;
        double m_ijk;
        for (int i = 0; i < ncomp; i++) {
            for (int j = 0; j < ncomp; j++) {
                m_ij = sqrt(cppargs.m[i]*cppargs.m[j]);
                if (m_ij > 2) {
                    m_ij = 2;
                }
                J2 = 0.;
                for (int l = 0; l < 5; l++) {
                    adip[l] = a0dip[l] + (m_ij-1)/m_ij*a1dip[l] + (m_ij-1)/m_ij*(m_ij-2)/m_ij*a2dip[l];
                    bdip[l] = b0dip[l] + (m_ij-1)/m_ij*b1dip[l] + (m_ij-1)/m_ij*(m_ij-2)/m_ij*b2dip[l];
                    J2 += (adip[l] + bdip[l]*e_ij[j*ncomp+j]/t)*pow(eta, l); // j*ncomp+j needs to be used for e_ij because it is formatted as a 1D vector
                }
                A2 += cppargs.x[i]*cppargs.x[j]*e_ij[i*ncomp+i]/t*e_ij[j*ncomp+j]/t*pow(s_ij[i*ncomp+i],3)*pow(s_ij[j*ncomp+j],3)/
                    pow(s_ij[i*ncomp+j],3)*cppargs.dip_num[i]*cppargs.dip_num[j]*dipmSQ[i]*dipmSQ[j]*J2;

                for (int k = 0; k < ncomp; k++) {
                    m_ijk = pow((cppargs.m[i]*cppargs.m[j]*cppargs.m[k]),1/3.);
                    if (m_ijk > 2) {
                        m_ijk = 2;
                    }
                    J3 = 0.;
                    for (int l = 0; l < 5; l++) {
                        cdip[l] = c0dip[l] + (m_ijk-1)/m_ijk*c1dip[l] + (m_ijk-1)/m_ijk*(m_ijk-2)/m_ijk*c2dip[l];
                        J3 += cdip[l]*pow(eta, l);
                    }
                    A3 += cppargs.x[i]*cppargs.x[j]*cppargs.x[k]*e_ij[i*ncomp+i]/t*e_ij[j*ncomp+j]/t*e_ij[k*ncomp+k]/t*
                        pow(s_ij[i*ncomp+i],3)*pow(s_ij[j*ncomp+j],3)*pow(s_ij[k*ncomp+k],3)/s_ij[i*ncomp+j]/s_ij[i*ncomp+k]/
                        s_ij[j*ncomp+k]*cppargs.dip_num[i]*cppargs.dip_num[j]*cppargs.dip_num[k]*dipmSQ[i]*
                        dipmSQ[j]*dipmSQ[k]*J3;
                }
            }
        }

        A2 = -PI*den*A2;
        A3 = -4/3.*PI*PI*den*den*A3;

        ares_polar = A2/(1-A3/A2);
    }

    // Association term -------------------------------------------------------
    // only the 2B association type is currently implemented
    double ares_assoc = 0.;
    if (!cppargs.e_assoc.empty()) {
        int a_sites = 2;
        int ncA = count_if(cppargs.vol_a.begin(), cppargs.vol_a.end(), IsNotZero); // number of associating compounds in the fluid

        vector<int> iA (ncA, 0); //indices of associating compounds
        int ctr = 0;
        for (int i = 0; i < ncomp; i++) {
            if (cppargs.vol_a[i] != 0.0) {
                iA[ctr] = i;
                ctr += 1;
            }
        }

        vector<double> XA (ncA*a_sites, 0);
        vector<double> eABij (ncA*ncA, 0);
        vector<double> volABij (ncA*ncA, 0);
        vector<double> delta_ij (ncA*ncA, 0);

        // these indices are necessary because we are only using 1D vectors
        int idxa = -1; // index over only associating compounds
        int idxi = 0; // index for the ii-th compound
        int idxj = 0; // index for the jj-th compound
        for (int i = 0; i < ncA; i++) {
            idxi = iA[i]*ncomp+iA[i];
            for (int j = 0; j < ncA; j++) {
                idxa += 1;
                idxj = iA[j]*ncomp+iA[j];
                eABij[idxa] = (cppargs.e_assoc[iA[i]]+cppargs.e_assoc[iA[j]])/2.;
                if (cppargs.k_hb.empty()) {
                    volABij[idxa] = sqrt(cppargs.vol_a[iA[i]]*cppargs.vol_a[iA[j]])*pow(sqrt(s_ij[idxi]*
                        s_ij[idxj])/(0.5*(s_ij[idxi]+s_ij[idxj])), 3);
                }
                else {
                    volABij[idxa] = sqrt(cppargs.vol_a[iA[i]]*cppargs.vol_a[iA[j]])*pow(sqrt(s_ij[idxi]*
                        s_ij[idxj])/(0.5*(s_ij[idxi]+s_ij[idxj])), 3)*(1-cppargs.k_hb[iA[i]*ncomp+iA[j]]);
                }
                delta_ij[idxa] = ghs[iA[i]*ncomp+iA[j]]*(exp(eABij[idxa]/t)-1)*pow(s_ij[iA[i]*ncomp+iA[j]], 3)*volABij[idxa];
            }
            XA[i*2] = (-1 + sqrt(1+8*den*delta_ij[i*ncA+i]))/(4*den*delta_ij[i*ncA+i]);
            if (!std::isfinite(XA[i*2])) {
                XA[i*2] = 0.02;
            }
            XA[i*2+1] = XA[i*2];
        }

        vector<double> x_assoc(ncA); // mole fractions of only the associating compounds
        for (int i = 0; i < ncA; i++) {
            x_assoc[i] = cppargs.x[iA[i]];
        }

        ctr = 0;
        double dif = 1000.;
        vector<double> XA_old = XA;
        while ((ctr < 500) && (dif > 1e-9)) {
            ctr += 1;
            XA = XA_find(XA, ncA, delta_ij, den, x_assoc);
            dif = 0.;
            for (int i = 0; i < ncA*2; i++) {
                dif += abs(XA[i] - XA_old[i]);
            }
            XA_old = XA;
        }

        ares_assoc = 0.;
        for (int i = 0; i < ncA; i++) {
            for (int k = 0; k < a_sites; k++) {
                ares_assoc += cppargs.x[iA[i]]*(log(XA[i*a_sites+k])-0.5*XA[i*a_sites+k] + 0.5);
            }
        }
    }

    // Ion term ---------------------------------------------------------------
    double ares_ion = 0.;
    if (!cppargs.z.empty()) {
        vector<double> q(cppargs.z.begin(), cppargs.z.end());
        for (int i = 0; i < ncomp; i++) {
            q[i] = q[i]*E_CHRG;
        }

        summ = 0.;
        for (int i = 0; i < ncomp; i++) {
            summ += cppargs.z[i]*cppargs.z[i]*cppargs.x[i];
        }
        double kappa = sqrt(den*E_CHRG*E_CHRG/kb/t/(cppargs.dielc*perm_vac)*summ); // the inverse Debye screening length. Equation 4 in Held et al. 2008.

        if (kappa != 0) {
            vector<double> chi(ncomp);
            vector<double> sigma_k(ncomp);
            summ = 0.;
            for (int i = 0; i < ncomp; i++) {
                chi[i] = 3/pow(kappa*cppargs.s[i], 3)*(1.5 + log(1+kappa*cppargs.s[i]) - 2*(1+kappa*cppargs.s[i]) +
                    0.5*pow(1+kappa*cppargs.s[i], 2));
                summ += cppargs.x[i]*q[i]*q[i]*chi[i]*kappa;
            }

            ares_ion = -1/12./PI/kb/t/(cppargs.dielc*perm_vac)*summ;
        }
    }

    double ares = ares_hc + ares_disp + ares_polar + ares_assoc + ares_ion;
    return ares;
}


double pcsaft_dadt_cpp(double t, double rho, add_args &cppargs) {
    /**
    Calculate the temperature derivative of the residual Helmholtz energy at
    constant density.

    Parameters
    ----------
    x : vector<double>, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : vector<double>, shape (n,)
        Segment number for each component.
    s : vector<double>, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : vector<double>, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : double
        Temperature (K)
    rho : double
        Molar density (mol m^-3)
    cppargs : add_args
        A struct containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : vector<double>, shape (n*n,)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : vector<double>, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : vector<double>, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : vector<double>, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : vector<double>, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : vector<double>, shape (n,)
            Charge number of the ions
        dielc : double
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    dadt : double
        Temperature derivative of residual Helmholtz energy at constant density (J mol^-1 K^-1)
    */
    int ncomp = cppargs.x.size(); // number of components
    vector<double> d (ncomp), dd_dt(ncomp);
    for (int i = 0; i < ncomp; i++) {
        d[i] = cppargs.s[i]*(1-0.12*exp(-3*cppargs.e[i]/t));
        dd_dt[i] = cppargs.s[i]*-3*cppargs.e[i]/t/t*0.12*exp(-3*cppargs.e[i]/t);
    }
    if (!cppargs.z.empty()) {
        for (int i = 0; i < ncomp; i++) {
            if (cppargs.z[i] != 0) {
                d[i] = cppargs.s[i]*(1-0.12); // for ions the diameter is assumed to be temperature independent (see Held et al. 2014)
                dd_dt[i] = 0.;
            }
        }
    }

    double den = rho*N_AV/1.0e30;

    vector<double> zeta (4, 0);
    double summ;
    for (int i = 0; i < 4; i++) {
        summ = 0;
        for (int j = 0; j < ncomp; j++) {
            summ += cppargs.x[j]*cppargs.m[j]*pow(d[j], i);
        }
        zeta[i] = PI/6*den*summ;
    }

    vector<double> dzeta_dt (4, 0);
    for (int i = 1; i < 4; i++) {
        summ = 0;
        for (int j = 0; j < ncomp; j++) {
            summ += cppargs.x[j]*cppargs.m[j]*i*dd_dt[j]*pow(d[j],(i-1));
        }
        dzeta_dt[i] = PI/6*den*summ;
    }

    double eta = zeta[3];
    double m_avg = 0;
    for (int i = 0; i < ncomp; i++) {
        m_avg += cppargs.x[i]*cppargs.m[i];
    }

    vector<double> ghs (ncomp*ncomp, 0);
    vector<double> dghs_dt (ncomp*ncomp, 0);
    vector<double> e_ij (ncomp*ncomp, 0);
    vector<double> s_ij (ncomp*ncomp, 0);
    double m2es3 = 0.;
    double m2e2s3 = 0.;
    double ddij_dt;
    int idx = -1;
    for (int i = 0; i < ncomp; i++) {
        for (int j = 0; j < ncomp; j++) {
            idx += 1;
            if (cppargs.l_ij.empty()) {
                s_ij[idx] = (cppargs.s[i] + cppargs.s[j])/2.;
            }
            else {
                s_ij[idx] = (cppargs.s[i] + cppargs.s[j])/2.*(1-cppargs.l_ij[idx]);
            }
            if (!cppargs.z.empty()) {
                if (cppargs.z[i]*cppargs.z[j] <= 0) { // for two cations or two anions e_ij is kept at zero to avoid dispersion between like ions (see Held et al. 2014)
                    if (cppargs.k_ij.empty()) {
                        e_ij[idx] = sqrt(cppargs.e[i]*cppargs.e[j]);
                    }
                    else {
                        e_ij[idx] = sqrt(cppargs.e[i]*cppargs.e[j])*(1-cppargs.k_ij[idx]);
                    }
                }
            } else {
                if (cppargs.k_ij.empty()) {
                    e_ij[idx] = sqrt(cppargs.e[i]*cppargs.e[j]);
                }
                else {
                    e_ij[idx] = sqrt(cppargs.e[i]*cppargs.e[j])*(1-cppargs.k_ij[idx]);
                }
            }
            m2es3 = m2es3 + cppargs.x[i]*cppargs.x[j]*cppargs.m[i]*cppargs.m[j]*e_ij[idx]/t*pow(s_ij[idx], 3);
            m2e2s3 = m2e2s3 + cppargs.x[i]*cppargs.x[j]*cppargs.m[i]*cppargs.m[j]*pow(e_ij[idx]/t,2)*pow(s_ij[idx], 3);
            ghs[idx] = 1/(1-zeta[3]) + (d[i]*d[j]/(d[i]+d[j]))*3*zeta[2]/(1-zeta[3])/(1-zeta[3]) +
                    pow(d[i]*d[j]/(d[i]+d[j]), 2)*2*zeta[2]*zeta[2]/pow(1-zeta[3], 3);
            ddij_dt = (d[i]*d[j]/(d[i]+d[j]))*(dd_dt[i]/d[i]+dd_dt[j]/d[j]-(dd_dt[i]+dd_dt[j])/(d[i]+d[j]));
            dghs_dt[idx] = dzeta_dt[3]/pow(1-zeta[3], 2.)
                + 3*(ddij_dt*zeta[2]+(d[i]*d[j]/(d[i]+d[j]))*dzeta_dt[2])/pow(1-zeta[3], 2.)
                + 4*(d[i]*d[j]/(d[i]+d[j]))*zeta[2]*(1.5*dzeta_dt[3]+ddij_dt*zeta[2]
                + (d[i]*d[j]/(d[i]+d[j]))*dzeta_dt[2])/pow(1-zeta[3], 3.)
                + 6*pow((d[i]*d[j]/(d[i]+d[j]))*zeta[2], 2.)*dzeta_dt[3]/pow(1-zeta[3], 4.);
        }
    }

    double dadt_hs = 1/zeta[0]*(3*(dzeta_dt[1]*zeta[2] + zeta[1]*dzeta_dt[2])/(1-zeta[3])
        + 3*zeta[1]*zeta[2]*dzeta_dt[3]/pow(1-zeta[3], 2.)
        + 3*pow(zeta[2], 2.)*dzeta_dt[2]/zeta[3]/pow(1-zeta[3], 2.)
        + pow(zeta[2],3.)*dzeta_dt[3]*(3*zeta[3]-1)/pow(zeta[3], 2.)/pow(1-zeta[3], 3.)
        + (3*pow(zeta[2], 2.)*dzeta_dt[2]*zeta[3] - 2*pow(zeta[2], 3.)*dzeta_dt[3])/pow(zeta[3], 3.)
        * log(1-zeta[3])
        + (zeta[0]-pow(zeta[2],3)/pow(zeta[3],2.))*dzeta_dt[3]/(1-zeta[3]));

    static double a0[7] = { 0.910563145, 0.636128145, 2.686134789, -26.54736249, 97.75920878, -159.5915409, 91.29777408 };
    static double a1[7] = { -0.308401692, 0.186053116, -2.503004726, 21.41979363, -65.25588533, 83.31868048, -33.74692293 };
    static double a2[7] = { -0.090614835, 0.452784281, 0.596270073, -1.724182913, -4.130211253, 13.77663187, -8.672847037 };
    static double b0[7] = { 0.724094694, 2.238279186, -4.002584949, -21.00357682, 26.85564136, 206.5513384, -355.6023561 };
    static double b1[7] = { -0.575549808, 0.699509552, 3.892567339, -17.21547165, 192.6722645, -161.8264617, -165.2076935 };
    static double b2[7] = { 0.097688312, -0.255757498, -9.155856153, 20.64207597, -38.80443005, 93.62677408, -29.66690559 };

    vector<double> a (7, 0);
    vector<double> b (7, 0);
    for (int i = 0; i < 7; i++) {
        a[i] = a0[i] + (m_avg-1.)/m_avg*a1[i] + (m_avg-1.)/m_avg*(m_avg-2.)/m_avg*a2[i];
        b[i] = b0[i] + (m_avg-1.)/m_avg*b1[i] + (m_avg-1.)/m_avg*(m_avg-2.)/m_avg*b2[i];
    }

    double I1 = 0.0;
    double I2 = 0.0;
    double dI1_dt = 0.0, dI2_dt = 0.;
    for (int i = 0; i < 7; i++) {
        I1 += a[i]*pow(eta, i);
        I2 += b[i]*pow(eta, i);
        dI1_dt += a[i]*dzeta_dt[3]*i*pow(eta, i-1);
        dI2_dt += b[i]*dzeta_dt[3]*i*pow(eta, i-1);
    }
    double C1 = 1./(1. + m_avg*(8*eta-2*eta*eta)/pow(1-eta, 4) + (1-m_avg)*(20*eta-27*eta*eta+12*pow(eta, 3)-2*pow(eta, 4))/pow((1-eta)*(2-eta), 2.0));
    double C2 = -1*C1*C1*(m_avg*(-4*eta*eta+20*eta+8)/pow(1-eta,5.) + (1-m_avg)*(2*pow(eta,3)+12*eta*eta-48*eta+40)/pow((1-eta)*(2-eta),3));
    double dC1_dt = C2*dzeta_dt[3];

    summ = 0.;
    for (int i = 0; i < ncomp; i++) {
        summ += cppargs.x[i]*(cppargs.m[i]-1)*dghs_dt[i*ncomp+i]/ghs[i*ncomp+i];
    }

    double dadt_hc = m_avg*dadt_hs - summ;
    double dadt_disp = -2*PI*den*(dI1_dt-I1/t)*m2es3 - PI*den*m_avg*(dC1_dt*I2+C1*dI2_dt-2*C1*I2/t)*m2e2s3;

    // Dipole term (Gross and Vrabec term) --------------------------------------
    double dadt_polar = 0.;
    if (!cppargs.dipm.empty()) {
        double A2 = 0.;
        double A3 = 0.;
        double dA2_dt = 0.;
        double dA3_dt = 0.;
        vector<double> dipmSQ (ncomp, 0);

        static double a0dip[5] = { 0.3043504, -0.1358588, 1.4493329, 0.3556977, -2.0653308 };
        static double a1dip[5] = { 0.9534641, -1.8396383, 2.0131180, -7.3724958, 8.2374135 };
        static double a2dip[5] = { -1.1610080, 4.5258607, 0.9751222, -12.281038, 5.9397575 };
        static double b0dip[5] = { 0.2187939, -1.1896431, 1.1626889, 0, 0 };
        static double b1dip[5] = { -0.5873164, 1.2489132, -0.5085280, 0, 0 };
        static double b2dip[5] = { 3.4869576, -14.915974, 15.372022, 0, 0 };
        static double c0dip[5] = { -0.0646774, 0.1975882, -0.8087562, 0.6902849, 0 };
        static double c1dip[5] = { -0.9520876, 2.9924258, -2.3802636, -0.2701261, 0 };
        static double c2dip[5] = { -0.6260979, 1.2924686, 1.6542783, -3.4396744, 0 };

        const static double conv = 7242.702976750923; // conversion factor, see the note below Table 2 in Gross and Vrabec 2006

        for (int i = 0; i < ncomp; i++) {
            dipmSQ[i] = pow(cppargs.dipm[i], 2.)/(cppargs.m[i]*cppargs.e[i]*pow(cppargs.s[i],3.))*conv;
        }


        vector<double> adip (5, 0);
        vector<double> bdip (5, 0);
        vector<double> cdip (5, 0);
        double J2, J3, dJ2_dt, dJ3_dt;
        double m_ij;
        double m_ijk;
        for (int i = 0; i < ncomp; i++) {
            for (int j = 0; j < ncomp; j++) {
                m_ij = sqrt(cppargs.m[i]*cppargs.m[j]);
                if (m_ij > 2) {
                    m_ij = 2;
                }
                J2 = 0.;
                dJ2_dt = 0.;
                for (int l = 0; l < 5; l++) {
                    adip[l] = a0dip[l] + (m_ij-1)/m_ij*a1dip[l] + (m_ij-1)/m_ij*(m_ij-2)/m_ij*a2dip[l];
                    bdip[l] = b0dip[l] + (m_ij-1)/m_ij*b1dip[l] + (m_ij-1)/m_ij*(m_ij-2)/m_ij*b2dip[l];
                    J2 += (adip[l] + bdip[l]*e_ij[j*ncomp+j]/t)*pow(eta, l); // j*ncomp+j needs to be used for e_ij because it is formatted as a 1D vector
                    dJ2_dt += adip[l]*l*pow(eta, l-1)*dzeta_dt[3]
                        + bdip[l]*e_ij[j*ncomp+j]*(1/t*l*pow(eta, l-1)*dzeta_dt[3]
                        - 1/pow(t,2.)*pow(eta,l));
                }
                A2 += cppargs.x[i]*cppargs.x[j]*e_ij[i*ncomp+i]/t*e_ij[j*ncomp+j]/t*pow(s_ij[i*ncomp+i],3)*pow(s_ij[j*ncomp+j],3)/
                    pow(s_ij[i*ncomp+j],3)*cppargs.dip_num[i]*cppargs.dip_num[j]*dipmSQ[i]*dipmSQ[j]*J2;
                dA2_dt += cppargs.x[i]*cppargs.x[j]*e_ij[i*ncomp+i]*e_ij[j*ncomp+j]*pow(s_ij[i*ncomp+i],3)*pow(s_ij[j*ncomp+j],3)
                    /pow(s_ij[i*ncomp+j],3)*cppargs.dip_num[i]*cppargs.dip_num[j]*dipmSQ[i]*dipmSQ[j]*
                    (dJ2_dt/pow(t,2)-2*J2/pow(t,3));

                for (int k = 0; k < ncomp; k++) {
                    m_ijk = pow((cppargs.m[i]*cppargs.m[j]*cppargs.m[k]),1/3.);
                    if (m_ijk > 2) {
                        m_ijk = 2;
                    }
                    J3 = 0.;
                    dJ3_dt = 0.;
                    for (int l = 0; l < 5; l++) {
                        cdip[l] = c0dip[l] + (m_ijk-1)/m_ijk*c1dip[l] + (m_ijk-1)/m_ijk*(m_ijk-2)/m_ijk*c2dip[l];
                        J3 += cdip[l]*pow(eta, l);
                        dJ3_dt += cdip[l]*l*pow(eta, l-1)*dzeta_dt[3];
                    }
                    A3 += cppargs.x[i]*cppargs.x[j]*cppargs.x[k]*e_ij[i*ncomp+i]/t*e_ij[j*ncomp+j]/t*e_ij[k*ncomp+k]/t*
                        pow(s_ij[i*ncomp+i],3)*pow(s_ij[j*ncomp+j],3)*pow(s_ij[k*ncomp+k],3)/s_ij[i*ncomp+j]/s_ij[i*ncomp+k]/
                        s_ij[j*ncomp+k]*cppargs.dip_num[i]*cppargs.dip_num[j]*cppargs.dip_num[k]*dipmSQ[i]*
                        dipmSQ[j]*dipmSQ[k]*J3;
                    dA3_dt += cppargs.x[i]*cppargs.x[j]*cppargs.x[k]*e_ij[i*ncomp+i]*e_ij[j*ncomp+j]*e_ij[k*ncomp+k]*
                        pow(s_ij[i*ncomp+i],3)*pow(s_ij[j*ncomp+j],3)*pow(s_ij[k*ncomp+k],3)/s_ij[i*ncomp+j]/s_ij[i*ncomp+k]
                        /s_ij[j*ncomp+k]*cppargs.dip_num[i]*cppargs.dip_num[j]*cppargs.dip_num[k]*dipmSQ[i]
                        *dipmSQ[j]*dipmSQ[k]*(-3*J3/pow(t,4) + dJ3_dt/pow(t,3));
                }
            }
        }

        A2 = -PI*den*A2;
        A3 = -4/3.*PI*PI*den*den*A3;
        dA2_dt = -PI*den*dA2_dt;
        dA3_dt = -4/3.*PI*PI*den*den*dA3_dt;

        dadt_polar = (dA2_dt-2*A3/A2*dA2_dt+dA3_dt)/pow(1-A3/A2, 2.);
    }

    // Association term -------------------------------------------------------
    // only the 2B association type is currently implemented
    double dadt_assoc = 0.;
    if (!cppargs.e_assoc.empty()) {
        int a_sites = 2;
        int ncA = count_if(cppargs.vol_a.begin(), cppargs.vol_a.end(), IsNotZero); // number of associating compounds in the fluid

        vector<int> iA (ncA, 0); //indices of associating compounds
        int ctr = 0;
        for (int i = 0; i < ncomp; i++) {
            if (cppargs.vol_a[i] != 0.0) {
                iA[ctr] = i;
                ctr += 1;
            }
        }

        vector<double> XA (ncA*a_sites, 0);
        vector<double> eABij (ncA*ncA, 0);
        vector<double> volABij (ncA*ncA, 0);
        vector<double> delta_ij (ncA*ncA, 0);
        vector<double> ddelta_dt (ncA*ncA, 0);

        // these indices are necessary because we are only using 1D vectors
        int idxa = -1; // index over only associating compounds
        int idxi = 0; // index for the ii-th compound
        int idxj = 0; // index for the jj-th compound
        for (int i = 0; i < ncA; i++) {
            idxi = iA[i]*ncomp+iA[i];
            for (int j = 0; j < ncA; j++) {
                idxa += 1;
                idxj = iA[j]*ncomp+iA[j];
                eABij[idxa] = (cppargs.e_assoc[iA[i]]+cppargs.e_assoc[iA[j]])/2.;
                if (cppargs.k_hb.empty()) {
                    volABij[idxa] = sqrt(cppargs.vol_a[iA[i]]*cppargs.vol_a[iA[j]])*pow(sqrt(s_ij[idxi]*
                        s_ij[idxj])/(0.5*(s_ij[idxi]+s_ij[idxj])), 3);
                }
                else {
                    volABij[idxa] = sqrt(cppargs.vol_a[iA[i]]*cppargs.vol_a[iA[j]])*pow(sqrt(s_ij[idxi]*
                        s_ij[idxj])/(0.5*(s_ij[idxi]+s_ij[idxj])), 3)*(1-cppargs.k_hb[iA[i]*ncomp+iA[j]]);
                }
                delta_ij[idxa] = ghs[iA[i]*ncomp+iA[j]]*(exp(eABij[idxa]/t)-1)*pow(s_ij[iA[i]*ncomp+iA[j]], 3)*volABij[idxa];
                ddelta_dt[idxa] = pow(s_ij[idxj],3)*volABij[idxa]*(-eABij[idxa]/pow(t,2)
                    *exp(eABij[idxa]/t)*ghs[iA[i]*ncomp+iA[j]] + dghs_dt[iA[i]*ncomp+iA[j]]
                    *(exp(eABij[idxa]/t)-1));
            }
            XA[i*2] = (-1 + sqrt(1+8*den*delta_ij[i*ncA+i]))/(4*den*delta_ij[i*ncA+i]);
            if (!std::isfinite(XA[i*2])) {
                XA[i*2] = 0.02;
            }
            XA[i*2+1] = XA[i*2];
        }

        vector<double> x_assoc(ncA); // mole fractions of only the associating compounds
        for (int i = 0; i < ncA; i++) {
            x_assoc[i] = cppargs.x[iA[i]];
        }

        ctr = 0;
        double dif = 1000.;
        vector<double> XA_old = XA;
        while ((ctr < 500) && (dif > 1e-9)) {
            ctr += 1;
            XA = XA_find(XA, ncA, delta_ij, den, x_assoc);
            dif = 0.;
            for (int i = 0; i < ncA*2; i++) {
                dif += abs(XA[i] - XA_old[i]);
            }
            XA_old = XA;
        }

        vector<double> dXA_dt (ncA*a_sites, 0);
        dXA_dt = dXAdt_find(ncA, delta_ij, den, XA, ddelta_dt, x_assoc, a_sites);

        int idx = -1;
        for (int i = 0; i < ncA; i++) {
            for (int j = 0; j < a_sites; j++) {
                idx += 1;
                dadt_assoc += cppargs.x[iA[i]]*(1/XA[idx]-0.5)*dXA_dt[idx];
            }
        }
    }

    // Ion term ---------------------------------------------------------------
    double dadt_ion = 0.;
    if (!cppargs.z.empty()) {
        vector<double> q(cppargs.z.begin(), cppargs.z.end());
        for (int i = 0; i < ncomp; i++) {
            q[i] = q[i]*E_CHRG;
        }

        summ = 0.;
        for (int i = 0; i < ncomp; i++) {
            summ += cppargs.z[i]*cppargs.z[i]*cppargs.x[i];
        }
        double kappa = sqrt(den*E_CHRG*E_CHRG/kb/t/(cppargs.dielc*perm_vac)*summ); // the inverse Debye screening length. Equation 4 in Held et al. 2008.

        double dkappa_dt;
        if (kappa != 0) {
            vector<double> chi(ncomp);
            vector<double> dchikap_dk(ncomp);
            summ = 0.;
            for (int i = 0; i < ncomp; i++) {
                chi[i] = 3/pow(kappa*cppargs.s[i], 3)*(1.5 + log(1+kappa*cppargs.s[i]) - 2*(1+kappa*cppargs.s[i]) +
                    0.5*pow(1+kappa*cppargs.s[i], 2));
                dchikap_dk[i] = -2*chi[i]+3/(1+kappa*cppargs.s[i]);
                summ += cppargs.x[i]*cppargs.z[i]*cppargs.z[i];
            }
            dkappa_dt = -0.5*den*E_CHRG*E_CHRG/kb/t/t/(cppargs.dielc*perm_vac)*summ/kappa;

            summ = 0.;
            for (int i = 0; i < ncomp; i++) {
                summ += cppargs.x[i]*q[i]*q[i]*(dchikap_dk[i]*dkappa_dt/t-kappa*chi[i]/t/t);
            }
            dadt_ion = -1/12./PI/kb/(cppargs.dielc*perm_vac)*summ;
        }
    }

    double dadt = dadt_hc + dadt_disp + dadt_assoc + dadt_polar + dadt_ion;
    return dadt;
}


double pcsaft_hres_cpp(double t, double rho, add_args &cppargs) {
    /**
    Calculate the residual enthalpy for one phase of the system.

    Parameters
    ----------
    x : vector<double>, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : vector<double>, shape (n,)
        Segment number for each component.
    s : vector<double>, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : vector<double>, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : double
        Temperature (K)
    rho : double
        Molar density (mol m^-3)
    cppargs : add_args
        A struct containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : vector<double>, shape (n*n,)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : vector<double>, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : vector<double>, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : vector<double>, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : vector<double>, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : vector<double>, shape (n,)
            Charge number of the ions
        dielc : double
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    hres : double
        Residual enthalpy (J mol^-1)
    */
    double Z = pcsaft_Z_cpp(t, rho, cppargs);
    double dares_dt = pcsaft_dadt_cpp(t, rho, cppargs);

    double hres = (-t*dares_dt + (Z-1))*kb*N_AV*t; // Equation A.46 from Gross and Sadowski 2001
    return hres;
}


double pcsaft_sres_cpp(double t, double rho, add_args &cppargs) {
    /**
    Calculate the residual entropy (constant volume) for one phase of the system.

    Parameters
    ----------
    x : vector<double>, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : vector<double>, shape (n,)
        Segment number for each component.
    s : vector<double>, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : vector<double>, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : double
        Temperature (K)
    rho : double
        Molar density (mol m^-3)
    cppargs : add_args
        A struct containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : vector<double>, shape (n*n,)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : vector<double>, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : vector<double>, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : vector<double>, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : vector<double>, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : vector<double>, shape (n,)
            Charge number of the ions
        dielc : double
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    sres : double
        Residual entropy (J mol^-1 K^-1)
    */
    double gres = pcsaft_gres_cpp(t, rho, cppargs);
    double hres = pcsaft_hres_cpp(t, rho, cppargs);

    double sres = (hres - gres)/t;
    return sres;
}

double pcsaft_gres_cpp(double t, double rho, add_args &cppargs) {
    /**
    Calculate the residual Gibbs energy for one phase of the system.

    Parameters
    ----------
    x : vector<double>, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    m : vector<double>, shape (n,)
        Segment number for each component.
    s : vector<double>, shape (n,)
        Segment diameter for each component. For ions this is the diameter of
        the hydrated ion. Units of Angstrom.
    e : vector<double>, shape (n,)
        Dispersion energy of each component. For ions this is the dispersion
        energy of the hydrated ion. Units of K.
    t : double
        Temperature (K)
    rho : double
        Molar density (mol m^-3)
    cppargs : add_args
        A struct containing additional arguments that can be passed for
        use in PC-SAFT:

        k_ij : vector<double>, shape (n*n,)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : vector<double>, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : vector<double>, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : vector<double>, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : vector<double>, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : vector<double>, shape (n,)
            Charge number of the ions
        dielc : double
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    gres : double
        Residual Gibbs energy (J mol^-1)
    */
    double ares = pcsaft_ares_cpp(t, rho, cppargs);
    double Z = pcsaft_Z_cpp(t, rho, cppargs);

    double gres = (ares + (Z - 1) - log(Z))*kb*N_AV*t; // Equation A.50 from Gross and Sadowski 2001
    return gres;
}


vector<double> flashTQ_cpp(double t, double Q, add_args &cppargs) {
    vector<double> result;
    try {
        result = scan_pressure(t, Q, cppargs, 30);
        if (result[0] == _HUGE) {
              throw SolutionError("A suitable initial guess for pressure could not be found for the PQ flash.");
        }
    }
    catch (const SolutionError& ex) {
        result = scan_pressure(t, Q, cppargs, 500);
    }

    double p_guess = result[0];
    double x_lo = result[1];
    double x_hi = result[2];

    double p;
    try {
        p = BoundedSecantBubPressure(t, Q, cppargs, p_guess, x_lo, x_hi, 0.01*p_guess, 1e-8, 200);
    }
    catch (const SolutionError& ex) {
        p = BoundedSecantBubPressure(t, Q, cppargs, p_guess, x_lo, x_hi, 0.01*p_guess, 0.1, 200);
    }

    vector<double> output = findx_bub_pressure(p, t, Q, cppargs);
    output[0] = p; // replace error with pressure for final output
    return output;
}


vector<double> flashTQ_cpp(double t, double Q, add_args &cppargs, double p_guess) {
    double x_lo = p_guess / 5;
    double x_hi = 5 * p_guess;

    double p;
    vector<double> output;
    try {
        p = BoundedSecantBubPressure(t, Q, cppargs, p_guess, x_lo, x_hi, 0.01*p_guess, 1e-8, 200);
        output = findx_bub_pressure(p, t, Q, cppargs);
        output[0] = p; // replace error with pressure for final output
    }
    catch (const SolutionError& ex) {
        output = flashTQ_cpp(t, Q, cppargs); // call function without an initial guess
    }

    return output;
}


vector<double> scan_pressure(double t, double Q, add_args &cppargs, int npts) {
    double x_lbound = -8;
    double x_ubound = 9;

    double err_min = _HUGE;
    double p_guess = _HUGE;
    double x_lo = _HUGE;
    double x_hi = _HUGE;
    int ctr_increasing = 0; // keeps track of the number of steps where the error is increasing instead of decreasing
    for (int i = 0; i < npts; i++) {
        double p_i = pow(10, ((x_ubound - x_lbound) / (double)npts * i + x_lbound));
        double err = resid_bub_pressure(p_i, t, Q, cppargs);

        if (err < err_min) {
            err_min = err;
            p_guess = p_i;
            x_lo = pow(10, ((x_ubound - x_lbound) / (double)npts * (i - 1) + x_lbound));
            x_hi = pow(10, ((x_ubound - x_lbound) / (double)npts * (i + 1) + x_lbound));
            ctr_increasing = 0;
        }
        else if (err_min < 1e9) {
            ctr_increasing += 1;
        }

        if (ctr_increasing > 2) { // this is necessary because PC-SAFT often gives a second, erroneous VLE at a higher pressure. Reference: Privat R, Gani R, Jaubert JN. Are safe results obtained when the PC-SAFT equation of state is applied to ordinary pure chemicals?. Fluid Phase Equilibria. 2010 Aug 15;295(1):76-92.
            break;
        }
    }

    if (err_min == _HUGE) {
        throw SolutionError("scan_pressure did not find any pressure with a finite error.");
    }

    vector<double> result(3);
    result[0] = p_guess;
    result[1] = x_lo;
    result[2] = x_hi;
    return result;
}


vector<double> findx_bub_pressure(double p, double t, double Q, add_args &cppargs) {
    double error = 0;
    vector<double> result;

    int ncomp = cppargs.x.size(); // number of components
    double rhol, rhov;
    vector<double> fugcoef_l(ncomp), fugcoef_v(ncomp);

    if (ncomp == 1) {
        rhol = pcsaft_den_cpp(t, p, 0, cppargs);
        fugcoef_l = pcsaft_fugcoef_cpp(t, rhol, cppargs);
        rhov = pcsaft_den_cpp(t, p, 1, cppargs);
        fugcoef_v = pcsaft_fugcoef_cpp(t, rhov, cppargs);
        error += 100000 * pow(fugcoef_l[0] - fugcoef_v[0], 2.);

        result.push_back(error);
        result.insert(result.end(), cppargs.x.begin(), cppargs.x.end());
        result.insert(result.end(), cppargs.x.begin(), cppargs.x.end());
    }
    else {
        int itr = 0;
        double dif = 10000.;

        double summ;
        vector<double> xv_old(ncomp);
        add_args cppargs_v = cppargs;
        add_args cppargs_l = cppargs;
        double x_ions = 0.; // overall mole fraction of ions in the system
        for (int i = 0; i < ncomp; i++) {
            if (!cppargs.z.empty() && cppargs.z[i] != 0) {
                x_ions += cppargs.x[i];
            }
        }
        while ((dif>1e-9) && (itr<100)) {
            xv_old = cppargs_v.x;
            rhol = pcsaft_den_cpp(t, p, 0, cppargs_l);
            fugcoef_l = pcsaft_fugcoef_cpp(t, rhol, cppargs_l);
            rhov = pcsaft_den_cpp(t, p, 1, cppargs_v);
            fugcoef_v = pcsaft_fugcoef_cpp(t, rhov, cppargs_v);

            if (Q > 0.5) {
                summ = 0.;
                for (int i = 0; i < ncomp; i++) {
                    if (cppargs.z.empty() || cppargs.z[i] == 0) {
                        cppargs_l.x[i] = fugcoef_v[i]*cppargs_v.x[i]/fugcoef_l[i];
                        summ += cppargs_l.x[i];
                    }
                }
                for (int i = 0; i < ncomp; i++) {
                    if (cppargs.z.empty() || cppargs.z[i] == 0) {
                        cppargs_l.x[i] = cppargs_l.x[i]/summ*(((1-Q) - x_ions)/(1-Q)); // ensures that mole fractions add up to 1
                        cppargs_v.x[i] = (cppargs.x[i] - (1-Q)*cppargs_l.x[i])/Q; // if PCSAFT->_Q is close to zero then this equation behaves poorly, and that is why we use this if statement to switch the equation around
                    }
                    else {
                        cppargs_l.x[i] = cppargs.x[i]/(1-Q);
                        cppargs_v.x[i] = 0.;
                    }
                }
            }
            else {
                summ = 0.;
                for (int i = 0; i < ncomp; i++) {
                    if (cppargs.z.empty() || cppargs.z[i] == 0) {
                        cppargs_v.x[i] = fugcoef_l[i]*cppargs_l.x[i]/fugcoef_v[i];
                    }
                    summ += cppargs_v.x[i];
                }
                for (int i = 0; i < ncomp; i++) {
                    cppargs_v.x[i] = cppargs_v.x[i]/summ;
                    cppargs_l.x[i] = (cppargs.x[i] - (Q)*cppargs_v.x[i])/(1-Q);
                }
            }

            dif = 0;
            for (int i = 0; i < ncomp; i++) {
                dif += abs(cppargs_v.x[i] - xv_old[i]);
            }
            itr += 1;
        }

        for (int i = 0; i < ncomp; i++) {
            if (cppargs.z.empty() || cppargs.z[i] == 0) {
                error += pow(cppargs_l.x[i]*fugcoef_l[i] - cppargs_v.x[i]*fugcoef_v[i], 2.);
            }
            error += pow((cppargs.x[i] - Q*cppargs_v.x[i] - (1-Q)*cppargs_l.x[i]), 2.);
        }

        result.push_back(error);
        result.insert(result.end(), cppargs_l.x.begin(), cppargs_l.x.end());
        result.insert(result.end(), cppargs_v.x.begin(), cppargs_v.x.end());
    }

    if (!std::isfinite(error) || (rhol - rhov) < 1e-5) {
        error = _HUGE;
        result[0] = error;
    }

    return result;
}


vector<double> flashPQ_cpp(double p, double Q, add_args &cppargs){
    vector<double> result;
    try {
        result = scan_temp(p, Q, cppargs, 40);
        if (result[0] == _HUGE) {
              throw SolutionError("A suitable initial guess for temperature could not be found for the PQ flash.");
        }
    }
    catch (const SolutionError& ex) {
        result = scan_temp(p, Q, cppargs, 1000);
    }

    double t_guess = result[0];
    double x_lo = result[1];
    double x_hi = result[2];

    double t;
    try {
        t = BoundedSecantBubTemp(p, Q, cppargs, t_guess, x_lo, x_hi, 0.01*t_guess, 1e-8, 200);
    }
    catch (const SolutionError& ex) {
        t = BoundedSecantBubTemp(p, Q, cppargs, t_guess, x_lo, x_hi, 0.01*t_guess, 0.1, 200);
    }

    vector<double> output = findx_bub_temp(t, p, Q, cppargs);
    output[0] = t; // replace error with temperature for final output
    return output;
}


vector<double> flashPQ_cpp(double p, double Q, add_args &cppargs, double t_guess){
    double x_lo = t_guess - 40;
    double x_hi = t_guess + 40;

    double t;
    vector<double> output;
    try {
        t = BoundedSecantBubTemp(p, Q, cppargs, t_guess, x_lo, x_hi, 0.01*t_guess, 1e-8, 200);
        output = findx_bub_temp(t, p, Q, cppargs);
        output[0] = t; // replace error with temperature for final output
    }
    catch (const SolutionError& ex) {
        output = flashPQ_cpp(p, Q, cppargs); // call function without an initial guess
    }

    return output;
}


vector<double> scan_temp(double p, double Q, add_args &cppargs, int npts) {
    double x_lbound = 1;
    double x_ubound = 800;

    double err_min = _HUGE;
    double t_guess = _HUGE;
    double x_lo = _HUGE;
    double x_hi = _HUGE;
    int ctr_increasing = 0; // keeps track of the number of steps where the error is increasing instead of decreasing
    for (int i = npts; i >= 0; i--) { // here we need to scan in the opposite direction (high T to low T) because a second, erroneous VLE occurs at lower temperatures. Reference: Privat R, Gani R, Jaubert JN. Are safe results obtained when the PC-SAFT equation of state is applied to ordinary pure chemicals?. Fluid Phase Equilibria. 2010 Aug 15;295(1):76-92.
        double t_i = ((x_ubound - x_lbound) / (double)npts * i + x_lbound);
        double err = resid_bub_temp(t_i, p, Q, cppargs);

        if (err < err_min) {
            err_min = err;
            t_guess = t_i;
            x_lo = ((x_ubound - x_lbound) / (double)npts * (i - 1) + x_lbound);
            x_hi = ((x_ubound - x_lbound) / (double)npts * (i + 1) + x_lbound);
            ctr_increasing = 0;
        }
        else if (err_min < 1e9) {
            ctr_increasing += 1;
        }

        if (ctr_increasing > 2) { // this is necessary because PC-SAFT often gives a second, erroneous VLE at a higher pressure. Reference: Privat R, Gani R, Jaubert JN. Are safe results obtained when the PC-SAFT equation of state is applied to ordinary pure chemicals?. Fluid Phase Equilibria. 2010 Aug 15;295(1):76-92.
            break;
        }
    }

    if (err_min == _HUGE) {
        throw SolutionError("scan_temp did not find any temperature with a finite error.");
    }

    vector<double> result(3);
    result[0] = t_guess;
    result[1] = x_lo;
    result[2] = x_hi;
    return result;
}


vector<double> findx_bub_temp(double t, double p, double Q, add_args &cppargs) {
    double error = 0;
    vector<double> result;

    std::vector<double>::iterator water_iter = std::find(cppargs.e.begin(), cppargs.e.end(), 353.9449);
    if (water_iter != cppargs.e.end()) {
        int water_idx = std::distance(cppargs.e.begin(), water_iter);
        try {
            cppargs.s[water_idx] = calc_water_sigma(t);
            cppargs.dielc = dielc_water(t); // Right now only aqueous mixtures are supported. Other solvents could be modeled by replacing the dielc_water function.
        } catch (const ValueError& ex) {
            error = _HUGE;
            result.push_back(error);
            result.insert(result.end(), cppargs.x.begin(), cppargs.x.end());
            result.insert(result.end(), cppargs.x.begin(), cppargs.x.end());
            return result;
        }
    }

    int ncomp = cppargs.x.size(); // number of components
    vector<double> fugcoef_l(ncomp), fugcoef_v(ncomp);
    double rhol, rhov;
    if (ncomp == 0) {
        rhol = pcsaft_den_cpp(t, p, 0, cppargs);
        fugcoef_l = pcsaft_fugcoef_cpp(t, rhol, cppargs);
        rhov = pcsaft_den_cpp(t, p, 1, cppargs);
        fugcoef_v = pcsaft_fugcoef_cpp(t, rhov, cppargs);
        error += 100000 * pow(fugcoef_l[0] - fugcoef_v[0], 2.);

        result.push_back(error);
        result.insert(result.end(), cppargs.x.begin(), cppargs.x.end());
        result.insert(result.end(), cppargs.x.begin(), cppargs.x.end());
    }
    else {
        int itr = 0;
        double dif = 10000.;

        double summ;
        vector<double> xv_old(ncomp);
        add_args cppargs_v = cppargs;
        add_args cppargs_l = cppargs;
        double x_ions = 0.; // overall mole fraction of ions in the system
        for (int i = 0; i < ncomp; i++) {
            if (!cppargs.z.empty() && cppargs.z[i] != 0) {
                x_ions += cppargs.x[i];
            }
        }
        while ((dif>1e-9) && (itr<100)) {
            xv_old = cppargs_v.x;
            rhol = pcsaft_den_cpp(t, p, 0, cppargs_l);
            fugcoef_l = pcsaft_fugcoef_cpp(t, rhol, cppargs_l);
            rhov = pcsaft_den_cpp(t, p, 1, cppargs_v);
            fugcoef_v = pcsaft_fugcoef_cpp(t, rhov, cppargs_v);

            if (Q > 0.5) {
                summ = 0.;
                for (int i = 0; i < ncomp; i++) {
                    if (cppargs.z.empty() || cppargs.z[i] == 0) {
                        cppargs_l.x[i] = fugcoef_v[i]*cppargs_v.x[i]/fugcoef_l[i];
                        summ += cppargs_l.x[i];
                    }
                }
                for (int i = 0; i < ncomp; i++) {
                    if (cppargs.z.empty() || cppargs.z[i] == 0) {
                        cppargs_l.x[i] = cppargs_l.x[i]/summ*(((1-Q) - x_ions)/(1-Q)); // ensures that mole fractions add up to 1
                        cppargs_v.x[i] = (cppargs.x[i] - (1-Q)*cppargs_l.x[i])/Q; // if PCSAFT->_Q is close to zero then this equation behaves poorly, and that is why we use this if statement to switch the equation around
                    }
                    else {
                        cppargs_l.x[i] = cppargs.x[i]/(1-Q);
                        cppargs_v.x[i] = 0.;
                    }
                }
            }
            else {
                summ = 0.;
                for (int i = 0; i < ncomp; i++) {
                    if (cppargs.z.empty() || cppargs.z[i] == 0) {
                        cppargs_v.x[i] = fugcoef_l[i]*cppargs_l.x[i]/fugcoef_v[i];
                    }
                    summ += cppargs_v.x[i];
                }
                for (int i = 0; i < ncomp; i++) {
                    cppargs_v.x[i] = cppargs_v.x[i]/summ;
                    cppargs_l.x[i] = (cppargs.x[i] - (Q)*cppargs_v.x[i])/(1-Q);
                }
            }

            dif = 0;
            for (int i = 0; i < ncomp; i++) {
                dif += abs(cppargs_v.x[i] - xv_old[i]);
            }
            itr += 1;
        }

        for (int i = 0; i < ncomp; i++) {
            if (cppargs.z.empty() || cppargs.z[i] == 0) {
                error += pow(cppargs_l.x[i]*fugcoef_l[i] - cppargs_v.x[i]*fugcoef_v[i], 2.);
            }
            error += pow((cppargs.x[i] - Q*cppargs_v.x[i] - (1-Q)*cppargs_l.x[i]), 2.);
        }

        result.push_back(error);
        result.insert(result.end(), cppargs_l.x.begin(), cppargs_l.x.end());
        result.insert(result.end(), cppargs_v.x.begin(), cppargs_v.x.end());
    }

    if (!std::isfinite(error) || (rhol - rhov) < 1e-5) {
        error = _HUGE;
        result[0] = error;
    }

    return result;
}


double pcsaft_den_cpp(double t, double p, int phase, add_args &cppargs) {
    /**
    Solve for the molar density when temperature and pressure are given.

    Parameters
    ----------
    t : double
        Temperature (K)
    p : double
        Pressure (Pa)
    phase : int
        The phase for which the calculation is performed. Options: 0 (liquid),
        1 (vapor).
    cppargs : add_args
        A struct containing additional arguments that can be passed for
        use in PC-SAFT:

        x : vector<double>, shape (n,)
            Mole fractions of each component. It has a length of n, where n is
            the number of components in the system.
        m : vector<double>, shape (n,)
            Segment number for each component.
        s : vector<double>, shape (n,)
            Segment diameter for each component. For ions this is the diameter of
            the hydrated ion. Units of Angstrom.
        e : vector<double>, shape (n,)
            Dispersion energy of each component. For ions this is the dispersion
            energy of the hydrated ion. Units of K.
        k_ij : vector<double>, shape (n*n,)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : vector<double>, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : vector<double>, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : vector<double>, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : vector<double>, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Some implementations use this as an adjustable parameter
            that is fit to data.
        z : vector<double>, shape (n,)
            Charge number of the ions
        dielc : double
            Dielectric constant of the medium to be used for electrolyte
            calculations.

    Returns
    -------
    rho : double
        Molar density (mol m^-3)
    */
    // split into grid and find bounds for each root
    int ncomp = cppargs.x.size(); // number of components
    vector<double> x_lo, x_hi;
    int num_pts = 25;
    double err;
    double rho_guess = 1e-13;
    double rho_guess_prev = rho_guess;
    double err_prev = resid_rho(reduced_to_molar(rho_guess, t, ncomp, cppargs), t, p, cppargs);
    for (int i = 0; i < num_pts; i++) {
        rho_guess = 0.7405 / (double)num_pts * i + 6e-3;
        err = resid_rho(reduced_to_molar(rho_guess, t, ncomp, cppargs), t, p, cppargs);
        if (err * err_prev < 0) {
            x_lo.push_back(rho_guess_prev);
            x_hi.push_back(rho_guess);
        }
        err_prev = err;
        rho_guess_prev = rho_guess;
    }

    // solve for appropriate root(s)
    double rho = _HUGE;
    double x_lo_molar, x_hi_molar;

    if (x_lo.size() == 1) {
        rho_guess = reduced_to_molar((x_lo[0] + x_hi[0]) / 2., t, ncomp, cppargs);
        x_lo_molar = reduced_to_molar(x_lo[0], t, ncomp, cppargs);
        x_hi_molar = reduced_to_molar(x_hi[0], t, ncomp, cppargs);
        rho = BrentRho(t, p, phase, cppargs, x_lo_molar, x_hi_molar, DBL_EPSILON, 1e-8, 200);
    }
    else if (x_lo.size() <= 3 && !x_lo.empty()) {
        if (phase == 0) {
            rho_guess = reduced_to_molar((x_lo.back() + x_hi.back()) / 2., t, ncomp, cppargs);
            x_lo_molar = reduced_to_molar(x_lo.back(), t, ncomp, cppargs);
            x_hi_molar = reduced_to_molar(x_hi.back(), t, ncomp, cppargs);
            rho = BrentRho(t, p, phase, cppargs, x_lo_molar, x_hi_molar, DBL_EPSILON, 1e-8, 200);
        }
        else if (phase == 1) {
            rho_guess = reduced_to_molar((x_lo[0] + x_hi[0]) / 40., t, ncomp, cppargs); // starting with a lower guess often provides better results
            x_lo_molar = reduced_to_molar(x_lo[0], t, ncomp, cppargs);
            x_hi_molar = reduced_to_molar(x_hi[0], t, ncomp, cppargs);
            rho = BrentRho(t, p, phase, cppargs, x_lo_molar, x_hi_molar, DBL_EPSILON, 1e-8, 200);
        }
    }
    else if (x_lo.size() > 3) {
        // if multiple roots to check, then find the one with the minimum gibbs energy. Reference: Privat R, Gani R, Jaubert JN. Are safe results obtained when the PC-SAFT equation of state is applied to ordinary pure chemicals?. Fluid Phase Equilibria. 2010 Aug 15;295(1):76-92.
        double g_min = 1e60;
        for (unsigned int i = 0; i < x_lo.size(); i++) {
            rho_guess = reduced_to_molar((x_lo[i] + x_hi[i]) / 2., t, ncomp, cppargs);
            x_lo_molar = reduced_to_molar(x_lo[i], t, ncomp, cppargs);
            x_hi_molar = reduced_to_molar(x_hi[i], t, ncomp, cppargs);
            double rho_i = BrentRho(t, p, phase, cppargs, x_lo_molar, x_hi_molar, DBL_EPSILON, 1e-8, 200);
            double g_i = pcsaft_gres_cpp(t, rho_i, cppargs);
            if (g_i < g_min) {
                g_min = g_i;
                rho = rho_i;
            }
        }
    }
    else {
        int num_pts = 25;
        double err_min = 1e40;
        double rho_min = _HUGE;
        double err, rho_guess;
        for (int i = 0; i < num_pts; i++) {
            rho_guess = 0.7405 / (double)num_pts * i + 1e-8;
            err = resid_rho(reduced_to_molar(rho_guess, t, ncomp, cppargs), t, p, cppargs);
            if (abs(err) < err_min) {
                err_min = abs(err);
                rho_min = reduced_to_molar(rho_guess, t, ncomp, cppargs);
            }
        }
        rho = rho_min;
    }

    return rho;
}


double reduced_to_molar(double nu, double t, int ncomp, add_args &cppargs) {

    vector<double> d(ncomp);
    double summ = 0.;
    for (int i = 0; i < ncomp; i++) {
        d[i] = cppargs.s[i]*(1-0.12*exp(-3*cppargs.e[i] / t));
        summ += cppargs.x[i]*cppargs.m[i]*pow(d[i],3.);
    }

    return 6/PI*nu/summ*1.0e30/N_AV;
}

double dielc_water(double t) {
    /**
    Return the dielectric constant of water at the given temperature.

    t : double
        Temperature (K)

    This equation was fit to values given in the reference. For temperatures
    from 263.15 to 368.15 K values at 1 bar were used. For temperatures from
    368.15 to 443.15 K values at 10 bar were used.

    Reference:
    D. G. Archer and P. Wang, “The Dielectric Constant of Water and Debye‐Hückel
    Limiting Law Slopes,” J. Phys. Chem. Ref. Data, vol. 19, no. 2, pp. 371–411,
    Mar. 1990.
    */
    double dielc;
    if (t < 263.15) {
        throw ValueError("The current function for the dielectric constant for water is only valid for temperatures above 263.15 K.");
    }
    else if (t <= 368.15) {
        dielc = 7.6555618295E-04*t*t - 8.1783881423E-01*t + 2.5419616803E+02;
    }
    else if (t <= 443.15) {
        dielc = 0.0005003272124*t*t - 0.6285556029*t + 220.4467027;
    }
    else {
        throw ValueError("The current function for the dielectric constant for water is only valid for temperatures less than 443.15 K.");
    }
    return dielc;
}

double calc_water_sigma(double t) {
    return 3.8395 + 1.2828 * exp(-0.0074944 * t) - 1.3939 * exp(-0.00056029 * t);
}

/*
----------------------------------------------------------------------------------------------------------------------
The code for the solvers was taken from CoolProp (https://github.com/CoolProp/CoolProp) and somewhat modified.
*/
/**

This function implements a 1-D bounded solver using the algorithm from BrentRho, R. P., Algorithms for Minimization Without Derivatives.
Englewood Cliffs, NJ: Prentice-Hall, 1973. Ch. 3-4.

a and b must bound the solution of interest and f(a) and f(b) must have opposite signs.  If the function is continuous, there must be
at least one solution in the interval [a,b].

@param a The minimum bound for the solution of f=0
@param b The maximum bound for the solution of f=0
@param macheps The machine precision
@param tol_abs Tolerance (absolute)
@param maxiter Maximum number of steps allowed.  Will throw a SolutionError if the solution cannot be found
*/
double BrentRho(double t, double p, int phase, add_args &cppargs, double a, double b,
    double macheps, double tol_abs, int maxiter)
{
    int iter;
    double fa,fb,c,fc,m,tol,d,e,pp,q,s,r;
    fa = resid_rho(a, t, p, cppargs);
    fb = resid_rho(b, t, p, cppargs);

    // If one of the boundaries is to within tolerance, just stop
    if (std::abs(fb) < tol_abs) { return b;}
    if (isnan(fb)){
        throw ValueError("BrentRho's method f(b) is NAN for b");
    }
    if (std::abs(fa) < tol_abs) { return a;}
    if (isnan(fa)){
        throw ValueError("BrentRho's method f(a) is NAN for a");
    }
    if (fa*fb>0){
        throw ValueError("Inputs in BrentRho do not bracket the root");
    }

    c=a;
    fc=fa;
    iter=1;
    if (std::abs(fc)<std::abs(fb)){
        // Goto ext: from BrentRho ALGOL code
        a=b;
        b=c;
        c=a;
        fa=fb;
        fb=fc;
        fc=fa;
    }
    d=b-a;
    e=b-a;
    m=0.5*(c-b);
    tol=2*macheps*std::abs(b)+tol_abs;
    while (std::abs(m)>tol && fb!=0){
        // See if a bisection is forced
        if (std::abs(e)<tol || std::abs(fa) <= std::abs(fb)){
            m=0.5*(c-b);
            d=e=m;
        }
        else{
            s=fb/fa;
            if (a==c){
                //Linear interpolation
                pp=2*m*s;
                q=1-s;
            }
            else{
                //Inverse quadratic interpolation
                q=fa/fc;
                r=fb/fc;
                m=0.5*(c-b);
                pp=s*(2*m*q*(q-r)-(b-a)*(r-1));
                q=(q-1)*(r-1)*(s-1);
            }
            if (pp>0){
                q=-q;
            }
            else{
                pp=-pp;
            }
            s=e;
            e=d;
            m=0.5*(c-b);
            if (2*pp<3*m*q-std::abs(tol*q) || pp<std::abs(0.5*s*q)){
                d=pp/q;
            }
            else{
                m=0.5*(c-b);
                d=e=m;
            }
        }
        a=b;
        fa=fb;
        if (std::abs(d)>tol){
            b+=d;
        }
        else if (m>0){
            b+=tol;
        }
        else{
            b+=-tol;
        }
        fb=resid_rho(b, t, p, cppargs);
        if (isnan(fb)){
            throw ValueError("BrentRho's method f(t) is NAN for t");
        }
        if (std::abs(fb) < macheps){
            return b;
        }
        if (fb*fc>0){
            // Goto int: from BrentRho ALGOL code
            c=a;
            fc=fa;
            d=e=b-a;
        }
        if (std::abs(fc)<std::abs(fb)){
            // Goto ext: from BrentRho ALGOL code
            a=b;
            b=c;
            c=a;
            fa=fb;
            fb=fc;
            fc=fa;
        }
        m=0.5*(c-b);
        tol=2*macheps*std::abs(b)+tol_abs;
        iter+=1;
        if (isnan(a)){
            throw ValueError("BrentRho's method a is NAN");}
        if (isnan(b)){
            throw ValueError("BrentRho's method b is NAN");}
        if (isnan(c)){
            throw ValueError("BrentRho's method c is NAN");}
        if (iter>maxiter){
            throw SolutionError("BrentRho's method reached maximum number of steps");}
        if (std::abs(fb)< 2*macheps*std::abs(b)){
            return b;
        }
    }
    return b;
}

double resid_rho(double rhomolar, double t, double p, add_args &cppargs){
    double peos = pcsaft_p_cpp(t, rhomolar, cppargs);
    double cost = (peos-p)/p;
    if (std::isfinite(cost)) {
        return cost;
    }
    else {
        return _HUGE;
    }
}

/**
In the secant function, a 1-D Newton-Raphson solver is implemented.  An initial guess for the solution is provided.

@param x0 The initial guess for the solution
@param xmax The upper bound for the solution
@param xmin The lower bound for the solution
@param dx The initial amount that is added to x in order to build the numerical derivative
@param tol The absolute value of the tolerance accepted for the objective function
@param maxiter Maximum number of iterations
@returns If no errors are found, the solution, otherwise the value _HUGE, the value for infinity
*/
double BoundedSecantBubPressure(double t, double Q, add_args &cppargs, double x0, double xmin,
    double xmax, double dx, double tol, int maxiter) {
    double x1=0,x2=0,x3=0,y1=0,y2=0,p,fval=999;
    int iter=1;
    if (std::abs(dx)==0){ throw ValueError("dx cannot be zero"); }
    while (iter<=3 || std::abs(fval)>tol)
    {
        if (iter==1){x1=x0; p=x1;}
        else if (iter==2){x2=x0+dx; p=x2;}
        else {p=x2;}
            fval=resid_bub_pressure(p, t, Q, cppargs);
        if (iter==1){y1=fval;}
        else
        {
            if (std::isfinite(fval)) {
                y2 = fval;
            }
            else {
                y2 = 1e40;
            }
            x3 = x2 - y2/(y2-y1)*(x2-x1);
            // Check bounds, go half the way to the limit if limit is exceeded
            if (x3 < xmin)
            {
                x3 = (xmin + x2)/2;
            }
            if (x3 > xmax)
            {
                x3 = (xmax + x2)/2;
            }
            y1=y2; x1=x2; x2=x3;

        }
        if (iter>maxiter){
            throw SolutionError("BoundedSecant reached maximum number of iterations");
        }
        iter=iter+1;
    }
    return x3;
}

double resid_bub_pressure(double p, double t, double Q, add_args &cppargs) {
    double error = 0;
    if (p <= 0) {
        error = _HUGE;
    }
    else {
        vector<double> result = findx_bub_pressure(p, t, Q, cppargs);
        error = result[0];
    }

    return error;
}


double BoundedSecantBubTemp(double p, double Q, add_args &cppargs, double x0, double xmin,
    double xmax, double dx, double tol, int maxiter) {
    double x1=0,x2=0,x3=0,y1=0,y2=0,t,fval=999;
    int iter=1;
    if (std::abs(dx)==0){ throw ValueError("dx cannot be zero"); }
    while (iter<=3 || std::abs(fval)>tol)
    {
        if (iter==1){x1=x0; t=x1;}
        else if (iter==2){x2=x0+dx; t=x2;}
        else {t=x2;}
            fval=resid_bub_temp(t, p, Q, cppargs);
        if (iter==1){y1=fval;}
        else
        {
            if (std::isfinite(fval)) {
                y2 = fval;
            }
            else {
                y2 = 1e40;
            }
            x3=x2-y2/(y2-y1)*(x2-x1);
            // Check bounds, go half the way to the limit if limit is exceeded
            if (x3 < xmin)
            {
                x3 = (xmin + x2)/2;
            }
            if (x3 > xmax)
            {
                x3 = (xmax + x2)/2;
            }
            y1=y2; x1=x2; x2=x3;

        }
        if (iter>maxiter){
            throw SolutionError("BoundedSecant reached maximum number of iterations");
        }
        iter=iter+1;
    }
    return x3;
}

double resid_bub_temp(double t, double p, double Q, add_args &cppargs) {
    double error = 0;
    if (t <= 0) {
        error = _HUGE;
    }
    else {
        vector<double> result = findx_bub_temp(t, p, Q, cppargs);
        error = result[0];
    }

    return error;
}
