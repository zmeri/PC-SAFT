#include <vector>

using namespace std;

const static double kb = 1.380648465952442093e-23; // Boltzmann constant, J K^-1
const static double PI = 3.141592653589793;
const static double N_AV = 6.022140857e23; // Avagadro's number
const static double E_CHRG = 1.6021766208e-19; // elementary charge, units of coulomb
const static double perm_vac = 8.854187817e-22; //permittivity in vacuum, C V^-1 Angstrom^-1

struct add_args {
    vector<double> k_ij;
    vector<double> e_assoc;
    vector<double> vol_a;
    vector<double> dipm;
    vector<double> dip_num;
    vector<double> z;
    double dielc;
    vector<double> k_hb;
    vector<double> l_ij;
};

bool IsNotZero (double x) {return x != 0.0;}

double pcsaft_Z_cpp(vector<double> x, vector<double> m, vector<double> s, vector<double> e,
    double t, double rho, add_args &cppargs);
vector<double> pcsaft_fugcoef_cpp(vector<double> x, vector<double> m, vector<double> s, vector<double> e,
    double t, double rho, add_args &cppargs);
double pcsaft_p_cpp(vector<double> x, vector<double> m, vector<double> s, vector<double> e,
    double t, double rho, add_args &cppargs);
double pcsaft_den_cpp(vector<double> x, vector<double> m, vector<double> s, vector<double> e,
    double t, double p, int phase, add_args &cppargs);
double pcsaft_ares_cpp(vector<double> x, vector<double> m, vector<double> s, vector<double> e,
    double t, double rho, add_args &cppargs);
double pcsaft_dadt_cpp(vector<double> x, vector<double> m, vector<double> s, vector<double> e,
    double t, double rho, add_args &cppargs);
double pcsaft_hres_cpp(vector<double> x, vector<double> m, vector<double> s, vector<double> e,
    double t, double rho, add_args &cppargs);
double pcsaft_sres_cpp(vector<double> x, vector<double> m, vector<double> s, vector<double> e,
    double t, double rho, add_args &cppargs);
double pcsaft_gres_cpp(vector<double> x, vector<double> m, vector<double> s, vector<double> e,
    double t, double rho, add_args &cppargs);

double bubblePfit_cpp(double p_guess, vector<double> xv_guess, vector<double> x, vector<double> m, vector<double> s, vector<double> e,
    double t, add_args &cppargs);
double PTzfit_cpp(double p_guess, vector<double> x_guess, double beta_guess, double mol, 
    double vol, vector<double> x_total, vector<double> m, vector<double> s, vector<double> e,
    double t, add_args &cppargs);

vector<double> XA_find(vector<double> XA_guess, int ncomp, vector<double> delta_ij, double den,
    vector<double> x);
vector<double> dXA_find(int ncA, int ncomp, vector<int> iA, vector<double> delta_ij, 
    double den, vector<double> XA, vector<double> ddelta_dd, vector<double> x, int n_sites);
vector<double> dXAdt_find(int ncA, vector<double> delta_ij, double den, 
    vector<double> XA, vector<double> ddelta_dt, vector<double> x, int n_sites);
