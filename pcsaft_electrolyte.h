#include <vector>
#include <string>

using std::vector;

const static double kb = 1.380648465952442093e-23; // Boltzmann constant, J K^-1
const static double PI = 3.141592653589793;
const static double N_AV = 6.022140857e23; // Avagadro's number
const static double E_CHRG = 1.6021766208e-19; // elementary charge, units of coulomb
const static double perm_vac = 8.854187817e-22; //permittivity in vacuum, C V^-1 Angstrom^-1

#ifndef DBL_EPSILON
    #include <limits>
    #define DBL_EPSILON std::numeric_limits<double>::epsilon()
#endif

struct add_args {
    vector<double> m;
    vector<double> s;
    vector<double> e;
    vector<double> k_ij;
    vector<double> e_assoc;
    vector<double> vol_a;
    vector<double> dipm;
    vector<double> dip_num;
    vector<double> z;
    double dielc;
    vector<int> assoc_num;
    vector<int> assoc_scheme;
    vector<double> k_hb;
    vector<double> l_ij;
};

double pcsaft_Z_cpp(double t, double rho, vector<double> x, add_args &cppargs);
vector<double> pcsaft_fugcoef_cpp(double t, double rho, vector<double> x, add_args &cppargs);
double pcsaft_p_cpp(double t, double rho, vector<double> x, add_args &cppargs);
double pcsaft_den_cpp(double t, double p, vector<double> x, int phase, add_args &cppargs);
double pcsaft_ares_cpp(double t, double rho, vector<double> x, add_args &cppargs);
double pcsaft_dadt_cpp(double t, double rho, vector<double> x, add_args &cppargs);
double pcsaft_hres_cpp(double t, double rho, vector<double> x, add_args &cppargs);
double pcsaft_sres_cpp(double t, double rho, vector<double> x, add_args &cppargs);
double pcsaft_gres_cpp(double t, double rho, vector<double> x, add_args &cppargs);

vector<double> flashTQ_cpp(double t, double Q, vector<double> x, add_args &cppargs);
vector<double> flashTQ_cpp(double t, double Q, vector<double> x, add_args &cppargs, double p_guess); // used if a guess value is given
vector<double> flashPQ_cpp(double p, double Q, vector<double> x, add_args &cppargs);
vector<double> flashPQ_cpp(double p, double Q, vector<double> x, add_args &cppargs, double t_guess); // used if a guess value is given

// functions used to solve for XA and its derivatives
vector<double> XA_find(vector<double> XA_guess, vector<double> delta_ij, double den,
    vector<double> x);
vector<double> dXAdx_find(vector<int> assoc_num, vector<double> delta_ij,
    double den, vector<double> XA, vector<double> ddelta_dx, vector<double> x);
vector<double> dXAdt_find(vector<double> delta_ij, double den,
    vector<double> XA, vector<double> ddelta_dt, vector<double> x);

// helper functions
bool IsNotZero (double x) {return x != 0.0;}
double reduced_to_molar(double nu, double t, int ncomp, vector<double> x, add_args &cppargs);
double dielc_water(double t);
double calc_water_sigma(double t);
add_args get_single_component(add_args cppargs, int i);

class ValueError: public std::exception
{
public:
    ValueError(const std::string &err) throw() : m_err(err) {}
    ~ValueError() throw() {};
    virtual const char* what() const throw() { return m_err.c_str(); }
private:
    std::string m_err;
};

class SolutionError: public std::exception
{
public:
    SolutionError(const std::string &err) throw() : m_err(err) {}
    ~SolutionError() throw() {};
    virtual const char* what() const throw() { return m_err.c_str(); }
private:
    std::string m_err;
};

// functions used in solving for density
double resid_rho(double rhomolar, double t, double p, vector<double> x, add_args &cppargs);
double BrentRho(double t, double p, vector<double> x, int phase, add_args &cppargs, double a, double b,
    double macheps, double tol_abs, int maxiter);

// functions used for flash calculations
vector<double> scan_pressure(double t, double Q, vector<double> x, add_args &cppargs, int npts);
vector<double> scan_temp(double p, double Q, vector<double> x, add_args &cppargs, int npts);
vector<double> findx_bub_pressure(double p, double t, double Q, vector<double> x, add_args &cppargs);
vector<double> findx_bub_temp(double t, double p, double Q, vector<double> x, add_args &cppargs);
double resid_bub_pressure(double p, double t, double Q, vector<double> x, add_args &cppargs);
double resid_bub_temp(double t, double p, double Q, vector<double> x, add_args &cppargs);
double BoundedSecantBubPressure(double t, double Q, vector<double> x, add_args &cppargs, double x0, double xmin,
    double xmax, double dx, double tol, int maxiter);
double BoundedSecantBubTemp(double p, double Q, vector<double> x, add_args &cppargs, double x0, double xmin,
    double xmax, double dx, double tol, int maxiter);
