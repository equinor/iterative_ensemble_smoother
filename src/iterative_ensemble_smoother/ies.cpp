#include <algorithm>
#include <memory>
#include <variant>

#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

using Eigen::ComputeThinU;
using Eigen::ComputeThinV;
using Eigen::ComputeFullU;
using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace py = pybind11;

enum struct Inversion {
    exact = 0,
    subspace_exact_r = 1,
    subspace_ee_r = 2,
    subspace_re = 3
};

class Data {
public:
    std::vector<bool> ens_mask{};
    std::vector<bool> obs_mask{};

    /** Coefficient matrix used to compute Omega = I + W (I -11'/N)/sqrt(N-1) */
    MatrixXd W;

    Data(int ens_size);

    void store_initial_obs_mask(const std::vector<bool> &mask);
    void update_obs_mask(const std::vector<bool> &mask);

    void store_initialE(const MatrixXd &E0);
    void augment_initialE(const MatrixXd &E0);
    void store_initialA(const MatrixXd &A);

    MatrixXd make_activeE() const;
    MatrixXd make_activeW() const;
    MatrixXd make_activeA() const;

    int iteration_nr = 1;

private:
    bool m_converged = false;

    std::vector<bool> m_obs_mask0{};
    /** Prior ensemble used in Ei=A0 Omega_i */
    MatrixXd A0{};
    /** Prior ensemble of measurement perturations (should be the same for all iterations) */
    MatrixXd E;
};

int calc_num_significant(const VectorXd &singular_values, double truncation) {
    int num_significant = 0;
    double total_sigma2 = singular_values.squaredNorm();

    /*
     * Determine the number of singular values by enforcing that
     * less than a fraction @truncation of the total variance be
     * accounted for.
     */
    double running_sigma2 = 0;
    for (auto sig : singular_values) {
        if (running_sigma2 / total_sigma2 <
            truncation) { /* Include one more singular value ? */
            num_significant++;
            running_sigma2 += sig * sig;
        } else
            break;
    }

    return num_significant;
}

/**
 * Implements parts of Eq. 14.31 in the book Data Assimilation,
 * The Ensemble Kalman Filter, 2nd Edition by Geir Evensen.
 * Specifically, this implements
 * X_1 (I + \Lambda_1)^{-1} X_1^T (D - M[A^f])
*/
MatrixXd genX3(const MatrixXd &W, const MatrixXd &D, const VectorXd &eig) {
    const int nrmin = std::min(D.rows(), D.cols());
    // Corresponds to (I + \Lambda_1)^{-1} since `eig` has already been transformed.
    MatrixXd Lambda_inv = eig(Eigen::seq(0, nrmin - 1)).asDiagonal();
    MatrixXd X1 = Lambda_inv * W.transpose();

    MatrixXd X2 = X1 * D;
    MatrixXd X3 = W * X2;

    return X3;
}

int svdS(const MatrixXd &S, const std::variant<double, int> &truncation,
         VectorXd &inv_sig0, MatrixXd &U0) {

    int num_significant = 0;

    auto svd = S.bdcSvd(ComputeThinU);
    U0 = svd.matrixU();
    VectorXd singular_values = svd.singularValues();

    if (std::holds_alternative<int>(truncation)) {
        num_significant = std::get<int>(truncation);
    } else {
        num_significant =
            calc_num_significant(singular_values, std::get<double>(truncation));
    }

    inv_sig0 = singular_values.cwiseInverse();

    inv_sig0(Eigen::seq(num_significant, Eigen::last)).setZero();

    return num_significant;
}

/**
 Routine computes X1 and eig corresponding to Eqs 14.54-14.55
 Geir Evensen
*/
void lowrankE(
    const MatrixXd &S, /* (nrobs x nrens) */
    const MatrixXd &E, /* (nrobs x nrens) */
    MatrixXd &W, /* (nrobs x nrmin) Corresponding to X1 from Eqs. 14.54-14.55 */
    VectorXd &eig, /* (nrmin) Corresponding to 1 / (1 + Lambda1^2) (14.54) */
    const std::variant<double, int> &truncation) {

    const int nrobs = S.rows();
    const int nrens = S.cols();
    const int nrmin = std::min(nrobs, nrens);

    VectorXd inv_sig0(nrmin);
    MatrixXd U0(nrobs, nrmin);

    /* Compute SVD of S=HA`  ->  U0, invsig0=sig0^(-1) */
    svdS(S, truncation, inv_sig0, U0);

    MatrixXd Sigma_inv = inv_sig0.asDiagonal();

    /* X0(nrmin x nrens) =  Sigma0^(+) * U0'* E  (14.51)  */
    MatrixXd X0 = Sigma_inv * U0.transpose() * E;

    /* Compute SVD of X0->  U1*eig*V1   14.52 */
    auto svd = X0.bdcSvd(ComputeThinU);
    const auto &sig1 = svd.singularValues();

    /* Lambda1 = 1/(I + Lambda^2)  in 14.56 */
    for (int i = 0; i < nrmin; i++)
        eig[i] = 1.0 / (1.0 + sig1[i] * sig1[i]);

    /* Compute X1 = W = U0 * (U1=sig0^+ U1) = U0 * Sigma0^(+') * U1  (14.55) */
    W = U0 * Sigma_inv.transpose() * svd.matrixU();
}

void lowrankCinv(
    const MatrixXd &S, const MatrixXd &R,
    MatrixXd &W,   /* Corresponding to X1 from Eq. 14.29 */
    VectorXd &eig, /* Corresponding to 1 / (1 + Lambda_1) (14.29) */
    const std::variant<double, int> &truncation) {

    const int nrobs = S.rows();
    const int nrens = S.cols();
    const int nrmin = std::min(nrobs, nrens);

    MatrixXd U0(nrobs, nrmin);
    MatrixXd Z(nrmin, nrmin);

    VectorXd inv_sig0(nrmin);
    svdS(S, truncation, inv_sig0, U0);

    MatrixXd Sigma_inv = inv_sig0.asDiagonal();

    /* B = Xo = (N-1) * Sigma0^(+) * U0'* Cee * U0 * Sigma0^(+')  (14.26)*/
    MatrixXd B = (nrens - 1.0) * Sigma_inv * U0.transpose() * R * U0 *
                 Sigma_inv.transpose();

    auto svd = B.bdcSvd(ComputeThinU);
    Z = svd.matrixU();
    eig = svd.singularValues();

    /* Lambda1 = (I + Lambda)^(-1) */
    for (int i = 0; i < nrmin; i++)
        eig[i] = 1.0 / (1 + eig[i]);

    Z = Sigma_inv * Z;

    W = U0 * Z; /* X1 = W = U0 * Z2 = U0 * Sigma0^(+') * Z    */
}

/**
 * Implementation of algorithm as described in
 * "Efficient Implementation of an Iterative Ensemble Smoother for Data Assimilation and Reservoir History Matching"
 * https://www.frontiersin.org/articles/10.3389/fams.2019.00047/full
 *
 * Section 2.4.3
 */
void compute_AA_projection(const MatrixXd &A, MatrixXd &Y) {

    MatrixXd Ai = A;
    Ai = Ai.colwise() - Ai.rowwise().mean();
    auto svd = Ai.bdcSvd(ComputeThinV);
    MatrixXd VT = svd.matrixV().transpose();
    MatrixXd AAi = VT.transpose() * VT;
    Y *= AAi;
}

/**
 *  The standard inversion works on the equation
 *          S'*(S*S'+R)^{-1} H           (a)
 */
void subspace_inversion(MatrixXd &W0, const Inversion ies_inversion,
                        const MatrixXd &E, const MatrixXd &R, const MatrixXd &S,
                        const MatrixXd &H,
                        const std::variant<double, int> &truncation,
                        double ies_steplength) {
    int ens_size = S.cols();
    int nrobs = S.rows();
    double nsc = 1.0 / sqrt(ens_size - 1.0);
    MatrixXd X1 = MatrixXd::Zero(
        nrobs, std::min(ens_size, nrobs)); // Used in subspace inversion
    VectorXd eig(ens_size);

    switch (ies_inversion) {
    case Inversion::subspace_re:
        lowrankE(S, E * nsc, X1, eig, truncation);
        break;

    case Inversion::subspace_ee_r: {
        MatrixXd Et = E.transpose();
        MatrixXd Cee = E * Et;
        Cee *= 1.0 / ((ens_size - 1) * (ens_size - 1));

        lowrankCinv(S, Cee, X1, eig, truncation);
        break;
    }

    case Inversion::subspace_exact_r:
        lowrankCinv(S, R * nsc * nsc, X1, eig, truncation);
        break;

    default:
        break;
    }

    // X3 = X1 * diag(eig) * X1' * H (Similar to Eq. 14.31, Evensen (2007))
    Eigen::Map<VectorXd> eig_vector(eig.data(), eig.size());
    MatrixXd X3 = genX3(X1, H, eig_vector);

    // Update data->W = (1-ies_steplength) * data->W +  ies_steplength * S' * X3 (Line 9)
    W0 = ies_steplength * S.transpose() * X3 + (1.0 - ies_steplength) * W0;
}

/**
 * Section 3.2 - Exact inversion
 * This calculates (S^T*S + I_N)^{-1} by taking the SVD of (S^T*S + I_N),
 * and since (S^T*S + I_N) is symmetric positive semi-definite we have that U=V and hence
 * (S^T*S + I_N)^{-1} = U * \Sigma^{-1} * U^T.
 */
void exact_inversion(MatrixXd &W0, const MatrixXd &S, const MatrixXd &H,
                     double ies_steplength) {
    int ens_size = S.cols();

    MatrixXd StS = S.transpose() * S + MatrixXd::Identity(ens_size, ens_size);

    auto svd = StS.bdcSvd(ComputeFullU);
    MatrixXd Z = svd.matrixU();
    VectorXd eig = svd.singularValues();

    MatrixXd ZtStH = Z.transpose() * S.transpose() * H;

    for (int i = 0; i < ens_size; i++)
        ZtStH.row(i) /= eig[i];

    // Update data->W = (1-ies_steplength) * data->W +  ies_steplength * Z * (Lamda^{-1}) Z' S' H (Line 9)
    W0 = ies_steplength * Z * ZtStH + (1.0 - ies_steplength) * W0;
}

Data::Data(int ens_size) : W(MatrixXd::Zero(ens_size, ens_size)) {}

void Data::store_initial_obs_mask(const std::vector<bool> &mask) {
    if (this->m_obs_mask0.empty())
        this->m_obs_mask0 = mask;
}

/** We store the initial observation perturbations in E, corresponding to
 * active data->obs_mask0 in data->E. The unused rows in data->E corresponds to
 * false data->obs_mask0
 */
void Data::store_initialE(const MatrixXd &E0) {
    if (E.rows() != 0 || E.cols() != 0)
        return;
    this->E = MatrixXd::Zero(obs_mask.size(), ens_mask.size());
    this->E.setConstant(-999.9);

    int m = 0;
    for (size_t iobs{}; iobs < obs_mask.size(); iobs++) {
        if (this->m_obs_mask0[iobs]) {
            int active_idx = 0;
            for (size_t iens{}; iens < ens_mask.size(); iens++) {
                if (this->ens_mask[iens]) {
                    this->E(iobs, iens) = E0(m, active_idx);
                    active_idx++;
                }
            }
            m++;
        }
    }
}

/** We augment the additional observation perturbations arriving in later
 * iterations, that was not stored before, in data->E.
 */
void Data::augment_initialE(const MatrixXd &E0) {

    int m = 0;
    for (size_t iobs{}; iobs < obs_mask.size(); iobs++) {
        if (!this->m_obs_mask0[iobs] && this->obs_mask[iobs]) {
            int i = -1;
            for (size_t iens{}; iens < ens_mask.size(); iens++) {
                if (this->ens_mask[iens]) {
                    i++;
                    this->E(iobs, iens) = E0(m, i);
                }
            }
            this->m_obs_mask0[iobs] = true;
        }
        if (this->obs_mask[iobs]) {
            m++;
        }
    }
}

void Data::store_initialA(const MatrixXd &A0) {
    if (this->A0.rows() != 0 || this->A0.cols() != 0)
        return;
    this->A0 = MatrixXd::Zero(A0.rows(), ens_mask.size());
    for (int irow = 0; irow < this->A0.rows(); irow++) {
        int active_idx = 0;
        for (size_t iens = 0; iens < ens_mask.size(); iens++) {
            if (ens_mask[iens]) {
                this->A0(irow, iens) = A0(irow, active_idx);
                active_idx++;
            }
        }
    }
}

namespace {

MatrixXd make_active(const MatrixXd &full_matrix,
                     const std::vector<bool> &row_mask,
                     const std::vector<bool> &column_mask) {
    int rows = row_mask.size();
    int columns = column_mask.size();
    MatrixXd active = MatrixXd::Zero(
        std::count(row_mask.begin(), row_mask.end(), true),
        std::count(column_mask.begin(), column_mask.end(), true));
    int row = 0;
    for (int iobs = 0; iobs < rows; iobs++) {
        if (row_mask[iobs]) {
            int column = 0;
            for (int iens = 0; iens < columns; iens++) {
                if (column_mask[iens]) {
                    active(row, column) = full_matrix(iobs, iens);
                    column++;
                }
            }
            row++;
        }
    }

    return active;
}
} // namespace

/*
  During the iteration process both the number of realizations and the number of
  observations can change, the number of realizations can only be reduced but
  the number of (active) observations can both be reduced and increased. The
  iteration algorithm is based maintaining a state for the entire update
  process, in order to do this correctly we must create matrix representations
  with the correct active elements both in observation and realisation space.
*/

MatrixXd Data::make_activeE() const {
    return make_active(this->E, this->obs_mask, this->ens_mask);
}

MatrixXd Data::make_activeW() const {
    return make_active(this->W, this->ens_mask, this->ens_mask);
}

MatrixXd Data::make_activeA() const {
    std::vector<bool> row_mask(this->A0.rows(), true);
    return make_active(this->A0, row_mask, this->ens_mask);
}

void init_update(Data &module_data, const std::vector<bool> &ens_mask,
                 const std::vector<bool> &obs_mask) {
    module_data.ens_mask = ens_mask;
    module_data.store_initial_obs_mask(obs_mask);
    module_data.obs_mask = obs_mask;
}

MatrixXd makeX(const MatrixXd &A, const MatrixXd &Y0, const MatrixXd &R,
               const MatrixXd &E, const MatrixXd &D,
               const Inversion ies_inversion,
               const std::variant<double, int> &truncation, MatrixXd &W0,
               double ies_steplength, int iteration_nr)

{
    const int ens_size = Y0.cols();

    MatrixXd Y = Y0;

    /* Normalized predicted ensemble anomalies.
       Line 4 of Algorithm 1, also (Eq. 30)
    */
    Y = (1.0 / sqrt(ens_size - 1.0)) * (Y.colwise() - Y.rowwise().mean());

    /* A^+A projection is necessary when the parameter matrix has less rows than columns,
       and when the forward model is non-linear.
       Section 2.4.3
    */
    if (A.rows() > 0 && A.cols() > 0) {
        const int state_size = A.rows();
        if (state_size <= ens_size - 1) {
            compute_AA_projection(A, Y);
        }
    }

    /* Line 5 of Algorithm 1 */
    MatrixXd Omega =
        (1.0 / sqrt(ens_size - 1.0)) * (W0.colwise() - W0.rowwise().mean());
    Omega.diagonal().array() += 1.0;

    /* Solving for the average sensitivity matrix.
       Line 6 of Algorithm 1, also Section 5
    */
    Omega.transposeInPlace();
    MatrixXd S = Omega.fullPivLu().solve(Y.transpose()).transpose();

    /* Similar to the innovation term.
       Differs in that `D` here is defined as dobs + E - Y instead of just dobs + E as in the paper.
       Line 7 of Algorithm 1, also Section 2.6
    */
    MatrixXd H = D + S * W0;

    /* Store previous W for convergence test */
    MatrixXd W = W0;

    /*
     * COMPUTE NEW UPDATED W                                                                        (Line 9)
     *    W = W - ies_steplength * ( W - S'*(S*S'+R)^{-1} H )          (a)
     * which in the case when R=I can be rewritten as
     *    W = W - ies_steplength * ( W - (S'*S + I)^{-1} * S' * H )    (b)
     *
     * With R=I the subspace inversion (ies_inversion=1) solving Eq. (a) with singular value
     * trucation=1.000 gives exactly the same solution as the exact inversion (ies_inversion=0).
     *
     * Using ies_inversion=IES_INVERSION_SUBSPACE_EXACT_R(2), and a step length of 1.0,
     * one update gives identical result to STD as long as the same SVD
     * truncation is used.
     *
     * With very large data sets it is likely that the inversion becomes poorly
     * conditioned and a trucation=1.000 is not a good choice. In this case the
     * ies_inversion > 0 and truncation set to 0.99 or so, should stabelize
     * the algorithm.
     *
     * Using ies_inversion=IES_INVERSION_SUBSPACE_EE_R(3) and
     * ies_inversion=IES_INVERSION_SUBSPACE_RE(2) gives identical results but
     * ies_inversion=IES_INVERSION_SUBSPACE_RE is much faster (N^2m) than
     * ies_inversion=IES_INVERSION_SUBSPACE_EE_R (Nm^2).
     *
     * See the enum: ies_inverson in ies_config.hpp:
     *
     * ies_inversion=IES_INVERSION_EXACT(0)            -> exact inversion from (b) with exact R=I
     * ies_inversion=IES_INVERSION_SUBSPACE_EXACT_R(1) -> subspace inversion from (a) with exact R
     * ies_inversion=IES_INVERSION_SUBSPACE_EE_R(2)    -> subspace inversion from (a) with R=EE
     * ies_inversion=IES_INVERSION_SUBSPACE_RE(3)      -> subspace inversion from (a) with R represented by E * E^T
     */

    if (ies_inversion == Inversion::exact) {
        exact_inversion(W0, S, H, ies_steplength);
    } else {
        subspace_inversion(W0, ies_inversion, E, R, S, H, truncation,
                           ies_steplength);
    }

    /* Line 9 of Algorithm 1 */
    MatrixXd X = W0;
    X /= sqrt(ens_size - 1.0);
    X.diagonal().array() += 1;

    std::vector<double> costJ(ens_size);
    double local_costf = 0.0;
    for (int i = 0; i < ens_size; i++) {
        costJ[i] = W.col(i).dot(W.col(i)) + D.col(i).dot(D.col(i));
        local_costf += costJ[i];
    }
    local_costf = local_costf / ens_size;

    return X;
}

/**
* the updated W is stored for each iteration in data->W. If we have lost
* realizations we copy only the active rows and cols from W0 to data->W which
* is then used in the algorithm.  (note the definition of the pointer dataW to
* data->W)
*/
static void store_active_W(Data &data, const MatrixXd &W0) {
    size_t i = 0;
    size_t j;

    data.W.setConstant(0.0);
    for (size_t iens{}; iens < data.ens_mask.size(); iens++) {
        if (data.ens_mask[iens]) {
            j = 0;
            for (size_t jens{}; jens < data.ens_mask.size(); jens++) {
                if (data.ens_mask[jens]) {
                    data.W(iens, jens) = W0(i, j);
                    j++;
                }
            }
            i++;
        }
    }
}

void updateA(Data &data,
             // Updated ensemble A retured to ERT.
             Eigen::Ref<MatrixXd> A,
             // Ensemble of predicted measurements
             const MatrixXd &Yin,
             // Measurement error covariance matrix (not used)
             const MatrixXd &Rin,
             // Ensemble of observation perturbations
             const MatrixXd &Ein,
             // (d+E-Y) Ensemble of perturbed observations - Y
             const MatrixXd &Din, const Inversion ies_inversion,
             const std::variant<double, int> &truncation,
             double ies_steplength) {

    int iteration_nr = data.iteration_nr;
    /*
      Counting number of active observations for current iteration. If the
      observations have been used in previous iterations they are contained in
      data->E0. If they are introduced in the current iteration they will be
      augmented to data->E.
    */
    data.store_initialE(Ein);
    data.augment_initialE(Ein);
    data.store_initialA(A);

    /*
     * Allocates the local matrices to be used.
     * Copies the initial measurement perturbations for the active observations into the current E matrix.
     * Copies the inputs in D, Y and R into their local representations
     */
    MatrixXd E = data.make_activeE();
    MatrixXd D = Din;

    /* Subtract new measurement perturbations              D=D-E    */
    D -= Ein;
    /* Add old measurement perturbations */
    D += E;

    auto W0 = data.make_activeW();
    MatrixXd X;

    X = makeX(A, Yin, Rin, E, D, ies_inversion, truncation, W0, ies_steplength,
              iteration_nr);

    store_active_W(data, W0);

    /* COMPUTE NEW ENSEMBLE SOLUTION FOR CURRENT ITERATION  Ei=A0*X (Line 11)*/
    MatrixXd A0 = data.make_activeA();
    A = A0 * X;
}

MatrixXd makeE(const VectorXd &obs_errors, const MatrixXd &noise) {
    int active_obs_size = obs_errors.rows();
    int active_ens_size = noise.cols();

    MatrixXd E = noise;
    VectorXd pert_mean = E.rowwise().mean();
    E = E.colwise() - pert_mean;
    VectorXd pert_var = E.cwiseProduct(E).rowwise().sum();

    for (int i = 0; i < active_obs_size; i++) {
        double factor = obs_errors(i) * sqrt(active_ens_size / pert_var(i));
        E.row(i) *= factor;
    }

    return E;
}

MatrixXd makeD(const VectorXd &obs_values, const MatrixXd &E,
               const MatrixXd &S) {

    MatrixXd D = E - S;

    D.colwise() += obs_values;

    return D;
}

PYBIND11_MODULE(_ies, m) {
    using namespace py::literals;

    py::class_<Data, std::shared_ptr<Data>>(m, "ModuleData")
        .def(py::init<int>())
        .def_readwrite("iteration_nr", &Data::iteration_nr);
    m.def("make_X", &makeX, "A"_a, "Y0"_a, "R"_a, "E"_a, "D"_a,
          "ies_inversion"_a, "truncation"_a, "W0"_a, "ies_steplength"_a,
          "iteration_nr"_a);
    m.def("make_E", &makeE, "obs_errors"_a, "noise"_a);
    m.def("make_D", &makeD, "obs_values"_a, "E"_a, "S"_a);
    m.def("update_A", &updateA, "data"_a, "A"_a, "Yin"_a, "R"_a, "E"_a, "D"_a,
          "inversion"_a, "truncation"_a, "step_length"_a);
    m.def("init_update", init_update, "module_data"_a, "ens_mask"_a,
          "obs_mask"_a);

    py::enum_<Inversion>(m, "InversionType")
        .value("EXACT", Inversion::exact)
        .value("EE_R", Inversion::subspace_ee_r)
        .value("EXACT_R", Inversion::subspace_exact_r)
        .value("SUBSPACE_RE", Inversion::subspace_re)
        .export_values();
}
