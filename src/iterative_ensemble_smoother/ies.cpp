#include <algorithm>
#include <variant>
#include <vector>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "./ies.hpp"
#include "./ies_data.hpp"
#include "./linalg.hpp"

namespace py = pybind11;
using Eigen::MatrixXd;

/** Implementation of algorithm as described in
 * "Efficient Implementation of an Iterative Ensemble Smoother for Data Assimilation and Reservoir History Matching"
 * https://www.frontiersin.org/articles/10.3389/fams.2019.00047/full
 */
void compute_AA_projection(const Eigen::MatrixXd &A, Eigen::MatrixXd &Y);

void subspace_inversion(Eigen::MatrixXd &W0, const Inversion ies_inversion,
                        const Eigen::MatrixXd &E, const Eigen::MatrixXd &R,
                        const Eigen::MatrixXd &S, const Eigen::MatrixXd &H,
                        const std::variant<double, int> &truncation,
                        double ies_steplength);

void exact_inversion(Eigen::MatrixXd &W0, const Eigen::MatrixXd &S,
                     const Eigen::MatrixXd &H, double ies_steplength);

void init_update(Data &module_data, const std::vector<bool> &ens_mask,
                 const std::vector<bool> &obs_mask) {
    module_data.ens_mask = ens_mask;
    module_data.store_initial_obs_mask(obs_mask);
    module_data.obs_mask = obs_mask;
}

Eigen::MatrixXd makeX(const Eigen::MatrixXd &A, const Eigen::MatrixXd &Y0,
                      const Eigen::MatrixXd &R, const Eigen::MatrixXd &E,
                      const Eigen::MatrixXd &D, const Inversion ies_inversion,
                      const std::variant<double, int> &truncation,
                      Eigen::MatrixXd &W0, double ies_steplength,
                      int iteration_nr)

{
    const int ens_size = Y0.cols();

    Eigen::MatrixXd Y = Y0;

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
    Eigen::MatrixXd Omega =
        (1.0 / sqrt(ens_size - 1.0)) * (W0.colwise() - W0.rowwise().mean());
    Omega.diagonal().array() += 1.0;

    /* Solving for the average sensitivity matrix.
       Line 6 of Algorithm 1, also Section 5
    */
    Omega.transposeInPlace();
    Eigen::MatrixXd S = Omega.fullPivLu().solve(Y.transpose()).transpose();

    /* Similar to the innovation term.
       Differs in that `D` here is defined as dobs + E - Y instead of just dobs + E as in the paper.
       Line 7 of Algorithm 1, also Section 2.6
    */
    Eigen::MatrixXd H = D + S * W0;

    /* Store previous W for convergence test */
    Eigen::MatrixXd W = W0;

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
    Eigen::MatrixXd X = W0;
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
static void store_active_W(Data &data, const Eigen::MatrixXd &W0) {
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
             Eigen::Ref<Eigen::MatrixXd> A,
             // Ensemble of predicted measurements
             const Eigen::MatrixXd &Yin,
             // Measurement error covariance matrix (not used)
             const Eigen::MatrixXd &Rin,
             // Ensemble of observation perturbations
             const Eigen::MatrixXd &Ein,
             // (d+E-Y) Ensemble of perturbed observations - Y
             const Eigen::MatrixXd &Din, const Inversion ies_inversion,
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
    Eigen::MatrixXd E = data.make_activeE();
    Eigen::MatrixXd D = Din;

    /* Subtract new measurement perturbations              D=D-E    */
    D -= Ein;
    /* Add old measurement perturbations */
    D += E;

    auto W0 = data.make_activeW();
    Eigen::MatrixXd X;

    X = makeX(A, Yin, Rin, E, D, ies_inversion, truncation, W0, ies_steplength,
              iteration_nr);

    store_active_W(data, W0);

    /* COMPUTE NEW ENSEMBLE SOLUTION FOR CURRENT ITERATION  Ei=A0*X (Line 11)*/
    Eigen::MatrixXd A0 = data.make_activeA();
    A = A0 * X;
}

/* Section 2.4.3 */
void compute_AA_projection(const Eigen::MatrixXd &A, Eigen::MatrixXd &Y) {

    Eigen::MatrixXd Ai = A;
    Ai = Ai.colwise() - Ai.rowwise().mean();
    auto svd = Ai.bdcSvd(Eigen::ComputeThinV);
    Eigen::MatrixXd VT = svd.matrixV().transpose();
    Eigen::MatrixXd AAi = VT.transpose() * VT;
    Y *= AAi;
}

/*
*  The standard inversion works on the equation
*          S'*(S*S'+R)^{-1} H           (a)
*/
void subspace_inversion(Eigen::MatrixXd &W0, const Inversion ies_inversion,
                        const Eigen::MatrixXd &E, const Eigen::MatrixXd &R,
                        const Eigen::MatrixXd &S, const Eigen::MatrixXd &H,
                        const std::variant<double, int> &truncation,
                        double ies_steplength) {

    int ens_size = S.cols();
    int nrobs = S.rows();
    double nsc = 1.0 / sqrt(ens_size - 1.0);
    Eigen::MatrixXd X1 = Eigen::MatrixXd::Zero(
        nrobs, std::min(ens_size, nrobs)); // Used in subspace inversion
    Eigen::VectorXd eig(ens_size);

    switch (ies_inversion) {
    case Inversion::subspace_re:
        lowrankE(S, E * nsc, X1, eig, truncation);
        break;

        case Inversion::subspace_ee_r: {
        Eigen::MatrixXd Et = E.transpose();
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
    Eigen::Map<Eigen::VectorXd> eig_vector(eig.data(), eig.size());
    Eigen::MatrixXd X3 = genX3(X1, H, eig_vector);

    // Update data->W = (1-ies_steplength) * data->W +  ies_steplength * S' * X3 (Line 9)
    W0 = ies_steplength * S.transpose() * X3 + (1.0 - ies_steplength) * W0;
}

/** Section 3.2 - Exact inversion
* This calculates (S^T*S + I_N)^{-1} by taking the SVD of (S^T*S + I_N),
* and since (S^T*S + I_N) is symmetric positive semi-definite we have that U=V and hence
* (S^T*S + I_N)^{-1} = U * \Sigma^{-1} * U^T.
*/
void exact_inversion(Eigen::MatrixXd &W0, const Eigen::MatrixXd &S,
                     const Eigen::MatrixXd &H, double ies_steplength) {
    int ens_size = S.cols();

    MatrixXd StS = S.transpose() * S + MatrixXd::Identity(ens_size, ens_size);

    auto svd = StS.bdcSvd(Eigen::ComputeFullU);
    MatrixXd Z = svd.matrixU();
    Eigen::VectorXd eig = svd.singularValues();

    MatrixXd ZtStH = Z.transpose() * S.transpose() * H;

    for (int i = 0; i < ens_size; i++)
        ZtStH.row(i) /= eig[i];

    // Update data->W = (1-ies_steplength) * data->W +  ies_steplength * Z * (Lamda^{-1}) Z' S' H (Line 9)
    W0 = ies_steplength * Z * ZtStH + (1.0 - ies_steplength) * W0;
}

Eigen::MatrixXd makeE(const Eigen::VectorXd &obs_errors,
                      const Eigen::MatrixXd &noise) {
    int active_obs_size = obs_errors.rows();
    int active_ens_size = noise.cols();

    Eigen::MatrixXd E = noise;
    Eigen::VectorXd pert_mean = E.rowwise().mean();
    E = E.colwise() - pert_mean;
    Eigen::VectorXd pert_var = E.cwiseProduct(E).rowwise().sum();

    for (int i = 0; i < active_obs_size; i++) {
        double factor = obs_errors(i) * sqrt(active_ens_size / pert_var(i));
        E.row(i) *= factor;
    }

    return E;
}

Eigen::MatrixXd makeD(const Eigen::VectorXd &obs_values,
                      const Eigen::MatrixXd &E, const Eigen::MatrixXd &S) {

    Eigen::MatrixXd D = E - S;

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
