#include <algorithm>
#include <memory>
#include <variant>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using Eigen::ComputeFullU;
using Eigen::ComputeThinU;
using Eigen::ComputeThinV;
using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace py = pybind11;

enum struct Inversion {
  exact = 0,
  subspace_exact_r = 1,
  subspace_ee_r = 2,
  subspace_re = 3
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
  // Corresponds to (I + \Lambda_1)^{-1} since `eig` has already been
  // transformed.
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
 * "Efficient Implementation of an Iterative Ensemble Smoother for Data
 * Assimilation and Reservoir History Matching"
 * https://www.frontiersin.org/articles/10.3389/fams.2019.00047/full
 *
 * Section 2.4.3
 */
/**
 *  The standard inversion works on the equation
 *          S'*(S*S'+R)^{-1} H           (a)
 */
void subspace_inversion(MatrixXd &W, const Inversion ies_inversion,
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

  // Update data->W = (1-ies_steplength) * data->W +  ies_steplength * S' * X3
  // (Line 9)
  W = ies_steplength * S.transpose() * X3 + (1.0 - ies_steplength) * W;
}

/**
 * Section 3.2 - Exact inversion
 * This calculates (S^T*S + I_N)^{-1} by taking the SVD of (S^T*S + I_N),
 * and since (S^T*S + I_N) is symmetric positive semi-definite we have that U=V
 * and hence (S^T*S + I_N)^{-1} = U * \Sigma^{-1} * U^T.
 */
void exact_inversion(MatrixXd &W, const MatrixXd &S, const MatrixXd &H,
                     double ies_steplength) {
  int ens_size = S.cols();

  MatrixXd StS = S.transpose() * S + MatrixXd::Identity(ens_size, ens_size);

  auto svd = StS.bdcSvd(ComputeFullU);
  MatrixXd Z = svd.matrixU();
  VectorXd eig = svd.singularValues();

  MatrixXd ZtStH = Z.transpose() * S.transpose() * H;

  for (int i = 0; i < ens_size; i++)
    ZtStH.row(i) /= eig[i];

  // Update data->W = (1-ies_steplength) * data->W +  ies_steplength * Z *
  // (Lamda^{-1}) Z' S' H (Line 9)
  W = ies_steplength * Z * ZtStH + (1.0 - ies_steplength) * W;
}

/**
 * @param Y Predicted ensemble anomalies normalized by sqrt(N-1),
 *          where N is the number of realizations.
 *          See line 4 of Algorithm 1 and Eq. 30.
 */
MatrixXd
create_transition_matrix(py::EigenDRef<MatrixXd> Y, py::EigenDRef<MatrixXd> R,
                         py::EigenDRef<MatrixXd> E, py::EigenDRef<MatrixXd> D,
                         const Inversion ies_inversion,
                         const std::variant<double, int> &truncation,
                         MatrixXd &W, double ies_steplength)

{
  const int ens_size = Y.cols();

  /* Line 5 of Algorithm 1 */
  MatrixXd Omega =
      (1.0 / sqrt(ens_size - 1.0)) * (W.colwise() - W.rowwise().mean());
  Omega.diagonal().array() += 1.0;

  /* Solving for the average sensitivity matrix.
     Line 6 of Algorithm 1, also Section 5
  */
  Omega.transposeInPlace();
  MatrixXd S = Omega.fullPivLu().solve(Y.transpose()).transpose();

  /* Similar to the innovation term.
     Differs in that `D` here is defined as dobs + E - Y instead of just dobs +
     E as in the paper. Line 7 of Algorithm 1, also Section 2.6
  */
  MatrixXd H = D + S * W;

  /*
   * COMPUTE NEW UPDATED W (Line 9 of Algorithm 1)
   * W = W - ies_steplength * (W - S' * (S * S' + R)^{-1} * H)
   * When R=I Line 9 can be rewritten as
   * W = W - ies_steplength * ( W - (S'*S + I)^{-1} * S' * H )
   * Notice the expression being inverted.
   * Instead of S * S' which is a (num_obs, num_obs) sized matrix,
   * we get S' * S which is of size (ensemble_size, ensemble_size).
   * This is great since num_obs is usually much larger than ensemble_size.
   *
   * With R=I the subspace inversion (ies_inversion=1) with
   * singular value trucation=1.000 gives exactly the same solution as the exact
   * inversion (ies_inversion=0).
   *
   * Using ies_inversion=IES_INVERSION_SUBSPACE_EXACT_R(2), and a step length
   * of 1.0, one update gives identical result to STD as long as the same SVD
   * truncation is used.
   *
   * With very large data sets it is likely that the inversion becomes poorly
   * conditioned and a trucation=1.000 is not a good choice. In this case the
   * ies_inversion > 0 and truncation set to 0.99 or so, should stabelize
   * the algorithm.
   */

  if (ies_inversion == Inversion::exact) {
    exact_inversion(W, S, H, ies_steplength);
  } else {
    subspace_inversion(W, ies_inversion, E, R, S, H, truncation,
                       ies_steplength);
  }

  /* Line 9 of Algorithm 1 */
  MatrixXd X = W;
  X /= sqrt(ens_size - 1.0);
  X.diagonal().array() += 1;

  return X;
}

/**
 * @param Y Predicted ensemble anomalies normalized by sqrt(ensemble_size-1).
 *          See line 4 of Algorithm 1 and Eq. 30.
 */
MatrixXd updateA(
    // Updated ensemble A retured to ERT.
    py::EigenDRef<MatrixXd> A, py::EigenDRef<MatrixXd> Y,
    // Measurement error covariance matrix (not used)
    py::EigenDRef<MatrixXd> Rin,
    // Ensemble of observation perturbations
    py::EigenDRef<MatrixXd> Ein,
    // (d+E-Y) Ensemble of perturbed observations - Y
    py::EigenDRef<MatrixXd> D, MatrixXd coefficient_matrix,
    const Inversion ies_inversion, const std::variant<double, int> &truncation,
    double ies_steplength) {

  auto X = create_transition_matrix(Y, Rin, Ein, D, ies_inversion, truncation,
                                    coefficient_matrix, ies_steplength);

  A *= X;

  return X;
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

  m.def("create_transition_matrix", &create_transition_matrix, "Y0"_a, "R"_a,
        "E"_a, "D"_a, "ies_inversion"_a, "truncation"_a, "W"_a,
        "ies_steplength"_a);
  m.def("make_E", &makeE, "obs_errors"_a, "noise"_a);
  m.def("make_D", &makeD, "obs_values"_a, "E"_a, "S"_a);
  m.def("update_A", &updateA, "A"_a, "Y"_a, "R"_a, "E"_a, "D"_a,
        "coefficient_matrix"_a, "inversion"_a, "truncation"_a, "step_length"_a);

  py::enum_<Inversion>(m, "InversionType")
      .value("EXACT", Inversion::exact)
      .value("EE_R", Inversion::subspace_ee_r)
      .value("EXACT_R", Inversion::subspace_exact_r)
      .value("SUBSPACE_RE", Inversion::subspace_re)
      .export_values();
}
