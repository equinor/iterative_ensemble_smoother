#pragma once
#include <variant>

#include <Eigen/Dense>
#include <ies_data.hpp>

namespace ies {

typedef enum {
    IES_INVERSION_EXACT = 0,
    IES_INVERSION_SUBSPACE_EXACT_R = 1,
    IES_INVERSION_SUBSPACE_EE_R = 2,
    IES_INVERSION_SUBSPACE_RE = 3
} inversion_type;

void init_update(Data &module_data, const std::vector<bool> &ens_mask,
                 const std::vector<bool> &obs_mask);

Eigen::MatrixXd makeX(const Eigen::MatrixXd &A, const Eigen::MatrixXd &Y0,
                      const Eigen::MatrixXd &R, const Eigen::MatrixXd &E,
                      const Eigen::MatrixXd &D,
                      const ies::inversion_type ies_inversion,
                      const std::variant<double, int> &truncation,
                      Eigen::MatrixXd &W0, double ies_steplength,
                      int iteration_nr);

void updateA(Data &data,
             // Updated ensemble A returned to ERT.
             Eigen::Ref<Eigen::MatrixXd> A,
             // Ensemble of predicted measurements
             const Eigen::MatrixXd &Yin,
             // Measurement error covariance matrix (not used)
             const Eigen::MatrixXd &Rin,
             // Ensemble of observation perturbations
             const Eigen::MatrixXd &Ein,
             // (d+E-Y) Ensemble of perturbed observations - Y
             const Eigen::MatrixXd &Din,
             const ies::inversion_type ies_inversion,
             const std::variant<double, int> &truncation,
             double ies_steplength);

Eigen::MatrixXd makeE(const Eigen::VectorXd &obs_errors,
                      const Eigen::MatrixXd &noise);
Eigen::MatrixXd makeD(const Eigen::VectorXd &obs_values,
                      const Eigen::MatrixXd &E, const Eigen::MatrixXd &S);
} // namespace ies
