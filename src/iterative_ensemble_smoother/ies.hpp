#pragma once
#include <variant>

#include <Eigen/Dense>
#include "./ies_data.hpp"

enum struct Inversion {
    exact = 0,
    subspace_exact_r = 1,
    subspace_ee_r = 2,
    subspace_re = 3
};

void init_update(Data &module_data, const std::vector<bool> &ens_mask,
                 const std::vector<bool> &obs_mask);

Eigen::MatrixXd makeX(const Eigen::MatrixXd &A, const Eigen::MatrixXd &Y0,
                      const Eigen::MatrixXd &R, const Eigen::MatrixXd &E,
                      const Eigen::MatrixXd &D, const Inversion ies_inversion,
                      const std::variant<double, int> &truncation,
                      Eigen::MatrixXd &W0, double ies_steplength,
                      int iteration_nr);

/**
 *
 * @param A Updated ensemble
 * @param Yin Ensemble of predicted measurements
 * @param Rin Measurement error covariance matrix (not used)
 * @param Ein Ensemble of observation perturbations
 * @param Din (d+E-Y) Ensemble of perturbed observations - Y
 */
void updateA(Data &data, Eigen::Ref<Eigen::MatrixXd> A,
             const Eigen::MatrixXd &Yin, const Eigen::MatrixXd &Rin,
             const Eigen::MatrixXd &Ein, const Eigen::MatrixXd &Din,
             const Inversion ies_inversion,
             const std::variant<double, int> &truncation,
             double ies_steplength);

Eigen::MatrixXd makeE(const Eigen::VectorXd &obs_errors,
                      const Eigen::MatrixXd &noise);
Eigen::MatrixXd makeD(const Eigen::VectorXd &obs_values,
                      const Eigen::MatrixXd &E, const Eigen::MatrixXd &S);
