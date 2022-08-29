#pragma once
#include <Eigen/Dense>
#include <variant>

namespace ies {
namespace linalg {
void lowrankCinv(
    const Eigen::MatrixXd &S, const Eigen::MatrixXd &R,
    Eigen::MatrixXd &W,   /* Corresponding to X1 from Eq. 14.29 */
    Eigen::VectorXd &eig, /* Corresponding to 1 / (1 + Lambda_1) (14.29) */
    const std::variant<double, int> &truncation);

void lowrankE(
    const Eigen::MatrixXd &S, /* (nrobs x nrens) */
    const Eigen::MatrixXd &E, /* (nrobs x nrens) */
    Eigen::MatrixXd
        &W, /* (nrobs x nrmin) Corresponding to X1 from Eqs. 14.54-14.55 */
    Eigen::VectorXd
        &eig, /* (nrmin) Corresponding to 1 / (1 + Lambda1^2) (14.54) */
    const std::variant<double, int> &truncation);

Eigen::MatrixXd genX3(const Eigen::MatrixXd &W, const Eigen::MatrixXd &D,
                      const Eigen::VectorXd &eig);
} // namespace linalg
} // namespace ies
