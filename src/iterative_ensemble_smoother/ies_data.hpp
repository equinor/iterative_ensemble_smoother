#pragma once
#include <Eigen/Dense>
#include <vector>

namespace ies {

class Data {
public:
    Data(int ens_size);

    void update_ens_mask(const std::vector<bool> &mask);
    void store_initial_obs_mask(const std::vector<bool> &mask);
    void update_obs_mask(const std::vector<bool> &mask);

    const std::vector<bool> &obs_mask0() const;
    const std::vector<bool> &obs_mask() const;
    const std::vector<bool> &ens_mask() const;

    const Eigen::MatrixXd &getA0() const;
    const Eigen::MatrixXd &getW() const;
    Eigen::MatrixXd &getW();
    const Eigen::MatrixXd &getE() const;

    int obs_mask_size() const;
    int ens_mask_size() const;

    void store_initialE(const Eigen::MatrixXd &E0);
    void augment_initialE(const Eigen::MatrixXd &E0);
    void store_initialA(const Eigen::MatrixXd &A);

    Eigen::MatrixXd make_activeE() const;
    Eigen::MatrixXd make_activeW() const;
    Eigen::MatrixXd make_activeA() const;

    int iteration_nr = 1;

private:
    bool m_converged = false;
    /** Coefficient matrix used to compute Omega = I + W (I -11'/N)/sqrt(N-1) */
    Eigen::MatrixXd W;

    std::vector<bool> m_ens_mask{};
    std::vector<bool> m_obs_mask0{};
    std::vector<bool> m_obs_mask{};
    /** Prior ensemble used in Ei=A0 Omega_i */
    Eigen::MatrixXd A0{};
    /** Prior ensemble of measurement perturations (should be the same for all iterations) */
    Eigen::MatrixXd E;
};

} // namespace ies
