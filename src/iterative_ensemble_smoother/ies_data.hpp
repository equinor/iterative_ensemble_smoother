#pragma once
#include <Eigen/Dense>
#include <vector>

class Data {
public:
    std::vector<bool> ens_mask{};
    std::vector<bool> obs_mask{};

    /** Coefficient matrix used to compute Omega = I + W (I -11'/N)/sqrt(N-1) */
    Eigen::MatrixXd W;

    Data(int ens_size);

    void store_initial_obs_mask(const std::vector<bool> &mask);
    void update_obs_mask(const std::vector<bool> &mask);

    void store_initialE(const Eigen::MatrixXd &E0);
    void augment_initialE(const Eigen::MatrixXd &E0);
    void store_initialA(const Eigen::MatrixXd &A);

    Eigen::MatrixXd make_activeE() const;
    Eigen::MatrixXd make_activeW() const;
    Eigen::MatrixXd make_activeA() const;

    int iteration_nr = 1;

private:
    bool m_converged = false;

    std::vector<bool> m_obs_mask0{};
    /** Prior ensemble used in Ei=A0 Omega_i */
    Eigen::MatrixXd A0{};
    /** Prior ensemble of measurement perturations (should be the same for all iterations) */
    Eigen::MatrixXd E;
};
