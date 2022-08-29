#include <algorithm>
#include <memory>

#include <ies_data.hpp>
#include <pybind11/pybind11.h>

ies::Data::Data(int ens_size) : W(Eigen::MatrixXd::Zero(ens_size, ens_size)) {}

void ies::Data::update_ens_mask(const std::vector<bool> &mask) {
    this->m_ens_mask = mask;
}

void ies::Data::store_initial_obs_mask(const std::vector<bool> &mask) {
    if (this->m_obs_mask0.empty())
        this->m_obs_mask0 = mask;
}

void ies::Data::update_obs_mask(const std::vector<bool> &mask) {
    this->m_obs_mask = mask;
}

int ies::Data::obs_mask_size() const { return this->m_obs_mask.size(); }

int ies::Data::ens_mask_size() const { return (this->m_ens_mask.size()); }

/** We store the initial observation perturbations in E, corresponding to
 * active data->obs_mask0 in data->E. The unused rows in data->E corresponds to
 * false data->obs_mask0
 */
void ies::Data::store_initialE(const Eigen::MatrixXd &E0) {
    if (E.rows() != 0 || E.cols() != 0)
        return;
    int obs_size_msk = this->obs_mask_size();
    int ens_size_msk = this->ens_mask_size();
    this->E = Eigen::MatrixXd::Zero(obs_size_msk, ens_size_msk);
    this->E.setConstant(-999.9);

    int m = 0;
    for (int iobs = 0; iobs < obs_size_msk; iobs++) {
        if (this->m_obs_mask0[iobs]) {
            int active_idx = 0;
            for (int iens = 0; iens < ens_size_msk; iens++) {
                if (this->m_ens_mask[iens]) {
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
void ies::Data::augment_initialE(const Eigen::MatrixXd &E0) {

    int obs_size_msk = this->obs_mask_size();
    int ens_size_msk = this->ens_mask_size();
    int m = 0;
    for (int iobs = 0; iobs < obs_size_msk; iobs++) {
        if (!this->m_obs_mask0[iobs] && this->m_obs_mask[iobs]) {
            int i = -1;
            for (int iens = 0; iens < ens_size_msk; iens++) {
                if (this->m_ens_mask[iens]) {
                    i++;
                    this->E(iobs, iens) = E0(m, i);
                }
            }
            this->m_obs_mask0[iobs] = true;
        }
        if (this->m_obs_mask[iobs]) {
            m++;
        }
    }
}

void ies::Data::store_initialA(const Eigen::MatrixXd &A0) {
    if (this->A0.rows() != 0 || this->A0.cols() != 0)
        return;
    this->A0 = Eigen::MatrixXd::Zero(A0.rows(), this->m_ens_mask.size());
    for (int irow = 0; irow < this->A0.rows(); irow++) {
        int active_idx = 0;
        for (int iens = 0; iens < this->m_ens_mask.size(); iens++) {
            if (this->m_ens_mask[iens]) {
                this->A0(irow, iens) = A0(irow, active_idx);
                active_idx++;
            }
        }
    }
}

const std::vector<bool> &ies::Data::obs_mask0() const {
    return this->m_obs_mask0;
}

const std::vector<bool> &ies::Data::obs_mask() const {
    return this->m_obs_mask;
}

const std::vector<bool> &ies::Data::ens_mask() const {
    return this->m_ens_mask;
}

const Eigen::MatrixXd &ies::Data::getE() const { return this->E; }

Eigen::MatrixXd &ies::Data::getW() { return this->W; }

const Eigen::MatrixXd &ies::Data::getW() const { return this->W; }

const Eigen::MatrixXd &ies::Data::getA0() const { return this->A0; }

namespace {

Eigen::MatrixXd make_active(const Eigen::MatrixXd &full_matrix,
                            const std::vector<bool> &row_mask,
                            const std::vector<bool> &column_mask) {
    int rows = row_mask.size();
    int columns = column_mask.size();
    Eigen::MatrixXd active = Eigen::MatrixXd::Zero(
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

Eigen::MatrixXd ies::Data::make_activeE() const {
    return make_active(this->E, this->m_obs_mask, this->m_ens_mask);
}

Eigen::MatrixXd ies::Data::make_activeW() const {
    return make_active(this->W, this->m_ens_mask, this->m_ens_mask);
}

Eigen::MatrixXd ies::Data::make_activeA() const {
    std::vector<bool> row_mask(this->A0.rows(), true);
    return make_active(this->A0, row_mask, this->m_ens_mask);
}
