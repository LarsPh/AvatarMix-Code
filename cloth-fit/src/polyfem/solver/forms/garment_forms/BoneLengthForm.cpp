#include "BoneLengthForm.hpp"

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
    BoneLengthForm::BoneLengthForm(const Eigen::MatrixXd &skelV, const Eigen::MatrixXi &skelE)
        : V0_(skelV)
        , E_(skelE)
    {
        L2_.setZero(E_.rows());
        for (int e = 0; e < E_.rows(); ++e)
        {
            const Eigen::Vector3d d = V0_.row(E_(e, 1)) - V0_.row(E_(e, 0));
            L2_(e) = d.squaredNorm();
        }
    }

    double BoneLengthForm::value_unweighted(const Eigen::VectorXd &x) const
    {
        const Eigen::MatrixXd V = utils::unflatten(x, 3) + V0_;
        double val = 0.0;
        for (int e = 0; e < E_.rows(); ++e)
        {
            const Eigen::Vector3d d = V.row(E_(e, 1)) - V.row(E_(e, 0));
            const double s = d.squaredNorm() - L2_(e);
            val += s * s;
        }
        return val;
    }

    void BoneLengthForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
    {
        const Eigen::MatrixXd V = utils::unflatten(x, 3) + V0_;
        gradv.setZero(x.size());

        for (int e = 0; e < E_.rows(); ++e)
        {
            const int i = E_(e, 0), j = E_(e, 1);
            const Eigen::Vector3d d = V.row(j) - V.row(i);
            const double s = d.squaredNorm() - L2_(e);
            const Eigen::Vector3d g = 4.0 * s * d;
            for (int k = 0; k < 3; ++k)
            {
                gradv(3 * i + k) -= g(k);
                gradv(3 * j + k) += g(k);
            }
        }
    }

    void BoneLengthForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
    {
        const Eigen::MatrixXd V = utils::unflatten(x, 3) + V0_;

        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(E_.rows() * 36);

        for (int e = 0; e < E_.rows(); ++e)
        {
            const int i = E_(e, 0), j = E_(e, 1);
            const Eigen::Vector3d d = V.row(j) - V.row(i);
            const double s = d.squaredNorm() - L2_(e);
            const Eigen::Matrix3d Hd = 8.0 * (d * d.transpose()) + 4.0 * s * Eigen::Matrix3d::Identity();

            auto scatter_block = [&](int vi, int vj, const Eigen::Matrix3d &M)
            {
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        trips.emplace_back(3 * vi + r, 3 * vj + c, M(r, c));
            };

            scatter_block(i, i, Hd);
            scatter_block(j, j, Hd);
            scatter_block(i, j, -Hd);
            scatter_block(j, i, -Hd);
        }

        hessian.setZero();
        hessian.resize(x.size(), x.size());
        hessian.setFromTriplets(trips.begin(), trips.end());
    }
} // namespace polyfem::solver


