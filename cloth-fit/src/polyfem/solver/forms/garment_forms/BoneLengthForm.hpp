#pragma once

#include <polyfem/solver/forms/Form.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem::solver
{
    class BoneLengthForm : public Form
    {
    public:
        BoneLengthForm(const Eigen::MatrixXd &skelV, const Eigen::MatrixXi &skelE);
        virtual ~BoneLengthForm() = default;

        std::string name() const override { return "bone-length"; }

    protected:
        double value_unweighted(const Eigen::VectorXd &x) const override;
        void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;
        void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

    private:
        Eigen::MatrixXd V0_;
        Eigen::MatrixXi E_;
        Eigen::VectorXd L2_; // squared rest lengths
    };
} // namespace polyfem::solver


