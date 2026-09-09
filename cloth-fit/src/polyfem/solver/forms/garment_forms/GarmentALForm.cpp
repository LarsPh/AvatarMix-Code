#include "GarmentALForm.hpp"
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/Timer.hpp>

namespace polyfem::solver {
    
    double PointPenaltyForm::value_unweighted(const Eigen::VectorXd &x) const 
    {
        return (x(indices_) - target_).squaredNorm() / 2.;
    }

    void PointPenaltyForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const 
    {
        gradv.setZero(x.size());
        gradv(indices_) = x(indices_) - target_;
    }

    void PointPenaltyForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const 
    {
        POLYFEM_SCOPED_TIMER("penalty hessian");
        hessian.setZero();
        hessian.resize(x.size(), x.size());
        std::vector<Eigen::Triplet<double>> triplets;
        for (int i = 0; i < indices_.size(); i++)
            triplets.emplace_back(indices_[i], indices_[i], 1.);
        hessian.setFromTriplets(triplets.begin(), triplets.end());
    }


    double PointLagrangianForm::value_unweighted(const Eigen::VectorXd &x) const 
    {
        return -lagr_mults_.transpose() * (x(indices_) - target_);
    }

    void PointLagrangianForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const 
    {
        gradv.setZero(x.size());
        gradv(indices_) = -lagr_mults_;
    }

    void PointLagrangianForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const 
    {
        hessian.setZero();
        hessian.resize(x.size(), x.size());
    }

    void PointLagrangianForm::update_lagrangian(const Eigen::VectorXd &x, const double k_al)
    {
        lagr_mults_ -= k_al * (x(indices_) - target_);
    }
}