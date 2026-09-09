#pragma once

#include <polyfem/solver/forms/Form.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem::solver
{
	class PointPenaltyForm : public Form
	{
	public:
		PointPenaltyForm(const Eigen::VectorXd &target, const std::vector<int> &indices) : target_(target), indices_(indices) { assert(indices_.size() == target_.size()); }
		virtual ~PointPenaltyForm() = default;

		std::string name() const override { return "point-penalty"; }

		double compute_error(const Eigen::VectorXd &x) const
		{
			return (x(indices_) - target_).squaredNorm();
		}

	protected:
		/// @brief Compute the potential value
		/// @param x Current solution
		/// @return Value of the contact barrier potential
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param x Current solution
		/// @param hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	private:
        Eigen::VectorXd target_;
		std::vector<int> indices_;
	};

	class PointLagrangianForm : public Form
	{
	public:
		PointLagrangianForm(const Eigen::VectorXd &target, const std::vector<int> &indices) : target_(target), indices_(indices) { assert(indices_.size() == target_.size()); lagr_mults_.setZero(indices.size()); }

		std::string name() const override { return "point-lagrangian"; }

		/// @brief Compute the value of the form
		/// @param x Current solution
		/// @return Computed value
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] hessian Output Hessian of the value wrt x
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	public:
		void update_lagrangian(const Eigen::VectorXd &x, const double k_al);

	private:
        Eigen::VectorXd target_;
		std::vector<int> indices_;
		Eigen::VectorXd lagr_mults_;              ///< vector of lagrange multipliers
	};
} // namespace polyfem::solver
