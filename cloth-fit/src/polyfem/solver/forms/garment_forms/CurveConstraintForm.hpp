#pragma once

#include <polyfem/solver/forms/Form.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <vector>

namespace polyfem::solver
{
	std::vector<Eigen::VectorXi> boundary_curves(const Eigen::MatrixXi &F);

	Eigen::MatrixXd extract_curve_center_targets(
		const Eigen::MatrixXd &garment_v,
		const std::vector<Eigen::VectorXi> &curves,
		const Eigen::MatrixXd &skeleton_v,
		const Eigen::MatrixXi &skeleton_bones,
		const Eigen::MatrixXd &target_skeleton_v);

	class CurveCurvatureForm : public Form
	{
	public:
		CurveCurvatureForm(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
		CurveCurvatureForm(const Eigen::MatrixXd &V, const std::vector<Eigen::VectorXi> &curves);
		virtual ~CurveCurvatureForm() = default;

		std::string name() const override { return "curve-curvature"; }

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
        std::vector<Eigen::MatrixXd> compute_angles(const Eigen::MatrixXd &V) const;

		const Eigen::MatrixXd V_;
        std::vector<Eigen::VectorXi> curves_;

		std::vector<Eigen::MatrixXd> orig_angles;
	};

	class CurveSizeForm : public Form
	{
	public:
		CurveSizeForm(const Eigen::MatrixXd &V, const std::vector<Eigen::VectorXi> &curves);
		virtual ~CurveSizeForm() = default;

		std::string name() const override { return "curve-size"; }

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
		const Eigen::MatrixXd V_;
        std::vector<Eigen::VectorXi> curves_;
	};

	class CurveTorsionForm : public Form
	{
	public:
		CurveTorsionForm(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
		CurveTorsionForm(const Eigen::MatrixXd &V, const std::vector<Eigen::VectorXi> &curves);
		virtual ~CurveTorsionForm() = default;

		std::string name() const override { return "curve-torsion"; }

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
		const double c = 0.01;
		
        std::vector<Eigen::MatrixXd> compute_angles(const Eigen::MatrixXd &V) const;

		const Eigen::MatrixXd V_;
        std::vector<Eigen::VectorXi> curves_;

		std::vector<Eigen::MatrixXd> orig_angles;
	};

	class SymmetryForm : public Form
	{
	public:
		SymmetryForm(const Eigen::MatrixXd &V, const std::vector<Eigen::VectorXi> &curves);
		virtual ~SymmetryForm() = default;

		std::string name() const override { return "symmetry"; }

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
		const Eigen::MatrixXd V_;
        std::vector<Eigen::VectorXi> curves_;
		const double tol = 1e-1;
		const int dim = 0;

		std::vector<Eigen::VectorXi> correspondences_;

		mutable StiffnessMatrix hessian_cached;
	};
}
