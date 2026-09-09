#pragma once

#include <polyfem/solver/forms/Form.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <vector>

namespace polyfem::solver
{
	class CurveCenterTargetForm : public Form
	{
	public:
		// WARNING: V and curves should only contain garment mesh info
		CurveCenterTargetForm(
			const Eigen::MatrixXd &V, 
			const std::vector<Eigen::VectorXi> &curves,
			const Eigen::MatrixXd &source_skeleton_v,
			const Eigen::MatrixXd &target_skeleton_v,
			const Eigen::MatrixXi &skeleton_edges);
		virtual ~CurveCenterTargetForm() = default;

		std::string name() const override { return "curve-center-target"; }

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
		const Eigen::MatrixXd source_skeleton_v_;
		const Eigen::MatrixXd target_skeleton_v_;
		const Eigen::MatrixXi skeleton_edges_;

		Eigen::VectorXi bones;
		Eigen::VectorXd relative_positions;
	};

	class CurveTargetForm : public Form
	{
	public:
		// WARNING: V and curves should only contain garment mesh info
		CurveTargetForm(
			const Eigen::MatrixXd &V, 
			const std::vector<Eigen::VectorXi> &curves,
			const Eigen::MatrixXd &source_skeleton_v,
			const Eigen::MatrixXd &target_skeleton_v,
			const Eigen::MatrixXi &skeleton_edges,
			const bool is_skirt = false,
			const bool automatic_bone_generation = false);
		virtual ~CurveTargetForm() = default;

		std::string name() const override { return "curve-target"; }

		// Debug helpers (used by hem-boundary skirt mode).
		// Note: if is_skirt=true, these include the inserted mid-leg joints/edges.
		const Eigen::MatrixXd &debug_source_skeleton_v() const { return source_skeleton_v_; }
		const Eigen::MatrixXd &debug_target_skeleton_v() const { return target_skeleton_v_; }
		const Eigen::MatrixXi &debug_skeleton_edges() const { return skeleton_edges_; }
		bool debug_is_skirt() const { return is_skirt_; }

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
		const bool is_skirt_;
        const Eigen::MatrixXd V_;
        std::vector<Eigen::VectorXi> curves_;
		Eigen::MatrixXd source_skeleton_v_;
		Eigen::MatrixXd target_skeleton_v_;
		Eigen::MatrixXi skeleton_edges_;

		Eigen::VectorXi bones;
		std::vector<Eigen::VectorXd> relative_positions;

		const bool automatic_bone_generation_;
	};
}
