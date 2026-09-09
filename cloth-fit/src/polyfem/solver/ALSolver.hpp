#pragma once

#include <polyfem/solver/GarmentNLProblem.hpp>
#include <polysolve/nonlinear/Solver.hpp>
#include <polyfem/Common.hpp>

#include <Eigen/Core>

#include <functional>
#include <vector>

namespace polyfem::solver
{
	template <class Problem, class LagrangianForm, class PenaltyForm>
	class ALSolver
	{
		using NLSolver = polysolve::nonlinear::Solver;

	public:
		ALSolver(
			std::shared_ptr<LagrangianForm> lagr_form,
			std::shared_ptr<PenaltyForm> pen_form,
			const double initial_al_weight,
			const double scaling,
			const double max_al_weight,
			const double eta_tol,
			const double error_threshold,
			const int max_outer_steps,
			const std::function<void(const Eigen::VectorXd &)> &update_barrier_stiffness);
		virtual ~ALSolver() = default;

		void solve_al(std::shared_ptr<NLSolver> nl_solver, Problem &nl_problem, Eigen::MatrixXd &sol);
		void solve_reduced(std::shared_ptr<NLSolver> nl_solver, Problem &nl_problem, Eigen::MatrixXd &sol);

		std::function<void(const double)> post_subsolve = [](const double) {};

	protected:
		void set_al_weight(Problem &nl_problem, const Eigen::VectorXd &x, const double weight);

		double compute_error(Problem &nl_problem, const Eigen::MatrixXd &sol) const;
		void update_lagrangian(Problem &nl_problem, const Eigen::MatrixXd &sol, double al_weight);

		// forms are only called directly in protected functions
		std::shared_ptr<LagrangianForm> lagr_form;
		std::shared_ptr<PenaltyForm> pen_form;

		const double initial_al_weight;
		const double scaling;
		const double max_al_weight;
		const double eta_tol;
		const double error_threshold;
		const int max_outer_steps;

		// TODO: replace this with a member function
		std::function<void(const Eigen::VectorXd &)> update_barrier_stiffness;
	};
} // namespace polyfem::solver