#include "ALSolver.hpp"
#include <polyfem/solver/forms/garment_forms/GarmentALForm.hpp>

#include <polyfem/utils/Logger.hpp>

namespace polyfem::solver
{
	template <class Problem, class LagrangianForm, class PenaltyForm>
	ALSolver<Problem, LagrangianForm, PenaltyForm>::ALSolver(
		std::shared_ptr<LagrangianForm> lagr_form,
		std::shared_ptr<PenaltyForm> pen_form,
		const double initial_al_weight,
		const double scaling,
		const double max_al_weight,
		const double eta_tol,
		const double error_threshold,
		const int max_outer_steps,
		const std::function<void(const Eigen::VectorXd &)> &update_barrier_stiffness)
		: lagr_form(lagr_form),
		  pen_form(pen_form),
		  initial_al_weight(initial_al_weight),
		  scaling(scaling),
		  max_al_weight(max_al_weight),
		  eta_tol(eta_tol),
		  error_threshold(error_threshold),
		  max_outer_steps(max_outer_steps),
		  update_barrier_stiffness(update_barrier_stiffness)
	{
	}

	template <class Problem, class LagrangianForm, class PenaltyForm>
	void ALSolver<Problem, LagrangianForm, PenaltyForm>::solve_al(std::shared_ptr<NLSolver> nl_solver, Problem &nl_problem, Eigen::MatrixXd &sol)
	{
		assert(sol.size() == nl_problem.full_size());

		const Eigen::VectorXd initial_sol = sol;
		Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);
		assert(tmp_sol.size() == nl_problem.reduced_size());

		// --------------------------------------------------------------------

		double al_weight = initial_al_weight;
		int al_steps = 0;
		const int iters = nl_solver->stop_criteria().iterations;

		const double initial_error = compute_error(nl_problem, sol);
		double current_error = initial_error;

		nl_problem.line_search_begin(sol, tmp_sol);

		while (!std::isfinite(nl_problem.value(tmp_sol))
			   || !nl_problem.is_step_valid(sol, tmp_sol)
			   || !nl_problem.is_step_collision_free(sol, tmp_sol)
			   || current_error > error_threshold)
		{
			if (max_outer_steps > 0 && al_steps >= max_outer_steps)
				log_and_throw_error("Augmented Lagrangian exceeded max_outer_steps={} without reaching feasibility. Consider increasing solver.augmented_lagrangian.max_outer_steps or loosening error_threshold.", max_outer_steps);

			nl_problem.line_search_end();

			set_al_weight(nl_problem, sol, al_weight);
			logger().debug("Solving AL Problem with weight {}", al_weight);

			nl_problem.init(sol);
			update_barrier_stiffness(sol);
			tmp_sol = sol;

			try
			{
				nl_solver->minimize(nl_problem, tmp_sol);
			}
			catch (const std::runtime_error &e)
			{
			}

			sol = tmp_sol;
			set_al_weight(nl_problem, sol, -1);

			current_error = compute_error(nl_problem, sol);
			const double eta = 1 - sqrt(current_error / initial_error);

			logger().debug("Current eta = {}", eta);

			if (eta < 0)
			{
				logger().debug("Higher error than initial, increase weight and revert to previous solution");
				sol = initial_sol;
			}

			tmp_sol = nl_problem.full_to_reduced(sol);
			nl_problem.line_search_begin(sol, tmp_sol);

			if (eta < eta_tol && al_weight < max_al_weight)
				al_weight *= scaling;
			else
				update_lagrangian(nl_problem, sol, al_weight);

			post_subsolve(al_weight);
			++al_steps;
		}
		nl_problem.line_search_end();
		nl_solver->stop_criteria().iterations = iters;
	}

	template <class Problem, class LagrangianForm, class PenaltyForm>
	void ALSolver<Problem, LagrangianForm, PenaltyForm>::solve_reduced(std::shared_ptr<NLSolver> nl_solver, Problem &nl_problem, Eigen::MatrixXd &sol)
	{
		assert(sol.size() == nl_problem.full_size());

		Eigen::VectorXd tmp_sol = nl_problem.full_to_reduced(sol);
		nl_problem.line_search_begin(sol, tmp_sol);

		if (!std::isfinite(nl_problem.value(tmp_sol))
			|| !nl_problem.is_step_valid(sol, tmp_sol)
			|| !nl_problem.is_step_collision_free(sol, tmp_sol))
			log_and_throw_error("Failed to apply boundary conditions; solve with augmented lagrangian first!");

		// --------------------------------------------------------------------
		// Perform one final solve with the DBC projected out

		logger().debug("Successfully applied boundary conditions; solving in reduced space");

		nl_problem.init(sol);
		update_barrier_stiffness(sol);
		try
		{
			nl_solver->minimize(nl_problem, tmp_sol);
		}
		catch (const std::runtime_error &e)
		{
			sol = nl_problem.reduced_to_full(tmp_sol);
			throw e;
		}
		sol = nl_problem.reduced_to_full(tmp_sol);

		post_subsolve(0);
	}

	template <>
	double ALSolver<GarmentNLProblem, PointLagrangianForm, PointPenaltyForm>::compute_error(GarmentNLProblem &nl_problem, const Eigen::MatrixXd &sol) const
	{
		return pen_form->compute_error(nl_problem.full_to_complete(sol));
	}

	template <>
	void ALSolver<GarmentNLProblem, PointLagrangianForm, PointPenaltyForm>::update_lagrangian(GarmentNLProblem &nl_problem, const Eigen::MatrixXd &sol, double al_weight)
	{
		lagr_form->update_lagrangian(nl_problem.full_to_complete(sol), al_weight);
	}

	template <class Problem, class LagrangianForm, class PenaltyForm>
	void ALSolver<Problem, LagrangianForm, PenaltyForm>::set_al_weight(Problem &nl_problem, const Eigen::VectorXd &x, const double weight)
	{
		if (pen_form == nullptr || lagr_form == nullptr)
			return;
		if (weight > 0)
		{
			pen_form->enable();
			lagr_form->enable();
			pen_form->set_weight(weight);
			nl_problem.use_full_size();
			nl_problem.set_apply_DBC(x, false);
		}
		else
		{
			pen_form->disable();
			lagr_form->disable();
			nl_problem.use_reduced_size();
			nl_problem.set_apply_DBC(x, true);
		}
	}

	template class ALSolver<GarmentNLProblem, PointLagrangianForm, PointPenaltyForm>;
} // namespace polyfem::solver