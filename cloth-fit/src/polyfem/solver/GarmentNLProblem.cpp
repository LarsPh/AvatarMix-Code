#include "GarmentNLProblem.hpp"

#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

namespace polyfem::solver
{
	GarmentNLProblem::GarmentNLProblem(
		const int full_size,
		const Eigen::MatrixXd &target,
		const std::vector<std::shared_ptr<Form>> &forms,
		const std::vector<std::shared_ptr<Form>> &full_forms)
		: FullNLProblem(forms),
		  full_size_(full_size),
		  reduced_size_(full_size_ - 1),
		  target_(utils::flatten(target)),
		  full_forms_(full_forms)
	{
		use_reduced_size();

        P.setZero();
        P.resize(reduced_size_ + target_.size(), full_size_);
        std::vector<Eigen::Triplet<double>> T;
        for (int i = 0; i < target_.size(); i++)
            T.emplace_back(i, 0, target_(i));
        
        for (int i = 0; i < reduced_size_; i++)
            T.emplace_back(i + target_.size(), i + 1, 1.0);
        
        P.setFromTriplets(T.begin(), T.end());
	}

	void GarmentNLProblem::init_lagging(const TVector &x)
	{
		// WARNING: This does not call for full_forms_
		FullNLProblem::init_lagging(full_to_complete(reduced_to_full(x)));
	}

	void GarmentNLProblem::update_lagging(const TVector &x, const int iter_num)
	{
		// WARNING: This does not call for full_forms_
		FullNLProblem::update_lagging(full_to_complete(reduced_to_full(x)), iter_num);
	}

	void GarmentNLProblem::update_quantities(const double t, const TVector &x)
	{
		const TVector y = reduced_to_full(x);
		const TVector z = full_to_complete(y);
		for (auto &f : forms_)
			f->update_quantities(t, z);
		for (auto &f : full_forms_)
			f->update_quantities(t, y);
	}

	void GarmentNLProblem::line_search_begin(const TVector &x0, const TVector &x1)
	{
		// WARNING: This does not call for full_forms_
		FullNLProblem::line_search_begin(full_to_complete(reduced_to_full(x0)), full_to_complete(reduced_to_full(x1)));
	}

	double GarmentNLProblem::max_step_size(const TVector &x0, const TVector &x1)
	{
		// WARNING: This does not call for full_forms_
		return FullNLProblem::max_step_size(full_to_complete(reduced_to_full(x0)), full_to_complete(reduced_to_full(x1)));
	}

	bool GarmentNLProblem::is_step_valid(const TVector &x0, const TVector &x1)
	{
		// WARNING: This does not call for full_forms_
		return FullNLProblem::is_step_valid(full_to_complete(reduced_to_full(x0)), full_to_complete(reduced_to_full(x1)));
	}

	bool GarmentNLProblem::is_step_collision_free(const TVector &x0, const TVector &x1)
	{
		// WARNING: This does not call for full_forms_
		return FullNLProblem::is_step_collision_free(full_to_complete(reduced_to_full(x0)), full_to_complete(reduced_to_full(x1)));
	}

	double GarmentNLProblem::value(const TVector &x)
	{
		const TVector y = reduced_to_full(x);
		const TVector z = full_to_complete(y);

		double val = FullNLProblem::value(z);
		for (auto &f : full_forms_)
			if (f->enabled())
				val += f->value(y);
		return val;
	}

	void GarmentNLProblem::gradient(const TVector &x, TVector &grad)
	{
		const TVector y = reduced_to_full(x);
		const TVector z = full_to_complete(y);

		TVector g;
		FullNLProblem::gradient(z, g);
		grad = complete_to_full_grad(g);
		for (auto &f : full_forms_)
		{
			if (!f->enabled())
				continue;
			TVector tmp;
			f->first_derivative(y, tmp);
			logger().trace("[{}] Gradient norm: {}", f->name(), tmp.norm());
			grad += tmp;
		}
		grad = full_to_reduced_grad(grad);
	}

	void GarmentNLProblem::hessian(const TVector &x, THessian &hessian)
	{
		const TVector y = reduced_to_full(x);
		const TVector z = full_to_complete(y);

		THessian h;
		FullNLProblem::hessian(z, h);

		complete_hessian_to_full_hessian(h, hessian);
		for (auto &f : full_forms_)
		{
			if (!f->enabled())
				continue;
			THessian tmp;
			f->second_derivative(y, tmp);
			hessian += tmp;
		}
		std::swap(h, hessian);
		full_hessian_to_reduced_hessian(h, hessian);
	}

	void GarmentNLProblem::solution_changed(const TVector &newX)
	{
		const TVector y = reduced_to_full(newX);
		const TVector z = full_to_complete(y);

		FullNLProblem::solution_changed(z);

		for (auto &f : full_forms_)
			f->solution_changed(y);
	}

	void GarmentNLProblem::post_step(const polysolve::nonlinear::PostStepData &data)
	{
		if (post_step_call_back)
			post_step_call_back(data);

		const TVector y = reduced_to_full(data.x);
		const TVector z = full_to_complete(y);
		const TVector gy = reduced_to_full(data.grad);
		const TVector gz = full_to_complete(y);
		FullNLProblem::post_step(polysolve::nonlinear::PostStepData(data.iter_num, data.solver_info, z, gz));

		polysolve::nonlinear::PostStepData tmp_data(data.iter_num, data.solver_info, y, gy);
		for (auto &f : full_forms_)
			f->post_step(tmp_data);

		// TODO: add me back
		// if (state_.args["output"]["advanced"]["save_nl_solve_sequence"])
		// {
		// 	const Eigen::MatrixXd displacements = utils::unflatten(full_to_complete(reduced_to_full(x)), state_.mesh->dimension());
		// 	io::OBJWriter::write(
		// 		state_.resolve_output_path(fmt::format("nonlinear_solve_iter{:03d}.obj", iter_num)),
		// 		state_.collision_mesh.displace_vertices(displacements),
		// 		state_.collision_mesh.edges(), state_.collision_mesh.faces());
		// }
	}

	void GarmentNLProblem::set_apply_DBC(const TVector &x, const bool val)
	{
		TVector full = full_to_complete(reduced_to_full(x));
		for (auto &form : forms_)
			form->set_apply_DBC(full, val);
	}

	GarmentNLProblem::TVector GarmentNLProblem::full_to_reduced(const TVector &full) const
	{
		if (full.size() == current_size())
			return full;
		return full.tail(full.size() - 1);
	}

	GarmentNLProblem::TVector GarmentNLProblem::full_to_reduced_grad(const TVector &full) const
	{
		if (full.size() == current_size())
			return full;
		return full.tail(full.size() - 1);
	}

	GarmentNLProblem::TVector GarmentNLProblem::reduced_to_full(const TVector &reduced) const
	{
		if (reduced.size() == full_size_)
			return reduced;

		TVector full(reduced.size() + 1);
		full(0) = target_value;
		full.tail(reduced.size()) = reduced;
		return full;
	}

	void GarmentNLProblem::full_hessian_to_reduced_hessian(const THessian &full, THessian &reduced) const
	{
		if (full.rows() == current_size())
			reduced = full;
		else
		{
			reduced.setZero();
			reduced.resize(full.rows() - 1, full.cols() - 1);

			std::vector<Eigen::Triplet<double>> T;
			for (int k = 0; k < full.outerSize(); ++k)
				for (THessian::InnerIterator it(full,k); it; ++it)
				{
					if (it.row() > 0 && it.col() > 0)
						T.emplace_back(it.row() - 1, it.col() - 1, it.value());
				}
			
			reduced.setFromTriplets(T.begin(), T.end());
		}
	}

	GarmentNLProblem::TVector GarmentNLProblem::complete_to_full_grad(const TVector &complete) const
	{
		return P.transpose() * complete;
	}

	GarmentNLProblem::TVector GarmentNLProblem::full_to_complete(const TVector &full) const
	{
		return P * full;
	}

	void GarmentNLProblem::complete_hessian_to_full_hessian(const THessian &complete, THessian &full) const
	{
		full = P.transpose() * complete * P;
	}
} // namespace polyfem::solver
