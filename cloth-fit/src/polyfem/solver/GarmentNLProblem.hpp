#pragma once

#include <polyfem/solver/FullNLProblem.hpp>

namespace polyfem::solver
{
	class GarmentNLProblem : public FullNLProblem
	{
	public:
		using typename FullNLProblem::Scalar;
		using typename FullNLProblem::THessian;
		using typename FullNLProblem::TVector;

	public:
		GarmentNLProblem(const int full_size,
					const Eigen::MatrixXd &target,
				  	const std::vector<std::shared_ptr<Form>> &forms,
					const std::vector<std::shared_ptr<Form>> &full_forms);
		virtual ~GarmentNLProblem() = default;

		double value(const TVector &x) override;
		void gradient(const TVector &x, TVector &gradv) override;
		void hessian(const TVector &x, THessian &hessian) override;

		bool is_step_valid(const TVector &x0, const TVector &x1) override;
		bool is_step_collision_free(const TVector &x0, const TVector &x1) override;
		double max_step_size(const TVector &x0, const TVector &x1) override;

		void line_search_begin(const TVector &x0, const TVector &x1) override;
		void post_step(const polysolve::nonlinear::PostStepData &data) override;

		void solution_changed(const TVector &new_x) override;

		void init(const TVector &x0) override
		{
			TVector y = full_to_complete(reduced_to_full(x0));
			for (auto &f : forms_)
				f->init(y);
		}

		void init_lagging(const TVector &x) override;
		void update_lagging(const TVector &x, const int iter_num) override;

		// --------------------------------------------------------------------

		void update_quantities(const double t, const TVector &x);

		int full_size() const { return full_size_; }
		int reduced_size() const { return reduced_size_; }

		void use_full_size() { current_size_ = CurrentSize::FULL_SIZE; }
		void use_reduced_size() { current_size_ = CurrentSize::REDUCED_SIZE; }

		// full is used in Augmented Lagrangian
		TVector full_to_reduced(const TVector &full) const;
		TVector full_to_reduced_grad(const TVector &full) const;
		void full_hessian_to_reduced_hessian(const THessian &full, THessian &reduced) const;
		TVector reduced_to_full(const TVector &reduced) const;

		// complete is used in forms
		TVector complete_to_full_grad(const TVector &complete) const;
		void complete_hessian_to_full_hessian(const THessian &complete, THessian &full) const;
		TVector full_to_complete(const TVector &full) const;

		void set_apply_DBC(const TVector &x, const bool val);

		std::function<void(const polysolve::nonlinear::PostStepData &data)> post_step_call_back;

		void set_target_value(double val) { target_value = val; }

		const std::vector<std::shared_ptr<Form>> &full_forms() const { return full_forms_; }

	protected:
		const int full_size_;    ///< Size of the full problem
		const int reduced_size_; ///< Size of the reduced problem

		const Eigen::VectorXd target_;
		StiffnessMatrix P;

		enum class CurrentSize
		{
			FULL_SIZE,
			REDUCED_SIZE
		};
		CurrentSize current_size_; ///< Current size of the problem (either full or reduced size)
		int current_size() const
		{
			return current_size_ == CurrentSize::FULL_SIZE ? full_size() : reduced_size();
		}

		// forms that depend on the x_full but not x_complete
		std::vector<std::shared_ptr<Form>> full_forms_;

		double target_value = 1.;
	};
} // namespace polyfem::solver
