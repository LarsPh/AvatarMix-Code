#pragma once

#include <polyfem/solver/forms/Form.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>
#include <polyfem/utils/MatrixCache.hpp>

#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

namespace polyfem::solver
{
	template <int n_refs>
	class FitForm : public Form
	{
	public:
		constexpr static int n_loc_samples = ((n_refs+1)*(n_refs+2))/2;

		FitForm(
			const Eigen::MatrixXd &V, 
			const Eigen::MatrixXi &F, 
			const Eigen::MatrixXd &surface_v, 
			const Eigen::MatrixXi &surface_f, 
			const double voxel_size,
			const std::vector<int> &not_fit_faces,
			const std::string out_dir,
			const Eigen::VectorXd &vertex_fit_multipliers);

		std::string name() const override { return "garment-fit"; }

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

		void solution_changed(const Eigen::VectorXd &new_x) override;

		// Percentile of fit gap D = max(0, phi - d0) over cached SDF samples
		double gap_percentile(const double q, const double d0 = 0.0) const;

    private:
        const Eigen::MatrixXd V_;
		const Eigen::MatrixXi F_;
        const double voxel_size_;
        const bool use_spline = true;

		Eigen::Matrix<double, n_loc_samples, 3> P;
		Eigen::Vector<double, n_loc_samples> weights;

		std::vector<openvdb::tools::HessType<double>> totalP;

        openvdb::DoubleGrid::Ptr grid;

		mutable std::unique_ptr<utils::MatrixCache> mat_cache_;

		const int power = 2;

		Eigen::VectorXd initial_distance;

		std::vector<int> fit_faces_ids;

		Eigen::VectorXd vertex_fit_multipliers_;
	};

	template <int n_refs>
	class SDFCollisionForm : public Form
	{
	public:
		constexpr static int n_loc_samples = ((n_refs+1)*(n_refs+2))/2;

		SDFCollisionForm(
			const Eigen::MatrixXd &V, 
			const Eigen::MatrixXi &F, 
			const Eigen::MatrixXd &surface_v, 
			const Eigen::MatrixXi &surface_f, 
			const double voxel_size,
			const double separation_dist);

		std::string name() const override { return "sdf-collision"; }

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

		void solution_changed(const Eigen::VectorXd &new_x) override;

    private:
        const Eigen::MatrixXd V_;
		const Eigen::MatrixXi F_;
        const double voxel_size_;
        const bool use_spline = true;

		Eigen::Matrix<double, n_loc_samples, 3> P;
		Eigen::Vector<double, n_loc_samples> weights;

		std::vector<openvdb::tools::HessType<double>> totalP;

        openvdb::DoubleGrid::Ptr grid;

		mutable std::unique_ptr<utils::MatrixCache> mat_cache_;

		const int power = 2;
		const double separation_dist_ = 0.;
	};
} // namespace polyfem::solver
