#pragma once

#include <polyfem/solver/forms/Form.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <cstddef>
#include <vector>

namespace polyfem::solver
{
	class AreaForm : public Form
	{
	public:
		AreaForm(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const double threshold = 1e-5) : V_(V), F_(F), threshold_(threshold) {}
		virtual ~AreaForm() = default;

		std::string name() const override { return "area"; }

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
		const Eigen::MatrixXi F_;
		const double threshold_;
	};

	class DefGradForm : public Form
	{
	public:
		DefGradForm(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) : V_(V), F_(F) {}
		virtual ~DefGradForm() = default;

		std::string name() const override { return "deformation"; }

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
		const Eigen::MatrixXi F_;
	};

	class AngleForm : public Form
	{
	public:
		AngleForm(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
		virtual ~AngleForm() = default;

		std::string name() const override { return "angle"; }

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
		const Eigen::MatrixXi F_;
		Eigen::MatrixXi TT, TTi;
		Eigen::VectorXd areas;
		Eigen::MatrixXd orig_angles;
	};


	class RelativeScalingForm : public Form
	{
	public:
		RelativeScalingForm(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
		virtual ~RelativeScalingForm() = default;

		std::string name() const override { return "relative-scaling"; }

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
		const Eigen::MatrixXi F_;
		Eigen::MatrixXi TT, TTi;
		Eigen::VectorXd orig_areas;
		Eigen::MatrixXd orig_dists;
	};


	class SimilarityForm : public Form
	{
	public:
		SimilarityForm(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
		virtual ~SimilarityForm() = default;

		std::string name() const override { return "similarity"; }

		// Optional per-vertex multipliers (defaults to 1); used to modulate local rigidity
		void set_vertex_multipliers(const Eigen::VectorXd &m) { vertex_multipliers_ = m; }
		// Optional: process each interior adjacency once (undirected unique edges) instead of twice.
		// This reduces similarity gradient/Hessian work ~2× if manifold adjacency.
		void set_use_unique_adjacency(const bool val) { use_unique_adjacency_ = val; }
		// Optional: parallelize similarity Hessian assembly (default: true).
		// This only affects SimilarityForm; other forms still use the global threading settings.
		void set_parallel_hessian(const bool val) { parallel_hessian_ = val; }
		// Optional: use a row-major Eigen::Map of x to avoid allocating V = unflatten(x)+V_ (default: true).
		void set_use_dx_map(const bool val) { use_dx_map_ = val; }
		// Optional: reserve triplet capacity based on adjacency count (default: true).
		void set_reserve_triplets(const bool val) { reserve_triplets_ = val; }

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
		struct UniqueAdjacency
		{
			int tri = -1;   // i
			int tri_adj = -1; // TT(i,j)
			int k = -1;     // k = tri*3 + local_edge
			int v0 = -1;    // F(tri, le(j,0))
			int v1 = -1;    // F(tri, le(j,1))
			int v2 = -1;    // F(tri, lv(j))
			int v3 = -1;    // F(tri_adj, lv(TTi(tri,j)))
			double area_sum = 0.0; // orig_areas(tri) + orig_areas(tri_adj)
		};

		const Eigen::MatrixXd V_;
		const Eigen::MatrixXi F_;
		Eigen::MatrixXi TT, TTi;
		Eigen::VectorXd orig_areas;
		Eigen::MatrixXd orig_coeffs;

		Eigen::VectorXd vertex_multipliers_;
		size_t n_internal_adj_ = 0; // number of (i,j) where TT(i,j) >= 0 (counts both directions)
		size_t n_unique_adj_ = 0;   // number of unique undirected interior adjacencies
		std::vector<UniqueAdjacency> unique_adjs_;
		bool use_unique_adjacency_ = false;
		bool parallel_hessian_ = true;
		bool use_dx_map_ = true;
		bool reserve_triplets_ = true;
	};

	class NormalForm : public Form
	{
	public:
		NormalForm(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
		virtual ~NormalForm() = default;

		std::string name() const override { return "normal"; }

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
		const Eigen::MatrixXi F_;
		
		Eigen::VectorXd orig_areas;
	};

	// class GlobalPositionalForm : public Form
	// {
	// public:
	// 	GlobalPositionalForm(
	// 		const Eigen::MatrixXd &V, 
	// 		const Eigen::MatrixXi &F,
	// 		const Eigen::MatrixXd &source_skeleton_v,
	// 		const Eigen::MatrixXd &target_skeleton_v,
	// 		const Eigen::MatrixXi &skeleton_edges,
    //     	const Eigen::MatrixXd &skin_weights);
	// 	virtual ~GlobalPositionalForm() = default;

	// 	std::string name() const override { return "global-relative-position"; }

	// protected:
	// 	/// @brief Compute the potential value
	// 	/// @param x Current solution
	// 	/// @return Value of the contact barrier potential
	// 	double value_unweighted(const Eigen::VectorXd &x) const override;

	// 	/// @brief Compute the first derivative of the value wrt x
	// 	/// @param[in] x Current solution
	// 	/// @param[out] gradv Output gradient of the value wrt x
	// 	void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

	// 	/// @brief Compute the second derivative of the value wrt x
	// 	/// @param x Current solution
	// 	/// @param hessian Output Hessian of the value wrt x
	// 	void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	// private:
	// 	const Eigen::MatrixXd V_;

	// 	const Eigen::MatrixXd source_skeleton_v_;
	// 	const Eigen::MatrixXd target_skeleton_v_;
	// 	const Eigen::MatrixXi skeleton_edges_;
	// 	const Eigen::MatrixXd skin_weights_;

	// 	Eigen::VectorXi bones;
	// 	Eigen::VectorXd relative_positions;
	// };
} // namespace polyfem::solver
