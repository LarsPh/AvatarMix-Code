#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace polyfem
{

	namespace mesh
	{
		template <int dim>
		class Transformation {
		public:
			/// @brief y = A * x + b
			/// @param A 
			/// @param b 
			Transformation(const Eigen::Matrix<double, dim, dim> &A, const Eigen::Vector<double, dim> &b) : A_(A), b_(b)
			{
			}

			void apply(Eigen::MatrixXd &V) const
			{
				V = (V * A_.transpose()).eval().rowwise() + b_.transpose();
			}

			void invert(Eigen::MatrixXd &V) const
			{
				V = (V.rowwise() - b_).eval() * A_.transpose().inv();
			}

		private:
			const Eigen::Matrix<double, dim, dim> A_;
			const Eigen::Vector<double, dim> b_;
		};

		void read_edge_mesh(const std::string &path, Eigen::MatrixXd &V, Eigen::MatrixXi &E);
		void write_edge_mesh(const std::string &path, const Eigen::MatrixXd &V, const Eigen::MatrixXi &E);

		// returns svi and svj
		std::tuple<Eigen::VectorXi, Eigen::VectorXi> remove_duplicate_vertices(Eigen::MatrixXd &V, Eigen::MatrixXi &F, const double threshold);

		// Compute absolute volume of a closed triangle mesh (errors if non-watertight).
		double closed_mesh_volume(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F);
	} // namespace mesh
} // namespace polyfem
