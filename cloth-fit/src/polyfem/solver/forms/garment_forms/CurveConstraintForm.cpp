#include "CurveConstraintForm.hpp"

#include <polyfem/utils/Logger.hpp>
#include <polyfem/autogen/auto_derivatives.hpp>
#include <polyfem/utils/autodiff.h>
#include <polyfem/utils/Timer.hpp>

#include <igl/boundary_facets.h>
#include <igl/adjacency_matrix.h>
#include <igl/connected_components.h>
#include <igl/edges_to_path.h>

#include <unordered_map>

DECLARE_DIFFSCALAR_BASE();

using namespace polyfem::autogen;

namespace polyfem::solver
{
	namespace
	{
		Eigen::Vector2d point_edge_closest_distance(
			const Eigen::Vector3d &p,
			const Eigen::Vector3d &a,
			const Eigen::Vector3d &b)
		{
			const Eigen::Vector3d e = b - a;
			const Eigen::Vector3d d = p - a;
			double t = e.dot(d) / e.squaredNorm();
			t = std::min(1., std::max(0., t));
			const double dist = (d - t * e).squaredNorm();
			return Eigen::Vector2d(dist, t);
		}

		template <class T>
		T symmetry_penalty(const Eigen::Matrix<T, -1, -1> &V, const Eigen::VectorXi &correspondence, const int dim)
		{
			const Eigen::Vector<T, 3> center = V.colwise().sum() / (T)V.rows();

			Eigen::Matrix<T, -1, -1> tmp = V(correspondence, Eigen::all) - V;
			tmp.col(dim) = (V(correspondence, dim) + V.col(dim)).array() - 2 * center(dim);

			return tmp.squaredNorm() / 2.;
		}

		template <class T>
		Eigen::Vector<T, 2> torsion_penalty(
			const Eigen::Vector<T, 3> &a,
			const Eigen::Vector<T, 3> &b,
			const Eigen::Vector<T, 3> &c,
			const Eigen::Vector<T, 3> &d)
		{
			const Eigen::Vector<T, 3> mid = (c - b).normalized();
			const Eigen::Vector<T, 3> v1 = (b - a).normalized();
			const Eigen::Vector<T, 3> v2 = (d - c).normalized();

			const Eigen::Vector<T, 3> w1 = v1 - v1.dot(mid) * mid;
			const Eigen::Vector<T, 3> w2 = v2 - v2.dot(mid) * mid;

			Eigen::Vector<T, 2> out;
			out << w1.dot(w2), w1.cross(w2).dot(mid);
			return out;
		}

        template <class T> 
        T mollifier(const T &z)
        {
            if (z < 1)
                return z * (2 - z);
            else
                return T(1.);
        }

	} // namespace
	std::vector<Eigen::VectorXi> boundary_curves(const Eigen::MatrixXi &F)
	{
		Eigen::MatrixXi edges;
		igl::boundary_facets(F, edges);

		if (edges.size() == 0)
			return {};

		Eigen::SparseMatrix<int> A;
		igl::adjacency_matrix(edges, A);

		Eigen::VectorXi C, K1;
		const int n_connected = igl::connected_components(A, C, K1);

		// logger().debug("[{}] Number of curve loops: {}", name(), n_connected);

		std::vector<Eigen::VectorXi> curves;
		for (int i = 0; i < n_connected; i++)
		{
			if (K1(i) < 2)
				continue;

			std::vector<int> ind;
			for (int j = 0; j < edges.rows(); j++)
			{
				if (C(edges(j, 0)) == i)
					ind.push_back(j);
			}

			const Eigen::MatrixXi edges_tmp = edges(ind, Eigen::all);

			Eigen::VectorXi I2, J2, K2;
			igl::edges_to_path(edges_tmp, I2, J2, K2);
			assert(I2(0) == I2(I2.size() - 1));
			assert(I2.size() >= 4);

			// Eigen::VectorXi I_prime(I2.size() + 1);
			// I_prime << I2, I2(1);

			curves.push_back(std::move(I2));
		}
		return curves;
	}

	Eigen::MatrixXd extract_curve_center_targets(
		const Eigen::MatrixXd &garment_v,
		const std::vector<Eigen::VectorXi> &curves,
		const Eigen::MatrixXd &skeleton_v,
		const Eigen::MatrixXi &skeleton_bones,
		const Eigen::MatrixXd &target_skeleton_v)
	{
		Eigen::MatrixXd targets(curves.size(), 3);
		for (int j = 0; j < curves.size(); j++)
		{
			// Compute centers of curves
			Eigen::Vector3d center = garment_v(curves[j], Eigen::all).colwise().sum() / curves[j].size();

			// Project centers to original skeleton bones
			int id = 0;
			double closest_dist = std::numeric_limits<double>::max(), closest_uv = 0;
			for (int i = 0; i < skeleton_bones.rows(); i++)
			{
				Eigen::Vector2d tmp = point_edge_closest_distance(center, skeleton_v.row(skeleton_bones(i, 0)), skeleton_v.row(skeleton_bones(i, 1)));
				if (tmp(0) < closest_dist)
				{
					closest_dist = tmp(0);
					closest_uv = tmp(1);
					id = i;
				}
			}

			// Map positions to new skeleton bones
			targets.row(j) = closest_uv * (target_skeleton_v.row(skeleton_bones(id, 1)) - target_skeleton_v.row(skeleton_bones(id, 0))) + target_skeleton_v.row(skeleton_bones(id, 0));
		}

		return targets;
	}

	CurveCurvatureForm::CurveCurvatureForm(const Eigen::MatrixXd &V, const std::vector<Eigen::VectorXi> &curves) : V_(V)
	{
		for (const auto &c : curves)
		{
			assert(c(0) == c(c.size() - 1));

			Eigen::VectorXi c_(c.size() + 1);
			c_.head(c.size()) = c;
			c_(c.size()) = c(1);

			assert(c_(0) == c_(c_.size() - 2));
			assert(c_(1) == c_(c_.size() - 1));

			curves_.push_back(c_);
		}
		orig_angles = compute_angles(V);
	}

	CurveCurvatureForm::CurveCurvatureForm(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) : CurveCurvatureForm(V, boundary_curves(F))
	{
	}

	std::vector<Eigen::MatrixXd> CurveCurvatureForm::compute_angles(const Eigen::MatrixXd &V) const
	{
		std::vector<Eigen::MatrixXd> out;
		for (const auto &curve : curves_)
		{
			Eigen::MatrixXd result(curve.size() - 2, 1);
			for (int i = 0; i < curve.size() - 2; i++)
			{
				const Eigen::Ref<const Eigen::Vector3d> a = V.row(curve(i));
				const Eigen::Ref<const Eigen::Vector3d> b = V.row(curve(i + 1));
				const Eigen::Ref<const Eigen::Vector3d> c = V.row(curve(i + 2));
				const Eigen::Vector3d v1 = c - b;
				const Eigen::Vector3d v2 = a - b;

				result(i, 0) = v1.dot(v2) / (v1.norm() * v2.norm());
			}
			out.push_back(std::move(result));
		}

		return out;
	}

	double CurveCurvatureForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::MatrixXd V = utils::unflatten(x, 3) + V_;
		auto angles = compute_angles(V);

		double val = 0;
		for (int i = 0; i < angles.size(); i++)
			val += (angles[i] - orig_angles[i]).squaredNorm() / 2.;

		return val;
	}

	void CurveCurvatureForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		const Eigen::MatrixXd V = utils::unflatten(x, 3) + V_;

		gradv.setZero(x.size());
		for (int c = 0; c < curves_.size(); c++)
		{
			const auto &curve = curves_[c];
			for (int i = 0; i < curve.size() - 2; i++)
			{
				const Eigen::Ref<const Eigen::Vector3d> p1 = V.row(curve(i));
				const Eigen::Ref<const Eigen::Vector3d> p2 = V.row(curve(i + 1));
				const Eigen::Ref<const Eigen::Vector3d> p3 = V.row(curve(i + 2));
				const Eigen::Vector3d v1 = p3 - p2;
				const Eigen::Vector3d v2 = p1 - p2;
				const double len = v1.norm() * v2.norm();
				const double errA = v1.dot(v2) / len - orig_angles[c](i, 0);

				Eigen::Vector<double, 9> g;
				curve_dot_product_norm_gradient(
					p1(0), p1(1), p1(2),
					p2(0), p2(1), p2(2),
					p3(0), p3(1), p3(2), g.data());

				for (int k = 0; k < 3; k++)
					gradv.segment(curve(i + k) * 3, 3) += g.segment<3>(k * 3) * errA;
			}
		}
	}

	void CurveCurvatureForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		const Eigen::MatrixXd V = utils::unflatten(x, 3) + V_;

		std::vector<Eigen::Triplet<double>> triplets;
		for (int c = 0; c < curves_.size(); c++)
		{
			const auto &curve = curves_[c];
			for (int i = 0; i < curve.size() - 2; i++)
			{
				const Eigen::Ref<const Eigen::Vector3d> p1 = V.row(curve(i));
				const Eigen::Ref<const Eigen::Vector3d> p2 = V.row(curve(i + 1));
				const Eigen::Ref<const Eigen::Vector3d> p3 = V.row(curve(i + 2));
				const Eigen::Vector3d v1 = p3 - p2;
				const Eigen::Vector3d v2 = p1 - p2;
				const double len = v1.norm() * v2.norm();
				const double errA = v1.dot(v2) / len - orig_angles[c](i, 0);

				Eigen::Vector<double, 9> g;
				Eigen::Matrix<double, 9, 9> h, local_hess;

				curve_dot_product_norm_gradient(
					p1(0), p1(1), p1(2),
					p2(0), p2(1), p2(2),
					p3(0), p3(1), p3(2), g.data());

				curve_dot_product_norm_hessian(
					p1(0), p1(1), p1(2),
					p2(0), p2(1), p2(2),
					p3(0), p3(1), p3(2), h.data());

				local_hess = h * errA + g * g.transpose();

				if (is_project_to_psd())
					local_hess = ipc::project_to_psd(local_hess);

				for (int lj = 0; lj < 3; lj++)
					for (int dj = 0; dj < 3; dj++)
						for (int li = 0; li < 3; li++)
							for (int di = 0; di < 3; di++)
								triplets.emplace_back(curve(i + li) * 3 + di, curve(i + lj) * 3 + dj, local_hess(li * 3 + di, lj * 3 + dj));
			}
		}

		hessian.setZero();
		hessian.resize(x.size(), x.size());
		hessian.setFromTriplets(triplets.begin(), triplets.end());
	}

	CurveSizeForm::CurveSizeForm(const Eigen::MatrixXd &V, const std::vector<Eigen::VectorXi> &curves) : V_(V)
	{
		for (const auto &c : curves)
		{
			assert(c(0) == c(c.size() - 1));
			curves_.push_back(c);
		}
	}

	double CurveSizeForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::MatrixXd V = utils::unflatten(x, 3) + V_;

		double val = 0;
		for (const auto &c : curves_)
		{
			for (int i = 1; i < c.size(); i++)
				val += (V.row(c(i)) - V.row(c(i - 1))).norm();
		}

		return val;
	}

	void CurveSizeForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		const Eigen::MatrixXd V = utils::unflatten(x, 3) + V_;

		gradv.setZero(x.size());
		for (const auto &c : curves_)
		{
			for (int i = 1; i < c.size(); i++)
			{
				const Eigen::Vector3d vec = (V.row(c(i)) - V.row(c(i - 1))).normalized();
				gradv.template segment<3>(c(i) * 3) += vec;
				gradv.template segment<3>(c(i - 1) * 3) -= vec;
			}
		}
	}

	void CurveSizeForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		const Eigen::MatrixXd V = utils::unflatten(x, 3) + V_;

		std::vector<Eigen::Triplet<double>> triplets;
		for (const auto &c : curves_)
		{
			for (int i = 1; i < c.size(); i++)
			{
				const Eigen::Vector3d vec = V.row(c(i)) - V.row(c(i - 1));
				const double l = vec.norm();
				const Eigen::Matrix3d H = (Eigen::Matrix3d::Identity() - (vec * vec.transpose()) / (l * l)) / l;
				for (int j = 0; j < 3; j++)
				{
					for (int k = 0; k < 3; k++)
					{
						triplets.emplace_back(c(i) * 3 + j, c(i) * 3 + k, H(j, k));
						triplets.emplace_back(c(i - 1) * 3 + j, c(i) * 3 + k, -H(j, k));
						triplets.emplace_back(c(i) * 3 + j, c(i - 1) * 3 + k, -H(j, k));
						triplets.emplace_back(c(i - 1) * 3 + j, c(i - 1) * 3 + k, H(j, k));
					}
				}
			}
		}

		hessian.setZero();
		hessian.resize(x.size(), x.size());
		hessian.setFromTriplets(triplets.begin(), triplets.end());
	}

	CurveTorsionForm::CurveTorsionForm(const Eigen::MatrixXd &V, const std::vector<Eigen::VectorXi> &curves) : V_(V)
	{
		for (const auto &c : curves)
		{
			const int N = c.size() - 1;
			assert(c(0) == c(N));

			Eigen::VectorXi c_(N + 3);
			c_.head(N + 1) = c;
			c_(N + 1) = c(1);
			c_(N + 2) = c(2);

			assert(c_(0) == c_(c_.size() - 3));
			assert(c_(1) == c_(c_.size() - 2));
			assert(c_(2) == c_(c_.size() - 1));

			curves_.push_back(c_);
		}
		orig_angles = compute_angles(V);
	}

	CurveTorsionForm::CurveTorsionForm(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) : CurveTorsionForm(V, boundary_curves(F))
	{
	}

	std::vector<Eigen::MatrixXd> CurveTorsionForm::compute_angles(const Eigen::MatrixXd &V) const
	{
		std::vector<Eigen::MatrixXd> out;
		for (const auto &curve : curves_)
		{
			Eigen::MatrixXd result(curve.size() - 3, 2);
			for (int i = 0; i < curve.size() - 3; i++)
				result.row(i) = torsion_penalty<double>(V.row(curve(i)), V.row(curve(i + 1)), V.row(curve(i + 2)), V.row(curve(i + 3)));

			out.push_back(std::move(result));
		}

		return out;
	}

	double CurveTorsionForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::MatrixXd V = utils::unflatten(x, 3) + V_;
		auto angles = compute_angles(V);

		double val = 0;
		for (int c = 0; c < curves_.size(); c++)
        {
			const auto &curve = curves_[c];
			for (int i = 0; i < curve.size() - 3; i++)
			{
                Eigen::MatrixXd tmp_v = V(curve.segment<4>(i), Eigen::all);
			    val += (pow(angles[c](i, 0) - orig_angles[c](i, 0), 2) + pow(angles[c](i, 1) - orig_angles[c](i, 1), 2)) / 2.;
            }
        }

		return val;
	}

	void CurveTorsionForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		const Eigen::MatrixXd V = utils::unflatten(x, 3) + V_;

		typedef DScalar1<double, Eigen::Matrix<double, 12, 1>> ADGrad;
        DiffScalarBase::setVariableCount(12);

		gradv.setZero(x.size());
		for (int c = 0; c < curves_.size(); c++)
		{
			const auto &curve = curves_[c];
			for (int i = 0; i < curve.size() - 3; i++)
			{
                Eigen::VectorXd tmp_v = utils::flatten(V(curve.segment<4>(i), Eigen::all));
                Eigen::Matrix<ADGrad, 4, 3> diff_v;
                for (int j = 0; j < diff_v.rows(); j++)
                    for (int d = 0; d < diff_v.cols(); d++)
                        diff_v(j, d) = ADGrad(j * diff_v.cols() + d, tmp_v(j * diff_v.cols() + d));

                Eigen::Vector<ADGrad, 2> diff_val = torsion_penalty<ADGrad>(diff_v.row(0), diff_v.row(1), diff_v.row(2), diff_v.row(3));

				ADGrad err = (pow(diff_val(0) - orig_angles[c](i, 0), 2) + pow(diff_val(1) - orig_angles[c](i, 1), 2)) / 2.;

				for (int k = 0; k < 4; k++)
					gradv.segment<3>(curve(i + k) * 3) += err.getGradient().segment<3>(k * 3);
			}
		}
	}

	void CurveTorsionForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		POLYFEM_SCOPED_TIMER("curve torsion hessian");
		const Eigen::MatrixXd V = utils::unflatten(x, 3) + V_;

        typedef DScalar2<double, Eigen::Matrix<double, 12, 1>, Eigen::Matrix<double, 12, 12>> ADHess;
        DiffScalarBase::setVariableCount(12);

		std::vector<Eigen::Triplet<double>> triplets;
		for (int c = 0; c < curves_.size(); c++)
		{
			const auto &curve = curves_[c];
			for (int i = 0; i < curve.size() - 3; i++)
			{
                Eigen::VectorXd tmp_v = utils::flatten(V(curve.segment<4>(i), Eigen::all));
                Eigen::Matrix<ADHess, 4, 3> diff_v;
                for (int j = 0; j < diff_v.rows(); j++)
                    for (int d = 0; d < diff_v.cols(); d++)
                        diff_v(j, d) = ADHess(j * diff_v.cols() + d, tmp_v(j * diff_v.cols() + d));

                Eigen::Vector<ADHess, 2> diff_val = torsion_penalty<ADHess>(diff_v.row(0), diff_v.row(1), diff_v.row(2), diff_v.row(3));

				ADHess err = (pow(diff_val(0) - orig_angles[c](i, 0), 2) + pow(diff_val(1) - orig_angles[c](i, 1), 2)) / 2.;

				Eigen::Matrix<double, 12, 12> h = err.getHessian();
				if (is_project_to_psd())
					h = ipc::project_to_psd(h);

				for (int lj = 0; lj < 4; lj++)
					for (int dj = 0; dj < 3; dj++)
						for (int li = 0; li < 4; li++)
							for (int di = 0; di < 3; di++)
								triplets.emplace_back(curve(i + li) * 3 + di, curve(i + lj) * 3 + dj, h(li * 3 + di, lj * 3 + dj));
			}
		}

		hessian.setZero();
		hessian.resize(x.size(), x.size());
		hessian.setFromTriplets(triplets.begin(), triplets.end());
	}

	SymmetryForm::SymmetryForm(const Eigen::MatrixXd &V, const std::vector<Eigen::VectorXi> &curves) : V_(V)
	{
		hessian_cached.resize(0, 0);
		for (const auto &curve : curves)
		{
			assert(curve(0) == curve(curve.size() - 1));
			Eigen::VectorXi curve_ = curve.head(curve.size() - 1);

			const Eigen::MatrixXd P = V(curve_, Eigen::all);
			const Eigen::Vector3d bbox_min = P.colwise().minCoeff();
			const Eigen::Vector3d bbox_max = P.colwise().maxCoeff();
			const double bbox_size = (bbox_max - bbox_min).maxCoeff();

			const Eigen::Vector3d center = V(curve_, Eigen::all).colwise().sum() / curve_.size();

			std::vector<Eigen::Vector3d> coordinates;
			for (int i = 0; i < curve_.size(); i++)
			{
				Eigen::Vector3d x = P.row(i);
				x(dim) = P(i, dim) - center(dim);
				coordinates.push_back(x);
			}

			Eigen::VectorXi correspondence = Eigen::VectorXi::Zero(coordinates.size());
			double max_err = 0;
			bool has_correspondence = true;
			for (int i = 0; i < coordinates.size(); i++)
			{
				Eigen::Vector3d x = coordinates[i];
				bool found = false;
				double min_err = std::numeric_limits<double>::max();
				int min_id = -1;
				for (int j = 0; j < coordinates.size(); j++)
				{
					Eigen::Vector3d y = coordinates[j];
					double err = abs(x(dim) + y(dim));
					for (int d = 0; d < 3; d++)
						if (d != dim)
							err += abs(x(d) - y(d));
					if (err < tol * bbox_size)
					{
						max_err = std::max(max_err, err);
						found = true;
						correspondence(i) = j;
						break;
					}
					if (err < min_err)
					{
						min_err = err;
						min_id = j;
					}
				}
				if (!found)
				{
					logger().debug("Asymmetric vertex (ID {}, pos {}) on the curve with error {} (ID {}, pos {}) found! Set weight to zero!",
								  curve_(i), x.transpose(), min_err / bbox_size, curve_(min_id), coordinates[min_id].transpose());

					has_correspondence = false;
					break;
				}
			}

			if (has_correspondence)
			{
				logger().debug("Symmetric curve identified! Error is {}", max_err / bbox_size);
				curves_.push_back(curve_);
				correspondences_.push_back(correspondence);
			}
		}

		if (curves_.size() == 0)
			disable();
	}

	double SymmetryForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const Eigen::MatrixXd V = utils::unflatten(x, 3) + V_;
		double out = 0.;
		for (int i = 0; i < curves_.size(); i++)
		{
			const auto &curve = curves_[i];
			const auto &correspondence = correspondences_[i];

			out += symmetry_penalty<double>(V(curve, Eigen::all), correspondence, dim);
		}
		return out;
	}

	void SymmetryForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		const Eigen::MatrixXd V = utils::unflatten(x, 3) + V_;

		// typedef DScalar1<double, Eigen::Matrix<double, Eigen::Dynamic, 1>> ADGrad;

		Eigen::MatrixXd g = Eigen::MatrixXd::Zero(V.rows(), V.cols());
		for (int i = 0; i < curves_.size(); i++)
		{
			const auto &curve = curves_[i];
			const auto &correspondence = correspondences_[i];

			const Eigen::Vector3d center = V(curve, Eigen::all).colwise().sum() / curve.size();

			Eigen::MatrixXd tmp = V(curve(correspondence), Eigen::all) - V(curve, Eigen::all);
			tmp.col(dim) = (V(curve(correspondence), dim) + V(curve, dim)).array() - 2 * center(dim);

			for (int d = 0; d < 3; d++)
			{
				if (d != dim)
				{
					g(curve(correspondence), d) += tmp.col(d);
					g(curve, d) -= tmp.col(d);
				}
			}

			g(curve(correspondence), dim) += tmp.col(dim);
			g(curve, dim) += tmp.col(dim);

			const double deriv_wrt_c = -2 * tmp.col(dim).sum();
			g(curve, dim).array() += deriv_wrt_c / curve.size();

			// Eigen::VectorXd tmp_v = utils::flatten(V(curve, Eigen::all));
			// DiffScalarBase::setVariableCount(tmp_v.size());
			// Eigen::Matrix<ADGrad, -1, -1> diff_v(curve.size(), 3);
			// for (int j = 0; j < diff_v.rows(); j++)
			//     for (int d = 0; d < diff_v.cols(); d++)
			//         diff_v(j, d) = ADGrad(j * diff_v.cols() + d, tmp_v(j * diff_v.cols() + d));

			// ADGrad diff_val = symmetry_penalty<ADGrad>(diff_v, correspondence, dim);
			// Eigen::VectorXd tmp_g = diff_val.getGradient();
			// for (int j = 0; j < curve.size(); j++)
			//     for (int d = 0; d < 3; d++)
			//         g(curve(j), d) += tmp_g(j * 3 + d);
		}

		gradv = utils::flatten(g);
	}

	void SymmetryForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		POLYFEM_SCOPED_TIMER("symmetry hessian");
		const Eigen::MatrixXd V = utils::unflatten(x, 3) + V_;

		if (hessian_cached.rows() == V.size())
		{
			hessian = hessian_cached;
			return;
		}

		hessian_cached.setZero();
		hessian_cached.resize(x.size(), x.size());

		typedef DScalar2<double, Eigen::Matrix<double, Eigen::Dynamic, 1>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> ADHess;

		std::vector<Eigen::Triplet<double>> T;
		for (int i = 0; i < curves_.size(); i++)
		{
			const auto &curve = curves_[i];
			const auto &correspondence = correspondences_[i];

			const int N = curve.size();
			// Eigen::MatrixXd H(N, N);
			// for (int d = 0; d < 3; d++)
			// {
			//     H.setZero();
			//     H.diagonal().array() += 2;

			//     for (int k = 0; k < N; k++)
			//         H(k, correspondence(k)) += 2 * (d == dim ? 1 : -1);

			//     if (d == dim)
			//         H.array() += (-4. / N);

			//     for (int k = 0; k < N; k++)
			//         for (int j = 0; j < N; j++)
			//             if (H(k, j) != 0)
			//                 T.emplace_back(curve(k) * 3 + d, curve(j) * 3 + d, H(k, j));
			// }

			Eigen::VectorXd tmp_v = utils::flatten(V(curve, Eigen::all));
			DiffScalarBase::setVariableCount(tmp_v.size());
			Eigen::Matrix<ADHess, -1, -1> diff_v(curve.size(), 3);
			for (int j = 0; j < diff_v.rows(); j++)
				for (int d = 0; d < diff_v.cols(); d++)
					diff_v(j, d) = ADHess(j * diff_v.cols() + d, tmp_v(j * diff_v.cols() + d));

			ADHess diff_val = symmetry_penalty<ADHess>(diff_v, correspondence, dim);
			Eigen::MatrixXd tmp_h = diff_val.getHessian();
			for (int j0 = 0; j0 < N; j0++)
				for (int j1 = 0; j1 < N; j1++)
					for (int d = 0; d < 3; d++)
						T.emplace_back(curve(j0) * 3 + d, curve(j1) * 3 + d, tmp_h(j0 * 3 + d, j1 * 3 + d));
		}

		hessian_cached.setFromTriplets(T.begin(), T.end());
		hessian = hessian_cached;
	}
} // namespace polyfem::solver
