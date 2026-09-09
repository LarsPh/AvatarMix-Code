#include "CurveCenterTargetForm.hpp"

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/autogen/auto_derivatives.hpp>
#include <polyfem/utils/autodiff.h>
#include <polyfem/utils/Timer.hpp>

#include <polyfem/mesh/MeshUtils.hpp>
#include <igl/write_triangle_mesh.h>
#include <optional>

namespace polyfem::solver
{
    namespace {
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
        Eigen::Vector<T, 2> point_line_closest_distance(
            const Eigen::Vector<T, 3> &p,
            const Eigen::Vector<T, 3> &a,
            const Eigen::Vector<T, 3> &b)
        {
            const Eigen::Vector<T, 3> e = b - a;
            const Eigen::Vector<T, 3> d = p - a;
            T t = e.dot(d) / e.squaredNorm();
            const T dist = (d - t * e).squaredNorm();
            return Eigen::Vector<T, 2>(dist, t);
        }

        Eigen::Vector3d fit_plane(const Eigen::MatrixXd &points)
        {
            const int N = points.rows();
            Eigen::Vector3d center = points.colwise().sum() / N;

            Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
            for (int i = 0; i < N; i++)
            {
                Eigen::Vector3d v = points.row(i).transpose() - center;
                A += v * v.transpose();
            }
            A /= N;

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(A);
            Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();
            if (eigenvalues(0) > eigenvalues(1) * 0.1)
            {
                logger().warn("The curve is not well approximated by a plane, error is {}", eigenvalues(0) / eigenvalues(1));
            }

            return eigensolver.eigenvectors().col(0);
        }

        // Nvidia actively chooses to use the MIT license for this section of code.
        // SPDX-License-Identifier: MIT
        // #########################################################

		template <class T>
		Eigen::Matrix<T, -1, 3> project_points_to_plane(
			const Eigen::Matrix<T, -1, 3> &points,
			const Eigen::Vector<T, 3> &plane_normal,
			const Eigen::Vector<T, 3> &plane_point)
		{
			return points - (points * plane_normal).asDiagonal() * plane_normal.transpose().replicate(points.rows(), 1);
		}

		template <class T>
		Eigen::VectorXi graham_scan(const Eigen::Matrix<T, -1, 2> &points)
		{
			const int n = points.rows();
			if (n <= 2)
			{
				return Eigen::VectorXi::LinSpaced(n, 0, n - 1);
			}

			// Find point with lowest y-coordinate (and leftmost if tied)
			int lowest = 0;
			for (int i = 1; i < n; i++)
			{
				if (points(i, 1) < points(lowest, 1) || (points(i, 1) == points(lowest, 1) && points(i, 0) < points(lowest, 0)))
				{
					lowest = i;
				}
			}

			// Sort points by polar angle with lowest point
			std::vector<int> indices(n);
			std::iota(indices.begin(), indices.end(), 0);
			std::swap(indices[0], indices[lowest]);

			std::sort(indices.begin() + 1, indices.end(),
					  [&](int i, int j) {
						  const T angle_i = std::atan2(points(i, 1) - points(lowest, 1),
													   points(i, 0) - points(lowest, 0));
						  const T angle_j = std::atan2(points(j, 1) - points(lowest, 1),
													   points(j, 0) - points(lowest, 0));
						  if (std::abs(angle_i - angle_j) < 1e-10)
						  {
							  // If angles are equal, sort by distance
							  const T dist_i = (points.row(i) - points.row(lowest)).squaredNorm();
							  const T dist_j = (points.row(j) - points.row(lowest)).squaredNorm();
							  return dist_i < dist_j;
						  }
						  return angle_i < angle_j;
					  });

			// Remove collinear points
			std::vector<int> unique_indices;
			unique_indices.push_back(indices[0]);
			for (int i = 1; i < n; i++)
			{
				while (unique_indices.size() >= 2)
				{
					const int p1 = unique_indices[unique_indices.size() - 2];
					const int p2 = unique_indices[unique_indices.size() - 1];
					const int p3 = indices[i];

					const T cross = (points(p2, 0) - points(p1, 0)) * (points(p3, 1) - points(p1, 1)) - (points(p2, 1) - points(p1, 1)) * (points(p3, 0) - points(p1, 0));

					if (cross > 1e-10)
					{
						break;
					}
					unique_indices.pop_back();
				}
				unique_indices.push_back(indices[i]);
			}

			// Convert to Eigen vector
			Eigen::VectorXi hull(unique_indices.size());
			for (int i = 0; i < unique_indices.size(); i++)
			{
				hull(i) = unique_indices[i];
			}

			return hull;
		}

		template <class T>
		Eigen::Matrix<T, -1, 2> compute_curve_convex_hull_2d(
			const Eigen::Matrix<T, -1, 3> &points)
		{
			// Fit plane to points
			const Eigen::Vector3d plane_normal = fit_plane(points);

			// Project points onto plane
			const Eigen::Vector3d plane_point = points.colwise().mean();
			const Eigen::Matrix<T, -1, 3> projected_points = project_points_to_plane(points, plane_normal, plane_point);

			// Find orthonormal basis for plane
			Eigen::Vector3d u = plane_normal.cross(Eigen::Vector3d::UnitX());
			if (u.norm() < 1e-10)
			{
				u = plane_normal.cross(Eigen::Vector3d::UnitY());
			}
			u.normalize();
			const Eigen::Vector3d v = plane_normal.cross(u);

			// Project points onto 2D basis
			Eigen::Matrix<T, -1, 2> points_2d(points.rows(), 2);
			for (int i = 0; i < points.rows(); i++)
			{
				points_2d(i, 0) = projected_points.row(i).dot(u);
				points_2d(i, 1) = projected_points.row(i).dot(v);
			}

			// Compute convex hull in 2D
			Eigen::VectorXi hull;
			hull = graham_scan(points_2d);

			// Return hull vertices in 2D
			return points_2d(hull, Eigen::all);
		}

		template <class T>
		bool is_point_in_convex_hull_2d(
			const Eigen::Vector<T, 3> &point,
			const Eigen::Vector<T, 3> &plane_normal,
			const Eigen::Vector<T, 3> &plane_point,
			const Eigen::Matrix<T, -1, 2> &hull_2d)
		{
			// Project point onto plane
			const Eigen::Vector<T, 3> projected_point = point - (point - plane_point).dot(plane_normal) * plane_normal;

			// Find orthonormal basis for plane
			Eigen::Vector3d u = plane_normal.cross(Eigen::Vector3d::UnitX());
			if (u.norm() < 1e-10)
			{
				u = plane_normal.cross(Eigen::Vector3d::UnitY());
			}
			u.normalize();
			const Eigen::Vector3d v = plane_normal.cross(u);

			// Project point onto 2D basis
			const T point_2d_x = projected_point.dot(u);
			const T point_2d_y = projected_point.dot(v);

			// Check if point is inside convex hull using winding number algorithm
			int winding_number = 0;
			for (int i = 0; i < hull_2d.rows(); i++)
			{
				const int j = (i + 1) % hull_2d.rows();
				const T x1 = hull_2d(i, 0) - point_2d_x;
				const T y1 = hull_2d(i, 1) - point_2d_y;
				const T x2 = hull_2d(j, 0) - point_2d_x;
				const T y2 = hull_2d(j, 1) - point_2d_y;

				if (y1 <= 0 && y2 > 0)
				{
					if (x1 * y2 - x2 * y1 > 0)
					{
						winding_number++;
					}
				}
				else if (y1 > 0 && y2 <= 0)
				{
					if (x1 * y2 - x2 * y1 < 0)
					{
						winding_number--;
					}
				}
			}

			return winding_number != 0;
		}

		template <class T>
		std::optional<Eigen::Vector<T, 3>> bone_plane_intersection(
			const Eigen::Vector<T, 3> &bone_start,
			const Eigen::Vector<T, 3> &bone_end,
			const Eigen::Vector<T, 3> &plane_normal,
			const Eigen::Vector<T, 3> &plane_point)
		{
			// Compute bone direction vector
			const Eigen::Vector<T, 3> bone_dir = bone_end - bone_start;

			// Compute denominator for intersection calculation
			const T denom = bone_dir.dot(plane_normal);

			// Check if bone is parallel to plane
			if (std::abs(denom) < 1e-10)
			{
				return std::nullopt;
			}

			// Compute intersection parameter
			const T t = (plane_point - bone_start).dot(plane_normal) / denom;

			// Check if intersection is within bone segment
			if (t < 0 || t > 1)
			{
				return std::nullopt;
			}

			// Return intersection point
			return bone_start + t * bone_dir;
		}

		template <class T>
		std::vector<Eigen::Vector3<T>> count_bone_intersections(
			const Eigen::Matrix<T, Eigen::Dynamic, 3> &curve_points,
			const Eigen::Matrix<T, Eigen::Dynamic, 3> &skeleton_v,
			const Eigen::MatrixXi &skeleton_edges)
		{
			// Fit plane to curve points
			const Eigen::Vector3<T> plane_normal = fit_plane(curve_points).normalized();
			const Eigen::Vector3<T> plane_point = curve_points.colwise().mean();

			const Eigen::Matrix<T, Eigen::Dynamic, 2> hull_2d = compute_curve_convex_hull_2d<T>(curve_points);

			// Store intersecting bone indices
			std::vector<Eigen::Vector3<T>> intersection_points;
			for (int i = 0; i < skeleton_edges.rows(); i++)
			{
				const Eigen::Vector3<T> bone_start = skeleton_v.row(skeleton_edges(i, 0));
				const Eigen::Vector3<T> bone_end = skeleton_v.row(skeleton_edges(i, 1));

				// Find intersection point with plane
				auto intersection = bone_plane_intersection(bone_start, bone_end, plane_normal, plane_point);
				if (!intersection)
				{
					continue;
				}
				// Check if intersection point is inside convex hull
				if (is_point_in_convex_hull_2d<T>(*intersection, plane_normal, plane_point, hull_2d))
				{
					intersection_points.push_back(*intersection);
				}
			}

			return intersection_points;
		}

		void add_bone_to_both_skeletons_with_root_as_parent(
			const Eigen::Vector3d &start_point,
			const Eigen::Vector3d &end_point,
			Eigen::MatrixXd &source_skeleton_v,
			Eigen::MatrixXd &target_skeleton_v,
			Eigen::MatrixXi &skeleton_edges)
		{
			// Add new vertices to skeleton
			const int start_idx = source_skeleton_v.rows();
			const int end_idx = start_idx + 1;

			source_skeleton_v.conservativeResize(end_idx + 1, 3);
			source_skeleton_v.row(start_idx) = start_point;
			source_skeleton_v.row(end_idx) = end_point;

			target_skeleton_v.conservativeResize(end_idx + 1, 3);
			target_skeleton_v.row(start_idx) = start_point;
			target_skeleton_v.row(end_idx) = end_point;

			// Add new edge to skeleton
			const int new_edge_idx = skeleton_edges.rows();
			skeleton_edges.conservativeResize(new_edge_idx + 2, 2);
			skeleton_edges.row(new_edge_idx) << 0, start_idx; // Root vertex index is 0
			skeleton_edges.row(new_edge_idx + 1) << start_idx, end_idx;
		}
        // #########################################################
	}

    CurveCenterTargetForm::CurveCenterTargetForm(
        const Eigen::MatrixXd &V, 
        const std::vector<Eigen::VectorXi> &curves,
        const Eigen::MatrixXd &source_skeleton_v,
        const Eigen::MatrixXd &target_skeleton_v,
        const Eigen::MatrixXi &skeleton_edges): 
        V_(V), source_skeleton_v_(source_skeleton_v),
        target_skeleton_v_(target_skeleton_v), skeleton_edges_(skeleton_edges)
    {
        for (auto curve : curves)
        {
            curves_.push_back(curve.head(curve.size()-1));
        }
        bones.resize(curves_.size());
        relative_positions.resize(curves_.size());
        for (int j = 0; j < curves_.size(); j++)
        {
            const Eigen::Vector3d center = V(curves_[j], Eigen::all).colwise().sum() / curves_[j].size();

            // Project centers to original skeleton bones
            int id = 0;
            double closest_dist = std::numeric_limits<double>::max(), closest_uv = 0;
            for (int i = 0; i < skeleton_edges.rows(); i++)
            {
                Eigen::Vector2d tmp = point_edge_closest_distance(center, source_skeleton_v.row(skeleton_edges(i, 0)), source_skeleton_v.row(skeleton_edges(i, 1)));
                if (tmp(0) < closest_dist)
                {
                    closest_dist = tmp(0);
                    closest_uv = tmp(1);
                    id = i;
                }
            }

            bones(j) = id;
            relative_positions(j) = closest_uv;
        }
    }

    double CurveCenterTargetForm::value_unweighted(const Eigen::VectorXd &x) const
    {
        const double t = x(0);
        const Eigen::MatrixXd V = V_ + utils::unflatten(x.tail(x.size() - 1), V_.cols());

        double val = 0.;
        for (int j = 0; j < curves_.size(); j++)
        {
            const Eigen::Vector3d center = V(curves_[j], Eigen::all).colwise().sum() / curves_[j].size();

            const int id = bones(j);
            const double param0 = relative_positions(j);
            const Eigen::Vector3d targetS = param0 * (source_skeleton_v_.row(skeleton_edges_(id, 1)) - source_skeleton_v_.row(skeleton_edges_(id, 0))) + source_skeleton_v_.row(skeleton_edges_(id, 0));
            const Eigen::Vector3d targetT = param0 * (target_skeleton_v_.row(skeleton_edges_(id, 1)) - target_skeleton_v_.row(skeleton_edges_(id, 0))) + target_skeleton_v_.row(skeleton_edges_(id, 0));
            const Eigen::Vector3d target = t * (targetT - targetS) + targetS;

            val += (target - center).squaredNorm();
        }

        return val / 2.;
    }

    void CurveCenterTargetForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
    {
        const double t = x(0);
        const Eigen::MatrixXd V = V_ + utils::unflatten(x.tail(x.size() - 1), V_.cols());

        gradv.setZero(x.size());
        for (int j = 0; j < curves_.size(); j++)
        {
            const auto &curve = curves_[j];
            const Eigen::Vector3d center = V(curve, Eigen::all).colwise().sum() / curve.size();

            const int id = bones(j);
            const double param0 = relative_positions(j);
            
            const Eigen::Vector3d targetS = param0 * (source_skeleton_v_.row(skeleton_edges_(id, 1)) - source_skeleton_v_.row(skeleton_edges_(id, 0))) + source_skeleton_v_.row(skeleton_edges_(id, 0));
            const Eigen::Vector3d targetT = param0 * (target_skeleton_v_.row(skeleton_edges_(id, 1)) - target_skeleton_v_.row(skeleton_edges_(id, 0))) + target_skeleton_v_.row(skeleton_edges_(id, 0));
            const Eigen::Vector3d target = t * (targetT - targetS) + targetS;
            
            const Eigen::Vector3d tmp = center - target;
            for (int i = 0; i < curve.size(); i++)
                gradv.segment<3>(1 + curve(i) * 3) += tmp / curve.size();
            gradv(0) -= tmp.dot(targetT - targetS);
        }
    }

    void CurveCenterTargetForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
    {
        hessian.resize(x.size(), x.size());
        hessian.setZero();

        const double t = x(0);
        const Eigen::MatrixXd V = V_ + utils::unflatten(x.tail(x.size() - 1), V_.cols());

        std::vector<Eigen::Triplet<double>> T;
        for (int j = 0; j < curves_.size(); j++)
        {
            const auto &curve = curves_[j];
            const int N = curve.size();
            if (N <= 0)
                continue;
            const Eigen::Vector3d center = V(curve, Eigen::all).colwise().sum() / N;

            const int id = bones(j);
            const double param0 = relative_positions(j);
            
            const Eigen::Vector3d targetS = param0 * (source_skeleton_v_.row(skeleton_edges_(id, 1)) - source_skeleton_v_.row(skeleton_edges_(id, 0))) + source_skeleton_v_.row(skeleton_edges_(id, 0));
            const Eigen::Vector3d targetT = param0 * (target_skeleton_v_.row(skeleton_edges_(id, 1)) - target_skeleton_v_.row(skeleton_edges_(id, 0))) + target_skeleton_v_.row(skeleton_edges_(id, 0));
            const Eigen::Vector3d target = t * (targetT - targetS) + targetS;
            
            Eigen::Matrix4d h = Eigen::Matrix4d::Zero();
            h(1, 1) = 1; h(2, 2) = 1; h(3, 3) = 1;
            h(0, 0) = (targetT - targetS).dot(targetT - targetS);
            h.block<3, 1>(1, 0) = targetS - targetT;
            h.block<1, 3>(0, 1) = targetS - targetT;

            Eigen::MatrixXd local_hess(N * 3 + 1, N * 3 + 1);
            local_hess.setZero();
            assert(local_hess.rows() == N * 3 + 1 && local_hess.cols() == N * 3 + 1);
            local_hess(0, 0) = h(0, 0);
            for (int i0 = 0; i0 < N; i0++)
            {
                for (int d0 = 0; d0 < 3; d0++)
                {
                    local_hess(i0 * 3 + d0 + 1, 0) = h(d0 + 1, 0) / N;
                    local_hess(0, i0 * 3 + d0 + 1) = h(0, d0 + 1) / N;
                    for (int i1 = 0; i1 < N; i1++)
                    {
                        for (int d1 = 0; d1 < 3; d1++)
                        {
                            local_hess(i0 * 3 + d0 + 1, i1 * 3 + d1 + 1) = h(d0 + 1, d1 + 1) / (N * N);
                        }
                    }
                }
            }

            T.emplace_back(0, 0, local_hess(0, 0));
            for (int i0 = 0; i0 < N; i0++)
            for (int d0 = 0; d0 < 3; d0++)
            {
                T.emplace_back(1 + curve(i0) * 3 + d0, 0, local_hess(i0 * 3 + d0 + 1, 0));
                T.emplace_back(0, 1 + curve(i0) * 3 + d0, local_hess(0, i0 * 3 + d0 + 1));
                for (int i1 = 0; i1 < N; i1++)
                for (int d1 = 0; d1 < 3; d1++)
                {
                    T.emplace_back(1 + curve(i0) * 3 + d0, 1 + curve(i1) * 3 + d1, local_hess(i0 * 3 + d0 + 1, i1 * 3 + d1 + 1));
                }
            }
        }

        hessian.setFromTriplets(T.begin(), T.end());
    }


    CurveTargetForm::CurveTargetForm(
        const Eigen::MatrixXd &V, 
        const std::vector<Eigen::VectorXi> &curves,
        const Eigen::MatrixXd &source_skeleton_v,
        const Eigen::MatrixXd &target_skeleton_v,
        const Eigen::MatrixXi &skeleton_edges,
		const bool is_skirt, 
		const bool automatic_bone_generation
    ): 
        is_skirt_(is_skirt), V_(V), source_skeleton_v_(source_skeleton_v),
        target_skeleton_v_(target_skeleton_v), skeleton_edges_(skeleton_edges), automatic_bone_generation_(automatic_bone_generation)
    {
        // Insert one skeleton as the average of two legs
        if (is_skirt_)
        {
            if (source_skeleton_v_.rows() == 15)
            {
                const int nvert = source_skeleton_v_.rows();
                const int nbone = skeleton_edges_.rows();
                skeleton_edges_.conservativeResize(nbone + 2, skeleton_edges_.cols());
                source_skeleton_v_.conservativeResize(nvert + 2, source_skeleton_v_.cols());
                target_skeleton_v_.conservativeResize(nvert + 2, target_skeleton_v_.cols());
                source_skeleton_v_.row(nvert) = (source_skeleton_v_.row(10) + source_skeleton_v_.row(13)) / 2.;
                source_skeleton_v_.row(nvert+1) = (source_skeleton_v_.row(11) + source_skeleton_v_.row(14)) / 2.;
                target_skeleton_v_.row(nvert) = (target_skeleton_v_.row(10) + target_skeleton_v_.row(13)) / 2.;
                target_skeleton_v_.row(nvert+1) = (target_skeleton_v_.row(11) + target_skeleton_v_.row(14)) / 2.;
                skeleton_edges_.row(nbone) << 0, nvert;
                skeleton_edges_.row(nbone+1) << nvert, nvert + 1;
            }
            else if (source_skeleton_v_.rows() == 24)
            {
                const int nvert = source_skeleton_v_.rows();
                const int nbone = skeleton_edges_.rows();
                skeleton_edges_.conservativeResize(nbone + 2, skeleton_edges_.cols());
                source_skeleton_v_.conservativeResize(nvert + 2, source_skeleton_v_.cols());
                target_skeleton_v_.conservativeResize(nvert + 2, target_skeleton_v_.cols());
                source_skeleton_v_.row(nvert) = (source_skeleton_v_.row(2) + source_skeleton_v_.row(6)) / 2.;
                source_skeleton_v_.row(nvert+1) = (source_skeleton_v_.row(3) + source_skeleton_v_.row(7)) / 2.;
                target_skeleton_v_.row(nvert) = (target_skeleton_v_.row(2) + target_skeleton_v_.row(6)) / 2.;
                target_skeleton_v_.row(nvert+1) = (target_skeleton_v_.row(3) + target_skeleton_v_.row(7)) / 2.;
                skeleton_edges_.row(nbone) << 0, nvert;
                skeleton_edges_.row(nbone+1) << nvert, nvert + 1;
            }
            else
            {
                // Generic mid-leg insertion for other skeleton topologies.
                //
                // Our pipeline may export a simplified SMPL-X skeleton (e.g., 16 joints). The original
                // implementation only supported a couple of hard-coded topologies (15/24), but for the
                // skirt hem constraint we only need a reasonable "middle-leg" bone: from the average
                // of the two hips to the average of the two lowest leg joints.
                const int nvert = source_skeleton_v_.rows();
                const int nbone = skeleton_edges_.rows();

                if (nvert < 3 || nbone < 2)
                    log_and_throw_error("\"is_skirt\": invalid skeleton (nvert={}, nbone={})", nvert, nbone);
                if (skeleton_edges_.cols() != 2)
                    log_and_throw_error("\"is_skirt\": expected skeleton edge matrix with 2 columns, got {}", skeleton_edges_.cols());

                const int root = 0;
                if (root < 0 || root >= nvert)
                    log_and_throw_error("\"is_skirt\": invalid root index {} for nvert={}", root, nvert);

                std::vector<std::vector<int>> adj((size_t)nvert);
                for (int e = 0; e < nbone; ++e)
                {
                    const int a = skeleton_edges_(e, 0);
                    const int b = skeleton_edges_(e, 1);
                    if (a < 0 || a >= nvert || b < 0 || b >= nvert)
                        continue;
                    adj[(size_t)a].push_back(b);
                    adj[(size_t)b].push_back(a);
                }

                std::vector<int> root_children;
                root_children.reserve(adj[(size_t)root].size());
                for (const int nb : adj[(size_t)root])
                {
                    // Prefer subtrees going "down" from the root (legs), skip the upward spine branch.
                    if (source_skeleton_v_(nb, 1) <= source_skeleton_v_(root, 1))
                        root_children.push_back(nb);
                }
                if ((int)root_children.size() < 2)
                {
                    // Fallback: use any two neighbors of the root.
                    root_children = adj[(size_t)root];
                }
                if ((int)root_children.size() < 2)
                    log_and_throw_error("\"is_skirt\": could not find 2 leg candidates from root degree={}", (int)adj[(size_t)root].size());

                auto find_lowest_in_subtree = [&](const int start) -> int {
                    int best = start;
                    double best_y = source_skeleton_v_(start, 1);

                    std::vector<std::pair<int, int>> stack;
                    stack.reserve((size_t)nvert);
                    stack.emplace_back(start, root);
                    while (!stack.empty())
                    {
                        const auto [u, parent] = stack.back();
                        stack.pop_back();

                        const double y = source_skeleton_v_(u, 1);
                        if (y < best_y)
                        {
                            best_y = y;
                            best = u;
                        }

                        for (const int v : adj[(size_t)u])
                        {
                            if (v == parent)
                                continue;
                            if (v == root)
                                continue;
                            stack.emplace_back(v, u);
                        }
                    }
                    return best;
                };

                struct LegCand
                {
                    int hip = -1;
                    int lowest = -1;
                    double lowest_y = 0;
                };
                std::vector<LegCand> cands;
                cands.reserve(root_children.size());
                for (const int child : root_children)
                {
                    const int low = find_lowest_in_subtree(child);
                    cands.push_back({child, low, source_skeleton_v_(low, 1)});
                }

                // Pick the two lowest subtrees.
                std::partial_sort(
                    cands.begin(),
                    cands.begin() + 2,
                    cands.end(),
                    [](const LegCand &a, const LegCand &b) { return a.lowest_y < b.lowest_y; });

                const int hip_l = cands[0].hip;
                const int hip_r = cands[1].hip;
                const int low_l = cands[0].lowest;
                const int low_r = cands[1].lowest;

                skeleton_edges_.conservativeResize(nbone + 2, skeleton_edges_.cols());
                source_skeleton_v_.conservativeResize(nvert + 2, source_skeleton_v_.cols());
                target_skeleton_v_.conservativeResize(nvert + 2, target_skeleton_v_.cols());

                source_skeleton_v_.row(nvert) = (source_skeleton_v_.row(hip_l) + source_skeleton_v_.row(hip_r)) / 2.;
                source_skeleton_v_.row(nvert + 1) = (source_skeleton_v_.row(low_l) + source_skeleton_v_.row(low_r)) / 2.;
                target_skeleton_v_.row(nvert) = (target_skeleton_v_.row(hip_l) + target_skeleton_v_.row(hip_r)) / 2.;
                target_skeleton_v_.row(nvert + 1) = (target_skeleton_v_.row(low_l) + target_skeleton_v_.row(low_r)) / 2.;

                skeleton_edges_.row(nbone) << root, nvert;
                skeleton_edges_.row(nbone + 1) << nvert, nvert + 1;
            }
        }

        for (auto curve : curves)
        {
            curves_.push_back(curve.head(curve.size()-1));
        }
        bones.resize(curves_.size());
        relative_positions.resize(curves_.size());

        for (int j = 0; j < curves_.size(); j++)
        {

            const Eigen::Vector3d center = V(curves_[j], Eigen::all).colwise().sum() / curves_[j].size();
            const Eigen::Vector3d curve_normal = fit_plane(V(curves_[j], Eigen::all)).normalized();

            // Nvidia actively chooses to use the MIT license for this section of code.
            // SPDX-License-Identifier: MIT
            // #########################################################
            if (automatic_bone_generation_){
                auto bone_intersection_points = count_bone_intersections<double>(V(curves_[j], Eigen::all), source_skeleton_v_, skeleton_edges_);
                if (bone_intersection_points.size() > 0)
                {
                    logger().debug("curve {} has {} bone intersections", j, bone_intersection_points.size());
                } 

                if (bone_intersection_points.size() > 1)
                {
                    // If we have intersection points, create new bone as the average of the others.
                    
                    // Calculate average intersection point
                    Eigen::Vector3d avg_intersection = Eigen::Vector3d::Zero();
                    for (const auto& point : bone_intersection_points) {
                        avg_intersection += point;
                    }
                    avg_intersection /= bone_intersection_points.size();

                    // Create two vertices per bone
                    const double bone_length = 0.1; // Small fixed length
                    const Eigen::Vector3d start_point = avg_intersection - 0.5 * bone_length * curve_normal;
                    const Eigen::Vector3d end_point = avg_intersection + 0.5 * bone_length * curve_normal;

                    add_bone_to_both_skeletons_with_root_as_parent(start_point, end_point, source_skeleton_v_, target_skeleton_v_, skeleton_edges_);
                }
            }
            // #########################################################

            logger().debug("curve normal {}", curve_normal.transpose());

            // Project centers to original skeleton bones
            double closest_dist = std::numeric_limits<double>::max();
            Eigen::Vector3d sdirec;
            for (int i = 0; i < skeleton_edges_.rows(); i++)
            {
                Eigen::Vector2i edge = skeleton_edges_.row(i);
                Eigen::Vector3d e0 = source_skeleton_v_.row(edge(0));
                Eigen::Vector3d e1 = source_skeleton_v_.row(edge(1));
                Eigen::Vector2d tmp = point_edge_closest_distance(center, e0, e1);

                sdirec = (source_skeleton_v_.row(skeleton_edges_(i, 1)) - source_skeleton_v_.row(skeleton_edges_(i, 0))).normalized();

                // if (tmp(0) < closest_dist && is_bone_available(skeleton_edges_.rows(), edge))
                if (abs(sdirec.dot(curve_normal)) > 0.5 && tmp(0) < closest_dist)
                {
                    closest_dist = tmp(0);
                    bones(j) = i;
                }
            }
            
            logger().debug("CurveTargetForm set curve center to bone {} - {}, direction is {}", skeleton_edges_(bones(j), 0), skeleton_edges_(bones(j), 1), sdirec.transpose());
            
            relative_positions[j].resize(curves_[j].size());
            for (int i = 0; i < curves_[j].size(); i++)
            {
                Eigen::Vector2d tmp = point_line_closest_distance<double>(V.row(curves_[j](i)), source_skeleton_v_.row(skeleton_edges_(bones(j), 0)), source_skeleton_v_.row(skeleton_edges_(bones(j), 1)));
                relative_positions[j](i) = tmp(1);
            }
        }
    }

    double CurveTargetForm::value_unweighted(const Eigen::VectorXd &x) const
    {
        const double t = x(0);
        const Eigen::MatrixXd V = V_ + utils::unflatten(x.tail(x.size() - 1), V_.cols());

        double val = 0.;
        for (int j = 0; j < curves_.size(); j++)
        {
            const int id = bones(j);
            for (int i = 0; i < curves_[j].size(); i++)
            {
                const Eigen::Vector3d bone0 = t * (target_skeleton_v_.row(skeleton_edges_(id, 0)) - source_skeleton_v_.row(skeleton_edges_(id, 0))) + source_skeleton_v_.row(skeleton_edges_(id, 0));
                const Eigen::Vector3d bone1 = t * (target_skeleton_v_.row(skeleton_edges_(id, 1)) - source_skeleton_v_.row(skeleton_edges_(id, 1))) + source_skeleton_v_.row(skeleton_edges_(id, 1));
                Eigen::Vector2d tmp = point_line_closest_distance<double>(V.row(curves_[j](i)), bone0, bone1);

                val += pow(relative_positions[j](i) - tmp(1), 2);
            }
        }

        return val / 2.;
    }

    void CurveTargetForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
    {
        const double t = x(0);
        const Eigen::MatrixXd V = V_ + utils::unflatten(x.tail(x.size() - 1), V_.cols());

		typedef DScalar1<double, Eigen::Matrix<double, 4, 1>> ADGrad;
        DiffScalarBase::setVariableCount(4);

        gradv.setZero(x.size());
        for (int j = 0; j < curves_.size(); j++)
        {
            const auto &curve = curves_[j];

            const int id = bones(j);
            for (int i = 0; i < curve.size(); i++)
            {
                Eigen::Vector<ADGrad, 4> diff_v;
                for (int d = 0; d < 3; d++)
                    diff_v(d) = ADGrad(d, V(curve(i), d));
                diff_v(3) = ADGrad(3, t);

                const Eigen::Vector3d a0 = source_skeleton_v_.row(skeleton_edges_(id, 0));
                const Eigen::Vector3d b0 = target_skeleton_v_.row(skeleton_edges_(id, 0));
                const Eigen::Vector3d a1 = source_skeleton_v_.row(skeleton_edges_(id, 1));
                const Eigen::Vector3d b1 = target_skeleton_v_.row(skeleton_edges_(id, 1));

                Eigen::Vector<ADGrad, 3> bone0;
                bone0 << diff_v(3) * (b0(0) - a0(0)) + a0(0), diff_v(3) * (b0(1) - a0(1)) + a0(1), diff_v(3) * (b0(2) - a0(2)) + a0(2);
                Eigen::Vector<ADGrad, 3> bone1;
                bone1 << diff_v(3) * (b1(0) - a1(0)) + a1(0), diff_v(3) * (b1(1) - a1(1)) + a1(1), diff_v(3) * (b1(2) - a1(2)) + a1(2);
                Eigen::Vector<ADGrad, 2> tmp = point_line_closest_distance<ADGrad>(diff_v.head<3>(), bone0, bone1);

                ADGrad err = pow(relative_positions[j](i) - tmp(1), 2) / 2.;

                gradv.segment<3>(1 + curve(i) * 3) += err.getGradient().head<3>().transpose();
                gradv(0) += err.getGradient()(3);
            }
        }
    }

    void CurveTargetForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
    {
        POLYFEM_SCOPED_TIMER("curve target hessian");
        hessian.resize(x.size(), x.size());
        hessian.setZero();

        typedef DScalar2<double, Eigen::Matrix<double, 4, 1>, Eigen::Matrix<double, 4, 4>> ADHess;
        DiffScalarBase::setVariableCount(4);

        const double t = x(0);
        const Eigen::MatrixXd V = V_ + utils::unflatten(x.tail(x.size() - 1), V_.cols());

        std::vector<Eigen::Triplet<double>> T;
        for (int j = 0; j < curves_.size(); j++)
        {
            const auto &curve = curves_[j];

            // logger().debug("curve normal {}", fit_plane(V(curves_[j], Eigen::all)).transpose());

            const int id = bones(j);
            for (int i = 0; i < curve.size(); i++)
            {
                Eigen::Vector<ADHess, 4> diff_v;
                for (int d = 0; d < 3; d++)
                    diff_v(d) = ADHess(d, V(curve(i), d));
                diff_v(3) = ADHess(3, t);

                const Eigen::Vector3d a0 = source_skeleton_v_.row(skeleton_edges_(id, 0));
                const Eigen::Vector3d b0 = target_skeleton_v_.row(skeleton_edges_(id, 0));
                const Eigen::Vector3d a1 = source_skeleton_v_.row(skeleton_edges_(id, 1));
                const Eigen::Vector3d b1 = target_skeleton_v_.row(skeleton_edges_(id, 1));

                Eigen::Vector<ADHess, 3> bone0;
                bone0 << diff_v(3) * (b0(0) - a0(0)) + a0(0), diff_v(3) * (b0(1) - a0(1)) + a0(1), diff_v(3) * (b0(2) - a0(2)) + a0(2);
                Eigen::Vector<ADHess, 3> bone1;
                bone1 << diff_v(3) * (b1(0) - a1(0)) + a1(0), diff_v(3) * (b1(1) - a1(1)) + a1(1), diff_v(3) * (b1(2) - a1(2)) + a1(2);
                Eigen::Vector<ADHess, 2> tmp = point_line_closest_distance<ADHess>(diff_v.head<3>(), bone0, bone1);

                ADHess err = pow(relative_positions[j](i) - tmp(1), 2) / 2.;

				Eigen::Matrix<double, 4, 4> h = err.getHessian();
				if (is_project_to_psd())
					h = ipc::project_to_psd(h);

                std::array<int, 4> indices{{curve(i) * 3 + 1, curve(i) * 3 + 2, curve(i) * 3 + 3, 0}};
                for (int k = 0; k < 4; k++)
                    for (int l = 0; l < 4; l++)
                        T.emplace_back(indices[k], indices[l], h(k, l));
            }
        }

        hessian.setFromTriplets(T.begin(), T.end());
    }
}