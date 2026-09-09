#include "SkeletonInsideSDFForm.hpp"

#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>

#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/SignedFloodFill.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <polyfem/io/OBJWriter.hpp>
#include <igl/boundary_facets.h>
#include <unordered_map>
#include <unordered_set>
#include <fstream>

namespace polyfem::solver
{
    using namespace openvdb;
    using namespace openvdb::tools;

    SkeletonInsideSDFForm::SkeletonInsideSDFForm(
        const Eigen::MatrixXd &skelV,
        const Eigen::MatrixXi &skelE,
        const Eigen::MatrixXd &surfaceV,
        const Eigen::MatrixXi &surfaceF,
        const double voxel_size,
        const int samples_per_bone,
        const double inside_margin,
        const bool flood_fill_sign,
        const int close_holes_voxels,
        const bool cap_open_boundaries)
        : V0_(skelV)
        , E_(skelE)
        , voxel_size_(voxel_size)
        , samples_per_bone_(std::max(2, samples_per_bone))
        , inside_margin_(inside_margin)
        , flood_fill_sign_(flood_fill_sign)
        , close_holes_voxels_(std::max(0, close_holes_voxels))
        , cap_open_boundaries_(cap_open_boundaries)
    {
        // build SDF of the surface mesh (same approach as in FitForm)
        math::Transform::Ptr xform = math::Transform::createLinearTransform(voxel_size_);

        std::vector<Vec3s> points;
        std::vector<Vec3I> triangles;
        std::vector<Vec4I> quads;

        points.reserve(surfaceV.rows());
        for (int i = 0; i < surfaceV.rows(); i++)
            points.push_back(Vec3s(surfaceV(i, 0), surfaceV(i, 1), surfaceV(i, 2)));

        for (int i = 0; i < surfaceF.rows(); i++)
            triangles.push_back(Vec3I(surfaceF(i, 0), surfaceF(i, 1), surfaceF(i, 2)));

        // Optionally cap open boundaries by adding fan-triangle caps per boundary loop
        if (cap_open_boundaries_)
        {
            Eigen::MatrixXi B;
            igl::boundary_facets(surfaceF, B);
            if (B.rows() > 0)
            {
                // Build adjacency on boundary
                std::unordered_map<int, std::vector<int>> adj;
                adj.reserve(B.rows());
                for (int i = 0; i < B.rows(); ++i)
                {
                    const int a = B(i, 0), b = B(i, 1);
                    adj[a].push_back(b);
                    adj[b].push_back(a);
                }
                std::unordered_set<long long> used;
                auto key = [](int a, int b) -> long long { return (static_cast<long long>(a) << 32) ^ static_cast<unsigned long long>(b); };

                std::vector<std::vector<int>> loops;
                std::unordered_set<int> visited_vertices;
                for (auto &kv : adj)
                {
                    const int start = kv.first;
                    if (visited_vertices.count(start)) continue;
                    // Walk a loop starting at start
                    if (adj[start].empty()) continue;
                    int prev = -1;
                    int curr = start;
                    int next = adj[curr].front();
                    std::vector<int> loop;
                    loop.push_back(curr);
                    int guard = 0;
                    while (guard++ < 100000)
                    {
                        // advance
                        int nb0 = adj[curr].size() > 0 ? adj[curr][0] : -1;
                        int nb1 = adj[curr].size() > 1 ? adj[curr][1] : -1;
                        int cand = (nb0 != prev ? nb0 : nb1);
                        if (cand < 0) break;
                        prev = curr;
                        curr = cand;
                        loop.push_back(curr);
                        visited_vertices.insert(curr);
                        if (curr == start) break;
                    }
                    if (loop.size() >= 3 && loop.front() == loop.back())
                        loop.pop_back();
                    if (loop.size() >= 3)
                        loops.push_back(loop);
                }

                for (const auto &loop : loops)
                {
                    // Plane fit
                    Eigen::Vector3d c = Eigen::Vector3d::Zero();
                    for (int vi : loop) c += surfaceV.row(vi).transpose();
                    c /= (double)loop.size();
                    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
                    for (int vi : loop)
                    {
                        Eigen::Vector3d d = surfaceV.row(vi).transpose() - c;
                        cov += d * d.transpose();
                    }
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
                    Eigen::Vector3d n = es.eigenvectors().col(0).normalized();
                    // Build orthonormal basis
                    Eigen::Vector3d t = es.eigenvectors().col(2).normalized();
                    Eigen::Vector3d b = n.cross(t).normalized();

                    // Add center vertex
                    const int center_idx = (int)points.size();
                    points.push_back(openvdb::Vec3s(c(0), c(1), c(2)));

                    // Fan triangulation
                    for (size_t k = 0; k < loop.size(); ++k)
                    {
                        int vi = loop[k];
                        int vj = loop[(k + 1) % loop.size()];
                        triangles.push_back(openvdb::Vec3I(center_idx, vi, vj));
                    }
                }
            }
        }

        grid_ = tools::meshToSignedDistanceField<DoubleGrid>(*xform, points, triangles, quads, /*exteriorWidth=*/150, /*interiorWidth=*/1);

        // Fix sign for open/ambiguous surfaces if requested
        if (flood_fill_sign_)
            tools::signedFloodFill(grid_->tree());

        // Morphological closing to seal holes up to close_holes_voxels_ (optional)
        if (close_holes_voxels_ > 0)
        {
            tools::LevelSetFilter<DoubleGrid> filter(*grid_);
            filter.dilate(close_holes_voxels_);
            filter.erode(close_holes_voxels_);
        }

        // build samples along each bone, inclusive of endpoints
        samples_.clear();
        samples_.reserve(E_.rows() * samples_per_bone_);
        for (int e = 0; e < E_.rows(); ++e)
        {
            const int i = E_(e, 0), j = E_(e, 1);
            for (int k = 0; k < samples_per_bone_; ++k)
            {
                const double t = samples_per_bone_ == 1 ? 0.0 : (double)k / (double)(samples_per_bone_ - 1);
                samples_.push_back({i, j, t});
            }
        }

        // Diagnostics: log SDF sign statistics at initial skeleton positions
        try
        {
            typename DoubleGrid::ConstAccessor acc = grid_->getConstAccessor();
            int pos_cnt = 0, neg_cnt = 0, zero_cnt = 0;
            double min_phi = std::numeric_limits<double>::infinity();
            double max_phi = -std::numeric_limits<double>::infinity();
            for (const auto &s : samples_)
            {
                const Eigen::Vector3d p = (1.0 - s.t) * V0_.row(s.vi) + s.t * V0_.row(s.vj);
                math::Vec3<double> pv(p(0), p(1), p(2));
                const double phi = tools::SplineSampler::sample(acc, grid_->transformPtr()->worldToIndex(pv));
                min_phi = std::min(min_phi, phi);
                max_phi = std::max(max_phi, phi);
                if (phi > 0) pos_cnt++; else if (phi < 0) neg_cnt++; else zero_cnt++;
            }
            logger().info("[SkeletonInsideSDFForm] samples={} pos={} neg={} zero={} min_phi={} max_phi={} margin={}",
                          (int)samples_.size(), pos_cnt, neg_cnt, zero_cnt, min_phi, max_phi, inside_margin_);
        }
        catch (...) {}
    }

    double SkeletonInsideSDFForm::value_unweighted(const Eigen::VectorXd &x) const
    {
        const Eigen::MatrixXd V = utils::unflatten(x, 3) + V0_;

        double val = 0.0;
        typename DoubleGrid::ConstAccessor acc = grid_->getConstAccessor();

        for (const auto &s : samples_)
        {
            const Eigen::Vector3d p = (1.0 - s.t) * V.row(s.vi) + s.t * V.row(s.vj);
            math::Vec3<double> pv(p(0), p(1), p(2));
            const auto hess = tools::SplineSampler::sampleHessian(acc, grid_->transformPtr()->worldToIndex(pv));
            const double phi = hess.x;
            const double s_val = phi + inside_margin_;
            if (s_val > 0.0)
                val += s_val * s_val;
        }

        return val;
    }

    void SkeletonInsideSDFForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
    {
        const Eigen::MatrixXd V = utils::unflatten(x, 3) + V0_;
        gradv.setZero(x.size());

        typename DoubleGrid::ConstAccessor acc = grid_->getConstAccessor();

        for (const auto &s : samples_)
        {
            const Eigen::Vector3d p = (1.0 - s.t) * V.row(s.vi) + s.t * V.row(s.vj);
            math::Vec3<double> pv(p(0), p(1), p(2));
            auto hess = tools::SplineSampler::sampleHessian(acc, grid_->transformPtr()->worldToIndex(pv));
            hess.g = hess.g / voxel_size_;

            const double phi = hess.x;
            const double s_val = phi + inside_margin_;
            if (s_val <= 0.0)
                continue;

            Eigen::Vector3d gp(hess.g[0], hess.g[1], hess.g[2]);
            const Eigen::Vector3d contrib = 2.0 * s_val * gp;

            for (int d = 0; d < 3; ++d)
            {
                gradv(3 * s.vi + d) += (1.0 - s.t) * contrib(d);
                gradv(3 * s.vj + d) += s.t * contrib(d);
            }
        }
    }

    void SkeletonInsideSDFForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
    {
        const Eigen::MatrixXd V = utils::unflatten(x, 3) + V0_;

        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(samples_.size() * 36);

        typename DoubleGrid::ConstAccessor acc = grid_->getConstAccessor();

        for (const auto &s : samples_)
        {
            const Eigen::Vector3d p = (1.0 - s.t) * V.row(s.vi) + s.t * V.row(s.vj);
            math::Vec3<double> pv(p(0), p(1), p(2));
            auto hess = tools::SplineSampler::sampleHessian(acc, grid_->transformPtr()->worldToIndex(pv));
            hess.g = hess.g / voxel_size_;
            hess.h = hess.h * (1.0 / voxel_size_ / voxel_size_);

            const double phi = hess.x;
            const double s_val = phi + inside_margin_;
            if (s_val <= 0.0)
                continue;

            Eigen::Vector3d gp(hess.g[0], hess.g[1], hess.g[2]);
            Eigen::Matrix3d Hp;
            Hp << hess.h(0, 0), hess.h(0, 1), hess.h(0, 2),
                  hess.h(1, 0), hess.h(1, 1), hess.h(1, 2),
                  hess.h(2, 0), hess.h(2, 1), hess.h(2, 2);

            Eigen::Matrix3d Hlocal = 2.0 * (gp * gp.transpose() + s_val * Hp);

            const double a = (1.0 - s.t);
            const double b = s.t;

            const int ii = s.vi;
            const int jj = s.vj;

            // scatter 3x3 blocks
            auto scatter_block = [&](int vi, int vj, double w, const Eigen::Matrix3d &M)
            {
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        trips.emplace_back(3 * vi + r, 3 * vj + c, w * M(r, c));
            };

            scatter_block(ii, ii, a * a, Hlocal);
            scatter_block(jj, jj, b * b, Hlocal);
            scatter_block(ii, jj, a * b, Hlocal);
            scatter_block(jj, ii, a * b, Hlocal);
        }

        hessian.setZero();
        hessian.resize(x.size(), x.size());
        hessian.setFromTriplets(trips.begin(), trips.end());
    }

    int SkeletonInsideSDFForm::count_violations(const Eigen::MatrixXd &V) const
    {
        int cnt = 0;
        typename DoubleGrid::ConstAccessor acc = grid_->getConstAccessor();
        for (const auto &s : samples_)
        {
            const Eigen::Vector3d p = (1.0 - s.t) * V.row(s.vi) + s.t * V.row(s.vj);
            math::Vec3<double> pv(p(0), p(1), p(2));
            const double phi = tools::SplineSampler::sample(acc, grid_->transformPtr()->worldToIndex(pv));
            if (phi + inside_margin_ > 0.0)
                ++cnt;
        }
        return cnt;
    }

    void SkeletonInsideSDFForm::export_samples_ply(const std::string &path, const Eigen::MatrixXd &V, const Eigen::RowVector3d &output_translation, const double output_scale) const
    {
        std::ofstream ply(path, std::ios::out);
        if (!ply.is_open())
        {
            logger().warn("Failed to open {} for writing samples PLY", path);
            return;
        }

        // Write ASCII PLY header for colored points
        ply << "ply\n";
        ply << "format ascii 1.0\n";
        ply << "element vertex " << samples_.size() << "\n";
        ply << "property float x\n";
        ply << "property float y\n";
        ply << "property float z\n";
        ply << "property uchar red\n";
        ply << "property uchar green\n";
        ply << "property uchar blue\n";
        ply << "element face 0\n";
        ply << "property list uchar int vertex_indices\n";
        ply << "end_header\n";

        typename DoubleGrid::ConstAccessor acc = grid_->getConstAccessor();
        for (const auto &s : samples_)
        {
            const Eigen::Vector3d p = (1.0 - s.t) * V.row(s.vi) + s.t * V.row(s.vj);
            math::Vec3<double> pv(p(0), p(1), p(2));
            const double phi = tools::SplineSampler::sample(acc, grid_->transformPtr()->worldToIndex(pv));
            const bool outside = (phi + inside_margin_ > 0.0);
            const int r = outside ? 220 : 60;
            const int g = outside ? 40  : 200;
            const int b = 40;
            const Eigen::Vector3d pout = p * output_scale + output_translation.transpose();
            ply << static_cast<float>(pout(0)) << " "
                << static_cast<float>(pout(1)) << " "
                << static_cast<float>(pout(2)) << " "
                << r << " " << g << " " << b << "\n";
        }

        ply.close();
        logger().debug("Wrote samples PLY to {}", path);
    }

    void SkeletonInsideSDFForm::export_skeleton_and_samples_ply(const std::string &path, const Eigen::MatrixXd &V, const Eigen::MatrixXi &E, const Eigen::RowVector3d &output_translation, const double output_scale) const
    {
        std::ofstream ply(path, std::ios::out);
        if (!ply.is_open())
        {
            logger().warn("Failed to open {} for writing combined PLY", path);
            return;
        }

        const size_t n_vertices = static_cast<size_t>(V.rows()) + samples_.size();
        const size_t n_edges = static_cast<size_t>(E.rows());

        // Header: vertices with colors; edges for skeleton
        ply << "ply\n";
        ply << "format ascii 1.0\n";
        ply << "element vertex " << n_vertices << "\n";
        ply << "property float x\n";
        ply << "property float y\n";
        ply << "property float z\n";
        ply << "property uchar red\n";
        ply << "property uchar green\n";
        ply << "property uchar blue\n";
        ply << "element edge " << n_edges << "\n";
        ply << "property int vertex1\n";
        ply << "property int vertex2\n";
        ply << "end_header\n";

        // Skeleton vertices: colored blue-ish
        const int sv_r = 60, sv_g = 60, sv_b = 200;
        for (int i = 0; i < V.rows(); ++i)
        {
            const Eigen::RowVector3d vout = V.row(i) * output_scale + output_translation;
            ply << static_cast<float>(vout(0)) << " "
                << static_cast<float>(vout(1)) << " "
                << static_cast<float>(vout(2)) << " "
                << sv_r << " " << sv_g << " " << sv_b << "\n";
        }

        // Sample points: colored by violation
        typename DoubleGrid::ConstAccessor acc = grid_->getConstAccessor();
        for (const auto &s : samples_)
        {
            const Eigen::Vector3d p = (1.0 - s.t) * V.row(s.vi) + s.t * V.row(s.vj);
            math::Vec3<double> pv(p(0), p(1), p(2));
            const double phi = tools::SplineSampler::sample(acc, grid_->transformPtr()->worldToIndex(pv));
            const bool outside = (phi + inside_margin_ > 0.0);
            const int r = outside ? 220 : 60;
            const int g = outside ? 40  : 200;
            const int b = 40;
            const Eigen::Vector3d pout = p * output_scale + output_translation.transpose();
            ply << static_cast<float>(pout(0)) << " "
                << static_cast<float>(pout(1)) << " "
                << static_cast<float>(pout(2)) << " "
                << r << " " << g << " " << b << "\n";
        }

        // Edges over the first V.rows() vertices
        for (int e = 0; e < E.rows(); ++e)
        {
            ply << E(e, 0) << " " << E(e, 1) << "\n";
        }

        ply.close();
        logger().debug("Wrote combined skeleton+samples PLY to {}", path);
    }

    void SkeletonInsideSDFForm::export_sdf_isosurface(const std::string &path, const double iso, const Eigen::RowVector3d &output_translation, const double output_scale) const
    {
        std::vector<openvdb::Vec3s> points;
        std::vector<openvdb::Vec4I> quads;
        try
        {
            tools::volumeToMesh(*grid_, points, quads, iso);
        }
        catch (...)
        {
            logger().warn("volumeToMesh failed for iso {}", iso);
            return;
        }

        Eigen::MatrixXd outpoints(points.size(), 3);
        Eigen::MatrixXi outtriangles(2 * quads.size(), 3);
        for (int i = 0; i < (int)points.size(); ++i)
            outpoints.row(i) << points[i](0) * output_scale + output_translation(0),
                                points[i](1) * output_scale + output_translation(1),
                                points[i](2) * output_scale + output_translation(2);
        for (int i = 0; i < (int)quads.size(); ++i)
        {
            outtriangles.row(2 * i) << quads[i](0), quads[i](1), quads[i](2);
            outtriangles.row(2 * i + 1) << quads[i](0), quads[i](2), quads[i](3);
        }
        io::OBJWriter::write(path, outpoints, Eigen::MatrixXi(), outtriangles);
        logger().debug("Wrote SDF isosurface (iso={}) to {}", iso, path);
    }
} // namespace polyfem::solver


