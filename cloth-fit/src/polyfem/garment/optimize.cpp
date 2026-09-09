#include "optimize.hpp"

#include <polyfem/solver/GarmentNLProblem.hpp>
#include <polyfem/solver/FullNLProblem.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/garment_forms/GarmentALForm.hpp>
#include <polyfem/solver/forms/garment_forms/SkeletonInsideSDFForm.hpp>
#include <polyfem/solver/forms/garment_forms/BoneLengthForm.hpp>
#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/par_for.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <igl/edges.h>
#include <igl/read_triangle_mesh.h>
#include <igl/readOBJ.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/write_triangle_mesh.h>
#include <igl/writeOBJ.h>

#include <polysolve/linear/Solver.hpp>
#include <polysolve/nonlinear/Solver.hpp>
#include <openvdb/openvdb.h>

#ifdef POLYFEM_WITH_PARAVIEWO
#include <paraviewo/ParaviewWriter.hpp>
#include <paraviewo/VTUWriter.hpp>
#endif

#include <ipc/ipc.hpp>
#include <ipc/distance/point_edge.hpp>
#include <ipc/distance/point_line.hpp>
#include <ipc/utils/logger.hpp>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/ostream_sink.h>

#include <jse/jse.h>

#include <unordered_set>
#include <filesystem>
#include <array>
#include <fstream>
#include <algorithm>

using namespace polyfem::solver;
using namespace polyfem::mesh;

namespace spdlog::level
{
	NLOHMANN_JSON_SERIALIZE_ENUM(
		spdlog::level::level_enum,
		{{spdlog::level::level_enum::trace, "trace"},
		 {spdlog::level::level_enum::debug, "debug"},
		 {spdlog::level::level_enum::info, "info"},
		 {spdlog::level::level_enum::warn, "warning"},
		 {spdlog::level::level_enum::err, "error"},
		 {spdlog::level::level_enum::critical, "critical"},
		 {spdlog::level::level_enum::off, "off"},
		 {spdlog::level::level_enum::trace, 0},
		 {spdlog::level::level_enum::debug, 1},
		 {spdlog::level::level_enum::info, 2},
		 {spdlog::level::level_enum::warn, 3},
		 {spdlog::level::level_enum::err, 3},
		 {spdlog::level::level_enum::critical, 4},
		 {spdlog::level::level_enum::off, 5}})
}
namespace polyfem {
    namespace {
        Eigen::Vector2d project_to_line(
            const Eigen::Vector3d &a,
            const Eigen::Vector3d &b,
            const Eigen::Vector3d &p)
        {
            Eigen::Vector3d s = b - a;
            double t = (p - a).dot(s) / s.squaredNorm();
            double d = (p - (a + t * s)).squaredNorm();

            return Eigen::Vector2d(d, t);
        }

        /// @brief Returns the squared distance of p to edge ab, and the parametric coordinate of the closest point
        Eigen::Vector2d project_to_edge(
            const Eigen::Vector3d &a,
            const Eigen::Vector3d &b,
            const Eigen::Vector3d &p)
        {
            Eigen::Vector3d s = b - a;
            double t = (p - a).dot(s) / s.squaredNorm();
            t = std::min(1., std::max(t, 0.));
            double d = (p - (a + t * s)).squaredNorm();

            return Eigen::Vector2d(d, t);
        }

        void floydWarshall(const Eigen::MatrixXi &G, Eigen::MatrixXi &dist, Eigen::MatrixXi &parents)
        {
            int N = G.rows();
            dist = G;
            parents = -Eigen::MatrixXi::Ones(N, N);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    if (dist(i, j) > 0 && dist(i, j) <= N)
                        parents(i, j) = i;
            
            for (int k = 0; k < N; k++)
            {
                for (int i = 0; i < N; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        if (k == i || k == j || i == j)
                            continue;
                        if (dist(i, j) > dist(i, k) + dist(k, j))
                        {
                            dist(i, j) = dist(i, k) + dist(k, j);
                            parents(i, j) = parents(k, j);
                        }
                    }
                }
            }

            // validate
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    if (i == j)
                        continue;
                    const int p = parents(i, j);
                    if (dist(i, p) + G(p, j) != dist(i, j))
                        log_and_throw_error("[floydWarshall] Wrong closest distance!");
                    
                    int cur = j;
                    while (parents(i, cur) != i)
                        cur = parents(i, cur);
                    if (parents(i, cur) != i)
                        log_and_throw_error("[floydWarshall] Wrong closest distance!");
                }
            }
        }

        bool are_same_edges(const Eigen::MatrixXi &A, const Eigen::MatrixXi &B)
        {
            if (A.rows() != B.rows())
                return false;
            
            for (int i = 0; i < A.rows(); i++)
            {
                bool flag = false;
                for (int j = 0; j < B.rows(); j++)
                {
                    if ((std::min(A(i, 0), A(i, 1)) == std::min(B(j, 0), B(j, 1)))
                    && (std::max(A(i, 0), A(i, 1)) == std::max(B(j, 0), B(j, 1))))
                    {
                        flag = true;
                        break;
                    }
                }
                if (!flag)
                    return false;
            }

            return true;
        }

        bool is_end_node(const Eigen::MatrixXi &edges, const int vid)
        {
            int cnt = 0;
            for (int i = 0; i < edges.size(); i++)
            {
                if (edges(i) == vid)
                    cnt++;
            }
            if (cnt == 0)
                log_and_throw_error("vid not found in is_end_node()!");
            return (cnt == 1);
        }

        void explode_trimesh(
            const Eigen::MatrixXd &Vin,
            const Eigen::MatrixXi &Fin,
            Eigen::MatrixXd &Vout,
            Eigen::MatrixXi &Fout,
            Eigen::VectorXi &index_map)
        {
            Vout = Eigen::MatrixXd::Zero(Fin.size(), 3);
            Fout = Fin;
            index_map.setZero(Vout.rows());
            for (int f = 0; f < Fin.rows(); f++)
            {
                for (int i = 0; i < Fin.cols(); i++)
                {
                    Vout.row(f * Fin.cols() + i) = Vin.row(Fin(f, i));
                    Fout(f, i) = f * Fin.cols() + i;
                    index_map(f * Fin.cols() + i) = Fin(f, i);
                }
            }
        }
    }

    void GarmentSolver::prepass_optimize_source_skeleton_inside_garment(const json &in_args)
    {
        if (!utils::is_param_valid(in_args, "skeleton_prepass"))
            return;

        const json &pp = in_args["skeleton_prepass"];
        const bool enable = utils::is_param_valid(pp, "enable") ? (bool)pp["enable"] : false;
        if (!enable)
            return;

        const double voxel_size = utils::is_param_valid(pp, "voxel_size") ? (double)pp["voxel_size"] : 5e-3;
        const int samples_per_bone = utils::is_param_valid(pp, "samples_per_bone") ? (int)pp["samples_per_bone"] : 7;
        const double inside_margin = utils::is_param_valid(pp, "inside_margin") ? (double)pp["inside_margin"] : 0.0;
        const bool flood_fill_sign = utils::is_param_valid(pp, "flood_fill_sign") ? (bool)pp["flood_fill_sign"] : true;
        const int close_holes_voxels = utils::is_param_valid(pp, "close_holes_voxels") ? (int)pp["close_holes_voxels"] : 0;
        const double inside_weight = utils::is_param_valid(pp, "inside_weight") ? (double)pp["inside_weight"] : 1.0;
        const double length_weight = utils::is_param_valid(pp, "length_weight") ? (double)pp["length_weight"] : 10.0;
        const double anchor_weight = utils::is_param_valid(pp, "anchor_weight") ? (double)pp["anchor_weight"] : 50.0;
        const int root_vertex = utils::is_param_valid(pp, "root_vertex") ? (int)pp["root_vertex"] : 0;
        const bool cap_open_boundaries = utils::is_param_valid(pp, "cap_open_boundaries") ? (bool)pp["cap_open_boundaries"] : false;

        logger().info("[prepass] enable=true voxel_size={} samples_per_bone={} inside_margin={} inside_weight={} length_weight={} anchor_weight={} root_vertex={}",
                      voxel_size, samples_per_bone, inside_margin, inside_weight, length_weight, anchor_weight, root_vertex);

        // Ensure OpenVDB is initialized before constructing grids
        try { openvdb::initialize(); } catch (...) {}

        // Save original source skeleton to visualize delta
        {
            Eigen::MatrixXd skel_out = skeleton_v;
            const double inv_s_out = (normalization_scale_ == 0.0) ? 1.0 : (1.0 / normalization_scale_);
            skel_out *= inv_s_out;
            if (restore_output_translation_)
                skel_out.rowwise() += source_output_translation_offset_;
            write_edge_mesh(out_folder + "/source_skeleton_before_prepass.obj", skel_out, skeleton_b);
        }

        std::vector<std::shared_ptr<Form>> forms;

        std::shared_ptr<SkeletonInsideSDFForm> inside_form;
        inside_form = std::make_shared<SkeletonInsideSDFForm>(
            skeleton_v, skeleton_b, garment.v, garment.f,
            voxel_size, samples_per_bone, inside_margin, flood_fill_sign, close_holes_voxels, cap_open_boundaries);
        inside_form->set_weight(inside_weight);
        forms.push_back(inside_form);

        // Export samples before optimization
        const Eigen::RowVector3d prepass_out_t = restore_output_translation_ ? source_output_translation_offset_ : Eigen::RowVector3d::Zero();
        const double inv_s_out = (normalization_scale_ == 0.0) ? 1.0 : (1.0 / normalization_scale_);
        inside_form->export_samples_ply(out_folder + "/prepass_samples_before.ply", skeleton_v, prepass_out_t, inv_s_out);
        inside_form->export_skeleton_and_samples_ply(out_folder + "/prepass_skeleton_samples_before.ply", skeleton_v, skeleton_b, prepass_out_t, inv_s_out);
        // Export SDF isosurfaces for debugging (surface and inner offset)
        inside_form->export_sdf_isosurface(out_folder + "/prepass_sdf_iso0.obj", 0.0, prepass_out_t, inv_s_out);
        inside_form->export_sdf_isosurface(out_folder + "/prepass_sdf_iso_inner.obj", -inside_margin, prepass_out_t, inv_s_out);

        // Auto-skip if there are no violations at the initial pose
        {
            const int viol = inside_form->count_violations(skeleton_v);
            if (viol == 0)
            {
                logger().info("[prepass] skipped: no outside samples detected at initial pose");
                return;
            }
            logger().info("[prepass] initial outside samples: {}", viol);
        }

        {
            auto length_form = std::make_shared<BoneLengthForm>(skeleton_v, skeleton_b);
            length_form->set_weight(length_weight);
            forms.push_back(length_form);
        }

        std::shared_ptr<PointPenaltyForm> anchor_form;
        if (anchor_weight > 0.0 && root_vertex >= 0 && root_vertex < skeleton_v.rows())
        {
            std::vector<int> indices = {3 * root_vertex + 0, 3 * root_vertex + 1, 3 * root_vertex + 2};
            Eigen::VectorXd target_vec(3);
            target_vec.setZero();
            anchor_form = std::make_shared<PointPenaltyForm>(target_vec, indices);
            anchor_form->set_weight(anchor_weight);
            forms.push_back(anchor_form);
        }

        FullNLProblem nl_problem(forms);
        nl_problem.set_project_to_psd(true);

        Eigen::VectorXd x = Eigen::VectorXd::Zero(3 * skeleton_v.rows());

        std::shared_ptr<polysolve::nonlinear::Solver> nl_solver;
        if (utils::is_param_valid(in_args, "solver") && in_args["solver"].contains("nonlinear") && in_args["solver"].contains("linear"))
            nl_solver = polysolve::nonlinear::Solver::create(in_args["solver"]["nonlinear"], in_args["solver"]["linear"], 1., logger());
        else
            nl_solver = polysolve::nonlinear::Solver::create("newton", "Eigen::SimplicialLDLT", 1., logger());

        try
        {
            nl_solver->minimize(nl_problem, x);
        }
        catch (const std::runtime_error &e)
        {
            logger().warn("Skeleton pre-pass optimization failed: {}", e.what());
        }

        skeleton_v += utils::unflatten(x, 3);

        // Debug output (after prepass)
        {
            Eigen::MatrixXd skel_out = skeleton_v;
            const double inv_s_out2 = (normalization_scale_ == 0.0) ? 1.0 : (1.0 / normalization_scale_);
            skel_out *= inv_s_out2;
            if (restore_output_translation_)
                skel_out.rowwise() += source_output_translation_offset_;
            write_edge_mesh(out_folder + "/prepass_source_skeleton.obj", skel_out, skeleton_b);
        }
        inside_form->export_samples_ply(out_folder + "/prepass_samples_after.ply", skeleton_v, prepass_out_t, inv_s_out);
        inside_form->export_skeleton_and_samples_ply(out_folder + "/prepass_skeleton_samples_after.ply", skeleton_v, skeleton_b, prepass_out_t, inv_s_out);
    }

    void OBJMesh::read(const std::string &path)
    {
        igl::readOBJ(path, v, tc, cn, f, ftc, fn);
    }

    void OBJMesh::write(const std::string &path)
    {
        igl::writeOBJ(path, v, f, cn, fn, tc, ftc);
    }

    void GarmentSolver::save_result(
        const std::string &path,
        const int index,
        GarmentNLProblem &prob,
        const Eigen::MatrixXd &V,
        const Eigen::MatrixXi &F,
        const Eigen::VectorXd &sol)
    {
        const Eigen::VectorXd full_disp = prob.reduced_to_full(sol);
        const Eigen::VectorXd complete_disp = prob.full_to_complete(full_disp);
        const Eigen::MatrixXd current_vertices = utils::unflatten(complete_disp, V.cols()) + V;

#ifdef POLYFEM_WITH_PARAVIEWO
        std::shared_ptr<paraviewo::ParaviewWriter> tmpw = std::make_shared<paraviewo::VTUWriter>();
        paraviewo::ParaviewWriter &writer = *tmpw;

        if (false)
        {
            Eigen::VectorXd total_grad = Eigen::VectorXd::Zero(complete_disp.size());
            std::unordered_set<std::string> existing_names;
            for (const auto &form : prob.forms())
            {
                Eigen::VectorXd grad;
                form->first_derivative(complete_disp, grad);
                std::string name = "grad_" + form->name();
                while (existing_names.count(name) != 0)
                    name += "_";
                existing_names.insert(name);
                grad.head(nc_avatar_v.rows() * 3).setZero();
                writer.add_field(name, utils::unflatten(grad, 3));
                total_grad += grad;
            }
            for (const auto &form : prob.full_forms())
            {
                Eigen::VectorXd grad_full, grad;
                form->first_derivative(full_disp, grad_full);
                std::string name = "grad_" + form->name();
                while (existing_names.count(name) != 0)
                    name += "_";
                existing_names.insert(name);
                grad.setZero(total_grad.size());
                grad.tail(grad_full.size() - 1) = grad_full.tail(grad_full.size() - 1);
                writer.add_field(name, utils::unflatten(grad, 3));
                total_grad += grad;
            }
            total_grad.head(nc_avatar_v.rows() * 3).setZero();
            writer.add_field("grad", utils::unflatten(total_grad, 3));

            Eigen::VectorXd body_ids = Eigen::VectorXd::Zero(V.rows());
            body_ids.head(nc_avatar_v.rows()).array() = 1;
            writer.add_field("body_ids", body_ids);

            logger().debug("Save VTU to {}", path + "/step_" + std::to_string(index) + ".vtu");
            writer.write_mesh(path + "/step_" + std::to_string(index) + ".vtu", current_vertices, F);
        }
#endif
        garment.v = current_vertices.bottomRows(garment.v.rows());

        // Diagnostic dump for exploding vertices (normalized space).
        // Trigger when a garment vertex becomes an extreme outlier, which typically indicates
        // numerical instability in one of the energy terms (e.g., similarity / fit / contact).
        {
            const Eigen::MatrixXd &G = garment.v;
            if (G.rows() > 0)
            {
                // Median center (more robust than mean under outliers)
                Eigen::RowVector3d med = Eigen::RowVector3d::Zero();
                for (int c = 0; c < 3; ++c)
                {
                    std::vector<double> vals;
                    vals.reserve(G.rows());
                    for (int i = 0; i < G.rows(); ++i) vals.push_back(G(i, c));
                    const auto mid = vals.begin() + vals.size() / 2;
                    std::nth_element(vals.begin(), mid, vals.end());
                    med(c) = *mid;
                }

                int vi_max = -1;
                double dmax = -1.0;
                for (int i = 0; i < G.rows(); ++i)
                {
                    const double d = (G.row(i) - med).norm();
                    if (d > dmax)
                    {
                        dmax = d;
                        vi_max = i;
                    }
                }

                const double outlier_thresh = 5.0; // in normalized-space units (~meters)
                if (vi_max >= 0 && std::isfinite(dmax) && dmax > outlier_thresh)
                {
                    const int n_avatar = nc_avatar_v.rows();
                    const int outlier_collision = n_avatar + vi_max;

                    // 1-ring neighborhood in collision mesh
                    std::vector<int> nbr;
                    nbr.reserve(64);
                    auto add_unique = [&](int v) {
                        for (int x : nbr) if (x == v) return;
                        nbr.push_back(v);
                    };
                    add_unique(outlier_collision);
                    for (int fi = 0; fi < F.rows(); ++fi)
                    {
                        const int a = F(fi, 0), b = F(fi, 1), c = F(fi, 2);
                        if (a == outlier_collision || b == outlier_collision || c == outlier_collision)
                        {
                            add_unique(a);
                            add_unique(b);
                            add_unique(c);
                        }
                    }

                    // Per-form gradient norms restricted to neighborhood
                    auto grad_norm_on = [&](const Eigen::VectorXd &grad) -> double {
                        double s = 0.0;
                        for (int v : nbr)
                        {
                            if (v < 0 || 3 * v + 2 >= grad.size()) continue;
                            const Eigen::Vector3d g = grad.segment<3>(3 * v);
                            s += g.squaredNorm();
                        }
                        return std::sqrt(s);
                    };

                    json j;
                    j["index"] = index;
                    j["space"] = "normalized";
                    j["garment_vertex_outlier_local"] = vi_max;
                    j["collision_vertex_outlier"] = outlier_collision;
                    j["max_dist_from_garment_median"] = dmax;
                    j["garment_median_xyz"] = {med(0), med(1), med(2)};
                    j["outlier_xyz"] = {G(vi_max, 0), G(vi_max, 1), G(vi_max, 2)};
                    j["neighborhood_collision_vertices"] = nbr;

                    // Compute weighted gradients per-form in complete space
                    json forms = json::array();
                    for (const auto &form : prob.forms())
                    {
                        Eigen::VectorXd grad;
                        form->first_derivative(complete_disp, grad);
                        const bool finite = grad.allFinite();
                        forms.push_back({
                            {"group", "forms"},
                            {"name", form->name()},
                            {"grad_finite", finite},
                            {"grad_l2_neighborhood", grad_norm_on(grad)},
                            {"grad_l2_global", finite ? grad.norm() : -1.0}
                        });
                    }
                    // Full-forms are defined on full_disp; lift into complete indexing for neighborhood norm
                    for (const auto &form : prob.full_forms())
                    {
                        Eigen::VectorXd grad_full;
                        form->first_derivative(full_disp, grad_full);
                        // grad_full includes an extra scalar at index 0 (as in existing debug block)
                        Eigen::VectorXd grad = Eigen::VectorXd::Zero(complete_disp.size());
                        if (grad_full.size() >= 1 && grad.size() >= 1)
                            grad.tail(grad_full.size() - 1) = grad_full.tail(grad_full.size() - 1);
                        const bool finite = grad_full.allFinite();
                        forms.push_back({
                            {"group", "full_forms"},
                            {"name", form->name()},
                            {"grad_finite", finite},
                            {"grad_l2_neighborhood", grad_norm_on(grad)},
                            {"grad_l2_global", finite ? grad_full.norm() : -1.0}
                        });
                    }
                    j["per_form_gradients"] = forms;

                    const std::string out_path = path + "/vertex_diagnostics_step_" + std::to_string(index) + ".json";
                    std::ofstream out(out_path);
                    if (out.is_open())
                    {
                        out << j.dump(2) << std::endl;
                        out.close();
                        logger().warn("Wrote outlier vertex diagnostics to {} (vi_local={} dist={})", out_path, vi_max, dmax);
                    }
                    else
                    {
                        logger().warn("Failed to write outlier vertex diagnostics to {}", out_path);
                    }
                }
            }
        }

        // Debug: also save meshes in cloth-fit normalized space (before undoing normalization / restoring translation).
        // Enabled by default whenever a non-default normalization mode is in effect.
        {
            const double eps2 = 1e-24; // (1e-12)^2
            const double ds = source_output_scale_ - 1.0;
            const double dt = target_output_scale_ - 1.0;
            const bool special_norm_mode =
                restore_output_translation_
                || (ds * ds > eps2)
                || (dt * dt > eps2);
            if (special_norm_mode)
            {
                OBJMesh garment_norm = garment; // already in normalized space
                const Eigen::MatrixXd avatar_norm = current_vertices.topRows(nc_avatar_v.rows());
                garment_norm.write(path + "/step_garment_norm_" + std::to_string(index) + ".obj");
                igl::write_triangle_mesh(path + "/step_avatar_norm_" + std::to_string(index) + ".obj", avatar_norm, nc_avatar_f);
            }
        }

        // Optional output translation restoration (keep internal state in normalized space)
        OBJMesh garment_out = garment;
        Eigen::MatrixXd avatar_out = current_vertices.topRows(nc_avatar_v.rows());
        // Always undo common normalization scale (A0) on output
        const double inv_s = (normalization_scale_ == 0.0) ? 1.0 : (1.0 / normalization_scale_);
        garment_out.v *= inv_s;
        avatar_out *= inv_s;
        // Undo per-side subject scale for subject-scale normalization mode (defaults are 1.0).
        garment_out.v *= source_output_scale_;
        avatar_out *= target_output_scale_;
        if (restore_output_translation_)
        {
            garment_out.v.rowwise() += source_output_translation_offset_;
            avatar_out.rowwise() += target_output_translation_offset_;
        }

        garment_out.write(path + "/step_garment_" + std::to_string(index) + ".obj");
        logger().debug("Save OBJ to {}", path + "/step_garment_" + std::to_string(index) + ".obj");

        igl::write_triangle_mesh(path + "/step_avatar_" + std::to_string(index) + ".obj", avatar_out, nc_avatar_f);
    }

    Eigen::Vector3d bbox_size(const Eigen::Matrix<double, -1, 3> &V)
    {
        return V.colwise().maxCoeff() - V.colwise().minCoeff();
    }

    void GarmentSolver::load_garment_mesh(
        const std::string &mesh_path,
        const std::string &no_fit_spec_path)
	{
        garment.read(mesh_path);

        if (std::filesystem::exists(no_fit_spec_path))
        {   
            Eigen::MatrixXi tmp_vids;
            io::read_matrix<int>(no_fit_spec_path, tmp_vids);

            if (tmp_vids.maxCoeff() >= garment.v.rows() || tmp_vids.minCoeff() < 0)
                log_and_throw_error("Vertex ID {} in no-fit.txt out of range!");

            Eigen::VectorXi vmask = Eigen::VectorXi::Zero(garment.v.rows());
            for (int i = 0; i < tmp_vids.size(); i++)
                vmask(tmp_vids(i)) = 1;
                
            for (int i = 0; i < garment.f.rows(); i++)
                if (vmask(garment.f(i, 0)) && vmask(garment.f(i, 1)) && vmask(garment.f(i, 2)))
                    not_fit_fids.push_back(i);
        }
        else
            logger().debug("Cannot find {}, will fit the garment tightly everywhere...", no_fit_spec_path);

        // if (std::filesystem::exists(path + "/skin.txt"))
        // {
        //     log_and_throw_error("Utilizing garment skinning weight is not supported!");
            
        //     io::read_matrix(path + "/skin.txt", garment_skinning_weights);
        //     assert(garment_skinning_weights.rows() == skeleton_v.rows());
        //     assert(garment.v.rows() == garment_skinning_weights.cols());
        //     assert(garment_skinning_weights.minCoeff() >= 0. && garment_skinning_weights.maxCoeff() <= 1.);
        // }
        // else
        // if (n_refs > 0) {
        //     while (n_refs-- > 0)
        //     {
        //         std::tie(garment.v, garment.f) = refine(garment.v, garment.f);
                
        //         std::vector<int> not_fit_fids_new;
        //         for (int i = 0; i < not_fit_fids.size(); i++)
        //             for (int j = 0; j < 4; j++)
        //                 not_fit_fids_new.push_back(not_fit_fids[i] * 4 + j);
        //         std::swap(not_fit_fids, not_fit_fids_new);
        //     }
        //     assert(n_refs == 0);
        // }

		// remove duplicate vertices in the garment
        // remove_duplicate_vertices(garment.v, garment.f, 1e-6);
	}

    void GarmentSolver::check_intersections(
        const ipc::CollisionMesh &collision_mesh,
        const Eigen::MatrixXd &collision_vertices) const
    {
        auto ids = ipc::my_has_intersections(collision_mesh, collision_vertices, ipc::BroadPhaseMethod::BVH);
        if (ids[0] >= 0)
        {
            io::OBJWriter::write(
                out_folder + "/intersection.obj", collision_vertices,
                collision_mesh.edges(), collision_mesh.faces());
            Eigen::MatrixXi edge(1, 2);
            edge << ids[0], ids[1];
            Eigen::MatrixXi face(1, 3);
            face << ids[2], ids[3], ids[4];
            io::OBJWriter::write(
                out_folder + "/intersecting_pair.obj", collision_vertices,
                edge, face);
            log_and_throw_error("Unable to solve, initial solution has intersections!");
        }
    }

    void GarmentSolver::check_cross_intersections(
        const ipc::CollisionMesh &collision_mesh,
        const Eigen::MatrixXd &collision_vertices) const
    {
        // This respects collision_mesh.can_collide (set to avatar<->garment in main)
        const bool has_int = ipc::has_intersections(collision_mesh, collision_vertices, ipc::BroadPhaseMethod::BVH);
        if (has_int)
        {
            io::OBJWriter::write(
                out_folder + "/intersection.obj", collision_vertices,
                collision_mesh.edges(), collision_mesh.faces());
            log_and_throw_error("Unable to solve, initial solution has intersections (avatar↔garment)!");
        }
    }

    bool GarmentSolver::has_garment_self_intersections() const
    {
        Eigen::MatrixXi E;
        igl::edges(garment.f, E);
        ipc::CollisionMesh cm(garment.v, E, garment.f);
        return ipc::has_intersections(cm, garment.v, ipc::BroadPhaseMethod::BVH);
    }
        
    void GarmentSolver::read_meshes(
        const std::string &avatar_mesh_path,
        const std::string &source_skeleton_path,
        const std::string &target_skeleton_path,
        const std::string &target_avatar_skinning_weights_path)
    {
        igl::read_triangle_mesh(avatar_mesh_path, avatar_v, avatar_f);

        // Record original avatar centroid before any optional removal
        orig_avatar_centroid = avatar_v.colwise().mean();
        have_orig_avatar_centroid = true;
 
        read_edge_mesh(source_skeleton_path, skeleton_v, skeleton_b);
        read_edge_mesh(target_skeleton_path, target_skeleton_v, target_skeleton_b);
        if (!are_same_edges(skeleton_b, target_skeleton_b))
            log_and_throw_error("Inconsistent skeletons!");
        target_skeleton_b = skeleton_b;

        if (std::filesystem::exists(target_avatar_skinning_weights_path))
        {
            io::read_matrix(target_avatar_skinning_weights_path, target_avatar_skinning_weights);
            if (target_avatar_skinning_weights.rows() != skeleton_v.rows()
                || avatar_v.rows() != target_avatar_skinning_weights.cols())
                log_and_throw_error("Inconsistent skin weights dimension with the number of vertices and bones! Skin weights dimension: {}x{}, number of bones: {}, number of vertices: {}", target_avatar_skinning_weights.rows(), target_avatar_skinning_weights.cols(), skeleton_v.rows(), avatar_v.rows());
        }
        else
        {
            target_avatar_skinning_weights.setZero(0, 0);
            logger().warn("Cannot find target avatar skinning weights, use pure distance-based projection instead...");
        }
    }

    void GarmentSolver::remove_avatar_vertices(const std::string &indices_path)
    {
        if (indices_path.empty())
            return;

        if (!std::filesystem::exists(indices_path))
        {
            logger().warn("Avatar remove indices file not found: {}", indices_path);
            return;
        }

        Eigen::MatrixXi tmp_vids;
        io::read_matrix<int>(indices_path, tmp_vids);
        if (tmp_vids.size() == 0)
        {
            logger().info("Avatar remove list empty; skipping.");
            return;
        }

        const int n_old_v = (int)avatar_v.rows();
        std::vector<char> keep(n_old_v, 1);
        for (int i = 0; i < tmp_vids.size(); ++i)
        {
            const int v = tmp_vids(i);
            if (v < 0 || v >= n_old_v)
                log_and_throw_error("Avatar remove vertex id {} out of range (0..{}).", v, n_old_v - 1);
            keep[v] = 0;
        }

        // Build mapping old->new
        std::vector<int> map_old_to_new(n_old_v, -1);
        int n_new_v = 0;
        for (int v = 0; v < n_old_v; ++v)
            if (keep[v]) map_old_to_new[v] = n_new_v++;

        // New vertices
        Eigen::MatrixXd newV(n_new_v, 3);
        for (int v = 0; v < n_old_v; ++v)
            if (keep[v]) newV.row(map_old_to_new[v]) = avatar_v.row(v);

        // New faces: keep only faces with all 3 vertices kept
        std::vector<Eigen::Vector3i> newF_vec;
        newF_vec.reserve(avatar_f.rows());
        for (int f = 0; f < avatar_f.rows(); ++f)
        {
            int a = avatar_f(f, 0), b = avatar_f(f, 1), c = avatar_f(f, 2);
            if (keep[a] && keep[b] && keep[c])
                newF_vec.emplace_back(Eigen::Vector3i(map_old_to_new[a], map_old_to_new[b], map_old_to_new[c]));
        }
        Eigen::MatrixXi newF((int)newF_vec.size(), 3);
        for (int i = 0; i < (int)newF_vec.size(); ++i) newF.row(i) = newF_vec[i];

        // Update skin weights if loaded: rows=bones, cols=verts
        if (target_avatar_skinning_weights.size() > 0)
        {
            Eigen::MatrixXd newW(target_avatar_skinning_weights.rows(), n_new_v);
            int col = 0;
            for (int v = 0; v < n_old_v; ++v)
            {
                if (!keep[v]) continue;
                newW.col(col++) = target_avatar_skinning_weights.col(v);
            }
            target_avatar_skinning_weights = newW;
        }

        logger().info("Avatar vertices removed: {} of {} kept ({} faces -> {}).", n_old_v - n_new_v, n_old_v, avatar_f.rows(), newF.rows());

        avatar_v = newV;
        avatar_f = newF;

        // Optional debug
        io::OBJWriter::write(out_folder + "/avatar_after_removal.obj", avatar_v, Eigen::MatrixXi(), avatar_f);
        if (target_avatar_skinning_weights.size() > 0)
        {
            std::ofstream sw(out_folder + "/avatar_skin_weights_after_removal.txt");
            if (sw.is_open())
            {
                sw << target_avatar_skinning_weights.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, " ", "\n"));
                sw.close();
            }
        }
    }

    void GarmentSolver::project_avatar_to_skeleton()
    {
        Eigen::MatrixXi graph, shared_vtx, dist, parent;
        {
            graph = Eigen::MatrixXi::Ones(skeleton_b.rows(), skeleton_b.rows()) * (skeleton_b.rows() + 1);
            shared_vtx = -Eigen::MatrixXi::Ones(skeleton_b.rows(), skeleton_b.rows());
            for (int i = 0; i < skeleton_b.rows(); i++)
            {
                graph(i, i) = 0;
                for (int j = 0; j < skeleton_b.rows(); j++)
                {
                    bool adjacent = (skeleton_b(i, 0) == skeleton_b(j, 0)) ||
                                    (skeleton_b(i, 0) == skeleton_b(j, 1)) ||
                                    (skeleton_b(i, 1) == skeleton_b(j, 0)) ||
                                    (skeleton_b(i, 1) == skeleton_b(j, 1));
                    if (i != j && adjacent)
                    {
                        graph(i, j) = 1;
                        if ((skeleton_b(i, 0) == skeleton_b(j, 0)) || (skeleton_b(i, 0) == skeleton_b(j, 1)))
                            shared_vtx(i, j) = skeleton_b(i, 0);
                        else
                            shared_vtx(i, j) = skeleton_b(i, 1);
                    }
                }
            }
            floydWarshall(graph, dist, parent);
        }

        const bool has_target_avatar_skin_weights = target_avatar_skinning_weights.size() > 0;

        // explode avatar mesh
        Eigen::MatrixXd new_skinning_weights;
        {
            Eigen::VectorXi index_map;
            explode_trimesh(avatar_v, avatar_f, nc_avatar_v, nc_avatar_f, index_map);
            if (has_target_avatar_skin_weights)
                new_skinning_weights = target_avatar_skinning_weights(Eigen::all, index_map);
        }
        

        Eigen::VectorXi eid;
        Eigen::VectorXd coord;
        Eigen::VectorXi max_bone;
        Eigen::MatrixXd skinny_avatar_v_debug;
        // first pass
        {
            const int N = nc_avatar_v.rows();
            Eigen::VectorXd dists(N);
            dists.setConstant(std::numeric_limits<double>::max());
            coord.setZero(N);
            eid = -Eigen::VectorXi::Ones(N);
            if (has_target_avatar_skin_weights)
                max_bone = -Eigen::VectorXi::Ones(N);
            for (int i = 0; i < N; i++)
            {
                Eigen::Index maxRow = -1;
                if (has_target_avatar_skin_weights)
                {
                    Eigen::Index maxCol;
                    const double max_skin_weight = new_skinning_weights.col(i).maxCoeff(&maxRow, &maxCol);
                    assert(maxCol == 0);
                    max_bone(i) = static_cast<int>(maxRow);
                }

                for (int e = 0; e < target_skeleton_b.rows(); e++)
                {
                    if (!has_target_avatar_skin_weights)
                    {
                        Eigen::Vector2d tmp1 = project_to_edge(target_skeleton_v.row(target_skeleton_b(e, 0)), target_skeleton_v.row(target_skeleton_b(e, 1)), nc_avatar_v.row(i));
                        if (tmp1(0) < dists(i))
                        {
                            dists(i) = tmp1(0);
                            {
                                Eigen::Vector2d tmp2 = project_to_line(target_skeleton_v.row(target_skeleton_b(e, 0)), target_skeleton_v.row(target_skeleton_b(e, 1)), nc_avatar_v.row(i));
                                // Always clamp to the segment
                                tmp2(1) = std::min(1.0, std::max(0.0, tmp2(1)));
                                coord(i) = tmp2(1);
                            }
                            eid(i) = e;
                        }
                    }
                    else if (target_skeleton_b(e, 0) == maxRow || target_skeleton_b(e, 1) == maxRow)
                    {
                        Eigen::Vector2d tmp1 = project_to_edge(target_skeleton_v.row(target_skeleton_b(e, 0)), target_skeleton_v.row(target_skeleton_b(e, 1)), nc_avatar_v.row(i));
                        if (tmp1(0) < dists(i))
                        {
                            dists(i) = tmp1(0);
                            {
                                Eigen::Vector2d tmp2 = project_to_line(target_skeleton_v.row(target_skeleton_b(e, 0)), target_skeleton_v.row(target_skeleton_b(e, 1)), nc_avatar_v.row(i));
                                // if (!is_end_node(target_skeleton_b, target_skeleton_b(e, 0)))
                                //     tmp2(1) = std::max(0., tmp2(1));
                                // if (!is_end_node(target_skeleton_b, target_skeleton_b(e, 1)))
                                //     tmp2(1) = std::min(1., tmp2(1));
                                // Always clamp to the segment
                                tmp2(1) = std::min(1.0, std::max(0.0, tmp2(1)));
                                coord(i) = tmp2(1);
                            }
                            eid(i) = e;
                        }
                    }
                }
                if (eid(i) < 0)
                    log_and_throw_error("Failed to project vertex to the bone!");
            }

            skinny_avatar_v.setZero(nc_avatar_v.rows(), nc_avatar_v.cols());
            for (int i = 0; i < nc_avatar_v.rows(); i++)
                skinny_avatar_v.row(i) += coord(i) * (skeleton_v(skeleton_b(eid(i), 1), Eigen::all) - skeleton_v(skeleton_b(eid(i), 0), Eigen::all)) + skeleton_v(skeleton_b(eid(i), 0), Eigen::all);

            skinny_avatar_v_debug.setZero(nc_avatar_v.rows(), nc_avatar_v.cols());
            for (int i = 0; i < nc_avatar_v.rows(); i++)
                skinny_avatar_v_debug.row(i) += coord(i) * (target_skeleton_v(skeleton_b(eid(i), 1), Eigen::all) - target_skeleton_v(skeleton_b(eid(i), 0), Eigen::all)) + target_skeleton_v(skeleton_b(eid(i), 0), Eigen::all);

            skinny_avatar_f = nc_avatar_f;
        }

        // Debug visualization: color exploded avatar vertices by chosen bone (from skin weights)
        if (has_target_avatar_skin_weights)
        {
            const std::string ply_path = out_folder + "/projected_avatar_bone_colors.ply";

            std::ofstream ply(ply_path, std::ios::out);
            if (ply.is_open())
            {
                // Write ASCII PLY with per-vertex uchar RGB colors
                ply << "ply\n";
                ply << "format ascii 1.0\n";
                ply << "element vertex " << nc_avatar_v.rows() << "\n";
                ply << "property float x\n";
                ply << "property float y\n";
                ply << "property float z\n";
                ply << "property uchar red\n";
                ply << "property uchar green\n";
                ply << "property uchar blue\n";
                ply << "element face " << nc_avatar_f.rows() << "\n";
                ply << "property list uchar int vertex_indices\n";
                ply << "end_header\n";

                auto id_to_rgb = [](int id) -> std::array<unsigned char,3> {
                    // Simple hash-based color; stable across runs
                    unsigned int h = static_cast<unsigned int>(id) * 2654435761u;
                    unsigned char r = static_cast<unsigned char>((h      ) & 0xFF);
                    unsigned char g = static_cast<unsigned char>((h >> 8 ) & 0xFF);
                    unsigned char b = static_cast<unsigned char>((h >> 16) & 0xFF);
                    // Ensure not too dark
                    r = static_cast<unsigned char>(128 + (r >> 1));
                    g = static_cast<unsigned char>(128 + (g >> 1));
                    b = static_cast<unsigned char>(128 + (b >> 1));
                    return std::array<unsigned char,3>{{r, g, b}};
                };

                for (int i = 0; i < nc_avatar_v.rows(); ++i)
                {
                    const auto rgb = id_to_rgb(std::max(0, max_bone(i)));
                    ply << static_cast<float>(nc_avatar_v(i,0)) << " "
                        << static_cast<float>(nc_avatar_v(i,1)) << " "
                        << static_cast<float>(nc_avatar_v(i,2)) << " "
                        << static_cast<int>(rgb[0]) << " "
                        << static_cast<int>(rgb[1]) << " "
                        << static_cast<int>(rgb[2]) << "\n";
                }

                for (int i = 0; i < nc_avatar_f.rows(); ++i)
                {
                    ply << 3 << " "
                        << nc_avatar_f(i,0) << " "
                        << nc_avatar_f(i,1) << " "
                        << nc_avatar_f(i,2) << "\n";
                }

                ply.close();
                logger().debug("Wrote PLY with bone colors to {}", ply_path);
            }
            else
            {
                logger().warn("Failed to open {} for writing bone-colored PLY", ply_path);
            }
        }

        // igl::write_triangle_mesh(out_folder + "/avatar_old.obj", nc_avatar_v, nc_avatar_f);
        // igl::write_triangle_mesh(out_folder + "/projected_avatar_old_source.obj", skinny_avatar_v, nc_avatar_f);
        // igl::write_triangle_mesh(out_folder + "/projected_avatar_old_target.obj", skinny_avatar_v_debug, nc_avatar_f);

        // iteratively reduce distance
        int n_op = 0;
        int save_iter = 0;
        for (int iter = 0; iter < 10; iter++) {
            const int n_faces = skinny_avatar_f.rows();
            std::vector<Eigen::Matrix<double, 6, 3>> new_faces;
            for (int f = 0; f < n_faces; f++)
            {
                int max_dist = 0;
                int max_dist_le = -1;
                for (int le = 0; le < 3; le++)
                {
                    const int a = f * 3 + le;
                    const int b = f * 3 + (le + 1) % 3;
                    const int c = f * 3 + (le + 2) % 3;

                    bool edge_overlap_with_skeleton = false;
                    for (int i = 0; i < target_skeleton_b.rows(); i++) 
                    {
                        Eigen::Vector3d vb = skinny_avatar_v.row(b);
                        Eigen::Vector3d va = skinny_avatar_v.row(a);
                        Eigen::Vector3d b0 = skeleton_v.row(target_skeleton_b(i, 0));
                        Eigen::Vector3d b1 = skeleton_v.row(target_skeleton_b(i, 1));

                        if (ipc::point_line_distance(vb, b0, b1) < 1e-4 * (b1 - b0).squaredNorm() &&
                            ipc::point_line_distance(va, b0, b1) < 1e-4 * (b1 - b0).squaredNorm())
                        {
                            edge_overlap_with_skeleton = true;
                            break;
                        }
                    }

                    int source = eid(b);
                    int cur = eid(a);

                    if (edge_overlap_with_skeleton || source == cur)
                        continue;
                    
                    if (dist(cur, source) > max_dist)
                    {
                        max_dist = dist(cur, source);
                        max_dist_le = le;
                    }
                }

                for (int le = 0; le < 3; le++)
                {
                    if (max_dist_le != le)
                        continue;
                    
                    const int a = f * 3 + le;
                    const int b = f * 3 + (le + 1) % 3;
                    const int c = f * 3 + (le + 2) % 3;

                    int source = eid(b);
                    int cur = eid(a);

                    std::vector<std::array<int, 2>> inserted_tmp;
                    while (cur != source)
                    {
                        std::array<int, 2> tmp{{shared_vtx(cur, parent(source, cur)), cur}};
                        inserted_tmp.push_back(tmp);
                        cur = parent(source, cur);
                    }

                    Eigen::RowVector3d p_prev = skinny_avatar_v.row(a);
                    for (int k = 0; k < inserted_tmp.size(); k++)
                    {
                        Eigen::RowVector3d p = skeleton_v.row(inserted_tmp[k][0]);

                        Eigen::Matrix<double, 6, 3> X;
                        X << p_prev, p, skinny_avatar_v.row(c),
                             nc_avatar_v.row(a) + (nc_avatar_v.row(b) - nc_avatar_v.row(a)) * ((double)k / (inserted_tmp.size() + 1)),
                             nc_avatar_v.row(a) + (nc_avatar_v.row(b) - nc_avatar_v.row(a)) * ((double)(k+1) / (inserted_tmp.size() + 1)),
                             nc_avatar_v.row(c);
                        new_faces.push_back(X);
                        eid.conservativeResize(eid.size() + 3);
                        eid.tail(3) << (int)((k == 0) ? eid(a) : inserted_tmp[k-1][1]), inserted_tmp[k][1], eid(c);

                        p_prev = p;
                    }

                    skinny_avatar_v.row(a) = p_prev;
                    nc_avatar_v.row(a) += (nc_avatar_v.row(b) - nc_avatar_v.row(a)) * ((double)inserted_tmp.size() / (inserted_tmp.size() + 1));
                    eid(a) = inserted_tmp.back()[1];
                }

                n_op++;

                if (f == n_faces - 1 || new_faces.size() > 5)
                {
                    {
                        Eigen::MatrixXd tmp(skinny_avatar_v.rows() + new_faces.size() * 3, 3);
                        tmp.topRows(skinny_avatar_v.rows()) = skinny_avatar_v;
                        for (int i = 0; i < new_faces.size(); i++)
                            tmp.block(skinny_avatar_v.rows() + 3 * i, 0, 3, 3) = new_faces[i].topRows(3);
                        std::swap(skinny_avatar_v, tmp);
                    }
        
                    {
                        Eigen::MatrixXd tmp(nc_avatar_v.rows() + new_faces.size() * 3, 3);
                        tmp.topRows(nc_avatar_v.rows()) = nc_avatar_v;
                        for (int i = 0; i < new_faces.size(); i++)
                            tmp.block(nc_avatar_v.rows() + 3 * i, 0, 3, 3) = new_faces[i].bottomRows(3);
                        std::swap(nc_avatar_v, tmp);
                    }
        
                    skinny_avatar_f = Eigen::VectorXi::LinSpaced(skinny_avatar_v.rows(), 0, skinny_avatar_v.rows() - 1).reshaped(3, skinny_avatar_v.rows() / 3).transpose();
                    nc_avatar_f = skinny_avatar_f;
                
                    // igl::write_triangle_mesh(out_folder + "/projected_avatar_new_" + std::to_string(save_iter) + ".obj", skinny_avatar_v, skinny_avatar_f);
                    // igl::write_triangle_mesh(out_folder + "/avatar_new_" + std::to_string(save_iter) + ".obj", nc_avatar_v, nc_avatar_f);
                    // save_iter++;
                    
                    new_faces.clear();
                }
            }

            if (n_faces == nc_avatar_f.rows())
                break;
        }

        {
            Eigen::MatrixXd tmp_v(nc_avatar_v.rows(), nc_avatar_v.cols() + skinny_avatar_v.cols());
            tmp_v << nc_avatar_v, skinny_avatar_v;
            const auto [svi, svj] = remove_duplicate_vertices(tmp_v, nc_avatar_f, 1e-10);

            nc_avatar_v = tmp_v.template leftCols<3>();
            skinny_avatar_v = tmp_v.template rightCols<3>();
            skinny_avatar_f = nc_avatar_f;
        }
        
        skinny_avatar_v += (nc_avatar_v - skinny_avatar_v) * 1e-2;
    }

    void GarmentSolver::normalize_meshes(const json &in_args)
    {
        // Reset output restoration state
        restore_output_translation_ = false;
        target_output_translation_offset_.setZero();
        source_output_translation_offset_.setZero();
        target_output_scale_ = 1.0;
        source_output_scale_ = 1.0;
        normalization_scale_ = 1.0;

        std::string mode = "current";
        bool restore_out_translation = false;
        bool save_offsets = false;
        std::string save_offsets_path = "";
        std::string source_scale_path = "";
        std::string target_scale_path = "";
        // A0 scale/units safeguard (new modes only)
        bool scale_guard_enabled = false;
        double scale_guard_canonical = 1.65;
        double scale_guard_range_min = 1.3;
        double scale_guard_range_max = 2.8;
        double scale_guard_max_ratio = 1.35;
        std::string scale_guard_apply_when = "both_outside";
        bool scale_guard_error_on_ratio = true;
        if (in_args.contains("normalization"))
        {
            const json &norm = in_args["normalization"];
            if (norm.is_object())
            {
                mode = norm.value("mode", mode);
                restore_out_translation = norm.value("restore_output_translation", restore_out_translation);
                save_offsets = norm.value("save_offsets", save_offsets);
                save_offsets_path = norm.value("save_offsets_path", save_offsets_path);
                source_scale_path = norm.value("source_scale_path", source_scale_path);
                target_scale_path = norm.value("target_scale_path", target_scale_path);
                if (norm.contains("scale_guard") && norm["scale_guard"].is_object())
                {
                    const json &sg = norm["scale_guard"];
                    scale_guard_enabled = sg.value("enabled", scale_guard_enabled);
                    scale_guard_canonical = sg.value("canonical", scale_guard_canonical);
                    scale_guard_range_min = sg.value("range_min", scale_guard_range_min);
                    scale_guard_range_max = sg.value("range_max", scale_guard_range_max);
                    scale_guard_max_ratio = sg.value("max_ratio", scale_guard_max_ratio);
                    scale_guard_apply_when = sg.value("apply_when", scale_guard_apply_when);
                    scale_guard_error_on_ratio = sg.value("error_on_ratio_exceed", scale_guard_error_on_ratio);
                }
            }
        }

        auto maybe_write_offsets = [&](const std::string &mode_str) {
            if (!save_offsets)
                return;
            if (mode_str != "shared_translation_no_scale"
                && mode_str != "separate_translation_no_scale"
                && mode_str != "separate_translation_with_subject_scale")
                return;

            std::string path = out_folder + "/normalization_offsets.json";
            if (!save_offsets_path.empty())
                path = save_offsets_path;

            const Eigen::RowVector3d src = source_output_translation_offset_;
            const Eigen::RowVector3d tgt = target_output_translation_offset_;
            const double s = normalization_scale_;
            const double inv_s = (s == 0.0 ? 0.0 : (1.0 / s));
            const double s_src = source_output_scale_;
            const double s_tgt = target_output_scale_;

            json j;
            j["mode"] = mode_str;
            j["source_offset"] = {src(0), src(1), src(2)};
            j["target_offset"] = {tgt(0), tgt(1), tgt(2)};
            j["applied_scale"] = s;
            j["inverse_scale"] = inv_s;
            if (mode_str == "separate_translation_with_subject_scale")
            {
                j["source_scale"] = s_src;
                j["target_scale"] = s_tgt;

                auto mat_row_major = [](const Eigen::Matrix4d &M) {
                    std::vector<double> out(16);
                    for (int r = 0; r < 4; ++r)
                        for (int c = 0; c < 4; ++c)
                            out[r * 4 + c] = M(r, c);
                    return out;
                };

                auto make_to_cf = [](const Eigen::RowVector3d &off, const double sc) {
                    const double inv = (sc == 0.0 ? 0.0 : (1.0 / sc));
                    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
                    M(0, 0) = inv; M(1, 1) = inv; M(2, 2) = inv;
                    M(0, 3) = -off(0) * inv;
                    M(1, 3) = -off(1) * inv;
                    M(2, 3) = -off(2) * inv;
                    return M;
                };

                auto make_cf_to = [](const Eigen::RowVector3d &off, const double sc) {
                    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
                    M(0, 0) = sc; M(1, 1) = sc; M(2, 2) = sc;
                    M(0, 3) = off(0);
                    M(1, 3) = off(1);
                    M(2, 3) = off(2);
                    return M;
                };

                const Eigen::Matrix4d source_to_cf = make_to_cf(src, s_src);
                const Eigen::Matrix4d target_to_cf = make_to_cf(tgt, s_tgt);
                const Eigen::Matrix4d cf_to_source = make_cf_to(src, s_src);
                const Eigen::Matrix4d cf_to_target = make_cf_to(tgt, s_tgt);
                j["source_to_cf_row_major"] = mat_row_major(source_to_cf);
                j["target_to_cf_row_major"] = mat_row_major(target_to_cf);
                j["cf_to_source_row_major"] = mat_row_major(cf_to_source);
                j["cf_to_target_row_major"] = mat_row_major(cf_to_target);
            }
            j["input_to_optimization"] = {
                {"source", {{"translation", {-src(0), -src(1), -src(2)}}, {"scale", s}}},
                {"target", {{"translation", {-tgt(0), -tgt(1), -tgt(2)}}, {"scale", s}}}
            };
            j["optimization_to_input"] = {
                {"source", {{"scale", inv_s}, {"translation", {src(0), src(1), src(2)}}}},
                {"target", {{"scale", inv_s}, {"translation", {tgt(0), tgt(1), tgt(2)}}}}
            };

            std::ofstream out(path);
            if (out.is_open())
            {
                out << j.dump(2) << std::endl;
                out.close();
                logger().info("Wrote normalization offsets to {}", path);
            }
            else
            {
                logger().warn("Failed to write normalization offsets to {}", path);
            }
        };

        // Diagnostics (bbox sizes are translation-invariant; useful to confirm scaling/no-scaling)
        const Eigen::Vector3d skel_bbox0 = bbox_size(skeleton_v);
        const Eigen::Vector3d garment_bbox0 = bbox_size(garment.v);
        const Eigen::Vector3d avatar_bbox0 = bbox_size(avatar_v);
        const Eigen::Vector3d target_skel_bbox0 = bbox_size(target_skeleton_v);
        logger().info("[normalize] mode={} restore_output_translation={} (pre) bbox: skel={} garment={} avatar={} target_skel={}",
            mode, restore_out_translation, skel_bbox0.transpose(), garment_bbox0.transpose(), avatar_bbox0.transpose(), target_skel_bbox0.transpose());

        if (mode == "none")
        {
            logger().info("[normalize] disabled (mode=none).");
            return;
        }

        if (mode == "shared_translation_no_scale")
        {
            // Subtract a single global offset from ALL involved meshes/skeletons (no scaling).
            // This keeps relative placement, but improves numeric conditioning if desired.
            const Eigen::RowVector3d ref = have_orig_avatar_centroid ? orig_avatar_centroid : avatar_v.colwise().mean();

            skeleton_v.rowwise() -= ref;
            garment.v.rowwise() -= ref;
            avatar_v.rowwise() -= ref;
            target_skeleton_v.rowwise() -= ref;

            // Always record the applied offsets for downstream use
            target_output_translation_offset_ = ref;
            source_output_translation_offset_ = ref;

            // Optional A0: common rescale to canonical units (new modes only)
            if (scale_guard_enabled)
            {
                const double h_src = bbox_size(skeleton_v).maxCoeff();
                const double h_tgt = bbox_size(target_skeleton_v).maxCoeff();
                const double h_min = std::min(h_src, h_tgt);
                const double h_max = std::max(h_src, h_tgt);
                const double ratio = (h_min > 0.0) ? (h_max / h_min) : std::numeric_limits<double>::infinity();
                if (ratio > scale_guard_max_ratio)
                {
                    if (scale_guard_error_on_ratio)
                        log_and_throw_error("A0 scale_guard: relative scale mismatch too large: h_src={} h_tgt={} ratio={} > {}", h_src, h_tgt, ratio, scale_guard_max_ratio);
                    else
                        logger().warn("A0 scale_guard: relative scale mismatch too large: h_src={} h_tgt={} ratio={} > {} (continuing)", h_src, h_tgt, ratio, scale_guard_max_ratio);
                }

                const bool src_outside = (h_src < scale_guard_range_min || h_src > scale_guard_range_max);
                const bool tgt_outside = (h_tgt < scale_guard_range_min || h_tgt > scale_guard_range_max);
                const bool apply = (scale_guard_apply_when == "both_outside") ? (src_outside && tgt_outside) : false;
                if (apply)
                {
                    normalization_scale_ = scale_guard_canonical / h_min;
                    skeleton_v *= normalization_scale_;
                    garment.v *= normalization_scale_;
                    avatar_v *= normalization_scale_;
                    target_skeleton_v *= normalization_scale_;
                    logger().info("[A0] scale_guard applied: h_src={} h_tgt={} canonical={} scale={}", h_src, h_tgt, scale_guard_canonical, normalization_scale_);
                }
                else
                {
                    logger().info("[A0] scale_guard not applied: h_src={} h_tgt={} outside=({}, {}) range=[{},{}]",
                        h_src, h_tgt, src_outside, tgt_outside, scale_guard_range_min, scale_guard_range_max);
                }
            }

            if (restore_out_translation)
                restore_output_translation_ = true;

            // Save offsets immediately at start of pipeline
            maybe_write_offsets(mode);

            logger().info("[normalize] shared_translation_no_scale ref={} restore_offset={}",
                ref, restore_output_translation_ ? ref : Eigen::RowVector3d::Zero());
            logger().info("[normalize] (post) bbox: skel={} garment={} avatar={} target_skel={}",
                bbox_size(skeleton_v).transpose(), bbox_size(garment.v).transpose(), bbox_size(avatar_v).transpose(), bbox_size(target_skeleton_v).transpose());
            return;
        }

        if (mode == "separate_translation_no_scale")
        {
            // Legacy-like translation behavior without scaling:
            // - Source side: center source skeleton + garment by source skeleton centroid
            // - Target side: center target avatar + target skeleton by target avatar centroid
            const Eigen::RowVector3d source_ref = skeleton_v.colwise().sum() / skeleton_v.rows();
            skeleton_v.rowwise() -= source_ref;
            garment.v.rowwise() -= source_ref;

            const Eigen::RowVector3d target_ref = have_orig_avatar_centroid ? orig_avatar_centroid : avatar_v.colwise().mean();
            avatar_v.rowwise() -= target_ref;
            target_skeleton_v.rowwise() -= target_ref;

            // Always record the applied offsets for downstream use
            target_output_translation_offset_ = target_ref;
            source_output_translation_offset_ = source_ref;

            // Optional A0: common rescale to canonical units (new modes only)
            if (scale_guard_enabled)
            {
                const double h_src = bbox_size(skeleton_v).maxCoeff();
                const double h_tgt = bbox_size(target_skeleton_v).maxCoeff();
                const double h_min = std::min(h_src, h_tgt);
                const double h_max = std::max(h_src, h_tgt);
                const double ratio = (h_min > 0.0) ? (h_max / h_min) : std::numeric_limits<double>::infinity();
                if (ratio > scale_guard_max_ratio)
                {
                    if (scale_guard_error_on_ratio)
                        log_and_throw_error("A0 scale_guard: relative scale mismatch too large: h_src={} h_tgt={} ratio={} > {}", h_src, h_tgt, ratio, scale_guard_max_ratio);
                    else
                        logger().warn("A0 scale_guard: relative scale mismatch too large: h_src={} h_tgt={} ratio={} > {} (continuing)", h_src, h_tgt, ratio, scale_guard_max_ratio);
                }

                const bool src_outside = (h_src < scale_guard_range_min || h_src > scale_guard_range_max);
                const bool tgt_outside = (h_tgt < scale_guard_range_min || h_tgt > scale_guard_range_max);
                const bool apply = (scale_guard_apply_when == "both_outside") ? (src_outside && tgt_outside) : false;
                if (apply)
                {
                    normalization_scale_ = scale_guard_canonical / h_min;
                    skeleton_v *= normalization_scale_;
                    garment.v *= normalization_scale_;
                    avatar_v *= normalization_scale_;
                    target_skeleton_v *= normalization_scale_;
                    logger().info("[A0] scale_guard applied: h_src={} h_tgt={} canonical={} scale={}", h_src, h_tgt, scale_guard_canonical, normalization_scale_);
                }
                else
                {
                    logger().info("[A0] scale_guard not applied: h_src={} h_tgt={} outside=({}, {}) range=[{},{}]",
                        h_src, h_tgt, src_outside, tgt_outside, scale_guard_range_min, scale_guard_range_max);
                }
            }

            if (restore_out_translation)
                restore_output_translation_ = true;

            // Save offsets immediately at start of pipeline
            maybe_write_offsets(mode);

            logger().info("[normalize] separate_translation_no_scale source_ref={} target_ref={} restore_offset={}",
                source_ref, target_ref,
                restore_output_translation_ ? target_output_translation_offset_ : Eigen::RowVector3d::Zero());
            logger().info("[normalize] (post) bbox: skel={} garment={} avatar={} target_skel={}",
                bbox_size(skeleton_v).transpose(), bbox_size(garment.v).transpose(), bbox_size(avatar_v).transpose(), bbox_size(target_skeleton_v).transpose());
            return;
        }

        if (mode == "separate_translation_with_subject_scale")
        {
            // Separate translations like separate_translation_no_scale, but also cancel per-side subject scale.
            // Optimization runs in:
            //   source: (x - source_ref) / source_scale
            //   target: (x - target_ref) / target_scale
            //
            // Outputs are expected to be restored back to the original world scale/translation at write time.

            auto load_scale_from_path = [&](const std::string &path) -> double {
                if (path.empty())
                    log_and_throw_error("[normalize] mode={} requires normalization.source_scale_path and normalization.target_scale_path (got empty path).", mode);
                if (!std::filesystem::exists(path))
                    log_and_throw_error("[normalize] scale JSON not found: {}", path);
                std::ifstream in(path);
                if (!in.is_open())
                    log_and_throw_error("[normalize] cannot open scale JSON: {}", path);
                json sj;
                in >> sj;
                if (!sj.is_object())
                    log_and_throw_error("[normalize] scale JSON must be an object: {}", path);

                double s_val = std::numeric_limits<double>::quiet_NaN();
                if (sj.contains("scale"))
                    s_val = sj["scale"].get<double>();
                else if (sj.contains("inv_scale"))
                {
                    const double inv = sj["inv_scale"].get<double>();
                    s_val = (inv == 0.0) ? std::numeric_limits<double>::infinity() : (1.0 / inv);
                }
                else if (sj.contains("inverse_scale"))
                {
                    const double inv = sj["inverse_scale"].get<double>();
                    s_val = (inv == 0.0) ? std::numeric_limits<double>::infinity() : (1.0 / inv);
                }
                else
                    log_and_throw_error("[normalize] scale JSON missing 'scale' or 'inv_scale'/'inverse_scale': {}", path);

                if (!std::isfinite(s_val) || s_val <= 0.0)
                    log_and_throw_error("[normalize] invalid scale {} loaded from {}", s_val, path);
                return s_val;
            };

            const double source_scale = load_scale_from_path(source_scale_path);
            const double target_scale = load_scale_from_path(target_scale_path);

            const Eigen::RowVector3d source_ref = skeleton_v.colwise().sum() / skeleton_v.rows();
            skeleton_v.rowwise() -= source_ref;
            garment.v.rowwise() -= source_ref;
            skeleton_v *= (1.0 / source_scale);
            garment.v *= (1.0 / source_scale);

            const Eigen::RowVector3d target_ref = have_orig_avatar_centroid ? orig_avatar_centroid : avatar_v.colwise().mean();
            avatar_v.rowwise() -= target_ref;
            target_skeleton_v.rowwise() -= target_ref;
            avatar_v *= (1.0 / target_scale);
            target_skeleton_v *= (1.0 / target_scale);

            // Record offsets/scales for downstream restoration and tooling.
            target_output_translation_offset_ = target_ref;
            source_output_translation_offset_ = source_ref;
            source_output_scale_ = source_scale;
            target_output_scale_ = target_scale;

            // Optional A0: common rescale to canonical units (uniform, after per-side scaling)
            if (scale_guard_enabled)
            {
                const double h_src = bbox_size(skeleton_v).maxCoeff();
                const double h_tgt = bbox_size(target_skeleton_v).maxCoeff();
                const double h_min = std::min(h_src, h_tgt);
                const double h_max = std::max(h_src, h_tgt);
                const double ratio = (h_min > 0.0) ? (h_max / h_min) : std::numeric_limits<double>::infinity();
                if (ratio > scale_guard_max_ratio)
                {
                    if (scale_guard_error_on_ratio)
                        log_and_throw_error("A0 scale_guard: relative scale mismatch too large: h_src={} h_tgt={} ratio={} > {}", h_src, h_tgt, ratio, scale_guard_max_ratio);
                    else
                        logger().warn("A0 scale_guard: relative scale mismatch too large: h_src={} h_tgt={} ratio={} > {} (continuing)", h_src, h_tgt, ratio, scale_guard_max_ratio);
                }

                const bool src_outside = (h_src < scale_guard_range_min || h_src > scale_guard_range_max);
                const bool tgt_outside = (h_tgt < scale_guard_range_min || h_tgt > scale_guard_range_max);
                const bool apply = (scale_guard_apply_when == "both_outside") ? (src_outside && tgt_outside) : false;
                if (apply)
                {
                    normalization_scale_ = scale_guard_canonical / h_min;
                    skeleton_v *= normalization_scale_;
                    garment.v *= normalization_scale_;
                    avatar_v *= normalization_scale_;
                    target_skeleton_v *= normalization_scale_;
                    logger().info("[A0] scale_guard applied: h_src={} h_tgt={} canonical={} scale={}", h_src, h_tgt, scale_guard_canonical, normalization_scale_);
                }
                else
                {
                    logger().info("[A0] scale_guard not applied: h_src={} h_tgt={} outside=({}, {}) range=[{},{}]",
                        h_src, h_tgt, src_outside, tgt_outside, scale_guard_range_min, scale_guard_range_max);
                }
            }

            if (restore_out_translation)
                restore_output_translation_ = true;

            maybe_write_offsets(mode);

            logger().info("[normalize] separate_translation_with_subject_scale source_ref={} target_ref={} source_scale={} target_scale={} restore_offset={}",
                source_ref, target_ref, source_scale, target_scale,
                restore_output_translation_ ? target_output_translation_offset_ : Eigen::RowVector3d::Zero());
            logger().info("[normalize] (post) bbox: skel={} garment={} avatar={} target_skel={}",
                bbox_size(skeleton_v).transpose(), bbox_size(garment.v).transpose(), bbox_size(avatar_v).transpose(), bbox_size(target_skeleton_v).transpose());
            return;
        }

        // Default / legacy behavior (kept for backward compatibility)
        // Center offset
        const Eigen::RowVector3d center_offset = skeleton_v.colwise().sum() / skeleton_v.rows();
        skeleton_v.rowwise() -= center_offset;
        garment.v.rowwise() -= center_offset;

        // Source side
        const double source_scaling = 2. / bbox_size(skeleton_v).maxCoeff();
        skeleton_v *= source_scaling;
        garment.v *= source_scaling;
        // skinny_avatar_v *= source_scaling;

        // Target side
        const double target_scaling = bbox_size(skeleton_v).maxCoeff() / bbox_size(target_skeleton_v).maxCoeff();
        const Eigen::RowVector3d avatar_mean_ref = have_orig_avatar_centroid ? orig_avatar_centroid : avatar_v.colwise().mean();
        const Eigen::Vector3d center = (skeleton_v.colwise().mean()).transpose() - target_scaling * avatar_mean_ref.transpose();
        Transformation<3> trans(target_scaling * Eigen::Matrix3d::Identity(), center);

        trans.apply(avatar_v);
        trans.apply(target_skeleton_v);

        logger().info("[normalize] legacy/current source_scaling={} target_scaling={} center_offset={} center={}",
            source_scaling, target_scaling, center_offset, center.transpose());
        logger().info("[normalize] (post) bbox: skel={} garment={} avatar={} target_skel={}",
            bbox_size(skeleton_v).transpose(), bbox_size(garment.v).transpose(), bbox_size(avatar_v).transpose(), bbox_size(target_skeleton_v).transpose());
    }

	json init(const json &p_args_in, const bool strict_validation)
	{
		json args_in = p_args_in; // mutable copy
        json args;

		utils::apply_common_params(args_in);

		// CHECK validity json
		json rules;
		jse::JSE jse;
		{
			jse.strict = strict_validation;
			const std::string polyfem_input_spec = POLYFEM_INPUT_SPEC;
			std::ifstream file(polyfem_input_spec);

			if (file.is_open())
				file >> rules;
			else
			{
				logger().error("unable to open {} rules", polyfem_input_spec);
				throw std::runtime_error("Invald spec file");
			}

			jse.include_directories.push_back(POLYFEM_JSON_SPEC_DIR);
			jse.include_directories.push_back(POLYSOLVE_JSON_SPEC_DIR);
			rules = jse.inject_include(rules);

			polysolve::linear::Solver::apply_default_solver(rules, "/solver/linear");
		}

		polysolve::linear::Solver::select_valid_solver(args_in["solver"]["linear"], logger());

		// Use the /solver/nonlinear settings as the default for /solver/augmented_lagrangian/nonlinear
		if (args_in.contains("/solver/nonlinear"_json_pointer))
		{
			if (args_in.contains("/solver/augmented_lagrangian/nonlinear"_json_pointer))
			{
				assert(args_in["solver"]["augmented_lagrangian"]["nonlinear"].is_object());
				// Merge the augmented lagrangian settings into the nonlinear settings,
				// and then replace the augmented lagrangian settings with the merged settings.
				json nonlinear = args_in["solver"]["nonlinear"]; // copy
				nonlinear.merge_patch(args_in["solver"]["augmented_lagrangian"]["nonlinear"]);
				args_in["solver"]["augmented_lagrangian"]["nonlinear"] = nonlinear;
			}
			else
			{
				// Copy the nonlinear settings to the augmented_lagrangian settings
				args_in["solver"]["augmented_lagrangian"]["nonlinear"] = args_in["solver"]["nonlinear"];
			}
		}

		const bool valid_input = jse.verify_json(args_in, rules);

		if (!valid_input)
		{
			logger().error("invalid input json:\n{}", jse.log2str());
			throw std::runtime_error("Invald input json file");
		}
		// end of check

		args = jse.inject_defaults(args_in, rules);

		// Save output directory and resolve output paths dynamically
		const std::string output_dir = utils::resolve_path(args["output"]["directory"], 
            utils::is_param_valid(args, "root_path") ? args["root_path"].get<std::string>() : "", 
            false);
        
		if (!output_dir.empty())
		{
			std::filesystem::create_directories(output_dir);
		}

        // set logger
        {
            spdlog::level::level_enum log_level = args["output"]["log"]["level"];
			const bool file_enabled = args["output"]["log"]["file_enabled"];
			const std::string file_path_in = args["output"]["log"]["file_path"];

			if (file_enabled)
			{
				std::filesystem::path log_path;
				if (file_path_in.empty())
				{
					log_path = std::filesystem::path(output_dir) / "run.log";
				}
				else
				{
					log_path = utils::resolve_path(file_path_in, output_dir, false);
				}

				if (!log_path.empty())
				{
					std::error_code ec;
					std::filesystem::create_directories(std::filesystem::path(log_path).parent_path(), ec);

					auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
					auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path.string(), true);
					auto combined_logger = std::make_shared<spdlog::logger>(
						"polyfem-run",
						spdlog::sinks_init_list{stdout_sink, file_sink});

					polyfem::set_logger(combined_logger);
					polyfem::logger().info("Logging to file {}", log_path.string());
				}
			}

            spdlog::set_level(log_level);
			logger().set_level(log_level);
			ipc::logger().set_level(log_level);
            
            spdlog::flush_every(std::chrono::seconds(3));
        }

		logger().info("Saving output to {}", output_dir);

		const unsigned int thread_in = args["solver"]["max_threads"];
		utils::NThread::get().set_num_threads(thread_in);

        return args;
	}
}