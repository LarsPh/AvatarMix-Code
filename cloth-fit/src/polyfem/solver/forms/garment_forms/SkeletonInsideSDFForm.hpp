#pragma once

#include <polyfem/solver/forms/Form.hpp>

#include <polyfem/Common.hpp>
#include <polyfem/utils/Types.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

#include <vector>

namespace polyfem::solver
{
    class SkeletonInsideSDFForm : public Form
    {
    public:
        SkeletonInsideSDFForm(
            const Eigen::MatrixXd &skelV,
            const Eigen::MatrixXi &skelE,
            const Eigen::MatrixXd &surfaceV,
            const Eigen::MatrixXi &surfaceF,
            const double voxel_size,
            const int samples_per_bone,
            const double inside_margin,
            const bool flood_fill_sign,
            const int close_holes_voxels,
            const bool cap_open_boundaries);

        virtual ~SkeletonInsideSDFForm() = default;

        std::string name() const override { return "skeleton-inside-sdf"; }

    protected:
        double value_unweighted(const Eigen::VectorXd &x) const override;
        void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;
        void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

    private:
        struct Sample { int vi; int vj; double t; };

        Eigen::MatrixXd V0_;
        Eigen::MatrixXi E_;
        std::vector<Sample> samples_;

        openvdb::DoubleGrid::Ptr grid_;
        double voxel_size_;
        int samples_per_bone_;
        double inside_margin_;
        bool flood_fill_sign_;
        int close_holes_voxels_;
        bool cap_open_boundaries_;

    public:
        // Count samples with (phi + margin) > 0 for the given absolute positions V
        int count_violations(const Eigen::MatrixXd &V) const;

        // Export sample points as PLY colored by inside/outside for visualization
        void export_samples_ply(
            const std::string &path,
            const Eigen::MatrixXd &V,
            const Eigen::RowVector3d &output_translation = Eigen::RowVector3d::Zero(),
            const double output_scale = 1.0) const;

        // Export combined skeleton (as edges) and samples into a single PLY
        void export_skeleton_and_samples_ply(
            const std::string &path,
            const Eigen::MatrixXd &V,
            const Eigen::MatrixXi &E,
            const Eigen::RowVector3d &output_translation = Eigen::RowVector3d::Zero(),
            const double output_scale = 1.0) const;

        // Export an SDF isosurface (e.g., iso=0 for surface, iso=-inside_margin for inner offset)
        void export_sdf_isosurface(
            const std::string &path,
            const double iso,
            const Eigen::RowVector3d &output_translation = Eigen::RowVector3d::Zero(),
            const double output_scale = 1.0) const;
    };
} // namespace polyfem::solver


