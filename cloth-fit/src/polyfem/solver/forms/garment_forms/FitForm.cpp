#include "FitForm.hpp"

#include <openvdb/openvdb.h>
#include <openvdb/math/Vec3.h>
#include <openvdb/tools/SignedFloodFill.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VolumeToMesh.h>

#include <polyfem/io/OBJWriter.hpp>
#include <polyfem/utils/MaybeParallelFor.hpp>
#include <polyfem/utils/Timer.hpp>
#include <polyfem/utils/Logger.hpp>

#include <ipc/utils/eigen_ext.hpp>

#include <algorithm>

using namespace openvdb;
using namespace polyfem::utils;

namespace {
    class LocalThreadMatStorage
    {
    public:
        Eigen::MatrixXd mat;
        // Eigen::MatrixXd samples;

        LocalThreadMatStorage(const int row, const int col)
        {
            mat.resize(row, col);
            mat.setZero();
        }
    };

    class LocalThreadSparseMatStorage
    {
    public:
        std::unique_ptr<MatrixCache> cache = nullptr;
        // Eigen::MatrixXd samples;

        LocalThreadSparseMatStorage() = delete;

        LocalThreadSparseMatStorage(const int buffer_size, const int rows, const int cols)
        {
            init(buffer_size, rows, cols);
        }

        LocalThreadSparseMatStorage(const int buffer_size, const MatrixCache &c)
        {
            init(buffer_size, c);
        }

        LocalThreadSparseMatStorage(const LocalThreadSparseMatStorage &other)
            : cache(other.cache->copy())
        {
        }

        LocalThreadSparseMatStorage &operator=(const LocalThreadSparseMatStorage &other)
        {
            assert(other.cache != nullptr);
            cache = other.cache->copy();
            return *this;
        }

        void init(const int buffer_size, const int rows, const int cols)
        {
            // assert(rows == cols);
            // cache = std::make_unique<DenseMatrixCache>();
            cache = std::make_unique<SparseMatrixCache>();
            cache->reserve(buffer_size);
            cache->init(rows, cols);
        }

        void init(const int buffer_size, const MatrixCache &c)
        {
            if (cache == nullptr)
                cache = c.copy();
            cache->reserve(buffer_size);
            cache->init(c);
        }
    };

    template <int N>
    constexpr Eigen::Matrix<double, ((N+1)*(N+2))/2, 4> upsample_standard()
    {
        constexpr int num = ((N+1)*(N+2))/2;

        Eigen::Matrix<double, num, 4> out;
        for (int i = 0, k = 0; i <= N; i++)
            for (int j = 0; i + j <= N; j++, k++)
            {
                std::array<int, 3> arr = {{i, j, N - i - j}};
                std::sort(arr.begin(), arr.end());

                double w = 6;
                if (arr[1] == 0)        // vertex node
                    w = 1;
                else if (arr[0] == 0)   // edge node
                    w = 3;
                else                    // face node
                    w = 6;
                
                out.row(k) << i, j, N - i - j, w;
            }
        
        out.template leftCols<3>() /= N;
        out.col(3) /= N * N;
        return out;
    }
}

namespace polyfem::solver
{
	template <int n_refs>
	double FitForm<n_refs>::gap_percentile(const double q, const double d0) const
	{
		if (q < 0.0 || q > 1.0)
			log_and_throw_error("FitForm::gap_percentile: q must be in [0,1], got {}", q);

		std::vector<double> gaps;
		gaps.reserve(fit_faces_ids.size() * P.rows());

		for (int f : fit_faces_ids)
		{
			for (int i = 0; i < P.rows(); ++i)
			{
				const double phi = totalP[f * n_loc_samples + i].x;
				const double d = std::max(0.0, phi - d0);
				gaps.push_back(d);
			}
		}

		if (gaps.empty())
			return 0.0;

		const size_t k = static_cast<size_t>(std::floor(q * (gaps.size() - 1)));
		std::nth_element(gaps.begin(), gaps.begin() + k, gaps.end());
		return gaps[k];
	}

    template <int n_refs>
    FitForm<n_refs>::FitForm(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXd &surface_v, const Eigen::MatrixXi &surface_f, const double voxel_size, const std::vector<int> &not_fit_faces, const std::string out_dir, const Eigen::VectorXd &vertex_fit_multipliers) : V_(V), F_(F), voxel_size_(voxel_size), totalP(std::vector<openvdb::tools::HessType<double>>(F_.rows() * n_loc_samples, openvdb::tools::HessType<double>(0.)))
    {
        math::Transform::Ptr xform = math::Transform::createLinearTransform(voxel_size);

        std::vector<Vec3s> points;
        std::vector<Vec3I> triangles;
        std::vector<Vec4I> quads;

        points.reserve(surface_v.rows());
        for (int i = 0; i < surface_v.rows(); i++)
            points.push_back(Vec3s(surface_v(i, 0), surface_v(i, 1), surface_v(i, 2)));
        
        for (int i = 0; i < surface_f.rows(); i++)
            triangles.push_back(Vec3I(surface_f(i, 0), surface_f(i, 1), surface_f(i, 2)));

        grid = tools::meshToSignedDistanceField<DoubleGrid>(*xform, points, triangles, quads, 150, 1);
        
        // build upsampling scheme on the garment mesh
        {
            Eigen::Matrix<double, n_loc_samples, 4> tmp = upsample_standard<n_refs>();
            P = tmp.template leftCols<3>();
            weights = tmp.col(3);
        }

        {
            for (int i = 0; i < F.rows(); i++)
            {
                if (std::find(not_fit_faces.begin(), not_fit_faces.end(), i) == not_fit_faces.end())
                    fit_faces_ids.push_back(i);
            }
        }

        io::OBJWriter::write(
            out_dir + "/fit-faces-debug.obj", V, Eigen::MatrixXi(), F(fit_faces_ids, Eigen::all));

        // {
            initial_distance.setZero(F_.rows() * n_loc_samples);
        //     typename DoubleGrid::ConstAccessor acc = grid->getConstAccessor();
        //     for (int f = 0; f < F_.rows(); f++) {
        //         const Eigen::Matrix3d M = V_({F_(f, 0),F_(f, 1),F_(f, 2)}, Eigen::all);
        //         Eigen::Matrix<double, n_loc_samples, 3> samples = P * M;
        //         for (int i = 0; i < P.rows(); i++) {
        //             math::Vec3<double> p(samples(i, 0), samples(i, 1), samples(i, 2));
        //             initial_distance(f * n_loc_samples + i) = tools::SplineSampler::sample(acc, grid->transformPtr()->worldToIndex(p));
        //         }
        //     }
        // }

        // export SDF as a triangle mesh
        {
            tools::volumeToMesh(*grid, points, quads, 0.);
            Eigen::MatrixXd outpoints(points.size(), 3);
            Eigen::MatrixXi outtriangles(2 * quads.size(), 3);
            for (int i = 0; i < points.size(); i++)
            {
                outpoints.row(i) << points[i](0), points[i](1), points[i](2);
            }
            for (int i = 0; i < quads.size(); i++)
            {
                outtriangles.row(2 * i) << quads[i](0), quads[i](1), quads[i](2);
                outtriangles.row(2 * i + 1) << quads[i](0), quads[i](2), quads[i](3);
            }

			io::OBJWriter::write(
				out_dir + "/sdf.obj", outpoints, Eigen::MatrixXi(), outtriangles);
        }

        // Validate and store per-vertex multipliers (clamped to >= 0)
        if (vertex_fit_multipliers.size() != V_.rows())
            log_and_throw_error("FitForm: vertex_fit_multipliers wrong size (got {}, expected {}).", vertex_fit_multipliers.size(), V_.rows());
        vertex_fit_multipliers_ = vertex_fit_multipliers.cwiseMax(0.);

        // Logging: report whether per-vertex multipliers are active
        // int non_unit = 0;
        // for (int i = 0; i < vertex_fit_multipliers_.size(); ++i)
        //     if (std::abs(vertex_fit_multipliers_(i) - 1.0) > 1e-12)
        //         ++non_unit;
        // if (non_unit > 0)
        //     logger().info("FitForm: per-vertex fit multipliers enabled ({} non-unit of {}).", non_unit, int(vertex_fit_multipliers_.size()));
        // else
        //     logger().info("FitForm: per-vertex fit multipliers all 1 (feature inactive).");
    }

    template <int n_refs>
    double FitForm<n_refs>::value_unweighted(const Eigen::VectorXd &x) const 
    {
        const Eigen::MatrixXd V = unflatten(x, 3) + V_;

        double val = 0;
        double max_dist = 0;
        typename DoubleGrid::ConstAccessor acc = grid->getConstAccessor();
        for (int f : fit_faces_ids) {
            const double area = (V_.row(F_(f, 1)) - V_.row(F_(f, 0))).template head<3>().cross((V_.row(F_(f, 2)) - V_.row(F_(f, 0))).template head<3>()).norm() / 2;
            for (int i = 0; i < P.rows(); i++) {
                const double tmp = totalP[f * n_loc_samples + i].x;

                if (std::isnan(tmp))
                    log_and_throw_error("Invalid sdf values!");
                
                if (tmp > initial_distance(f * n_loc_samples + i))
                {
                    const double m = P(i, 0) * vertex_fit_multipliers_(F_(f, 0))
                                   + P(i, 1) * vertex_fit_multipliers_(F_(f, 1))
                                   + P(i, 2) * vertex_fit_multipliers_(F_(f, 2));
                    val += m * area * weights(i) * pow(tmp - initial_distance(f * n_loc_samples + i), power);
                }
            }
        }

        return val;
    }

    template <int n_refs>
    void FitForm<n_refs>::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const 
    {
        POLYFEM_SCOPED_TIMER("fit gradient");
        const Eigen::MatrixXd V = unflatten(x, 3) + V_;

        Eigen::MatrixXd g = Eigen::MatrixXd::Zero(V.rows(), V.cols());

        auto storage = create_thread_storage(LocalThreadMatStorage(g.rows(), g.cols()));

        maybe_parallel_for(fit_faces_ids.size(), [&](int start, int end, int thread_id) {
            LocalThreadMatStorage &local_storage = get_local_thread_storage(storage, thread_id);
            for (int f_aux = start; f_aux < end; f_aux++) {
                int f = fit_faces_ids[f_aux];
                const double area = (V_.row(F_(f, 1)) - V_.row(F_(f, 0))).template head<3>().cross((V_.row(F_(f, 2)) - V_.row(F_(f, 0))).template head<3>()).norm() / 2;
                for (int i = 0; i < P.rows(); i++) {
                    const auto &tmp = totalP[f * n_loc_samples + i];
                    
                    if (tmp.x > initial_distance(f * n_loc_samples + i))
                    {
                        const double m = P(i, 0) * vertex_fit_multipliers_(F_(f, 0))
                                       + P(i, 1) * vertex_fit_multipliers_(F_(f, 1))
                                       + P(i, 2) * vertex_fit_multipliers_(F_(f, 2));
                        for (int d = 0; d < 3; d++)
                            local_storage.mat(F_.row(f), d) += (m * pow(tmp.x - initial_distance(f * n_loc_samples + i), power-1) * power * area * weights(i) * tmp.g(d)) * P.row(i);
                    }
                }
            }
        });

		// Serially merge local storages
		for (const LocalThreadMatStorage &local_storage : storage)
			g += local_storage.mat;
        
        gradv = flatten(g);
    }

    template <int n_refs>
    void FitForm<n_refs>::solution_changed(const Eigen::VectorXd &new_x)
    {
        POLYFEM_SCOPED_TIMER("sample SDF");

        const Eigen::MatrixXd V = unflatten(new_x, 3) + V_;

        maybe_parallel_for(fit_faces_ids.size(), [&](int start, int end, int thread_id) {
            typename DoubleGrid::ConstAccessor acc = grid->getConstAccessor();
            for (int f_aux = start; f_aux < end; f_aux++) {
                int f = fit_faces_ids[f_aux];
                const Eigen::Matrix3d M = V({F_(f, 0),F_(f, 1),F_(f, 2)}, Eigen::all);
                Eigen::Matrix<double, n_loc_samples, 3> samples = P * M;
                const double area = (V_.row(F_(f, 1)) - V_.row(F_(f, 0))).template head<3>().cross((V_.row(F_(f, 2)) - V_.row(F_(f, 0))).template head<3>()).norm() / 2;
                for (int i = 0; i < P.rows(); i++) {
                    math::Vec3<double> p(samples(i, 0), samples(i, 1), samples(i, 2));
                    totalP[f * n_loc_samples + i] = tools::SplineSampler::sampleHessian(acc, grid->transformPtr()->worldToIndex(p));
                    auto &tmp = totalP[f * n_loc_samples + i];
                    tmp.g = tmp.g / voxel_size_;
                    tmp.h = tmp.h * (1. / voxel_size_ / voxel_size_);
                }
            }
        });
    }

    template <int n_refs>
    void FitForm<n_refs>::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const 
    {
        POLYFEM_SCOPED_TIMER("fit hessian");
        hessian.setZero();
        hessian.resize(x.size(), x.size());   
        if (!use_spline)
            return;
        
        const Eigen::MatrixXd V = unflatten(x, 3) + V_;

        auto storage = create_thread_storage(LocalThreadSparseMatStorage(long(1e7), hessian.rows(), hessian.cols()));

        igl::Timer timer;
        timer.start();
        maybe_parallel_for(fit_faces_ids.size(), [&](int start, int end, int thread_id) {
            LocalThreadSparseMatStorage &local_storage = get_local_thread_storage(storage, thread_id);
            Eigen::Matrix<double, 9, 9> local_hess;
            for (int f_aux = start; f_aux < end; f_aux++) {
                int f = fit_faces_ids[f_aux];
                const double area = (V_.row(F_(f, 1)) - V_.row(F_(f, 0))).template head<3>().cross((V_.row(F_(f, 2)) - V_.row(F_(f, 0))).template head<3>()).norm() / 2;
                local_hess.setZero();
                for (int i = 0; i < P.rows(); i++) {
                    const auto &tmp = totalP[f * n_loc_samples + i];

                    if (tmp.x <= initial_distance(f * n_loc_samples + i))
                        continue;
                    
                    Eigen::Vector3d g;
                    g << tmp.g(0), tmp.g(1), tmp.g(2);
                    Eigen::Matrix3d h;
                    h << tmp.h(0, 0), tmp.h(0, 1), tmp.h(0, 2), 
                        tmp.h(1, 0), tmp.h(1, 1), tmp.h(1, 2),
                        tmp.h(2, 0), tmp.h(2, 1), tmp.h(2, 2);
                    h *= pow(tmp.x - initial_distance(f * n_loc_samples + i), power-1) * power;
                    h += g * g.transpose() * (pow(tmp.x - initial_distance(f * n_loc_samples + i), power-2) * power * (power-1));
                    const double m = P(i, 0) * vertex_fit_multipliers_(F_(f, 0))
                                   + P(i, 1) * vertex_fit_multipliers_(F_(f, 1))
                                   + P(i, 2) * vertex_fit_multipliers_(F_(f, 2));
                    h *= m * area * weights(i);

                    for (int d = 0; d < 3; d++)
                        for (int k = 0; k < 3; k++)
                            for (int s = 0; s < 3; s++)
                                for (int l = 0; l < 3; l++)
                                    local_hess(s * 3 + d, l * 3 + k) += P(i, s) * P(i, l) * h(d, k);
                }
                for (int d = 0; d < 3; d++)
                    for (int k = 0; k < 3; k++)
                        for (int s = 0; s < 3; s++)
                            for (int l = 0; l < 3; l++)
                                local_storage.cache->add_value(f, F_(f, s) * 3 + d, F_(f, l) * 3 + k, local_hess(s * 3 + d, l * 3 + k));
            }
        });

        timer.stop();
        logger().trace("done separate assembly {}s...", timer.getElapsedTime());

        // Assemble the hessian matrix by concatenating the tuples in each local storage

        // Collect thread storages
        std::vector<LocalThreadSparseMatStorage *> storages(storage.size());
        long int index = 0;
        for (auto &local_storage : storage)
        {
            storages[index++] = &local_storage;
        }

        timer.start();
        maybe_parallel_for(storages.size(), [&](int i) {
            storages[i]->cache->prune();
        });
        timer.stop();
        logger().trace("done pruning triplets {}s...", timer.getElapsedTime());

        // Prepares for parallel concatenation
        std::vector<long int> offsets(storage.size());

        index = 0;
        long int triplet_count = 0;
        for (auto &local_storage : storage)
        {
            offsets[index++] = triplet_count;
            triplet_count += local_storage.cache->triplet_count();
        }

        std::vector<Eigen::Triplet<double>> triplets;

        assert(storages.size() >= 1);
        if (storages[0]->cache->is_dense())
        {
            timer.start();
            // Serially merge local storages
            Eigen::MatrixXd tmp(hessian);
            for (const LocalThreadSparseMatStorage &local_storage : storage)
                tmp += dynamic_cast<const DenseMatrixCache &>(*local_storage.cache).mat();
            hessian = tmp.sparseView();
            hessian.makeCompressed();
            timer.stop();

            logger().trace("Serial assembly time: {}s...", timer.getElapsedTime());
        }
        else if (triplet_count >= triplets.max_size())
        {
            // Serial fallback version in case the vector of triplets cannot be allocated

            logger().error("Cannot allocate space for triplets, switching to serial assembly.");

            timer.start();
            // Serially merge local storages
            for (LocalThreadSparseMatStorage &local_storage : storage)
                hessian += local_storage.cache->get_matrix(false); // will also prune
            hessian.makeCompressed();
            timer.stop();

            logger().trace("Serial assembly time: {}s...", timer.getElapsedTime());
        }
        else
        {
            timer.start();
            triplets.resize(triplet_count);
            timer.stop();

            logger().trace("done allocate triplets {}s...", timer.getElapsedTime());
            logger().trace("Triplets Count: {}", triplet_count);

            timer.start();
            // Parallel copy into triplets
            maybe_parallel_for(storages.size(), [&](int i) {
                const SparseMatrixCache &cache = dynamic_cast<const SparseMatrixCache &>(*storages[i]->cache);
                long int offset = offsets[i];

                std::copy(cache.entries().begin(), cache.entries().end(), triplets.begin() + offset);
                offset += cache.entries().size();

                if (cache.mat().nonZeros() > 0)
                {
                    long int count = 0;
                    for (int k = 0; k < cache.mat().outerSize(); ++k)
                    {
                        for (Eigen::SparseMatrix<double>::InnerIterator it(cache.mat(), k); it; ++it)
                        {
                            assert(count < cache.mat().nonZeros());
                            triplets[offset + count++] = Eigen::Triplet<double>(it.row(), it.col(), it.value());
                        }
                    }
                }
            });

            timer.stop();
            logger().trace("done concatenate triplets {}s...", timer.getElapsedTime());

            timer.start();
            // Sort and assemble
            hessian.setFromTriplets(triplets.begin(), triplets.end());
            timer.stop();

            logger().trace("done setFromTriplets assembly {}s...", timer.getElapsedTime());
        }
    }

    template class FitForm<4>;


    template <int n_refs>
    SDFCollisionForm<n_refs>::SDFCollisionForm(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const Eigen::MatrixXd &surface_v, const Eigen::MatrixXi &surface_f, const double voxel_size, const double separation_dist) : V_(V), F_(F), voxel_size_(voxel_size), totalP(std::vector<openvdb::tools::HessType<double>>(F_.rows() * n_loc_samples, openvdb::tools::HessType<double>(0.))), separation_dist_(separation_dist)
    {
        math::Transform::Ptr xform = math::Transform::createLinearTransform(voxel_size);

        std::vector<Vec3s> points;
        std::vector<Vec3I> triangles;
        std::vector<Vec4I> quads;

        points.reserve(surface_v.rows());
        for (int i = 0; i < surface_v.rows(); i++)
            points.push_back(Vec3s(surface_v(i, 0), surface_v(i, 1), surface_v(i, 2)));
        
        for (int i = 0; i < surface_f.rows(); i++)
            triangles.push_back(Vec3I(surface_f(i, 0), surface_f(i, 1), surface_f(i, 2)));

        grid = tools::meshToSignedDistanceField<DoubleGrid>(*xform, points, triangles, quads, 150, 1);
        
        // build upsampling scheme on the garment mesh
        {
            Eigen::Matrix<double, n_loc_samples, 4> tmp = upsample_standard<n_refs>();
            P = tmp.template leftCols<3>();
            weights = tmp.col(3);
        }
    }

    template <int n_refs>
    double SDFCollisionForm<n_refs>::value_unweighted(const Eigen::VectorXd &x) const 
    {
        const Eigen::MatrixXd V = unflatten(x, 3) + V_;

        double val = 0;
        double max_dist = 0;
        typename DoubleGrid::ConstAccessor acc = grid->getConstAccessor();
        for (int f = 0; f < F_.rows(); f++) {
            const double area = (V_.row(F_(f, 1)) - V_.row(F_(f, 0))).template head<3>().cross((V_.row(F_(f, 2)) - V_.row(F_(f, 0))).template head<3>()).norm() / 2;
            for (int i = 0; i < P.rows(); i++) {
                const double tmp = totalP[f * n_loc_samples + i].x;

                if (std::isnan(tmp))
                    log_and_throw_error("Invalid sdf values!");
                
                if (tmp < separation_dist_)
                    val += area * weights(i) * pow(separation_dist_-tmp, power);
            }
        }

        return val;
    }

    template <int n_refs>
    void SDFCollisionForm<n_refs>::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const 
    {
        POLYFEM_SCOPED_TIMER("fit gradient");
        const Eigen::MatrixXd V = unflatten(x, 3) + V_;

        Eigen::MatrixXd g = Eigen::MatrixXd::Zero(V.rows(), V.cols());

        auto storage = create_thread_storage(LocalThreadMatStorage(g.rows(), g.cols()));

        maybe_parallel_for(F_.rows(), [&](int start, int end, int thread_id) {
            LocalThreadMatStorage &local_storage = get_local_thread_storage(storage, thread_id);
            for (int f = start; f < end; f++) {
                const double area = (V_.row(F_(f, 1)) - V_.row(F_(f, 0))).template head<3>().cross((V_.row(F_(f, 2)) - V_.row(F_(f, 0))).template head<3>()).norm() / 2;
                for (int i = 0; i < P.rows(); i++) {
                    const auto &tmp = totalP[f * n_loc_samples + i];
                    
                    if (tmp.x < separation_dist_)
                        for (int d = 0; d < 3; d++)
                            local_storage.mat(F_.row(f), d) -= (pow(separation_dist_-tmp.x, power-1) * power * area * weights(i) * tmp.g(d)) * P.row(i);
                }
            }
        });

		// Serially merge local storages
		for (const LocalThreadMatStorage &local_storage : storage)
			g += local_storage.mat;
        
        gradv = flatten(g);
    }

    template <int n_refs>
    void SDFCollisionForm<n_refs>::solution_changed(const Eigen::VectorXd &new_x)
    {
        POLYFEM_SCOPED_TIMER("sample SDF");

        const Eigen::MatrixXd V = unflatten(new_x, 3) + V_;

        maybe_parallel_for(F_.rows(), [&](int start, int end, int thread_id) {
            typename DoubleGrid::ConstAccessor acc = grid->getConstAccessor();
            for (int f = start; f < end; f++) {
                const Eigen::Matrix3d M = V({F_(f, 0),F_(f, 1),F_(f, 2)}, Eigen::all);
                Eigen::Matrix<double, n_loc_samples, 3> samples = P * M;
                const double area = (V_.row(F_(f, 1)) - V_.row(F_(f, 0))).template head<3>().cross((V_.row(F_(f, 2)) - V_.row(F_(f, 0))).template head<3>()).norm() / 2;
                for (int i = 0; i < P.rows(); i++) {
                    math::Vec3<double> p(samples(i, 0), samples(i, 1), samples(i, 2));
                    totalP[f * n_loc_samples + i] = tools::SplineSampler::sampleHessian(acc, grid->transformPtr()->worldToIndex(p));
                    auto &tmp = totalP[f * n_loc_samples + i];
                    tmp.g = tmp.g / voxel_size_;
                    tmp.h = tmp.h * (1. / voxel_size_ / voxel_size_);
                }
            }
        });
    }

    template <int n_refs>
    void SDFCollisionForm<n_refs>::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const 
    {
        POLYFEM_SCOPED_TIMER("sdf collision hessian");
        hessian.setZero();
        hessian.resize(x.size(), x.size());   
        if (!use_spline)
            return;
        
        const Eigen::MatrixXd V = unflatten(x, 3) + V_;

        auto storage = create_thread_storage(LocalThreadSparseMatStorage(long(1e7), hessian.rows(), hessian.cols()));

        igl::Timer timer;
        timer.start();
        maybe_parallel_for(F_.rows(), [&](int start, int end, int thread_id) {
            LocalThreadSparseMatStorage &local_storage = get_local_thread_storage(storage, thread_id);
            Eigen::Matrix<double, 9, 9> local_hess;
            for (int f = start; f < end; f++) {
                const double area = (V_.row(F_(f, 1)) - V_.row(F_(f, 0))).template head<3>().cross((V_.row(F_(f, 2)) - V_.row(F_(f, 0))).template head<3>()).norm() / 2;
                local_hess.setZero();
                for (int i = 0; i < P.rows(); i++) {
                    const auto &tmp = totalP[f * n_loc_samples + i];

                    if (tmp.x >= separation_dist_)
                        continue;
                    
                    Eigen::Vector3d g;
                    g << tmp.g(0), tmp.g(1), tmp.g(2);
                    Eigen::Matrix3d h;
                    h << tmp.h(0, 0), tmp.h(0, 1), tmp.h(0, 2), 
                        tmp.h(1, 0), tmp.h(1, 1), tmp.h(1, 2),
                        tmp.h(2, 0), tmp.h(2, 1), tmp.h(2, 2);
                    h *= -pow(separation_dist_-tmp.x, power-1) * power;
                    h += g * g.transpose() * (pow(separation_dist_-tmp.x, power-2) * power * (power-1));
                    h *= area * weights(i);

                    for (int d = 0; d < 3; d++)
                        for (int k = 0; k < 3; k++)
                            for (int s = 0; s < 3; s++)
                                for (int l = 0; l < 3; l++)
                                    local_hess(s * 3 + d, l * 3 + k) += P(i, s) * P(i, l) * h(d, k);
                }
                for (int d = 0; d < 3; d++)
                    for (int k = 0; k < 3; k++)
                        for (int s = 0; s < 3; s++)
                            for (int l = 0; l < 3; l++)
                                local_storage.cache->add_value(f, F_(f, s) * 3 + d, F_(f, l) * 3 + k, local_hess(s * 3 + d, l * 3 + k));
            }
        });

        timer.stop();
        logger().trace("done separate assembly {}s...", timer.getElapsedTime());

        // Assemble the hessian matrix by concatenating the tuples in each local storage

        // Collect thread storages
        std::vector<LocalThreadSparseMatStorage *> storages(storage.size());
        long int index = 0;
        for (auto &local_storage : storage)
        {
            storages[index++] = &local_storage;
        }

        timer.start();
        maybe_parallel_for(storages.size(), [&](int i) {
            storages[i]->cache->prune();
        });
        timer.stop();
        logger().trace("done pruning triplets {}s...", timer.getElapsedTime());

        // Prepares for parallel concatenation
        std::vector<long int> offsets(storage.size());

        index = 0;
        long int triplet_count = 0;
        for (auto &local_storage : storage)
        {
            offsets[index++] = triplet_count;
            triplet_count += local_storage.cache->triplet_count();
        }

        std::vector<Eigen::Triplet<double>> triplets;

        assert(storages.size() >= 1);
        if (storages[0]->cache->is_dense())
        {
            timer.start();
            // Serially merge local storages
            Eigen::MatrixXd tmp(hessian);
            for (const LocalThreadSparseMatStorage &local_storage : storage)
                tmp += dynamic_cast<const DenseMatrixCache &>(*local_storage.cache).mat();
            hessian = tmp.sparseView();
            hessian.makeCompressed();
            timer.stop();

            logger().trace("Serial assembly time: {}s...", timer.getElapsedTime());
        }
        else if (triplet_count >= triplets.max_size())
        {
            // Serial fallback version in case the vector of triplets cannot be allocated

            logger().error("Cannot allocate space for triplets, switching to serial assembly.");

            timer.start();
            // Serially merge local storages
            for (LocalThreadSparseMatStorage &local_storage : storage)
                hessian += local_storage.cache->get_matrix(false); // will also prune
            hessian.makeCompressed();
            timer.stop();

            logger().trace("Serial assembly time: {}s...", timer.getElapsedTime());
        }
        else
        {
            timer.start();
            triplets.resize(triplet_count);
            timer.stop();

            logger().trace("done allocate triplets {}s...", timer.getElapsedTime());
            logger().trace("Triplets Count: {}", triplet_count);

            timer.start();
            // Parallel copy into triplets
            maybe_parallel_for(storages.size(), [&](int i) {
                const SparseMatrixCache &cache = dynamic_cast<const SparseMatrixCache &>(*storages[i]->cache);
                long int offset = offsets[i];

                std::copy(cache.entries().begin(), cache.entries().end(), triplets.begin() + offset);
                offset += cache.entries().size();

                if (cache.mat().nonZeros() > 0)
                {
                    long int count = 0;
                    for (int k = 0; k < cache.mat().outerSize(); ++k)
                    {
                        for (Eigen::SparseMatrix<double>::InnerIterator it(cache.mat(), k); it; ++it)
                        {
                            assert(count < cache.mat().nonZeros());
                            triplets[offset + count++] = Eigen::Triplet<double>(it.row(), it.col(), it.value());
                        }
                    }
                }
            });

            timer.stop();
            logger().trace("done concatenate triplets {}s...", timer.getElapsedTime());

            timer.start();
            // Sort and assemble
            hessian.setFromTriplets(triplets.begin(), triplets.end());
            timer.stop();

            logger().trace("done setFromTriplets assembly {}s...", timer.getElapsedTime());
        }
    }

    template class SDFCollisionForm<4>;
}
