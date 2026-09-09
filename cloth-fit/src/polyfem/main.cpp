#include <filesystem>

#include <CLI/CLI.hpp>

#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <igl/edges.h>

#include <polysolve/nonlinear/Solver.hpp>

#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/io/MatrixIO.hpp>
#include <polyfem/solver/forms/ContactForm.hpp>
#include <polyfem/solver/forms/garment_forms/GarmentForm.hpp>
#include <polyfem/solver/forms/garment_forms/GarmentALForm.hpp>
#include <polyfem/solver/forms/garment_forms/CurveConstraintForm.hpp>
#include <polyfem/solver/forms/garment_forms/CurveCenterTargetForm.hpp>
#include <polyfem/solver/forms/garment_forms/FitForm.hpp>
#include <polyfem/solver/GarmentNLProblem.hpp>
#include <polyfem/solver/ALSolver.hpp>
#include <polyfem/utils/JSONUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/ChromeTrace.hpp>
#include <polyfem/utils/TensorboardLogger.hpp>
#include <polyfem/utils/TimingRegistry.hpp>
#include <polyfem/mesh/MeshUtils.hpp>
#include <polyfem/garment/optimize.hpp>

#include <fstream>
#include <algorithm>
#include <limits>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>

using namespace polyfem;
using namespace solver;
using namespace mesh;

namespace
{
	struct RGB8
	{
		unsigned char r = 0;
		unsigned char g = 0;
		unsigned char b = 0;
	};

	inline RGB8 jet_like_colormap(double t01)
	{
		t01 = std::min(std::max(t01, 0.0), 1.0);
		double r = 0, g = 0, b = 0;
		if (t01 < 0.25)
		{
			r = 0.0;
			g = 4.0 * t01;
			b = 1.0;
		}
		else if (t01 < 0.50)
		{
			r = 0.0;
			g = 1.0;
			b = 1.0 - 4.0 * (t01 - 0.25);
		}
		else if (t01 < 0.75)
		{
			r = 4.0 * (t01 - 0.50);
			g = 1.0;
			b = 0.0;
		}
		else
		{
			r = 1.0;
			g = 1.0 - 4.0 * (t01 - 0.75);
			b = 0.0;
		}
		auto to_u8 = [](double x) -> unsigned char {
			x = std::min(std::max(x, 0.0), 1.0);
			return static_cast<unsigned char>(std::llround(255.0 * x));
		};
		return RGB8{to_u8(r), to_u8(g), to_u8(b)};
	}

	inline RGB8 diverging_colormap(double t_signed)
	{
		// Map [-1,1] to blue->white->red
		t_signed = std::min(std::max(t_signed, -1.0), 1.0);
		const double a = std::abs(t_signed);
		const double w = 1.0 - a;
		double r = w, g = w, b = w;
		if (t_signed >= 0)
			r += a;
		else
			b += a;
		auto to_u8 = [](double x) -> unsigned char {
			x = std::min(std::max(x, 0.0), 1.0);
			return static_cast<unsigned char>(std::llround(255.0 * x));
		};
		return RGB8{to_u8(r), to_u8(g), to_u8(b)};
	}

	static void write_ply_colored_vertices(
		const std::string &path,
		const Eigen::MatrixXd &V,
		const Eigen::MatrixXi &F,
		const std::vector<RGB8> &colors)
	{
		if (V.rows() != (int)colors.size())
			log_and_throw_error("PLY export: V.rows()={} does not match colors.size()={}", V.rows(), colors.size());
		if (V.cols() < 3)
			log_and_throw_error("PLY export: expected V with at least 3 columns, got {}", V.cols());
		if (F.cols() != 3)
			log_and_throw_error("PLY export: expected triangle F (3 cols), got {}", F.cols());

		std::ofstream out(path, std::ios::out);
		if (!out.is_open())
			log_and_throw_error("PLY export: cannot open file for write: {}", path);

		out << "ply\nformat ascii 1.0\n";
		out << "element vertex " << V.rows() << "\n";
		out << "property float x\nproperty float y\nproperty float z\n";
		out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
		out << "element face " << F.rows() << "\n";
		out << "property list uchar int vertex_indices\n";
		out << "end_header\n";

		out.setf(std::ios::fixed);
		out.precision(8);
		for (int i = 0; i < V.rows(); ++i)
		{
			out << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << " ";
			out << int(colors[i].r) << " " << int(colors[i].g) << " " << int(colors[i].b) << "\n";
		}
		for (int f = 0; f < F.rows(); ++f)
		{
			out << "3 " << F(f, 0) << " " << F(f, 1) << " " << F(f, 2) << "\n";
		}
	}

	struct A2CorrEntry
	{
		int tri_id = -1;
		double dist = std::numeric_limits<double>::quiet_NaN();
		Eigen::Vector3d bary = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
		bool has_dist = false;
		bool has_bary = false;
	};

	static std::unordered_set<int> read_vertex_id_set_ascii(
		const std::string &path,
		const int index_base,
		const int n_vertices,
		const std::string &label)
	{
		std::ifstream in(path);
		if (!in.is_open())
			log_and_throw_error("{}: cannot open vertex id file: {}", label, path);

		std::unordered_set<int> s;
		s.reserve(4096);
		long long raw = 0;
		while (in >> raw)
		{
			const long long v0 = raw - (long long)index_base;
			if (v0 < 0 || v0 >= (long long)n_vertices)
				log_and_throw_error("{}: vertex id {} (index_base={}) out of range [0,{}). File={}", label, raw, index_base, n_vertices, path);
			s.insert((int)v0);
		}
		if (s.empty())
			logger().warn("{}: vertex id file is empty: {}", label, path);
		return s;
	}

	static std::vector<std::vector<int>> extract_vertexset_loop_curves(
		const Eigen::MatrixXi &F,
		const std::unordered_set<int> &vertex_set,
		const int n_vertices,
		const int max_curves,
		const std::string &label)
	{
		if (n_vertices <= 0)
			log_and_throw_error("{}: invalid n_vertices={}", label, n_vertices);
		if (max_curves <= 0)
			return {};

		// Build induced adjacency graph on vertex_set using triangle edges.
		std::vector<unsigned char> in_set((size_t)n_vertices, 0);
		std::vector<int> set_vertices;
		set_vertices.reserve(vertex_set.size());
		for (const int v : vertex_set)
		{
			if (v < 0 || v >= n_vertices)
				continue;
			in_set[(size_t)v] = 1;
			set_vertices.push_back(v);
		}

		std::vector<std::vector<int>> adj((size_t)n_vertices);
		adj.shrink_to_fit(); // keep memory sane if n_vertices huge
		auto add_edge = [&](const int a, const int b) {
			if (a < 0 || b < 0 || a >= n_vertices || b >= n_vertices)
				return;
			if (!in_set[(size_t)a] || !in_set[(size_t)b])
				return;
			adj[(size_t)a].push_back(b);
			adj[(size_t)b].push_back(a);
		};
		for (int fi = 0; fi < F.rows(); ++fi)
		{
			const int a = F(fi, 0);
			const int b = F(fi, 1);
			const int c = F(fi, 2);
			add_edge(a, b);
			add_edge(b, c);
			add_edge(c, a);
		}
		for (const int v : set_vertices)
		{
			auto &nb = adj[(size_t)v];
			std::sort(nb.begin(), nb.end());
			nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
		}

		// Connected components on the induced subgraph.
		std::vector<unsigned char> visited((size_t)n_vertices, 0);
		std::vector<std::vector<int>> comps;
		comps.reserve(8);
		for (const int seed : set_vertices)
		{
			if (visited[(size_t)seed])
				continue;
			std::vector<int> stack;
			std::vector<int> comp;
			stack.push_back(seed);
			visited[(size_t)seed] = 1;
			while (!stack.empty())
			{
				const int u = stack.back();
				stack.pop_back();
				comp.push_back(u);
				for (const int v : adj[(size_t)u])
				{
					if (!visited[(size_t)v])
					{
						visited[(size_t)v] = 1;
						stack.push_back(v);
					}
				}
			}
			if (!comp.empty())
				comps.push_back(std::move(comp));
		}

		struct LoopCand
		{
			std::vector<int> loop; // includes repeated start at end
			int size = 0;          // unique vertex count (loop.size()-1)
		};
		std::vector<LoopCand> all_loops;
		all_loops.reserve(8);

		// Workspace arrays reused across components.
		std::vector<unsigned char> active((size_t)n_vertices, 0);
		std::vector<unsigned char> seen((size_t)n_vertices, 0);
		std::vector<int> deg((size_t)n_vertices, 0);

		for (auto &comp : comps)
		{
			if ((int)comp.size() < 3)
				continue;

			// Initialize active set for pruning (2-core extraction).
			for (const int v : comp)
			{
				active[(size_t)v] = 1;
				seen[(size_t)v] = 0;
				deg[(size_t)v] = 0;
			}
			for (const int v : comp)
			{
				int d = 0;
				for (const int nb : adj[(size_t)v])
					if (active[(size_t)nb])
						++d;
				deg[(size_t)v] = d;
			}

			std::vector<int> prune_stack;
			prune_stack.reserve(comp.size());
			for (const int v : comp)
				if (deg[(size_t)v] < 2)
					prune_stack.push_back(v);

			while (!prune_stack.empty())
			{
				const int v = prune_stack.back();
				prune_stack.pop_back();
				if (!active[(size_t)v])
					continue;
				active[(size_t)v] = 0;
				for (const int nb : adj[(size_t)v])
				{
					if (!active[(size_t)nb])
						continue;
					deg[(size_t)nb] -= 1;
					if (deg[(size_t)nb] == 1) // just crossed below 2
						prune_stack.push_back(nb);
				}
			}

			int core_count = 0;
			for (const int v : comp)
				if (active[(size_t)v])
					++core_count;
			if (core_count < 3)
			{
				for (const int v : comp)
					active[(size_t)v] = 0;
				continue;
			}

			// Extract cycle(s) from the remaining 2-core.
			for (const int start : comp)
			{
				if (!active[(size_t)start] || seen[(size_t)start])
					continue;

				std::vector<int> loop;
				loop.reserve((size_t)core_count + 1);
				int prev = -1;
				int cur = start;

				const int max_steps = core_count + 5;
				bool closed = false;
				for (int step = 0; step < max_steps; ++step)
				{
					loop.push_back(cur);
					seen[(size_t)cur] = 1;

					int next = -1;
					int n_active_nb = 0;
					for (const int nb : adj[(size_t)cur])
					{
						if (!active[(size_t)nb])
							continue;
						++n_active_nb;
						if (nb == prev)
							continue;
						if (next == -1)
							next = nb; // deterministic: first neighbor (adj is sorted)
					}

					if (n_active_nb != 2)
					{
						logger().warn("{}: non-cycle vertex degree {} at v={} (expect 2). Loop extraction may be ambiguous.", label, n_active_nb, cur);
					}

					if (next == -1)
						break;
					prev = cur;
					cur = next;
					if (cur == start)
					{
						loop.push_back(start);
						closed = true;
						break;
					}
				}

				if (!closed)
				{
					logger().warn("{}: failed to close a loop from start v={} (loop_len={}). Ignoring this partial trace.", label, start, (int)loop.size());
					continue;
				}

				const int uniq = (int)loop.size() - 1;
				if (uniq >= 3)
					all_loops.push_back(LoopCand{std::move(loop), uniq});
			}

			// Reset workspace marks for this component.
			for (const int v : comp)
			{
				active[(size_t)v] = 0;
				seen[(size_t)v] = 0;
			}
		}

		if (all_loops.empty())
			return {};

		std::sort(all_loops.begin(), all_loops.end(), [](const LoopCand &a, const LoopCand &b) {
			return a.size > b.size;
		});

		std::vector<std::vector<int>> out;
		out.reserve((size_t)std::min<int>(max_curves, (int)all_loops.size()));
		for (int i = 0; i < (int)all_loops.size() && (int)out.size() < max_curves; ++i)
			out.push_back(std::move(all_loops[i].loop));
		return out;
	}

	static std::vector<std::vector<int>> extract_vertexset_region_boundary_loops(
		const Eigen::MatrixXd &V,
		const Eigen::MatrixXi &F,
		const std::unordered_set<int> &vertex_set,
		const int n_vertices,
		const int max_curves,
		const std::string &label,
		std::vector<unsigned char> *out_used_in_set = nullptr,
		int *out_dilate_steps = nullptr,
		const int force_dilate_steps = -1,
		const int max_auto_dilate_steps = 4,
		const std::string &select_mode = std::string("overlap"),
		const double accept_loop_overlap = 0.50,
		const double accept_loop_coverage = 0.70)
	{
		if (n_vertices <= 0)
			log_and_throw_error("{}: invalid n_vertices={}", label, n_vertices);
		if (V.rows() != n_vertices)
			log_and_throw_error("{}: V.rows()={} does not match n_vertices={}", label, V.rows(), n_vertices);
		if (max_curves <= 0)
			return {};

		std::vector<unsigned char> base_in_set((size_t)n_vertices, 0);
		for (const int v : vertex_set)
			if (v >= 0 && v < n_vertices)
				base_in_set[(size_t)v] = 1;

		// Build vertex adjacency once (for optional auto-dilation).
		std::vector<std::vector<int>> v_adj((size_t)n_vertices);
		v_adj.reserve((size_t)n_vertices);
		for (int fi = 0; fi < F.rows(); ++fi)
		{
			const int a = F(fi, 0);
			const int b = F(fi, 1);
			const int c = F(fi, 2);
			if (a < 0 || b < 0 || c < 0 || a >= n_vertices || b >= n_vertices || c >= n_vertices)
				continue;
			v_adj[(size_t)a].push_back(b);
			v_adj[(size_t)a].push_back(c);
			v_adj[(size_t)b].push_back(a);
			v_adj[(size_t)b].push_back(c);
			v_adj[(size_t)c].push_back(a);
			v_adj[(size_t)c].push_back(b);
		}
		for (int v = 0; v < n_vertices; ++v)
		{
			auto &nb = v_adj[(size_t)v];
			std::sort(nb.begin(), nb.end());
			nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
		}

		auto count_mask = [&](const std::vector<unsigned char> &m) -> size_t {
			size_t n = 0;
			for (unsigned char x : m)
				n += (x != 0);
			return n;
		};

		auto dilate_one_ring = [&](std::vector<unsigned char> &m) {
			std::vector<unsigned char> out = m;
			for (int v = 0; v < n_vertices; ++v)
			{
				if (!m[(size_t)v])
					continue;
				for (const int nb : v_adj[(size_t)v])
					out[(size_t)nb] = 1;
			}
			m.swap(out);
		};

		auto edge_key = [&](int a, int b) -> uint64_t {
			if (a > b) std::swap(a, b);
			return (uint64_t(uint32_t(a)) << 32) | uint64_t(uint32_t(b));
		};

		// Manual selections are often a "thin ring" where very few faces have all 3 vertices in the set,
		// making the "region boundary" ill-defined. We optionally auto-dilate by a few 1-ring steps to
		// create a triangle-complete region and retry.
		const int kMaxAuto = std::max(0, max_auto_dilate_steps);
		const double kAcceptCov = std::max(0.0, std::min(1.0, accept_loop_coverage));
		const double kAcceptOv  = std::max(0.0, std::min(1.0, accept_loop_overlap));
		const std::string mode = select_mode;

		std::vector<unsigned char> in_set = base_in_set;
		int dilate_steps = 0;
		if (force_dilate_steps >= 0)
		{
			// Fixed/manual dilation: apply exactly N 1-ring dilations and do not run auto-dilation.
			const int n = std::min(force_dilate_steps, 64); // hard cap for safety
			for (int i = 0; i < n; ++i)
				dilate_one_ring(in_set);
			dilate_steps = n;
			logger().info("{}: using manual dilate_steps={} (auto_dilate_max_steps ignored)", label, dilate_steps);
		}

		auto edge_len = [&](const int a, const int b) -> double {
			return (V.row(a).head<3>() - V.row(b).head<3>()).norm();
		};

		// Keep the best attempt even if we don't meet acceptance criteria.
		std::vector<std::vector<int>> best_out;
		std::vector<unsigned char> best_in_set;
		double best_perim = -1.0;
		int best_dilate = 0;
		double best_overlap = -1.0;

		const int max_attempts = (force_dilate_steps >= 0) ? 0 : kMaxAuto;
		for (int attempt = 0; attempt <= max_attempts; ++attempt)
		{
			const size_t in_size = count_mask(in_set);
			logger().info("{}: attempt {} (auto_dilate_steps={} in_set_vertices={})", label, attempt, dilate_steps, in_size);

			// Define the region as faces whose all 3 vertices are in in_set.
			// The region boundary is then edges that belong to exactly one such face (count==1).
			std::unordered_map<uint64_t, int> edge_in_face_count;
			edge_in_face_count.reserve(in_size * 4 + 1024);
			for (int fi = 0; fi < F.rows(); ++fi)
			{
				const int a = F(fi, 0);
				const int b = F(fi, 1);
				const int c = F(fi, 2);
				if (a < 0 || b < 0 || c < 0 || a >= n_vertices || b >= n_vertices || c >= n_vertices)
					continue;
				if (!in_set[(size_t)a] || !in_set[(size_t)b] || !in_set[(size_t)c])
					continue;
				edge_in_face_count[edge_key(a, b)] += 1;
				edge_in_face_count[edge_key(b, c)] += 1;
				edge_in_face_count[edge_key(c, a)] += 1;
			}

			std::unordered_set<uint64_t> edge_set;
			edge_set.reserve(edge_in_face_count.size() * 2);
			std::vector<std::pair<int, int>> edges;
			edges.reserve(edge_in_face_count.size());
			for (const auto &kv : edge_in_face_count)
			{
				if (kv.second != 1)
					continue;
				const uint64_t k = kv.first;
				const int a = int(uint32_t(k >> 32));
				const int b = int(uint32_t(k & 0xffffffffu));
				if (a < 0 || b < 0 || a >= n_vertices || b >= n_vertices)
					continue;
				edge_set.insert(k);
				edges.emplace_back(a, b);
			}

			if (edges.empty())
			{
				logger().warn("{}: no region boundary edges found from in-face region (in_set_vertices={}).", label, in_size);
				if (attempt < max_attempts)
				{
					dilate_one_ring(in_set);
					++dilate_steps;
					continue;
				}
				break;
			}

			// Boundary edge graph adjacency.
			std::vector<std::vector<int>> adj((size_t)n_vertices);
			for (const auto &e : edges)
			{
				adj[(size_t)e.first].push_back(e.second);
				adj[(size_t)e.second].push_back(e.first);
			}
			std::vector<int> boundary_vertices;
			boundary_vertices.reserve(edges.size());
			int deg2 = 0, degNon2 = 0;
			for (int v = 0; v < n_vertices; ++v)
			{
				auto &nb = adj[(size_t)v];
				if (nb.empty())
					continue;
				std::sort(nb.begin(), nb.end());
				nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
				boundary_vertices.push_back(v);
				if ((int)nb.size() == 2) ++deg2; else ++degNon2;
			}

			logger().info("{}: region_boundary edges={} boundary_vertices={} (deg2={}, deg!=2={})",
				label, edges.size(), boundary_vertices.size(), deg2, degNon2);
			if (degNon2 > 0)
				logger().warn("{}: boundary graph has {} vertices with degree != 2; loop tracing may be ambiguous.", label, degNon2);

			// Trace loops by walking unvisited boundary edges.
			std::unordered_set<uint64_t> visited;
			visited.reserve(edge_set.size() * 2);

			struct LoopCand
			{
				std::vector<int> loop; // includes repeated start at end
				double perimeter = 0.0;
				int n_unique = 0;
			};
			std::vector<LoopCand> loops;
			loops.reserve(8);

			for (const auto &e0 : edges)
			{
				const uint64_t k0 = edge_key(e0.first, e0.second);
				if (visited.find(k0) != visited.end())
					continue;

				const int start = e0.first;
				int prev = start;
				int cur = e0.second;

				std::vector<int> loop;
				loop.reserve(1024);
				loop.push_back(start);
				loop.push_back(cur);
				visited.insert(k0);

				const int max_steps = (int)edges.size() + 5;
				bool closed = false;
				for (int step = 0; step < max_steps; ++step)
				{
					if (cur == start)
					{
						closed = true;
						break;
					}

					int next = -1;
					for (const int nb : adj[(size_t)cur])
					{
						if (nb == prev)
							continue;
						const uint64_t kk = edge_key(cur, nb);
						if (visited.find(kk) != visited.end())
							continue;
						next = nb;
						break; // deterministic (neighbors sorted)
					}

					if (next == -1)
					{
						// Try closing back to start if that edge exists and is unvisited.
						const uint64_t kclose = edge_key(cur, start);
						if (visited.find(kclose) == visited.end() && edge_set.find(kclose) != edge_set.end() && (int)loop.size() >= 3)
						{
							loop.push_back(start);
							visited.insert(kclose);
							closed = true;
						}
						break;
					}

					const uint64_t knext = edge_key(cur, next);
					visited.insert(knext);
					prev = cur;
					cur = next;
					loop.push_back(cur);
				}

				if (!closed || loop.size() < 4 || loop.front() != loop.back())
				{
					// Not a closed loop: ignore (but we keep visited marks to avoid O(E^2) retries).
					continue;
				}

				double perim = 0.0;
				for (int i = 0; i + 1 < (int)loop.size(); ++i)
					perim += edge_len(loop[i], loop[i + 1]);

				const int nuniq = (int)loop.size() - 1;
				loops.push_back(LoopCand{std::move(loop), perim, nuniq});
			}

			if (loops.empty())
			{
				logger().warn("{}: no closed boundary loops could be traced from region boundary edges.", label);
				if (attempt < max_attempts)
				{
					dilate_one_ring(in_set);
					++dilate_steps;
					continue;
				}
				break;
			}

			for (auto &lc : loops)
				lc.n_unique = (int)lc.loop.size() - 1;

			// Rank loops.
			// - mode="overlap": prefer overlap with original input set, then perimeter.
			// - mode="perimeter": prefer perimeter (full loop), then overlap, then fewer dilations at attempt-selection stage.
			auto loop_overlap = [&](const LoopCand &lc) -> double {
				if (lc.n_unique <= 0)
					return 0.0;
				int hit = 0;
				// loop includes repeated start at end; ignore last.
				for (int i = 0; i + 1 < (int)lc.loop.size(); ++i)
				{
					const int v = lc.loop[i];
					if (v >= 0 && v < n_vertices && base_in_set[(size_t)v])
						++hit;
				}
				return double(hit) / double(lc.n_unique);
			};

			std::vector<double> overlaps;
			overlaps.reserve(loops.size());
			for (const auto &lc : loops)
				overlaps.push_back(loop_overlap(lc));

			std::vector<int> order(loops.size());
			for (int i = 0; i < (int)order.size(); ++i) order[i] = i;
			std::sort(order.begin(), order.end(), [&](int ia, int ib) {
				const double oa = overlaps[(size_t)ia], ob = overlaps[(size_t)ib];
				const double pa = loops[(size_t)ia].perimeter, pb = loops[(size_t)ib].perimeter;
				if (mode == "perimeter")
				{
					if (pa != pb) return pa > pb;
					if (oa != ob) return oa > ob;
					return loops[(size_t)ia].n_unique > loops[(size_t)ib].n_unique;
				}
				// default: overlap
				if (oa != ob) return oa > ob;
				if (pa != pb) return pa > pb;
				return loops[(size_t)ia].n_unique > loops[(size_t)ib].n_unique;
			});

			logger().info("{}: traced loops={} (showing top 3 by overlap)", label, loops.size());
			{
				// Also show top-3 by overlap (the actual selection criterion).
				const int show_n = std::min<int>(3, (int)order.size());
				for (int r = 0; r < show_n; ++r)
				{
					const int i = order[(size_t)r];
					logger().info("{}: top_overlap[{}] overlap={:.3f} perimeter={} n_unique={}",
						label, r, overlaps[(size_t)i], loops[(size_t)i].perimeter, loops[(size_t)i].n_unique);
				}
			}

			std::vector<std::vector<int>> out;
			out.reserve((size_t)std::min<int>(max_curves, (int)loops.size()));
			for (int r = 0; r < (int)order.size() && (int)out.size() < max_curves; ++r)
				out.push_back(loops[(size_t)order[(size_t)r]].loop);

			const int top_i = order[0];
			const double top_overlap = overlaps[(size_t)top_i];
			const double top_perim = loops[(size_t)top_i].perimeter;
			const int top_nuniq = loops[(size_t)top_i].n_unique;
			// Track best attempt across dilation levels.
			// - mode="overlap": maximize overlap, then fewer dilations, then perimeter.
			// - mode="perimeter": maximize perimeter, then fewer dilations, then overlap.
			bool take = false;
			if (!out.empty())
			{
				if (mode == "perimeter")
				{
					take = (top_perim > best_perim)
						|| (top_perim == best_perim && dilate_steps < best_dilate)
						|| (top_perim == best_perim && dilate_steps == best_dilate && top_overlap > best_overlap);
				}
				else
				{
					take = (top_overlap > best_overlap)
						|| (top_overlap == best_overlap && dilate_steps < best_dilate)
						|| (top_overlap == best_overlap && dilate_steps == best_dilate && top_perim > best_perim);
				}
			}
			if (take)
			{
				best_overlap = top_overlap;
				best_perim = top_perim;
				best_out = out;
				best_in_set = in_set;
				best_dilate = dilate_steps;
			}

			const double coverage = boundary_vertices.empty() ? 0.0 : (double)top_nuniq / (double)boundary_vertices.size();
			logger().info("{}: best loop metrics: coverage={:.3f} overlap={:.3f} (boundary_vertices={})",
				label, coverage, top_overlap, boundary_vertices.size());

			// Accept when boundary graph is a clean cycle and the best loop covers most boundary vertices.
			const double min_overlap_guard = (mode == "perimeter") ? 0.05 : kAcceptOv;
			if (degNon2 == 0 && coverage >= kAcceptCov && top_overlap >= min_overlap_guard)
			{
				if (out_used_in_set) *out_used_in_set = in_set;
				if (out_dilate_steps) *out_dilate_steps = dilate_steps;
				return out;
			}

			if (attempt < max_attempts)
			{
				dilate_one_ring(in_set);
				++dilate_steps;
			}
		}

		if (!best_out.empty())
		{
			logger().warn("{}: using best loop(s) from auto-dilate attempt (dilate_steps={}, top_perimeter={}).",
				label, best_dilate, best_perim);
			if (mode == "overlap")
				logger().warn("{}: best attempt had overlap={:.3f} (higher is closer to input hem set).", label, best_overlap);
			else
				logger().warn("{}: best attempt overlap={:.3f} (note: select_mode='{}' may ignore overlap).", label, best_overlap, mode);
			if (out_used_in_set) *out_used_in_set = best_in_set;
			if (out_dilate_steps) *out_dilate_steps = best_dilate;
			return best_out;
		}

		logger().warn("{}: failed to extract any closed boundary loops.", label);
		if (out_used_in_set) *out_used_in_set = in_set;
		if (out_dilate_steps) *out_dilate_steps = dilate_steps;
		return {};
	}

	std::vector<A2CorrEntry> read_a2_correspondence_ascii(
		const std::string &path,
		const int n_garment_verts,
		const int n_body_tris)
	{
		std::ifstream in(path);
		if (!in.is_open())
			log_and_throw_error("A2: cannot open correspondence file: {}", path);

		std::vector<A2CorrEntry> corr;
		corr.reserve(n_garment_verts);

		std::string line;
		int line_idx = 0;
		while (std::getline(in, line))
		{
			// Reject blank lines: we require strict one-line-per-vertex indexing.
			bool only_ws = true;
			for (char c : line)
				if (!std::isspace(static_cast<unsigned char>(c)))
				{
					only_ws = false;
					break;
				}
			if (only_ws)
				log_and_throw_error("A2: correspondence file contains a blank line at line {} (must be exactly N lines).", line_idx + 1);

			std::istringstream ss(line);
			std::vector<double> vals;
			{
				double v;
				while (ss >> v)
					vals.push_back(v);
			}

			if (vals.size() != 1 && vals.size() != 2 && vals.size() != 5)
				log_and_throw_error("A2: invalid correspondence line {}: expected 1, 2, or 5 numbers, got {}.", line_idx + 1, vals.size());

			A2CorrEntry e;
			e.tri_id = int(std::llround(vals[0]));
			// Sentinel: tri_id == -1 means "no correspondence" (keep w=1 for this garment vertex).
			if (e.tri_id == -1)
			{
				// Ignore any extra tokens if provided; this keeps Python-side generation simple.
				e.has_dist = false;
				e.has_bary = false;
				corr.push_back(e);
				++line_idx;
				continue;
			}
			if (e.tri_id < 0 || e.tri_id >= n_body_tris)
				log_and_throw_error("A2: tri_id out of range at line {}: {} not in [-1,{}].", line_idx + 1, e.tri_id, n_body_tris - 1);

			if (vals.size() >= 2)
			{
				e.has_dist = true;
				e.dist = vals[1];
				if (!std::isfinite(e.dist) || e.dist < 0.0)
					log_and_throw_error("A2: invalid dist at line {}: {} (must be finite and >= 0).", line_idx + 1, e.dist);
			}

			if (vals.size() == 5)
			{
				e.has_bary = true;
				e.bary << vals[2], vals[3], vals[4];
				// Barycentric is not used in core A2, so we only do lightweight sanity.
				if (!e.bary.allFinite())
					log_and_throw_error("A2: invalid barycentric coords at line {} (must be finite).", line_idx + 1);
			}

			corr.push_back(e);
			++line_idx;
		}

		if (corr.size() != size_t(n_garment_verts))
			log_and_throw_error("A2: correspondence line count mismatch (got {}, expected {}).", corr.size(), n_garment_verts);

		return corr;
	}

	inline double tri_area(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, const int f)
	{
		const Eigen::Vector3d a = V.row(F(f, 0)).head<3>();
		const Eigen::Vector3d b = V.row(F(f, 1)).head<3>();
		const Eigen::Vector3d c = V.row(F(f, 2)).head<3>();
		return 0.5 * (b - a).cross(c - a).norm();
	}
} // namespace

bool has_arg(const CLI::App &command_line, const std::string &value)
{
	const auto *opt = command_line.get_option_no_throw(value.size() == 1 ? ("-" + value) : ("--" + value));
	if (!opt)
		return false;

	return opt->count() > 0;
}

bool load_json(const std::string &json_file, json &out)
{
	std::ifstream file(json_file);

	if (!file.is_open())
		return false;

	file >> out;

	if (!out.contains("root_path"))
		out["root_path"] = json_file;

	return true;
}

int main(int argc, char **argv)
{
	using namespace polyfem;

	CLI::App command_line{"polyfem"};

	command_line.ignore_case();
	command_line.ignore_underscore();

	// Eigen::setNbThreads(1);
	unsigned max_threads = 16;
	command_line.add_option("--max_threads", max_threads, "Maximum number of threads");

	auto input = command_line.add_option_group("input");

	std::string json_file = "";
	input->add_option("-j,--json", json_file, "Simulation JSON file")->check(CLI::ExistingFile);

	input->require_option(1);

	std::string output_dir = "";
	command_line.add_option("-o,--output_dir", output_dir, "Directory for output files")->check(CLI::ExistingDirectory | CLI::NonexistentPath);

	const std::vector<std::pair<std::string, spdlog::level::level_enum>>
		SPDLOG_LEVEL_NAMES_TO_LEVELS = {
			{"trace", spdlog::level::trace},
			{"debug", spdlog::level::debug},
			{"info", spdlog::level::info},
			{"warning", spdlog::level::warn},
			{"error", spdlog::level::err},
			{"critical", spdlog::level::critical},
			{"off", spdlog::level::off}};
	spdlog::level::level_enum log_level = spdlog::level::debug;
	command_line.add_option("--log_level", log_level, "Log level")
		->transform(CLI::CheckedTransformer(SPDLOG_LEVEL_NAMES_TO_LEVELS, CLI::ignore_case));

	CLI11_PARSE(command_line, argc, argv);

	json in_args = json({});
	{
		if (!json_file.empty())
		{
			const bool ok = load_json(json_file, in_args);

			if (!ok)
				log_and_throw_error(fmt::format("unable to open {} file", json_file));
		}

		if (in_args.empty())
		{
			logger().error("No input file specified!");
			return command_line.exit(CLI::RequiredError("--json"));
		}

		json tmp = json::object();
		if (has_arg(command_line, "log_level"))
			tmp["/output/log/level"_json_pointer] = int(log_level);
		if (has_arg(command_line, "max_threads"))
			tmp["/solver/max_threads"_json_pointer] = max_threads;
		if (has_arg(command_line, "output_dir"))
			tmp["/output/directory"_json_pointer] = std::filesystem::absolute(output_dir);

		assert(tmp.is_object());
		in_args.merge_patch(tmp);
	}

	{
		in_args = init(in_args, false);
	}

	GarmentSolver gstate;

	const std::string out_folder = in_args["/output/directory"_json_pointer];
	const std::string avatar_mesh_path = in_args["avatar_mesh_path"];
	const std::string garment_mesh_path = in_args["garment_mesh_path"];
	const std::string source_skeleton_path = in_args["source_skeleton_path"];
	const std::string target_skeleton_path = in_args["target_skeleton_path"];
	const std::string avatar_skin_weights_path = in_args["avatar_skin_weights_path"];
	const bool self_collision = in_args["contact"]["enabled"];

	if (!std::filesystem::exists(avatar_mesh_path))
		log_and_throw_error("Invalid avatar mesh path: {}", avatar_mesh_path);

	if (!std::filesystem::exists(garment_mesh_path))
		log_and_throw_error("Invalid garment mesh path: {}", garment_mesh_path);

	if (!std::filesystem::exists(source_skeleton_path))
		log_and_throw_error("Invalid source skeleton mesh path: {}", source_skeleton_path);

	if (!std::filesystem::exists(target_skeleton_path))
		log_and_throw_error("Invalid target skeleton mesh path: {}", target_skeleton_path);

	gstate.out_folder = out_folder;

	// Optional profiling (Chrome trace + aggregated timers)
	{
		auto get_profiling_obj = [&]() -> const json * {
			if (utils::is_param_valid(in_args, "output") && in_args["output"].is_object()
				&& in_args["output"].contains("profiling") && in_args["output"]["profiling"].is_object())
				return &in_args["output"]["profiling"];
			if (in_args.contains("profiling") && in_args["profiling"].is_object())
				return &in_args["profiling"];
			return nullptr;
		};

		const json *profiling = get_profiling_obj();
		if (profiling)
		{
			// Chrome trace timeline
			if (profiling->contains("chrome_trace") && (*profiling)["chrome_trace"].is_object())
			{
				const auto &ct = (*profiling)["chrome_trace"];
				const bool enabled = ct.value("enabled", false);
				if (enabled)
				{
					std::string path = ct.value("path", std::string(""));
					if (path.empty())
						path = out_folder + "/trace.json";
					utils::ChromeTrace::instance().init(path);
				}
			}

			// Aggregated timing summary (by POLYFEM_SCOPED_TIMER name)
			if (profiling->contains("timing_summary") && (*profiling)["timing_summary"].is_object())
			{
				const auto &ts = (*profiling)["timing_summary"];
				const bool enabled = ts.value("enabled", false);
				utils::TimingRegistry::instance().set_enabled(enabled);
			}
		}
	}

	// Optional TensorBoard scalar logging (native event files)
	bool tb_detailed_energies = false;
	utils::TensorboardLogger tb_logger;
	{
		utils::TensorboardLogger::Options tbopt;
		std::string tb_log_dir = "";

		if (utils::is_param_valid(in_args, "output") && in_args["output"].is_object())
		{
			const auto &out = in_args["output"];
			if (out.contains("tensorboard") && out["tensorboard"].is_object())
			{
				const auto &tb = out["tensorboard"];
				tbopt.enabled = tb.value("enabled", false);
				tb_log_dir = tb.value("log_dir", std::string(""));
				tbopt.log_every = tb.value("log_every", tbopt.log_every);
				tbopt.energy_every = tb.value("energy_every", tbopt.energy_every);
				tbopt.flush_period_s = tb.value("flush_period_s", size_t(3));
				tbopt.max_queue_size = tb.value("max_queue_size", size_t(100000));
				tbopt.resume = tb.value("resume", false);
				tb_detailed_energies = tb.value("detailed_energies", false);
			}
		}

		tb_logger.init(out_folder, tb_log_dir, tbopt);
	}

	int exit_code = EXIT_SUCCESS;
	try
	{
		utils::ChromeTraceScope trace_total("total_run");

		gstate.read_meshes(avatar_mesh_path, source_skeleton_path, target_skeleton_path, avatar_skin_weights_path);
		if (utils::is_param_valid(in_args, "avatar_remove_indices_path"))
			gstate.remove_avatar_vertices(in_args["avatar_remove_indices_path"]);
		gstate.load_garment_mesh(in_args["garment_mesh_path"], in_args["no_fit_spec_path"]);
		gstate.normalize_meshes(in_args);

		// Optional skeleton pre-pass: make source skeleton inside garment mesh
		gstate.prepass_optimize_source_skeleton_inside_garment(in_args);
		gstate.project_avatar_to_skeleton();

	// Optional A2: area-aware per-vertex multipliers for Step3 fit term (computed once)
	bool a2_enabled = false;
	bool a2_final_substep_only = true;
	bool a2_use_distance_gate = true;
	bool a2_export_visualization = false;
	bool a2_enable_surf_relax = false;
	double a2_lambda = 6.0;
	double a2_w_max = 4.0;
	double a2_d_gate = 0.05;
	double a2_surf_k = 0.8;
	double a2_surf_min = 0.7;
	Eigen::VectorXd a2_w = Eigen::VectorXd::Ones(gstate.n_garment_vertices());
	Eigen::VectorXd a2_surf = Eigen::VectorXd::Ones(gstate.n_garment_vertices());
	bool a2_has_dist = false;
	std::vector<A2CorrEntry> a2_corr;
	Eigen::MatrixXd a2_body_Vs, a2_body_Vt;
	Eigen::MatrixXi a2_body_F;
	Eigen::VectorXd a2_body_rho;

	// Optional: vertex-fit-weight visualization (colored PLY).
	// This is meant to complement A2's visualization: instead of visualizing `a2_w` itself, we export
	// per-garment-vertex fit multipliers coming from:
	//  - vertex masks only (hands/feet/semantic/etc.) and
	//  - vertex masks * A2 (overall effect in Step3).
	bool fit_weight_vis_enabled = false;
	bool fit_weight_vis_export_mask_only = true;
	bool fit_weight_vis_export_final = true;
	std::string fit_weight_vis_mask_only_filename = "fit_weight_vertex_masks_only.ply";
	std::string fit_weight_vis_final_filename = "fit_weight_vertex_masks_times_area.ply";
	if (in_args.contains("fit_weight_vis") && in_args["fit_weight_vis"].is_object())
	{
		const auto &vis = in_args["fit_weight_vis"];
		fit_weight_vis_enabled = vis.value("enabled", fit_weight_vis_enabled);
		fit_weight_vis_export_mask_only = vis.value("export_mask_only", fit_weight_vis_export_mask_only);
		fit_weight_vis_export_final = vis.value("export_final", fit_weight_vis_export_final);
		fit_weight_vis_mask_only_filename = vis.value("mask_only_filename", fit_weight_vis_mask_only_filename);
		fit_weight_vis_final_filename = vis.value("final_filename", fit_weight_vis_final_filename);
	}
	bool fit_weight_vis_written = false;
	{
		if (in_args.contains("a2") && in_args["a2"].is_object())
		{
			const auto &a2 = in_args["a2"];
			a2_enabled = a2.value("enabled", false);
			a2_final_substep_only = a2.value("final_substep_only", true);
			a2_use_distance_gate = a2.value("use_distance_gate", true);
			a2_export_visualization = a2.value("export_visualization", false);
			a2_enable_surf_relax = a2.value("enable_surf_relax", false);
			a2_lambda = a2.value("lambda", a2_lambda);
			a2_w_max = a2.value("w_max", a2_w_max);
			a2_d_gate = a2.value("d_gate", a2_d_gate);
			a2_surf_k = a2.value("surf_k", a2_surf_k);
			a2_surf_min = a2.value("surf_min", a2_surf_min);

			if (a2_enabled)
			{
				const std::string src_body_path = a2.value("source_body_trim_path", std::string(""));
				const std::string tgt_body_path = a2.value("target_body_trim_path", std::string(""));
				const std::string corr_path = a2.value("correspondence_path", std::string(""));

				if (src_body_path.empty() || tgt_body_path.empty() || corr_path.empty())
					log_and_throw_error("A2 enabled but a2.source_body_trim_path / a2.target_body_trim_path / a2.correspondence_path is missing/empty.");
				if (!std::filesystem::exists(src_body_path))
					log_and_throw_error("A2 source_body_trim_path not found: {}", src_body_path);
				if (!std::filesystem::exists(tgt_body_path))
					log_and_throw_error("A2 target_body_trim_path not found: {}", tgt_body_path);
				if (!std::filesystem::exists(corr_path))
					log_and_throw_error("A2 correspondence_path not found: {}", corr_path);

				Eigen::MatrixXd Vs, Vt;
				Eigen::MatrixXi Fs, Ft;
				if (!igl::read_triangle_mesh(src_body_path, Vs, Fs))
					log_and_throw_error("A2 failed to read source body mesh: {}", src_body_path);
				if (!igl::read_triangle_mesh(tgt_body_path, Vt, Ft))
					log_and_throw_error("A2 failed to read target body mesh: {}", tgt_body_path);

				if (Fs.rows() != Ft.rows() || Fs.cols() != Ft.cols() || (Fs.array() != Ft.array()).any())
					log_and_throw_error("A2 requires source and target trimmed body meshes to have identical face ordering/topology (Fs must equal Ft).");

				const int n_tris = Fs.rows();
				const double eps_area = 1e-12;
				Eigen::VectorXd rho = Eigen::VectorXd::Ones(n_tris);
				for (int f = 0; f < n_tris; ++f)
				{
					const double As = tri_area(Vs, Fs, f);
					const double At = tri_area(Vt, Ft, f);
					if (!std::isfinite(As) || !std::isfinite(At))
						log_and_throw_error("A2: non-finite triangle area encountered at f={}.", f);
					rho(f) = At / std::max(As, eps_area);
				}
				a2_body_Vs = Vs;
				a2_body_Vt = Vt;
				a2_body_F = Fs;
				a2_body_rho = rho;

				a2_corr = read_a2_correspondence_ascii(corr_path, gstate.n_garment_vertices(), n_tris);
				a2_has_dist = false;
				for (const auto &e : a2_corr)
					if (e.has_dist)
					{
						a2_has_dist = true;
						break;
					}

				// If dist is not provided, distance gating is effectively disabled.
				const bool use_gate = a2_use_distance_gate && a2_has_dist && (a2_d_gate > 0.0);
				if (a2_enable_surf_relax)
				{
					if (!(a2_surf_min >= 0.0 && a2_surf_min <= 1.0))
						log_and_throw_error("A2 surf relax: surf_min must be in [0,1], got {}", a2_surf_min);
					if (!(a2_surf_k >= 0.0))
						log_and_throw_error("A2 surf relax: surf_k must be >= 0, got {}", a2_surf_k);
				}

				for (int i = 0; i < gstate.n_garment_vertices(); ++i)
				{
					const int f = a2_corr[i].tri_id;
					if (f < 0)
					{
						a2_w(i) = 1.0;
						a2_surf(i) = 1.0;
						continue;
					}
					const double r = rho(f);
					const double shrink = std::max(0.0, 1.0 - r);
					double w = 1.0 + a2_lambda * shrink;
					w = std::min(std::max(w, 1.0), a2_w_max);
					if (use_gate && a2_corr[i].has_dist && a2_corr[i].dist > a2_d_gate)
						w = 1.0;
					if (!std::isfinite(w))
						log_and_throw_error("A2: computed non-finite weight at garment vertex {}.", i);
					a2_w(i) = w;

					if (a2_enable_surf_relax)
					{
						double s = 1.0 - a2_surf_k * shrink;
						s = std::min(std::max(s, a2_surf_min), 1.0);
						if (use_gate && a2_corr[i].has_dist && a2_corr[i].dist > a2_d_gate)
							s = 1.0;
						if (!std::isfinite(s))
							log_and_throw_error("A2 surf relax: computed non-finite multiplier at garment vertex {}.", i);
						a2_surf(i) = s;
					}
					else
					{
						a2_surf(i) = 1.0;
					}
				}

				// Log summary stats
				{
					std::vector<double> wvals(a2_w.data(), a2_w.data() + a2_w.size());
					std::sort(wvals.begin(), wvals.end());
					const double w_min = wvals.front();
					const double w_med = wvals[wvals.size() / 2];
					const double w_max = wvals.back();

					int gated = 0;
					if (use_gate)
						for (const auto &e : a2_corr)
							if (e.has_dist && e.dist > a2_d_gate)
								++gated;

					logger().info("[A2] enabled=true lambda={} w_max={} dist_gate={} (use_gate={} gated={}/{})",
						a2_lambda, a2_w_max, a2_d_gate, use_gate, gated, gstate.n_garment_vertices());
					logger().info("[A2] garment w stats: min={} med={} max={}", w_min, w_med, w_max);
				}
				if (a2_enable_surf_relax)
				{
					std::vector<double> svals(a2_surf.data(), a2_surf.data() + a2_surf.size());
					std::sort(svals.begin(), svals.end());
					const double s_min = svals.front();
					const double s_med = svals[svals.size() / 2];
					const double s_max = svals.back();
					logger().info("[A2] surf_relax enabled=true k={} s_min={} stats: min={} med={} max={}",
						a2_surf_k, a2_surf_min, s_min, s_med, s_max);
				}

				// Optional debug export for Python-side sanity checks
				{
					std::ofstream out(out_folder + "/a2_weights.txt", std::ios::out);
					if (out.is_open())
					{
						out << "# i tri_id dist b0 b1 b2 w\n";
						for (int i = 0; i < gstate.n_garment_vertices(); ++i)
						{
							const auto &e = a2_corr[i];
							out << i << " " << e.tri_id << " ";
							out << (e.has_dist ? e.dist : -1.0) << " ";
							out << (e.has_bary ? e.bary(0) : -1.0) << " "
								<< (e.has_bary ? e.bary(1) : -1.0) << " "
								<< (e.has_bary ? e.bary(2) : -1.0) << " ";
							out << a2_w(i) << "\n";
						}
					}
				}

				// Optional visualization exports (colored PLY)
				if (a2_export_visualization)
				{
					// Body: per-face relative area delta (rho-1), mapped to vertices by averaging incident faces.
					const Eigen::VectorXd face_delta = a2_body_rho.array() - 1.0;
					Eigen::VectorXd v_delta = Eigen::VectorXd::Zero(a2_body_Vs.rows());
					Eigen::VectorXd v_cnt = Eigen::VectorXd::Zero(a2_body_Vs.rows());
					for (int f = 0; f < a2_body_F.rows(); ++f)
					{
						const double d = face_delta(f);
						for (int k = 0; k < 3; ++k)
						{
							const int vi = a2_body_F(f, k);
							v_delta(vi) += d;
							v_cnt(vi) += 1.0;
						}
					}
					for (int i = 0; i < v_delta.size(); ++i)
						if (v_cnt(i) > 0)
							v_delta(i) /= v_cnt(i);

					double max_abs = 0.0;
					for (int i = 0; i < v_delta.size(); ++i)
						if (std::isfinite(v_delta(i)))
							max_abs = std::max(max_abs, std::abs(v_delta(i)));
					max_abs = std::max(max_abs, 1e-12);

					std::vector<RGB8> body_colors(a2_body_Vs.rows());
					for (int i = 0; i < a2_body_Vs.rows(); ++i)
					{
						const double s = std::isfinite(v_delta(i)) ? (v_delta(i) / max_abs) : 0.0;
						body_colors[i] = diverging_colormap(s);
					}
					write_ply_colored_vertices(out_folder + "/a2_body_source_delta_area.ply", a2_body_Vs, a2_body_F, body_colors);
					write_ply_colored_vertices(out_folder + "/a2_body_target_delta_area.ply", a2_body_Vt, a2_body_F, body_colors);

					// Garment: A2 per-vertex weights (w in [1,w_max]) colored with a sequential colormap.
					const double denom = std::max(1e-12, a2_w_max - 1.0);
					std::vector<RGB8> garment_colors(gstate.n_garment_vertices());
					for (int i = 0; i < gstate.n_garment_vertices(); ++i)
					{
						const double t = (a2_w(i) - 1.0) / denom; // expected in [0,1]
						garment_colors[i] = jet_like_colormap(t);
					}
					write_ply_colored_vertices(out_folder + "/a2_garment_vertex_weight.ply", gstate.garment.v, gstate.garment.f, garment_colors);

					logger().info("[A2] wrote PLY visualizations: a2_body_*_delta_area.ply, a2_garment_vertex_weight.ply");
				}
			}
		}
	}

	// Optional: restore output translation (keep internal state normalized/shifted)
	Eigen::MatrixXd out_avatar_v = gstate.avatar_v;
	Eigen::MatrixXd out_projected_avatar_v = gstate.skinny_avatar_v;
	Eigen::MatrixXd out_target_skel_v = gstate.target_skeleton_v;
	Eigen::MatrixXd out_source_skel_v = gstate.skeleton_v;

	// Always undo common normalization scale (A0) on outputs
	const double inv_s_out = (gstate.normalization_scale() == 0.0) ? 1.0 : (1.0 / gstate.normalization_scale());
	out_avatar_v *= inv_s_out;
	out_projected_avatar_v *= inv_s_out;
	out_target_skel_v *= inv_s_out;
	out_source_skel_v *= inv_s_out;

	// Undo per-side subject scale (for normalization.mode=separate_translation_with_subject_scale).
	// Defaults are 1.0, so this is a no-op for legacy modes.
	out_avatar_v *= gstate.target_output_scale();
	out_projected_avatar_v *= gstate.target_output_scale();
	out_target_skel_v *= gstate.target_output_scale();
	out_source_skel_v *= gstate.source_output_scale();

	if (gstate.restore_output_translation())
	{
		const Eigen::RowVector3d &t_target = gstate.target_output_translation_offset();
		const Eigen::RowVector3d &t_source = gstate.source_output_translation_offset();
		out_avatar_v.rowwise() += t_target;
		out_projected_avatar_v.rowwise() += t_target;
		out_target_skel_v.rowwise() += t_target;
		out_source_skel_v.rowwise() += t_source;
	}

	igl::write_triangle_mesh(out_folder + "/target_avatar.obj", out_avatar_v, gstate.avatar_f);
	igl::write_triangle_mesh(out_folder + "/projected_avatar.obj", out_projected_avatar_v, gstate.nc_avatar_f);
	write_edge_mesh(out_folder + "/target_skeleton.obj", out_target_skel_v, gstate.target_skeleton_b);
	write_edge_mesh(out_folder + "/source_skeleton.obj", out_source_skel_v, gstate.skeleton_b);

	logger().info("avatar n_verts: {}, garment n_verts: {}, total n_verts: {}", gstate.nc_avatar_v.rows(), gstate.n_garment_vertices(), gstate.nc_avatar_v.rows() + gstate.n_garment_vertices());

	Eigen::MatrixXi collision_triangles(gstate.nc_avatar_f.rows() + gstate.n_garment_faces(), gstate.garment.f.cols());
	collision_triangles << gstate.nc_avatar_f, gstate.garment.f.array() + gstate.nc_avatar_v.rows();
	Eigen::MatrixXi collision_edges;
	igl::edges(collision_triangles, collision_edges);

	Eigen::MatrixXd collision_vertices(gstate.nc_avatar_v.rows() + gstate.n_garment_vertices(), gstate.garment.v.cols());
	collision_vertices << gstate.skinny_avatar_v, gstate.garment.v;

	ipc::CollisionMesh collision_mesh;
	{
		collision_mesh = ipc::CollisionMesh(
			collision_vertices, collision_edges, collision_triangles);

		const int n_avatar_verts = gstate.nc_avatar_v.rows();
		collision_mesh.can_collide = [n_avatar_verts, self_collision](size_t vi, size_t vj) {
			if (self_collision)
				return vi >= n_avatar_verts || vj >= n_avatar_verts;
			else
				return (vi >= n_avatar_verts && vj < n_avatar_verts) || (vi < n_avatar_verts && vj >= n_avatar_verts);
		};

		// 1) Cross (avatar↔garment) intersections (force cross-only regardless of self_collision setting)
		ipc::CollisionMesh cm_cross = collision_mesh;
		cm_cross.can_collide = [n_avatar_verts](size_t vi, size_t vj) {
			return (vi >= n_avatar_verts && vj < n_avatar_verts) || (vi < n_avatar_verts && vj >= n_avatar_verts);
		};
		gstate.check_cross_intersections(cm_cross, collision_vertices);
	}

	// 2) Optional: Garment self-intersections (configurable error/warn)
	{
		bool check_self = true;
		bool error_on_self = true;
		if (utils::is_param_valid(in_args, "initial_checks"))
		{
			const auto &ic = in_args["initial_checks"];
			check_self = ic.value("check_garment_self_intersection", true);
			error_on_self = ic.value("error_on_garment_self_intersection", true);
		}
		if (check_self)
		{
			const bool has_self = gstate.has_garment_self_intersections();
			if (has_self)
			{
				logger().warn("Garment self-intersections detected.");
				if (error_on_self)
					log_and_throw_error("Initial garment has self-intersections (config says to error).");
				else
				{
					// If user allows self-intersections, force-disable contact barriers and warn
					if (in_args["contact"]["enabled"]) {
						logger().warn("contact.enabled is true but garment self-intersection errors are disabled; forcing contact.enabled=false to avoid BC failures.");
						in_args["contact"]["enabled"] = false;
					}
				}
			}
		}
	}

	auto curves = boundary_curves(collision_triangles.bottomRows(gstate.n_garment_faces()));
	const Eigen::MatrixXd source_curve_centers = extract_curve_center_targets(collision_vertices, curves, gstate.skeleton_v, gstate.skeleton_b, gstate.skeleton_v);
	const Eigen::MatrixXd target_curve_centers = extract_curve_center_targets(collision_vertices, curves, gstate.skeleton_v, gstate.skeleton_b, gstate.target_skeleton_v);

	const Eigen::MatrixXd initial_garment_v = gstate.garment.v;
	Eigen::MatrixXd cur_garment_v = gstate.garment.v;
	int save_id = 0;
	const int total_steps = in_args["incremental_steps"];
	const int stride = in_args["output"]["skip_frame"];

	// ------------------------------------------------------------
	// A1 Step3 annealing (compute once; applied only on final substep Step3)
	bool step3_anneal_enabled = false;
	double step3_r0 = 0.2, step3_a = 4.0, step3_s1 = 0.3, step3_a2 = 0.4;
	double step3_percentile = 0.9, step3_tau = 0.008, step3_d0 = 0.0;
	int step3_check_every = 10, step3_stall_K = 8;
	double step3_stall_eps = 0.01;
	double a1_h_global = 0.0;
	double a1_fit_w_phase1_global = in_args["fit_weight"];
	double a1_fit_w_phase2_global = in_args["fit_weight"];
	double a1_sim_w_phase1_global = in_args["similarity_penalty_weight"];
	double a1_sim_w_phase2_global = in_args["similarity_penalty_weight"];
	if (in_args.contains("step3_anneal") && in_args["step3_anneal"].is_object())
	{
		const auto &a1 = in_args["step3_anneal"];
		step3_anneal_enabled = a1.value("enabled", step3_anneal_enabled);
		step3_r0 = a1.value("r0", step3_r0);
		step3_a = a1.value("a", step3_a);
		step3_s1 = a1.value("s1", step3_s1);
		step3_a2 = a1.value("a2", step3_a2);
		step3_percentile = a1.value("percentile", step3_percentile);
		step3_tau = a1.value("tau", step3_tau);
		step3_d0 = a1.value("d0", step3_d0);
		step3_check_every = a1.value("check_every", step3_check_every);
		step3_stall_eps = a1.value("stall_eps", step3_stall_eps);
		step3_stall_K = a1.value("stall_K", step3_stall_K);
	}

	// Fail fast: if enabled, validate source mesh and compute shrink severity h upfront.
	if (step3_anneal_enabled)
	{
		if (!utils::is_param_valid(in_args, "source_avatar_mesh_path") || in_args["source_avatar_mesh_path"].get<std::string>().empty())
			log_and_throw_error("A1 step3_anneal enabled but source_avatar_mesh_path is missing/empty.");

		const std::string src_path = in_args["source_avatar_mesh_path"];
		if (!std::filesystem::exists(src_path))
			log_and_throw_error("A1 source_avatar_mesh_path not found: {}", src_path);

		Eigen::MatrixXd srcV, tgtV;
		Eigen::MatrixXi srcF, tgtF;
		igl::read_triangle_mesh(src_path, srcV, srcF);
		igl::read_triangle_mesh(avatar_mesh_path, tgtV, tgtF);

		const double Vs = mesh::closed_mesh_volume(srcV, srcF);
		const double Vt = mesh::closed_mesh_volume(tgtV, tgtF);
		const double shrink = (Vs > 0.0) ? std::max(0.0, (Vs - Vt) / Vs) : 0.0;
		a1_h_global = std::min(1.0, std::max(0.0, shrink / std::max(step3_r0, 1e-12)));

		const double fit_base = in_args["fit_weight"];
		const double sim_base = in_args["similarity_penalty_weight"];
		a1_fit_w_phase1_global = fit_base * (1.0 + step3_a * a1_h_global);
		a1_fit_w_phase2_global = fit_base * (1.0 + step3_a2 * a1_h_global);
		a1_sim_w_phase1_global = sim_base * step3_s1;
		a1_sim_w_phase2_global = sim_base;

		logger().info("[A1] precheck Vs={} Vt={} shrink={} h={}", Vs, Vt, shrink, a1_h_global);
		if (a1_h_global > 0.0)
			logger().info("[A1] Phase1 (TIGHTEN): fit_weight={} similarity_weight={}", a1_fit_w_phase1_global, a1_sim_w_phase1_global);
		else
			logger().info("[A1] no shrink detected (h=0), annealing will be skipped.");
	}

	// Helper: read integer vertex indices from either `indices` or `indices_path`.
	// If `indices_path` exists but is empty/whitespace-only, treat it as "no indices" (no-op mask).
	auto read_indices_from_mask = [&](const json &mask, Eigen::MatrixXi &tmp_vids, const std::string &context) {
		if (utils::is_param_valid(mask, "indices_path")) {
			const std::string path = mask["indices_path"];
			if (!std::filesystem::exists(path)) {
				log_and_throw_error("{} indices_path file not found: {}", context, path);
			}
			std::ifstream in(path);
			if (!in.is_open()) {
				log_and_throw_error("{} unable to open indices_path file: {}", context, path);
			}
			std::vector<int> ids;
			ids.reserve(256);
			int id = -1;
			while (in >> id) {
				ids.push_back(id);
			}
			tmp_vids.resize((int)ids.size(), 1);
			for (int i = 0; i < (int)ids.size(); ++i)
				tmp_vids(i, 0) = ids[i];
			if (ids.empty()) {
				logger().debug("{} indices_path '{}' is empty; ignoring this mask.", context, path);
			}
		} else if (utils::is_param_valid(mask, "indices")) {
			const auto &arr = mask["indices"];
			tmp_vids.resize(arr.size(), 1);
			for (int i = 0; i < arr.size(); ++i)
				tmp_vids(i, 0) = arr[i];
		} else {
			log_and_throw_error("{} entry requires either indices or indices_path", context);
		}
	};

	std::vector<std::shared_ptr<Form>> persistent_forms;
	std::vector<std::shared_ptr<Form>> persistent_full_forms;
	std::shared_ptr<CurveSizeForm> curve_size_form;
	std::shared_ptr<SimilarityForm> similarity_form;
	Eigen::VectorXd sim_multipliers_base = Eigen::VectorXd::Ones(collision_vertices.rows());
	{
		similarity_form = std::make_shared<SimilarityForm>(collision_vertices, collision_triangles.bottomRows(gstate.n_garment_faces()));
		similarity_form->set_weight(in_args["similarity_penalty_weight"]);
		// Optional A/B/C switches for Similarity Hessian assembly only.
		// This keeps global threading (solver.max_threads, Fit SDF sampling, barrier, etc.) unchanged.
		if (utils::is_param_valid(in_args, "solver") && in_args["solver"].is_object())
		{
			const auto &solver = in_args["solver"];

			// High-level mode (recommended): optimized / optimized_serial / legacy
			if (solver.contains("similarity_hessian_mode") && solver["similarity_hessian_mode"].is_string())
			{
				const std::string mode = solver["similarity_hessian_mode"].get<std::string>();
				if (mode == "legacy")
				{
					similarity_form->set_parallel_hessian(false);
					similarity_form->set_use_dx_map(false);
					similarity_form->set_reserve_triplets(false);
				}
				else if (mode == "optimized_serial")
				{
					similarity_form->set_parallel_hessian(false);
					similarity_form->set_use_dx_map(true);
					similarity_form->set_reserve_triplets(true);
				}
				else if (mode == "optimized")
				{
					similarity_form->set_parallel_hessian(true);
					similarity_form->set_use_dx_map(true);
					similarity_form->set_reserve_triplets(true);
				}
				else
				{
					log_and_throw_error("solver.similarity_hessian_mode must be one of: 'optimized', 'optimized_serial', 'legacy'. Got '{}'.", mode);
				}
				logger().debug("[perf] similarity_hessian_mode={}", mode);
			}
			else
			{
				// Backward-compatible: old flag for parallel-only toggle
				const bool parallel_sim_hess = solver.value("parallel_similarity_hessian", true);
				similarity_form->set_parallel_hessian(parallel_sim_hess);
				logger().debug("[perf] parallel_similarity_hessian={}", parallel_sim_hess);
			}

			// Fine-grained overrides (optional)
			if (solver.contains("similarity_hessian_use_dx_map") && solver["similarity_hessian_use_dx_map"].is_boolean())
				similarity_form->set_use_dx_map(solver["similarity_hessian_use_dx_map"].get<bool>());
			if (solver.contains("similarity_hessian_reserve_triplets") && solver["similarity_hessian_reserve_triplets"].is_boolean())
				similarity_form->set_reserve_triplets(solver["similarity_hessian_reserve_triplets"].get<bool>());

			// Medium-risk optimization: process each interior adjacency once (unique undirected edges).
			const bool unique_adj = solver.value("similarity_unique_adjacency", false);
			similarity_form->set_use_unique_adjacency(unique_adj);
			logger().debug("[perf] similarity_unique_adjacency={}", unique_adj);
		}
		// Optional per-vertex multipliers for similarity
		if (in_args.contains("similarity_weight_masks"))
		{
			// Start from 1s; apply user masks (last one wins). We'll keep this as a reusable base.
			sim_multipliers_base.setOnes();
			for (const auto &mask : in_args["similarity_weight_masks"]) {
				double m = mask["multiplier"].get<double>();
				m = std::max(0.0, m);
				Eigen::MatrixXi tmp_vids;
				read_indices_from_mask(mask, tmp_vids, "similarity_weight_masks");

				// Optional mesh hint: "garment", "avatar", or "collision" (default: infer)
				std::string mesh_type = utils::is_param_valid(mask, "mesh") ? mask["mesh"].get<std::string>() : std::string("");
				const int n_avatar_verts_local = gstate.nc_avatar_v.rows();
				const int n_garment_verts_local = gstate.n_garment_vertices();
				const int total_rows = (int)collision_vertices.rows();
				for (int i = 0; i < tmp_vids.size(); i++) {
					const int v = tmp_vids(i);
					int mapped = v;
					if (mesh_type == "garment")
						mapped = n_avatar_verts_local + v;
					else if (mesh_type == "avatar")
						mapped = v; // already avatar space
					else if (mesh_type == "collision")
						mapped = v; // direct index into collision vertices
					else {
						// infer: if it looks like a garment index, offset it
						mapped = (v >= 0 && v < n_garment_verts_local) ? (n_avatar_verts_local + v) : v;
					}

					if (mapped < 0 || mapped >= total_rows)
						log_and_throw_error("Vertex ID {} (mapped {}) in similarity_weight_masks out of range!", v, mapped);
					sim_multipliers_base(mapped) = m;
				}
			}
		}
		// Always set (so Step3 can temporarily override and then restore).
		similarity_form->set_vertex_multipliers(sim_multipliers_base);
		persistent_forms.push_back(similarity_form);

		if (in_args["curvature_penalty_weight"] > 0)
		{
			auto curvature_form = std::make_shared<CurveCurvatureForm>(collision_vertices, curves);
			curvature_form->set_weight(in_args["curvature_penalty_weight"]);
			persistent_forms.push_back(curvature_form);
		}

		if (in_args["twist_penalty_weight"] > 0)
		{
			auto twist_form = std::make_shared<CurveTorsionForm>(collision_vertices, curves);
			twist_form->set_weight(in_args["twist_penalty_weight"]);
			persistent_forms.push_back(twist_form);
		}

		if (in_args["symmetry_weight"] > 0)
		{
			auto sym_form = std::make_shared<SymmetryForm>(collision_vertices, curves);
			sym_form->set_weight(in_args["symmetry_weight"]);
			if (sym_form->enabled())
				persistent_forms.push_back(sym_form);
		}

		if (in_args["curve_size_weight"] > 0)
		{
			curve_size_form = std::make_shared<CurveSizeForm>(collision_vertices, curves);
			curve_size_form->disable();
			curve_size_form->set_weight(in_args["curve_size_weight"]);
			persistent_forms.push_back(curve_size_form);
		}

		if (in_args["contact"]["enabled"]) {
			const double dhat = in_args["contact"]["dhat"];
			std::shared_ptr<ContactForm> contact_form = std::make_shared<ContactForm>(collision_mesh, dhat, 1, false, false, false, false, in_args["solver"]["contact"]["CCD"]["broad_phase"], in_args["solver"]["contact"]["CCD"]["tolerance"], in_args["solver"]["contact"]["CCD"]["max_iterations"]);
			contact_form->set_weight(1);
			contact_form->set_barrier_stiffness(in_args["solver"]["contact"]["barrier_stiffness"]);
			contact_form->save_ccd_debug_meshes = false;
			persistent_forms.push_back(contact_form);
		}

		// ------------------------------------------------------------
		// Curve-center target constraints:
		// - Default: apply to all boundary curves (legacy behavior).
		// - Optional hem-only mode: select hem boundary loop(s) by overlap with a manual vertex-id set,
		//   apply curve-center target to those loops only, and enable skirt mid-leg trick only for hem.
		const auto tmp_curves = boundary_curves(gstate.garment.f);

		// Optional: multi-ring manual constraints (backward-compatible extension).
		// If configured, this is additive to existing hem_boundary behavior.
		if (in_args.contains("ring_constraints") && in_args["ring_constraints"].is_object())
		{
			const auto &rc = in_args["ring_constraints"];
			if (rc.contains("manual") && rc["manual"].is_array())
			{
				auto sanitize_name = [](const std::string &s) -> std::string {
					std::string out;
					out.reserve(s.size());
					for (char c : s)
					{
						if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-')
							out.push_back(c);
						else
							out.push_back('_');
					}
					if (out.empty())
						out = "ring";
					return out;
				};

				int ring_ok = 0;
				for (int ri = 0; ri < (int)rc["manual"].size(); ++ri)
				{
					const auto &item = rc["manual"][ri];
					if (!item.is_object())
					{
						logger().warn("[ring_constraints] manual[{}] is not an object; skipped.", ri);
						continue;
					}
					if (!item.value("enabled", true))
						continue;

					const std::string ring_name = sanitize_name(item.value("name", std::string("ring_" + std::to_string(ri))));
					const std::string ring_label = "[ring_constraints(" + ring_name + ")]";
					const std::string ids_path = item.value("vertex_ids_path", std::string(""));
					if (ids_path.empty())
					{
						log_and_throw_error("{} missing required field vertex_ids_path", ring_label);
					}
					const int index_base = item.value("index_base", 1);
					if (index_base != 0 && index_base != 1)
						log_and_throw_error("{} index_base must be 0 or 1, got {}", ring_label, index_base);

					const int max_curves = std::max(1, item.value("max_curves", 1));
					const int dilate_steps = item.value("dilate_steps", -1);
					const int auto_dilate_max_steps = std::max(0, item.value("auto_dilate_max_steps", 4));
					const std::string select_mode = item.value("select_mode", std::string("overlap"));
					const double accept_overlap = item.value("accept_overlap", 0.5);
					const double accept_coverage = item.value("accept_coverage", 0.7);
					const double curv_w = item.value("curvature_penalty_weight", 0.0);
					const double twist_w = item.value("twist_penalty_weight", 0.0);
					const double center_w_override = item.value("curve_center_target_weight", -1.0);
					const double center_w = (center_w_override >= 0.0) ? center_w_override : in_args["curve_center_target_weight"].get<double>();
					const bool is_skirt_ring = item.value("is_skirt", false);
					const bool export_skel_dbg = item.value("export_skeleton_debug", false);
					std::string skel_dbg_prefix = item.value("skeleton_debug_prefix", std::string(""));
					if (skel_dbg_prefix.empty())
						skel_dbg_prefix = "ring_" + ring_name + "_skeleton";

					const auto ring_set = read_vertex_id_set_ascii(
						ids_path,
						index_base,
						gstate.n_garment_vertices(),
						ring_label);

					logger().info("{} extracting loop(s): max_curves={} select_mode={} dilate_steps={} auto_max={}",
						ring_label, max_curves, select_mode, dilate_steps, auto_dilate_max_steps);

					std::vector<unsigned char> used_in_set;
					int auto_steps_used = 0;
					const auto ring_loops = extract_vertexset_region_boundary_loops(
						gstate.garment.v,
						gstate.garment.f,
						ring_set,
						gstate.n_garment_vertices(),
						max_curves,
						ring_label,
						&used_in_set,
						&auto_steps_used,
						dilate_steps,
						auto_dilate_max_steps,
						select_mode,
						accept_overlap,
						accept_coverage);
					logger().info("{} auto_dilate_steps_used={}", ring_label, auto_steps_used);

					std::vector<Eigen::VectorXi> ring_curves;
					ring_curves.reserve(ring_loops.size());
					for (int li = 0; li < (int)ring_loops.size(); ++li)
					{
						const auto &loop = ring_loops[li];
						if ((int)loop.size() < 4)
							continue;
						Eigen::VectorXi c(loop.size());
						for (int k = 0; k < (int)loop.size(); ++k)
							c(k) = loop[k];
						ring_curves.push_back(std::move(c));
						logger().info("{} extracted loop {}: n_unique={}", ring_label, li, (int)loop.size() - 1);
					}
					if (ring_curves.empty())
					{
						logger().warn("{} failed to extract a closed loop; skipping this ring.", ring_label);
						continue;
					}
					++ring_ok;

					// Per-ring debug exports.
					try
					{
						std::unordered_set<int> sel;
						sel.reserve(8192);
						for (const auto &c : ring_curves)
						{
							const int n = c.size();
							if (n <= 1)
								continue;
							const int m = (n >= 2 && c(0) == c(n - 1)) ? (n - 1) : n;
							for (int k = 0; k < m; ++k)
								sel.insert((int)c(k));
						}

						std::vector<RGB8> colors;
						colors.resize(gstate.n_garment_vertices(), RGB8{128, 128, 128});
						if ((int)used_in_set.size() == gstate.n_garment_vertices())
							for (int v = 0; v < gstate.n_garment_vertices(); ++v)
								if (used_in_set[(size_t)v])
									colors[(size_t)v] = RGB8{0, 160, 255};
						for (const int v : ring_set)
							if (v >= 0 && v < gstate.n_garment_vertices())
								colors[v] = RGB8{0, 255, 0};
						for (const int v : sel)
							if (v >= 0 && v < gstate.n_garment_vertices())
								colors[v] = RGB8{255, 255, 0};

						const std::string dbg_prefix = "ring_" + ring_name;
						write_ply_colored_vertices(out_folder + "/" + dbg_prefix + "_vertexset_loops.ply", gstate.garment.v, gstate.garment.f, colors);
						std::ofstream txt(out_folder + "/" + dbg_prefix + "_vertexset_loops.txt", std::ios::out);
						if (txt.is_open())
						{
							for (int li = 0; li < (int)ring_curves.size(); ++li)
							{
								const auto &c = ring_curves[li];
								const int n = c.size();
								const int uniq = (n >= 2 && c(0) == c(n - 1)) ? (n - 1) : n;
								txt << "loop_" << li << " n_unique=" << uniq << " ";
								for (int k = 0; k < n; ++k)
								{
									txt << c(k);
									if (k + 1 < n)
										txt << " ";
								}
								txt << "\n";
							}
						}
					}
					catch (...)
					{
						logger().warn("{} failed to write debug loop exports.", ring_label);
					}

					// Map garment-local ring curves to collision index space.
					std::vector<Eigen::VectorXi> ring_curves_collision;
					ring_curves_collision.reserve(ring_curves.size());
					const int n_avatar_verts_local = gstate.nc_avatar_v.rows();
					for (const auto &c : ring_curves)
					{
						Eigen::VectorXi cc = c;
						for (int i = 0; i < cc.size(); ++i)
							cc(i) += n_avatar_verts_local;
						ring_curves_collision.push_back(std::move(cc));
					}

					if (curv_w > 0.0)
					{
						auto f = std::make_shared<CurveCurvatureForm>(collision_vertices, ring_curves_collision);
						f->set_weight(curv_w);
						persistent_forms.push_back(f);
						logger().info("{} enabled curvature regularization: weight={}", ring_label, curv_w);
					}
					if (twist_w > 0.0)
					{
						auto f = std::make_shared<CurveTorsionForm>(collision_vertices, ring_curves_collision);
						f->set_weight(twist_w);
						persistent_forms.push_back(f);
						logger().info("{} enabled twist regularization: weight={}", ring_label, twist_w);
					}

					if (center_w > 0.0)
					{
						auto ring_center_target_form = std::make_shared<CurveTargetForm>(
							initial_garment_v,
							ring_curves,
							gstate.skeleton_v,
							gstate.target_skeleton_v,
							gstate.skeleton_b,
							is_skirt_ring,
							in_args["curve_center_target_automatic_bone_generation"]);
						ring_center_target_form->set_weight(center_w);
						persistent_full_forms.push_back(ring_center_target_form);
						logger().info("{} enabled center-target: weight={} is_skirt={}", ring_label, center_w, is_skirt_ring);

						if (export_skel_dbg)
						{
							try
							{
								auto write_skeleton_points_ply = [&](const std::string &path, const Eigen::MatrixXd &SV) {
									std::ofstream ply(path, std::ios::out);
									if (!ply.is_open()) return;
									ply << "ply\nformat ascii 1.0\n";
									ply << "element vertex " << SV.rows() << "\n";
									ply << "property float x\nproperty float y\nproperty float z\n";
									ply << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
									ply << "element face 0\nproperty list uchar int vertex_indices\nend_header\n";
									const int n = SV.rows();
									const int mid0 = n - 2;
									const int mid1 = n - 1;
									for (int i = 0; i < n; ++i)
									{
										const bool is_mid = (ring_center_target_form->debug_is_skirt() && (i == mid0 || i == mid1));
										const int r = is_mid ? 255 : 220;
										const int g = is_mid ? 60 : 220;
										const int b = is_mid ? 60 : 220;
										ply << (float)SV(i, 0) << " " << (float)SV(i, 1) << " " << (float)SV(i, 2) << " " << r << " " << g << " " << b << "\n";
									}
								};
								auto write_skeleton_edges_txt = [&](const std::string &path, const Eigen::MatrixXi &SE) {
									std::ofstream txt(out_folder + "/" + path, std::ios::out);
									if (!txt.is_open()) return;
									const int m = SE.rows();
									for (int e = 0; e < m; ++e)
									{
										const bool is_mid = (ring_center_target_form->debug_is_skirt() && (e >= m - 2));
										txt << e << " " << SE(e, 0) << " " << SE(e, 1) << (is_mid ? " mid_leg\n" : "\n");
									}
								};
								const auto &SSV = ring_center_target_form->debug_source_skeleton_v();
								const auto &TSV = ring_center_target_form->debug_target_skeleton_v();
								const auto &SE = ring_center_target_form->debug_skeleton_edges();
								write_skeleton_points_ply(out_folder + "/" + skel_dbg_prefix + "_source_joints.ply", SSV);
								write_skeleton_points_ply(out_folder + "/" + skel_dbg_prefix + "_target_joints.ply", TSV);
								write_skeleton_edges_txt(skel_dbg_prefix + "_edges.txt", SE);
							}
							catch (...)
							{
								logger().warn("{} failed to write skeleton debug exports.", ring_label);
							}
						}
					}
				}
				logger().info("[ring_constraints] manual processed: {} successful entries", ring_ok);
			}
		}

		bool hem_enabled = false;
		std::string hem_ids_path;
		int hem_index_base = 1;
		double hem_overlap_threshold = 0.2;
		int hem_max_curves = 1;
		// Optional tuning knobs for region-loop extraction
		int hem_dilate_steps = -1;            // >=0: fixed/manual dilation; <0: auto
		int hem_auto_dilate_max_steps = 4;    // auto-dilate cap (ignored if hem_dilate_steps>=0)
		std::string hem_select_mode = "overlap"; // "overlap" or "perimeter"
		double hem_accept_overlap = 0.50;     // overlap acceptance threshold (for overlap mode)
		double hem_accept_coverage = 0.70;    // coverage acceptance threshold
		bool hem_export_skeleton_debug = false;
		std::string hem_skeleton_debug_prefix = "hem_boundary_skeleton";
		double hem_curvature_penalty_weight = 0.0;
		double hem_twist_penalty_weight = 0.0;
		if (in_args.contains("hem_boundary") && in_args["hem_boundary"].is_object())
		{
			const auto &hb = in_args["hem_boundary"];
			hem_enabled = hb.value("enabled", false);
			hem_ids_path = hb.value("vertex_ids_path", std::string(""));
			hem_index_base = hb.value("index_base", hem_index_base);
			hem_overlap_threshold = hb.value("overlap_threshold", hem_overlap_threshold);
			hem_max_curves = hb.value("max_curves", hem_max_curves);
			hem_dilate_steps = hb.value("dilate_steps", hem_dilate_steps);
			hem_auto_dilate_max_steps = hb.value("auto_dilate_max_steps", hem_auto_dilate_max_steps);
			hem_select_mode = hb.value("select_mode", hem_select_mode);
			hem_accept_overlap = hb.value("accept_overlap", hem_accept_overlap);
			hem_accept_coverage = hb.value("accept_coverage", hem_accept_coverage);
			hem_export_skeleton_debug = hb.value("export_skeleton_debug", hem_export_skeleton_debug);
			hem_skeleton_debug_prefix = hb.value("skeleton_debug_prefix", hem_skeleton_debug_prefix);
			hem_curvature_penalty_weight = hb.value("curvature_penalty_weight", hem_curvature_penalty_weight);
			hem_twist_penalty_weight = hb.value("twist_penalty_weight", hem_twist_penalty_weight);
			hem_max_curves = std::max(1, hem_max_curves);
			hem_auto_dilate_max_steps = std::max(0, hem_auto_dilate_max_steps);
			if (hem_dilate_steps >= 0)
				hem_dilate_steps = std::max(0, hem_dilate_steps);
			if (hem_index_base != 0 && hem_index_base != 1)
				log_and_throw_error("hem_boundary.index_base must be 0 or 1, got {}", hem_index_base);
		}

		if (hem_enabled)
		{
			if (hem_ids_path.empty())
				log_and_throw_error("hem_boundary.enabled=true but hem_boundary.vertex_ids_path is missing/empty.");

			const auto hem_set = read_vertex_id_set_ascii(
				hem_ids_path,
				hem_index_base,
				gstate.n_garment_vertices(),
				"hem_boundary");

			// For watertight clothed-human meshes, the true hem is typically NOT a topological open boundary.
			// In hem-boundary mode we therefore treat the provided vertex-id set as a REGION and extract its
			// boundary loop(s) from edges separating hem_set and non-hem_set. This is robust to thick-band labels.
			logger().info("[hem_boundary] extracting boundary loop(s) of vertex-id region. max_curves={} (overlap_threshold={} ignored)",
				hem_max_curves, hem_overlap_threshold);
			std::vector<unsigned char> hem_used_in_set;
			int hem_auto_dilate_steps = 0;
			const auto hem_loops = extract_vertexset_region_boundary_loops(
				gstate.garment.v,
				gstate.garment.f,
				hem_set,
				gstate.n_garment_vertices(),
				hem_max_curves,
				"[hem_boundary(region_boundary)]",
				&hem_used_in_set,
				&hem_auto_dilate_steps,
				hem_dilate_steps,
				hem_auto_dilate_max_steps,
				hem_select_mode,
				hem_accept_overlap,
				hem_accept_coverage);
			logger().info("[hem_boundary] auto_dilate_steps_used={}", hem_auto_dilate_steps);

			std::vector<Eigen::VectorXi> hem_curves;
			hem_curves.reserve(hem_loops.size());
			for (int li = 0; li < (int)hem_loops.size(); ++li)
			{
				const auto &loop = hem_loops[li];
				if ((int)loop.size() < 4) // includes repeated start
					continue;
				Eigen::VectorXi c(loop.size());
				for (int k = 0; k < (int)loop.size(); ++k)
					c(k) = loop[k];
				hem_curves.push_back(std::move(c));
				logger().info("[hem_boundary] extracted loop {}: n_unique={}", li, (int)loop.size() - 1);
			}

			// Debug exports: extracted loop(s) vs input vertex set.
			try
			{
				// Collect all loop vertices (unique, exclude repeated last).
				std::unordered_set<int> sel;
				sel.reserve(8192);
				for (const auto &c : hem_curves)
				{
					const int n = c.size();
					if (n <= 1)
						continue;
					const int m = (n >= 2 && c(0) == c(n - 1)) ? (n - 1) : n;
					for (int k = 0; k < m; ++k)
						sel.insert((int)c(k));
				}

				// 1) PLY visualization (garment mesh with per-vertex colors).
				std::vector<RGB8> colors;
				colors.resize(gstate.n_garment_vertices(), RGB8{128, 128, 128});
				// Visualize the (possibly auto-dilated) working set in blue, and the original input set in green.
				// This helps diagnose "thin ring" selections where dilation is needed to form triangle-complete regions.
				if ((int)hem_used_in_set.size() == gstate.n_garment_vertices())
				{
					for (int v = 0; v < gstate.n_garment_vertices(); ++v)
						if (hem_used_in_set[(size_t)v])
							colors[(size_t)v] = RGB8{0, 160, 255}; // used working set (blue)
				}
				for (const int v : hem_set)
				{
					if (v >= 0 && v < gstate.n_garment_vertices())
						colors[v] = RGB8{0, 255, 0}; // input set (green)
				}
				for (const int v : sel)
				{
					if (v < 0 || v >= gstate.n_garment_vertices())
						continue;
					colors[v] = RGB8{255, 255, 0}; // extracted loop vertices (yellow)
				}
				write_ply_colored_vertices(out_folder + "/hem_boundary_vertexset_loops.ply", gstate.garment.v, gstate.garment.f, colors);
				// Keep legacy filename for continuity.
				write_ply_colored_vertices(out_folder + "/hem_boundary_selected_ring.ply", gstate.garment.v, gstate.garment.f, colors);

				// 2) ASCII dump of ordered vertex ids (0-based).
				std::ofstream txt(out_folder + "/hem_boundary_vertexset_loops.txt", std::ios::out);
				if (txt.is_open())
				{
					for (int li = 0; li < (int)hem_curves.size(); ++li)
					{
						const auto &c = hem_curves[li];
						const int n = c.size();
						const int uniq = (n >= 2 && c(0) == c(n - 1)) ? (n - 1) : n;
						txt << "loop_" << li << " n_unique=" << uniq << " ";
						for (int k = 0; k < n; ++k)
						{
							txt << c(k);
							if (k + 1 < n)
								txt << " ";
						}
						txt << "\n";
					}
				}

				logger().info("[hem_boundary] wrote debug: hem_boundary_vertexset_loops.ply + hem_boundary_vertexset_loops.txt");
			}
			catch (...)
			{
				logger().warn("[hem_boundary] failed to write hem vertex-set debug exports (continuing).");
			}

			if (hem_curves.empty())
			{
				logger().warn("[hem_boundary] failed to extract a closed loop from vertex-id set. Skipping hem boundary constraint.");
			}
			else
			{
				// Map hem curves (garment-local vertex ids) to collision-vertex ids used by curve forms.
				// The "complete" vector stacks [avatar_vertices, garment_vertices].
				std::vector<Eigen::VectorXi> hem_curves_collision;
				hem_curves_collision.reserve(hem_curves.size());
				const int n_avatar_verts_local = gstate.nc_avatar_v.rows();
				for (const auto &c : hem_curves)
				{
					Eigen::VectorXi cc = c;
					for (int i = 0; i < cc.size(); ++i)
						cc(i) += n_avatar_verts_local;
					hem_curves_collision.push_back(std::move(cc));
				}

				// Optional: hem-only curvature/twist regularizers (operate on hem_curves, even if the mesh is watertight).
				if (hem_curvature_penalty_weight > 0.0)
				{
					auto hem_curv_form = std::make_shared<CurveCurvatureForm>(collision_vertices, hem_curves_collision);
					hem_curv_form->set_weight(hem_curvature_penalty_weight);
					persistent_forms.push_back(hem_curv_form);
					logger().info("[hem_boundary] enabled hem curvature regularization: curvature_penalty_weight={}", hem_curvature_penalty_weight);
				}
				if (hem_twist_penalty_weight > 0.0)
				{
					auto hem_twist_form = std::make_shared<CurveTorsionForm>(collision_vertices, hem_curves_collision);
					hem_twist_form->set_weight(hem_twist_penalty_weight);
					persistent_forms.push_back(hem_twist_form);
					logger().info("[hem_boundary] enabled hem twist regularization: twist_penalty_weight={}", hem_twist_penalty_weight);
				}

				// Hem-only CurveTargetForm with skirt mid-leg trick enabled.
				if (in_args["curve_center_target_weight"] > 0)
				{
					auto hem_center_target_form = std::make_shared<CurveTargetForm>(
						initial_garment_v,
						hem_curves,
						gstate.skeleton_v,
						gstate.target_skeleton_v,
						gstate.skeleton_b,
						true, // is_skirt: enable mid-leg trick only for hem
						in_args["curve_center_target_automatic_bone_generation"]);
					hem_center_target_form->set_weight(in_args["curve_center_target_weight"]);
					persistent_full_forms.push_back(hem_center_target_form);

					// Optional debug: export the (possibly modified) skeleton used by CurveTargetForm.
					// This is the only reliable way to visualize the inserted mid-leg bone, since insertion
					// happens inside CurveTargetForm (it keeps a private copy of skeleton vertices/edges).
					if (hem_export_skeleton_debug)
					{
						auto write_skeleton_points_ply = [&](const std::string &path, const Eigen::MatrixXd &SV) {
							std::ofstream ply(path, std::ios::out);
							if (!ply.is_open()) return;
							ply << "ply\nformat ascii 1.0\n";
							ply << "element vertex " << SV.rows() << "\n";
							ply << "property float x\nproperty float y\nproperty float z\n";
							ply << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
							ply << "element face 0\nproperty list uchar int vertex_indices\nend_header\n";
							const int n = SV.rows();
							const int mid0 = n - 2;
							const int mid1 = n - 1;
							for (int i = 0; i < n; ++i)
							{
								const bool is_mid = (hem_center_target_form->debug_is_skirt() && (i == mid0 || i == mid1));
								const int r = is_mid ? 255 : 220;
								const int g = is_mid ? 60  : 220;
								const int b = is_mid ? 60  : 220;
								ply << (float)SV(i,0) << " " << (float)SV(i,1) << " " << (float)SV(i,2)
									<< " " << r << " " << g << " " << b << "\n";
							}
						};

						auto write_skeleton_edges_txt = [&](const std::string &path, const Eigen::MatrixXi &SE) {
							std::ofstream txt(out_folder + "/" + path, std::ios::out);
							if (!txt.is_open()) return;
							const int m = SE.rows();
							for (int e = 0; e < m; ++e)
							{
								const bool is_mid = (hem_center_target_form->debug_is_skirt() && (e >= m - 2));
								txt << e << " " << SE(e, 0) << " " << SE(e, 1) << (is_mid ? " mid_leg\n" : "\n");
							}
						};

						const auto &SSV = hem_center_target_form->debug_source_skeleton_v();
						const auto &TSV = hem_center_target_form->debug_target_skeleton_v();
						const auto &SE  = hem_center_target_form->debug_skeleton_edges();
						write_skeleton_points_ply(out_folder + "/" + hem_skeleton_debug_prefix + "_source_joints.ply", SSV);
						write_skeleton_points_ply(out_folder + "/" + hem_skeleton_debug_prefix + "_target_joints.ply", TSV);
						write_skeleton_edges_txt(hem_skeleton_debug_prefix + "_edges.txt", SE);
						logger().info("[hem_boundary] wrote skeleton debug: {}_source_joints.ply, {}_target_joints.ply, {}_edges.txt",
							hem_skeleton_debug_prefix, hem_skeleton_debug_prefix, hem_skeleton_debug_prefix);
					}
				}
			}
		}
		else
		{
			// Legacy behavior: curve-center constraints on all boundary curves.
			auto center_target_form = std::make_shared<CurveTargetForm>(
				initial_garment_v,
				tmp_curves,
				gstate.skeleton_v,
				gstate.target_skeleton_v,
				gstate.skeleton_b,
				in_args["is_skirt"],
				in_args["curve_center_target_automatic_bone_generation"]);
			center_target_form->set_weight(in_args["curve_center_target_weight"]);
			persistent_full_forms.push_back(center_target_form);
		}
	}

	Eigen::MatrixXd sol = Eigen::MatrixXd::Zero(1 + initial_garment_v.size(), 1);
	int64_t tb_global_step = 0;

	// Optional progress logging (debug): substep + phase + nonlinear iteration
	struct ProgressOpt
	{
		bool enabled = false;
		int log_every = 10; // 1 => every nonlinear iteration
	};
	ProgressOpt prog;
	if (utils::is_param_valid(in_args, "output") && in_args["output"].is_object())
	{
		const auto &out = in_args["output"];
		if (out.contains("progress") && out["progress"].is_object())
		{
			const auto &p = out["progress"];
			prog.enabled = p.value("enabled", false);
			prog.log_every = std::max(1, p.value("log_every", prog.log_every));
		}
	}
	int64_t prog_step = 0;
	std::string prog_phase = "";
	int prog_substep = 0;
	int prog_total_substeps = std::max(1, total_steps);
	int prog_max_iter = 0;
	int prog_al_outer = 0;
	double prog_al_weight = 0.0;
	for (int substep = 0; substep < total_steps; ++substep)
	{
		utils::ChromeTraceScope trace_substep(fmt::format("substep_{}", substep));
		const double prev_alpha = substep / (double)total_steps;
		const double next_alpha = (substep + 1) / (double)total_steps;

		logger().info("Start substep {} out of {}", substep + 1, total_steps);

		// continuation
		const Eigen::MatrixXd next_avatar_v = (gstate.nc_avatar_v - gstate.skinny_avatar_v) * next_alpha + gstate.skinny_avatar_v;
		const Eigen::MatrixXd next_curve_centers = (target_curve_centers - source_curve_centers) * next_alpha + source_curve_centers;

		std::vector<std::shared_ptr<Form>> forms = persistent_forms;
		std::shared_ptr<PointPenaltyForm> pen_form;
		std::shared_ptr<PointLagrangianForm> lagr_form;
		std::shared_ptr<FitForm<4>> fit_form;
		{
			std::vector<int> indices(gstate.nc_avatar_v.size());
			for (int i = 0; i < indices.size(); i++)
				indices[i] = i;
			// Apply per-vertex continuation multipliers (like fit_weight_masks) to damp pull on selected regions
			Eigen::VectorXd disp = utils::flatten(next_avatar_v - gstate.skinny_avatar_v);
			Eigen::VectorXd cont_multipliers = Eigen::VectorXd::Ones(gstate.nc_avatar_v.rows());
			std::vector<int> vis_avatar_marked;
			std::vector<int> vis_garment_marked;
			if (in_args.contains("continuation_weight_masks"))
			{
				for (const auto &mask : in_args["continuation_weight_masks"]) {
					double m = mask["multiplier"].get<double>();
					m = std::max(0.0, m);
					Eigen::MatrixXi tmp_vids;
					if (utils::is_param_valid(mask, "indices_path")) {
						polyfem::io::read_matrix<int>(mask["indices_path"], tmp_vids);
					} else if (utils::is_param_valid(mask, "indices")) {
						const auto &arr = mask["indices"];
						tmp_vids.resize(arr.size(), 1);
						for (int i = 0; i < arr.size(); ++i) tmp_vids(i) = arr[i];
					} else {
						log_and_throw_error("continuation_weight_masks entry requires either indices or indices_path");
					}
					std::string mesh_type = utils::is_param_valid(mask, "mesh") ? mask["mesh"].get<std::string>() : std::string("");
					for (int i = 0; i < tmp_vids.size(); i++) {
						const int id = tmp_vids(i);
						if (mesh_type == "garment") {
							if (id < 0 || id >= gstate.n_garment_vertices())
								log_and_throw_error("Vertex ID {} in continuation_weight_masks (garment) out of range!", id);
							// Map garment vertex to nearest avatar vertex by position
							const Eigen::RowVector3d pg = gstate.garment.v.row(id);
							int best = -1; double bestd = std::numeric_limits<double>::infinity();
							for (int av = 0; av < gstate.nc_avatar_v.rows(); ++av) {
								double d = (gstate.nc_avatar_v.row(av) - pg).squaredNorm();
								if (d < bestd) { bestd = d; best = av; }
							}
							if (best >= 0) {
								cont_multipliers(best) = m;
								vis_garment_marked.push_back(id);
								vis_avatar_marked.push_back(best);
							}
						} else {
							// Assume avatar indices
							if (id < 0 || id >= gstate.nc_avatar_v.rows())
								log_and_throw_error("Vertex ID {} in continuation_weight_masks (avatar) out of range!", id);
							cont_multipliers(id) = m;
							vis_avatar_marked.push_back(id);
						}
					}
				}
			}
			for (int v = 0; v < gstate.nc_avatar_v.rows(); ++v)
				disp.segment<3>(3 * v) *= cont_multipliers(v);

			// Optional visualization of marked vertices (avatar/garment)
			auto write_colored_ply = [&](const std::string &path, const Eigen::MatrixXd &V, const std::vector<int> &marked) {
				std::ofstream ply(path, std::ios::out);
				if (!ply.is_open()) return;
				ply << "ply\nformat ascii 1.0\n";
				ply << "element vertex " << V.rows() << "\n";
				ply << "property float x\nproperty float y\nproperty float z\n";
				ply << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
				ply << "element face 0\nproperty list uchar int vertex_indices\nend_header\n";
				std::vector<char> is_marked(V.rows(), 0);
				for (int idx : marked) if (idx >= 0 && idx < V.rows()) is_marked[idx] = 1;
				for (int i = 0; i < V.rows(); ++i) {
					int r = is_marked[i] ? 220 : 60;
					int g = is_marked[i] ? 40  : 200;
					int b = 40;
					ply << (float)V(i,0) << " " << (float)V(i,1) << " " << (float)V(i,2) << " " << r << " " << g << " " << b << "\n";
				}
				ply.close();
			};
			if (!vis_avatar_marked.empty()) write_colored_ply(out_folder + "/continuation_avatar_mask.ply", gstate.nc_avatar_v, vis_avatar_marked);
			if (!vis_garment_marked.empty()) write_colored_ply(out_folder + "/continuation_garment_mask.ply", gstate.garment.v, vis_garment_marked);

			pen_form = std::make_shared<PointPenaltyForm>(disp, indices);
			forms.push_back(pen_form);

			lagr_form = std::make_shared<PointLagrangianForm>(disp, indices);
			forms.push_back(lagr_form);

			// Build per-vertex multipliers (default 1), with garment overrides from fit_weight_masks (last one wins)
			Eigen::VectorXd vertex_fit_multipliers = Eigen::VectorXd::Ones(collision_vertices.rows());
			Eigen::VectorXd garment_mask_multipliers = Eigen::VectorXd::Ones(gstate.n_garment_vertices());
			if (in_args.contains("fit_weight_masks"))
			{
				for (const auto &mask : in_args["fit_weight_masks"])
				{
					double m = mask["multiplier"].get<double>();
					m = std::max(0.0, m); // clamp to 0+
					Eigen::MatrixXi tmp_vids;
					read_indices_from_mask(mask, tmp_vids, "fit_weight_masks");
					for (int i = 0; i < tmp_vids.size(); i++)
					{
						const int v = tmp_vids(i);
						if (v < 0 || v >= gstate.n_garment_vertices())
							log_and_throw_error("Vertex ID {} in fit_weight_masks out of range!", v);
						garment_mask_multipliers(v) = m; // last one wins
					}
				}
			}

			// Apply A2 per-garment-vertex multipliers (Step3 only; typically final substep only).
			Eigen::VectorXd garment_final_multipliers = garment_mask_multipliers;
			const bool a2_active_this_substep = a2_enabled && (!a2_final_substep_only || (substep == total_steps - 1));
			if (a2_active_this_substep)
			{
				// Do not override suppression masks (<1). Only multiply where the existing multiplier is >= 1.
				for (int i = 0; i < garment_final_multipliers.size(); ++i)
					if (garment_final_multipliers(i) >= 1.0)
						garment_final_multipliers(i) *= a2_w(i);
			}

			const int n_avatar_verts = gstate.nc_avatar_v.rows();
			vertex_fit_multipliers.segment(n_avatar_verts, gstate.n_garment_vertices()) = garment_final_multipliers;

			// Optional: export fit-weight multipliers as heatmap PLYs (garment only).
			// Write once on the last substep, so it matches the "final" multipliers (esp. if A2 is final-substep-only).
			if (fit_weight_vis_enabled && !fit_weight_vis_written && (substep == total_steps - 1))
			{
				auto colors_from_multipliers = [&](const Eigen::VectorXd &m) -> std::vector<RGB8> {
					double max_abs_log = 0.0;
					for (int i = 0; i < m.size(); ++i)
					{
						const double x = std::max(1e-12, m(i));
						const double l = std::log(x);
						if (std::isfinite(l))
							max_abs_log = std::max(max_abs_log, std::abs(l));
					}
					max_abs_log = std::max(max_abs_log, 1e-12);
					std::vector<RGB8> c(m.size());
					for (int i = 0; i < m.size(); ++i)
					{
						const double x = std::max(1e-12, m(i));
						const double l = std::log(x);
						const double s = std::isfinite(l) ? (l / max_abs_log) : 0.0; // in [-1,1]
						c[i] = diverging_colormap(s);
					}
					return c;
				};

				if (fit_weight_vis_export_mask_only)
				{
					write_ply_colored_vertices(
						out_folder + "/" + fit_weight_vis_mask_only_filename,
						gstate.garment.v,
						gstate.garment.f,
						colors_from_multipliers(garment_mask_multipliers));
				}
				if (fit_weight_vis_export_final)
				{
					write_ply_colored_vertices(
						out_folder + "/" + fit_weight_vis_final_filename,
						gstate.garment.v,
						gstate.garment.f,
						colors_from_multipliers(garment_final_multipliers));
				}
				logger().info(
					"[fit_weight_vis] wrote PLY visualizations (mask_only={}, final={}, a2_active_last_substep={})",
					fit_weight_vis_export_mask_only ? fit_weight_vis_mask_only_filename : std::string("(disabled)"),
					fit_weight_vis_export_final ? fit_weight_vis_final_filename : std::string("(disabled)"),
					a2_active_this_substep);
				fit_weight_vis_written = true;
			}

			fit_form = std::make_shared<FitForm<4>>(collision_vertices, collision_triangles.bottomRows(gstate.n_garment_faces()), gstate.avatar_v, gstate.avatar_f, in_args["voxel_size"], gstate.not_fit_fids, out_folder, vertex_fit_multipliers);
			fit_form->disable();
			fit_form->set_weight(in_args["fit_weight"]);
			forms.push_back(fit_form);

			if (in_args["curve_size_weight"] > 0)
				curve_size_form->disable();
		}

		GarmentNLProblem nl_problem(1 + initial_garment_v.size(), utils::flatten(gstate.nc_avatar_v - gstate.skinny_avatar_v), forms, persistent_full_forms);
		nl_problem.set_target_value(next_alpha);

		nl_problem.line_search_begin(sol, sol);
		bool allow_nonfinite_bc = false;
		if (utils::is_param_valid(in_args, "initial_checks"))
		{
			const auto &ic = in_args["initial_checks"];
			const bool error_on_self = ic.value("error_on_garment_self_intersection", true);
			allow_nonfinite_bc = !error_on_self; // if user allows self-intersections, tolerate non-finite BC value
		}
		const double bc_val = nl_problem.value(sol);
		bool bc_finite = std::isfinite(bc_val);
		if (!bc_finite && allow_nonfinite_bc)
		{
			logger().warn("BC value is non-finite but proceeding (garment self-intersections allowed).");
			bc_finite = true;
		}
		const bool bc_valid = nl_problem.is_step_valid(sol, sol);
		const bool bc_collision_free = nl_problem.is_step_collision_free(sol, sol);
		if (!(bc_finite && bc_valid && bc_collision_free))
		{
			logger().error("BC check failed: finite={} valid_step={} collision_free={}", bc_finite, bc_valid, bc_collision_free);
			igl::write_triangle_mesh(out_folder + "/bc_check_projected_avatar.obj", gstate.skinny_avatar_v, gstate.nc_avatar_f);
			igl::write_triangle_mesh(out_folder + "/bc_check_garment.obj", gstate.garment.v, gstate.garment.f);
			log_and_throw_error("Failed to apply boundary conditions!");
		}

		std::shared_ptr<polysolve::nonlinear::Solver> nl_solver = polysolve::nonlinear::Solver::create(in_args["solver"]["augmented_lagrangian"]["nonlinear"], in_args["solver"]["linear"], 1., logger());
		const int max_iter_step2 = nl_solver->stop_criteria().iterations;

		double initial_weight = in_args["solver"]["augmented_lagrangian"]["initial_weight"];
		const double scaling = in_args["solver"]["augmented_lagrangian"]["scaling"];
		const double max_weight = in_args["solver"]["augmented_lagrangian"]["max_weight"].get<double>();
		const int max_outer_steps = in_args["solver"]["augmented_lagrangian"].value("max_outer_steps", 0);
		logger().debug("Set initial AL weight to {}", initial_weight);

		ALSolver<GarmentNLProblem, PointLagrangianForm, PointPenaltyForm> al_solver(
			lagr_form, pen_form,
			initial_weight, scaling, max_weight,
			in_args["solver"]["augmented_lagrangian"]["eta"],
			in_args["solver"]["augmented_lagrangian"]["error_threshold"],
			max_outer_steps,
			[](const Eigen::VectorXd &x) {});

		// Track AL outer steps/weight for progress logging
		prog_al_outer = 0;
		prog_al_weight = initial_weight;
		al_solver.post_subsolve = [&](const double w) {
			// Called after each AL outer solve with its current weight; also called after reduced solve with 0.
			if (prog_phase == "AL_step2")
			{
				++prog_al_outer;
				prog_al_weight = w;
			}
		};

		const bool is_last_substep = (substep == total_steps - 1);
		bool a1_active = false;
		bool a1_in_step3 = false;
		int a1_step3_iter = 0;
		int a1_stall_cnt = 0;
		double a1_last_p90 = std::numeric_limits<double>::infinity();
		bool a1_phase2 = false;
		const double a1_h = a1_h_global;
		const double a1_fit_w_phase1 = a1_fit_w_phase1_global;
		const double a1_fit_w_phase2 = a1_fit_w_phase2_global;
		const double a1_sim_w_phase1 = a1_sim_w_phase1_global;
		const double a1_sim_w_phase2 = a1_sim_w_phase2_global;

		nl_problem.post_step_call_back = [&](const polysolve::nonlinear::PostStepData &data) {
			const Eigen::VectorXd &sol = data.x;
			if (save_id % stride == 0)
				gstate.save_result(out_folder, save_id / stride, nl_problem, collision_vertices, collision_triangles, sol);
			++save_id;

			if (prog.enabled)
			{
				++prog_step;
				if ((prog_step % prog.log_every) == 0)
				{
					if (prog_phase == "AL_step2")
					{
						logger().debug("[progress] substep={}/{} phase={} outer={} weight={} iter={}/{}",
							prog_substep + 1, prog_total_substeps, prog_phase, prog_al_outer, prog_al_weight,
							data.iter_num + 1, prog_max_iter);
					}
					else if (prog_phase == "AL_step3")
					{
						logger().debug("[progress] substep={}/{} phase={} iter={}/{}",
							prog_substep + 1, prog_total_substeps, prog_phase,
							data.iter_num + 1, prog_max_iter);
					}
					else
					{
						logger().debug("[progress] substep={}/{} phase={} iter={}",
							prog_substep + 1, prog_total_substeps, prog_phase, data.iter_num + 1);
					}
				}
			}

			if (tb_logger.is_enabled())
			{
				++tb_global_step;
				const int log_every = std::max(1, tb_logger.options().log_every);
				if ((tb_global_step % log_every) == 0)
				{
					tb_logger.add_scalar("nl/grad_norm", tb_global_step, data.grad.norm());

					const int energy_every = std::max(1, tb_logger.options().energy_every);
					if ((tb_global_step % energy_every) == 0)
					{
						// Note: energy evaluation can be expensive; sample at a lower rate.
						if (tb_detailed_energies)
						{
							const Eigen::VectorXd y = nl_problem.reduced_to_full(data.x);
							const Eigen::VectorXd z = nl_problem.full_to_complete(y);

							double total = 0.0;
							for (const auto &f : nl_problem.forms())
							{
								if (!f->enabled())
									continue;
								if (f->weight() == 0.0)
									continue;
								const double e = f->value(z); // weighted
								total += e;
								tb_logger.add_scalar("energy/forms/" + f->name(), tb_global_step, e);
							}
							for (const auto &f : nl_problem.full_forms())
							{
								if (!f->enabled())
									continue;
								if (f->weight() == 0.0)
									continue;
								const double e = f->value(y); // weighted
								total += e;
								tb_logger.add_scalar("energy/full_forms/" + f->name(), tb_global_step, e);
							}

							// Keep legacy tag for backward compatibility, plus a clearer alias.
							tb_logger.add_scalar("nl/energy", tb_global_step, total);
							tb_logger.add_scalar("nl/energy_total", tb_global_step, total);
						}
						else
						{
							const double total = nl_problem.value(data.x);
							tb_logger.add_scalar("nl/energy", tb_global_step, total);
							tb_logger.add_scalar("nl/energy_total", tb_global_step, total);
						}
					}
				}
			}

			if (a1_active && a1_in_step3)
			{
				++a1_step3_iter;
				if (!a1_phase2 && (a1_step3_iter % step3_check_every == 0))
				{
					const double p90 = fit_form->gap_percentile(step3_percentile, step3_d0);
					logger().debug("[A1] iter={} P{}={}", a1_step3_iter, step3_percentile, p90);
					if (tb_logger.is_enabled())
						tb_logger.add_scalar("A1/p90_gap", tb_global_step, p90);

					if (p90 < step3_tau)
					{
						a1_phase2 = true;
					}
					else if (std::isfinite(a1_last_p90))
					{
						const double denom = std::max(a1_last_p90, 1e-12);
						const double rel_impr = (a1_last_p90 - p90) / denom;
						if (rel_impr < step3_stall_eps)
							++a1_stall_cnt;
						else
							a1_stall_cnt = 0;
						if (a1_stall_cnt >= step3_stall_K)
							a1_phase2 = true;
					}

					a1_last_p90 = p90;

					if (a1_phase2)
					{
						logger().info("[A1] switch to Phase2 (RESTORE): fit_weight={} similarity_weight={}", a1_fit_w_phase2, a1_sim_w_phase2);
						fit_form->set_weight(a1_fit_w_phase2);
						similarity_form->set_weight(a1_sim_w_phase2);
					}
				}
			}
		};

		{
			utils::ChromeTraceScope trace_step2("AL_step2");
			prog_phase = "AL_step2";
			prog_substep = substep;
			prog_max_iter = max_iter_step2;
		al_solver.solve_al(nl_solver, nl_problem, sol);
		}

		fit_form->enable();
		if (in_args["curve_size_weight"] > 0 && substep == total_steps - 1)
			curve_size_form->enable();

		// Activate A1 only for final substep Step3
		if (step3_anneal_enabled && is_last_substep)
		{
			if (a1_h > 0.0)
			{
				a1_active = true;
				a1_in_step3 = true;
				a1_step3_iter = 0;
				a1_stall_cnt = 0;
				a1_last_p90 = std::numeric_limits<double>::infinity();
				a1_phase2 = false;

				logger().info("[A1] Phase1 (TIGHTEN): fit_weight={} similarity_weight={}", a1_fit_w_phase1, a1_sim_w_phase1);
				fit_form->set_weight(a1_fit_w_phase1);
				similarity_form->set_weight(a1_sim_w_phase1);
			}
			else
			{
				logger().info("[A1] no shrink detected (h=0), skipping annealing.");
			}
		}

		nl_solver = polysolve::nonlinear::Solver::create(in_args["solver"]["nonlinear"], in_args["solver"]["linear"], 1., logger());
		const int max_iter_step3 = nl_solver->stop_criteria().iterations;
		{
			utils::ChromeTraceScope trace_step3("AL_step3");
			prog_phase = "AL_step3";
			prog_substep = substep;
			prog_max_iter = max_iter_step3;
			// A2 local surface similarity relaxation: apply only for Step3 (reduced solve),
			// and only on final substep if configured.
			bool a2_surf_applied = false;
			Eigen::VectorXd sim_multipliers_step3;
			if (a2_enabled && a2_enable_surf_relax && (!a2_final_substep_only || is_last_substep))
			{
				sim_multipliers_step3 = sim_multipliers_base;
				const int n_avatar_verts = gstate.nc_avatar_v.rows();
				sim_multipliers_step3.segment(n_avatar_verts, gstate.n_garment_vertices()).array() *= a2_surf.array();
				similarity_form->set_vertex_multipliers(sim_multipliers_step3);
				a2_surf_applied = true;
			}

			al_solver.solve_reduced(nl_solver, nl_problem, sol);

			// Restore base similarity multipliers for next Step2 / next substep.
			if (a2_surf_applied)
				similarity_form->set_vertex_multipliers(sim_multipliers_base);
		}

		a1_in_step3 = false;

		cur_garment_v = initial_garment_v + utils::unflatten(sol.bottomRows(cur_garment_v.size()), 3);

		// Always save the final step result regardless of stride
		if (substep == total_steps - 1)
			gstate.save_result(out_folder, (save_id / stride) + 1, nl_problem, collision_vertices, collision_triangles, sol);
	}

		logger().info("Garment retargeting succeeded!");
	}
	catch (const std::exception &e)
	{
		logger().error("Run terminated with error: {}", e.what());
		exit_code = EXIT_FAILURE;
	}

	// Optional aggregated timing summary (if enabled)
	if (utils::TimingRegistry::instance().enabled())
	{
		const int top_k = 20;
		const auto entries = utils::TimingRegistry::instance().snapshot_sorted();
		logger().info("[timing] Top {} scoped timers (total time):", std::min<int>(top_k, (int)entries.size()));
		for (int i = 0; i < std::min<int>(top_k, (int)entries.size()); ++i)
		{
			const auto &e = entries[i];
			logger().info("[timing] {:>2}. {:<32} total={:.3f}s count={}", i + 1, e.name, e.total_time_s, e.count);
		}
	}

	// Ensure Chrome trace file is closed even on early exit.
	utils::ChromeTrace::instance().shutdown();

	return exit_code;
}
