////////////////////////////////////////////////////////////////////////////////
#include "MeshUtils.hpp"

#include <polyfem/io/OBJReader.hpp>
#include <polyfem/utils/StringUtils.hpp>
#include <polyfem/utils/Logger.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <igl/PI.h>
#include <igl/read_triangle_mesh.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/boundary_facets.h>

////////////////////////////////////////////////////////////////////////////////

using namespace polyfem::io;
using namespace polyfem::utils;

////////////////////////////////////////////////////////////////////////////////

void polyfem::mesh::write_edge_mesh(
	const std::string &path, 
	const Eigen::MatrixXd &V, 
	const Eigen::MatrixXi &E)
{
	std::ofstream outfile(path);

	for (int i = 0; i < V.rows(); i++)
	{
		outfile << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << "\n";
	}

	for (int j = 0; j < E.rows(); j++)
	{
		outfile << "l " << E(j, 0) + 1 << " " << E(j, 1) + 1 << "\n";
	}
}

void polyfem::mesh::read_edge_mesh(
	const std::string &path,
	Eigen::MatrixXd &V,
	Eigen::MatrixXi &E)
{
	std::ifstream infile(path);

	V.resize(0, 3);
	E.resize(0, 2);
	std::string line;
	while (std::getline(infile, line))
	{
		std::istringstream iss(line.substr(2));
		if (utils::StringUtils::startswith(line, "v "))
		{
			V.conservativeResize(V.rows() + 1, 3);

			double a, b, c;
			if (iss >> a >> b >> c)
				V.row(V.rows()-1) << a, b, c;
			else
				log_and_throw_error("read_edge_mesh failed to load vertex {}", V.rows() - 1);
		}
		else if (utils::StringUtils::startswith(line, "l "))
		{
			E.conservativeResize(E.rows() + 1, 2);

			int a, b;
			if (iss >> a >> b)
				E.row(E.rows()-1) << a - 1, b - 1;
			else
				log_and_throw_error("read_edge_mesh failed to load edge {}", E.rows() - 1);
		}
		else if (utils::StringUtils::startswith(line, "f "))
		{
			log_and_throw_error("read_edge_mesh does not support faces!");
		}
	}
}

std::tuple<Eigen::VectorXi, Eigen::VectorXi> polyfem::mesh::remove_duplicate_vertices(Eigen::MatrixXd &V, Eigen::MatrixXi &F, const double threshold)
{
	Eigen::VectorXi svi, svj;
	Eigen::MatrixXi sf;
	Eigen::MatrixXd sv;
	igl::remove_duplicate_vertices(V, F, threshold, sv, svi, svj, sf);
	for (int i = 0; i < sf.rows(); i++)
	{
		if (sf(i, 0) == sf(i, 1) || sf(i, 2) == sf(i, 1) || sf(i, 0) == sf(i, 2))
			log_and_throw_error("Treshold in igl::remove_duplicate_vertices is too large!!");
	}
	std::swap(sv, V);
	std::swap(sf, F);

	return {svi, svj};
}

double polyfem::mesh::closed_mesh_volume(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
{
	Eigen::MatrixXi B;
	igl::boundary_facets(F, B);
	if (B.rows() > 0)
		log_and_throw_error("closed_mesh_volume: mesh is not watertight ({} boundary edges).", B.rows());

	// Signed volume by summing tetrahedra wrt origin: V = sum(a·(b×c))/6
	double vol6 = 0.0;
	for (int f = 0; f < F.rows(); ++f)
	{
		const Eigen::Vector3d a = V.row(F(f, 0)).transpose();
		const Eigen::Vector3d b = V.row(F(f, 1)).transpose();
		const Eigen::Vector3d c = V.row(F(f, 2)).transpose();
		vol6 += a.dot(b.cross(c));
	}
	const double vol = std::abs(vol6) / 6.0;
	if (!std::isfinite(vol) || vol <= 0.0)
		log_and_throw_error("closed_mesh_volume: invalid volume {}", vol);
    return vol;
}
