#pragma once

namespace polyfem {
namespace autogen {

    /// @brief Gradient of the python function
    /// def mix_product(t0, t1, t2):
    ///     return numpy.cross(t0, t1).dot(t2) / sympy.sqrt(t0.dot(t0) * t1.dot(t1) * t2.dot(t2))
    void angle_gradient(
        double t0_x,
        double t0_y,
        double t0_z,
        double t1_x,
        double t1_y,
        double t1_z,
        double t2_x,
        double t2_y,
        double t2_z,
        double dA[9]);

    /// @brief Hessian of the python function
    /// def mix_product(t0, t1, t2):
    ///     return numpy.cross(t0, t1).dot(t2) / sympy.sqrt(t0.dot(t0) * t1.dot(t1) * t2.dot(t2))
    // dA is (81×1) flattened in column-major order
    void angle_hessian(
        double t0_x,
        double t0_y,
        double t0_z,
        double t1_x,
        double t1_y,
        double t1_z,
        double t2_x,
        double t2_y,
        double t2_z,
        double dA[81]);

    /// @brief Gradient of the python function
    /// def triangle_area(t0, t1, t2):
    ///     n = numpy.cross(t1 - t0, t2 - t0)
    ///     return norm(n) / 2
    void triangle_area_gradient(
        double t0_x,
        double t0_y,
        double t0_z,
        double t1_x,
        double t1_y,
        double t1_z,
        double t2_x,
        double t2_y,
        double t2_z,
        double dA[9]);

    /// @brief Hessian of the python function
    /// def triangle_area(t0, t1, t2):
    ///     n = numpy.cross(t1 - t0, t2 - t0)
    ///     return norm(n) / 2
    // dA is (81×1) flattened in column-major order
    void triangle_area_hessian(
        double t0_x,
        double t0_y,
        double t0_z,
        double t1_x,
        double t1_y,
        double t1_z,
        double t2_x,
        double t2_y,
        double t2_z,
        double dA[81]);

    /// @brief Gradient of the python function
    /// def normalized_dot(a,b,c,d,e,f):
    ///     n1 = numpy.cross((b - a), (c - a))
    ///     n2 = numpy.cross((e - d), (f - d))
    ///     return n1.dot(n2) / sympy.sqrt(n1.dot(n1) * n2.dot(n2))
    void normal_dot_product_gradient(
        double t0_x,
        double t0_y,
        double t0_z,
        double t1_x,
        double t1_y,
        double t1_z,
        double t2_x,
        double t2_y,
        double t2_z,
        double t3_x,
        double t3_y,
        double t3_z,
        double dA[12]);

    /// @brief Hessian of the python function
    /// def normalized_dot(a,b,c,d,e,f):
    ///     n1 = numpy.cross((b - a), (c - a))
    ///     n2 = numpy.cross((e - d), (f - d))
    ///     return n1.dot(n2) / sympy.sqrt(n1.dot(n1) * n2.dot(n2))
    // dA is (324×1) flattened in column-major order
    void normal_dot_product_hessian(
        double t0_x,
        double t0_y,
        double t0_z,
        double t1_x,
        double t1_y,
        double t1_z,
        double t2_x,
        double t2_y,
        double t2_z,
        double t3_x,
        double t3_y,
        double t3_z,
        double dA[144]);

    void normal_triple_product_gradient(
        double t0_x,
        double t0_y,
        double t0_z,
        double t1_x,
        double t1_y,
        double t1_z,
        double t2_x,
        double t2_y,
        double t2_z,
        double t3_x,
        double t3_y,
        double t3_z,
        double dA[12]);
    // dA is (144×1) flattened in column-major order
    void normal_triple_product_hessian(
        double t0_x,
        double t0_y,
        double t0_z,
        double t1_x,
        double t1_y,
        double t1_z,
        double t2_x,
        double t2_y,
        double t2_z,
        double t3_x,
        double t3_y,
        double t3_z,
        double dA[144]);

    void point_edge_distance_gradient(
        double t0_x,
        double t0_y,
        double t0_z,
        double t1_x,
        double t1_y,
        double t1_z,
        double t2_x,
        double t2_y,
        double t2_z,
        double dA[9]);
    // dA is (81×1) flattened in column-major order
    void point_edge_distance_hessian(
        double t0_x,
        double t0_y,
        double t0_z,
        double t1_x,
        double t1_y,
        double t1_z,
        double t2_x,
        double t2_y,
        double t2_z,
        double dA[81]);


    void curve_dot_product_norm_gradient(
        double p1x,
        double p1y,
        double p1z,
        double p2x,
        double p2y,
        double p2z,
        double p3x,
        double p3y,
        double p3z,
        double grad[9]);
    // hess is (81×1) flattened in column-major order
    void curve_dot_product_norm_hessian(
        double p1x,
        double p1y,
        double p1z,
        double p2x,
        double p2y,
        double p2z,
        double p3x,
        double p3y,
        double p3z,
        double hess[81]);

    void curve_twist_gradient(double t0_x, double t0_y, double t0_z, double t1_x, double t1_y, double t1_z, double t2_x, double t2_y, double t2_z, double t3_x, double t3_y, double t3_z, double dA[12]);
    // dA is (144×1) flattened in column-major order
    void curve_twist_hessian(double t0_x, double t0_y, double t0_z, double t1_x, double t1_y, double t1_z, double t2_x, double t2_y, double t2_z, double t3_x, double t3_y, double t3_z, double dA[144]);

    void def_grad_gradient(
        double t0_x,
        double t0_y,
        double t0_z,
        double t1_x,
        double t1_y,
        double t1_z,
        double t2_x,
        double t2_y,
        double t2_z,
        double dA[9]);
    // dA is (81×1) flattened in column-major order
    void def_grad_hessian(
        double t0_x,
        double t0_y,
        double t0_z,
        double t1_x,
        double t1_y,
        double t1_z,
        double t2_x,
        double t2_y,
        double t2_z,
        double dA[81]);

    void line_projection_uv_gradient(
        double t, double vx, double vy, double vz, double dA[4],
        double a0x, double a0y, double a0z,
        double a1x, double a1y, double a1z,
        double b0x, double b0y, double b0z,
        double b1x, double b1y, double b1z);
    // dA is (16×1) flattened in column-major order
    void line_projection_uv_hessian(
        double t, double vx, double vy, double vz, double dA[16],
        double a0x, double a0y, double a0z,
        double a1x, double a1y, double a1z,
        double b0x, double b0y, double b0z,
        double b1x, double b1y, double b1z);

    // p0 = np.array(symbols('p0x p0y p0z'))
    // p1 = np.array(symbols('p1x p1y p1z'))
    // p2 = np.array(symbols('p2x p2y p2z'))
    // p3 = np.array(symbols('p3x p3y p3z'))
    // x = np.array([p0.T, p1.T, p2.T, p3.T]).reshape(-1)
    // mid = p2 - p1
    // v1 = p1 - p0
    // v2 = p3 - p2
    // w1 = project(v1, mid)
    // w2 = project(v2, mid)
    // T = np.cross(w1, w2).dot(mid)  / (mynorm(w1) * mynorm(w2) * mynorm(mid))
    void curve_torsion_sin_gradient(
        double p0x,
        double p0y,
        double p0z,
        double p1x,
        double p1y,
        double p1z,
        double p2x,
        double p2y,
        double p2z,
        double p3x,
        double p3y,
        double p3z,
        double grad[12]);
    // hess is (144×1) flattened in column-major order
    void curve_torsion_sin_hessian(
        double p0x,
        double p0y,
        double p0z,
        double p1x,
        double p1y,
        double p1z,
        double p2x,
        double p2y,
        double p2z,
        double p3x,
        double p3y,
        double p3z,
        double hess[144]);

    // p0 = np.array(symbols('p0x p0y p0z'))
    // p1 = np.array(symbols('p1x p1y p1z'))
    // p2 = np.array(symbols('p2x p2y p2z'))
    // p3 = np.array(symbols('p3x p3y p3z'))
    // x = np.array([p0.T, p1.T, p2.T, p3.T]).reshape(-1)
    // mid = p2 - p1
    // v1 = p1 - p0
    // v2 = p3 - p2
    // w1 = project(v1, mid)
    // w2 = project(v2, mid)
    // T = w1.dot(w2) / (mynorm(w1) * mynorm(w2))
    void curve_torsion_cos_gradient(
        double p0x,
        double p0y,
        double p0z,
        double p1x,
        double p1y,
        double p1z,
        double p2x,
        double p2y,
        double p2z,
        double p3x,
        double p3y,
        double p3z,
        double grad[12]);
    // hess is (144×1) flattened in column-major order
    void curve_torsion_cos_hessian(
        double p0x,
        double p0y,
        double p0z,
        double p1x,
        double p1y,
        double p1z,
        double p2x,
        double p2y,
        double p2z,
        double p3x,
        double p3y,
        double p3z,
        double hess[144]);

    // c0 = np.array(symbols('c0x c0y c0z'))
    // c1 = np.array(symbols('c1x c1y c1z'))
    // p0 = np.array(symbols('p0x p0y p0z'))
    // p1 = np.array(symbols('p1x p1y p1z'))
    // p2 = np.array(symbols('p2x p2y p2z'))
    // p3 = np.array(symbols('p3x p3y p3z'))
    // x = np.array([p0.T, p1.T, p2.T, p3.T]).reshape(-1)
    // e = p1 - p0
    // n0 = np.cross(e, p2 - p0)
    // n1 = np.cross(p3 - p0, e)
    // vec0 = c0[0] * e + c0[1] * (p2 - p0) + c0[2] * n0
    // vec1 = c1[0] * (p3 - p0) + c1[1] * e + c1[2] * n1
    // T = (vec0 - vec1).dot(vec0 - vec1)
    void similarity_gradient(double ve0_x, double ve0_y, double ve0_z, double ve1_x, double ve1_y, double ve1_z, double vf0_x, double vf0_y, double vf0_z, double vf1_x, double vf1_y, double vf1_z, double c1_x, double c1_y, double c1_z, double c2_x, double c2_y, double c2_z, double grad[12]);
    void similarity_hessian(double ve0_x, double ve0_y, double ve0_z, double ve1_x, double ve1_y, double ve1_z, double vf0_x, double vf0_y, double vf0_z, double vf1_x, double vf1_y, double vf1_z, double c1_x, double c1_y, double c1_z, double c2_x, double c2_y, double c2_z, double hess[144]);

    // def normal(a, b, c, d):
    //     normal = numpy.cross(b - a, c - a)
    //     return 1 - normal.dot(d)

    // x = numpy.array(sympy.symbols(" ".join([f"t{i}_{d}" for i in range(4) for d in "xyz"])))
    // CXXGradientGenerator(
    //     normal(*numpy.split(x, 4)), x,
    //     "normal_gradient", out_param_name="dA")
    // CXXHessianGenerator(
    //     normal(*numpy.split(x, 4)), x,
    //     "normal_hessian", out_param_name="dA")
    void normal_gradient(
        double t0_x,
        double t0_y,
        double t0_z,
        double t1_x,
        double t1_y,
        double t1_z,
        double t2_x,
        double t2_y,
        double t2_z,
        double t3_x,
        double t3_y,
        double t3_z,
        double dA[12]);
    // dA is (144×1) flattened in column-major order
    void normal_hessian(
        double t0_x,
        double t0_y,
        double t0_z,
        double t1_x,
        double t1_y,
        double t1_z,
        double t2_x,
        double t2_y,
        double t2_z,
        double t3_x,
        double t3_y,
        double t3_z,
        double dA[144]);
}
}