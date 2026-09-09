#include "auto_derivatives.hpp"
#include <cmath>

namespace polyfem::autogen {

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
        double dA[9])
    {
        const auto t0 =
            std::pow(t0_x, 2) + std::pow(t0_y, 2) + std::pow(t0_z, 2);
        const auto t1 =
            std::pow(t1_x, 2) + std::pow(t1_y, 2) + std::pow(t1_z, 2);
        const auto t2 =
            std::pow(t2_x, 2) + std::pow(t2_y, 2) + std::pow(t2_z, 2);
        const auto t3 = std::pow(t0 * t1 * t2, -1.0 / 2.0);
        const auto t4 = t0_z * t1_y;
        const auto t5 = t0_y * t1_x;
        const auto t6 = t0_x * t1_z - t0_z * t1_x;
        const auto t7 =
            t2_x * (t0_y * t1_z - t4) - t2_y * t6 + t2_z * (t0_x * t1_y - t5);
        const auto t8 = t7 / t0;
        const auto t9 = t7 / t1;
        const auto t10 = t7 / t2;
        dA[0] = t3 * (-t0_x * t8 + t1_y * t2_z - t1_z * t2_y);
        dA[1] = -t3 * (t0_y * t8 + t1_x * t2_z - t1_z * t2_x);
        dA[2] = t3 * (-t0_z * t8 + t1_x * t2_y - t1_y * t2_x);
        dA[3] = -t3 * (t0_y * t2_z - t0_z * t2_y + t1_x * t9);
        dA[4] = t3 * (t0_x * t2_z - t0_z * t2_x - t1_y * t9);
        dA[5] = -t3 * (t0_x * t2_y - t0_y * t2_x + t1_z * t9);
        dA[6] = t3 * (t0_y * t1_z - t10 * t2_x - t4);
        dA[7] = -t3 * (t10 * t2_y + t6);
        dA[8] = t3 * (t0_x * t1_y - t10 * t2_z - t5);
    }

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
        double dA[81])
    {
        const auto t0 = t1_y * t2_z;
        const auto t1 = t1_z * t2_y;
        const auto t2 = t0 - t1;
        const auto t3 = std::pow(t0_x, 2);
        const auto t4 = std::pow(t0_y, 2);
        const auto t5 = std::pow(t0_z, 2);
        const auto t6 = t3 + t4 + t5;
        const auto t7 = 1.0 / t6;
        const auto t8 = t0_x * t1_z;
        const auto t9 = t0_z * t1_x;
        const auto t10 = t8 - t9;
        const auto t11 = t10 * t2_y;
        const auto t12 = -t11;
        const auto t13 = t0_y * t1_z;
        const auto t14 = -t0_z * t1_y + t13;
        const auto t15 = t14 * t2_x;
        const auto t16 = t0_x * t1_y;
        const auto t17 = t0_y * t1_x;
        const auto t18 = t16 - t17;
        const auto t19 = t18 * t2_z;
        const auto t20 = t15 + t19;
        const auto t21 = t12 + t20;
        const auto t22 = std::pow(t1_x, 2);
        const auto t23 = std::pow(t1_y, 2);
        const auto t24 = std::pow(t1_z, 2);
        const auto t25 = t22 + t23 + t24;
        const auto t26 = std::pow(t2_x, 2);
        const auto t27 = std::pow(t2_y, 2);
        const auto t28 = std::pow(t2_z, 2);
        const auto t29 = t26 + t27 + t28;
        const auto t30 = std::pow(t25 * t29 * t6, -1.0 / 2.0);
        const auto t31 = t30 * t7;
        const auto t32 = t1_z * t2_x;
        const auto t33 = t1_x * t2_z - t32;
        const auto t34 = t0_x * t7;
        const auto t35 = 3 * t21;
        const auto t36 = t31 * (t0_x * t33 - t0_y * t2 + t0_y * t34 * t35);
        const auto t37 = t1_x * t2_y;
        const auto t38 = -t1_y * t2_x + t37;
        const auto t39 =
            t31 * (3 * t0_x * t0_z * t21 * t7 - t0_x * t38 - t0_z * t2);
        const auto t40 = t0_y * t2_z;
        const auto t41 = t0_z * t2_y;
        const auto t42 = t40 - t41;
        const auto t43 = 1.0 / t25;
        const auto t44 = t2 * t43;
        const auto t45 = t1_x * t43;
        const auto t46 = t30 * (-t1_x * t44 + t21 * t34 * t45 + t34 * t42);
        const auto t47 = t0_x * t2_z;
        const auto t48 = -t0_z * t2_x + t47;
        const auto t49 = t21 * t7;
        const auto t50 = t43 * t49;
        const auto t51 = t30 * (t16 * t50 - t1_y * t44 + t2_z - t34 * t48);
        const auto t52 = t0_x * t2_y;
        const auto t53 = t0_y * t2_x;
        const auto t54 = t52 - t53;
        const auto t55 = t30 * (-t1_z * t44 - t2_y + t34 * t54 + t50 * t8);
        const auto t56 = 1.0 / t29;
        const auto t57 = t2 * t56;
        const auto t58 =
            t30 * (t0_x * t21 * t2_x * t56 * t7 - t14 * t34 - t2_x * t57);
        const auto t59 = t49 * t56;
        const auto t60 = t30 * (t10 * t34 - t1_z - t2_y * t57 + t52 * t59);
        const auto t61 = t30 * (-t18 * t34 + t1_y - t2_z * t57 + t47 * t59);
        const auto t62 = t11 - t19;
        const auto t63 = -t15 + t62;
        const auto t64 = t0_y * t7;
        const auto t65 = t31 * (-t0_y * t38 + t0_z * t33 + t0_z * t35 * t64);
        const auto t66 = t33 * t43;
        const auto t67 = t30 * (t17 * t50 + t1_x * t66 - t2_z + t42 * t64);
        const auto t68 = t1_y * t43;
        const auto t69 = t21 * t64;
        const auto t70 = t30 * (t1_y * t66 - t48 * t64 + t68 * t69);
        const auto t71 = t30 * (t13 * t50 + t1_z * t66 + t2_x + t54 * t64);
        const auto t72 = t33 * t56;
        const auto t73 = t30 * (-t14 * t64 + t1_z + t2_x * t72 + t53 * t59);
        const auto t74 = t2_y * t56;
        const auto t75 = t30 * (t10 * t64 + t2_y * t72 + t69 * t74);
        const auto t76 = t30 * (-t18 * t64 - t1_x + t2_z * t72 + t40 * t59);
        const auto t77 = t0_z * t7;
        const auto t78 = t38 * t43;
        const auto t79 = t30 * (-t1_x * t78 + t2_y + t42 * t77 + t50 * t9);
        const auto t80 = t30
            * (t0_z * t1_y * t21 * t43 * t7 - t1_y * t78 - t2_x - t48 * t77);
        const auto t81 = t1_z * t43;
        const auto t82 = t30 * (-t1_z * t78 + t21 * t77 * t81 + t54 * t77);
        const auto t83 = t38 * t56;
        const auto t84 = t30
            * (t0_z * t21 * t2_x * t56 * t7 - t14 * t77 - t1_y - t2_x * t83);
        const auto t85 = t30 * (t10 * t77 + t1_x - t2_y * t83 + t41 * t59);
        const auto t86 =
            t30 * (t0_z * t21 * t2_z * t56 * t7 - t18 * t77 - t2_z * t83);
        const auto t87 = t35 * t43;
        const auto t88 = t30 * t43;
        const auto t89 = t35 * t45;
        const auto t90 = t88 * (-t1_x * t48 + t1_y * t42 + t1_y * t89);
        const auto t91 = t88 * (t1_x * t54 + t1_z * t42 + t1_z * t89);
        const auto t92 = t42 * t56;
        const auto t93 =
            t30 * (-t14 * t45 + t21 * t2_x * t45 * t56 + t2_x * t92);
        const auto t94 = t21 * t43 * t56;
        const auto t95 = t30 * (t0_z + t10 * t45 + t2_y * t92 + t37 * t94);
        const auto t96 = t30
            * (-t0_y - t18 * t45 + t1_x * t21 * t2_z * t43 * t56
               + t2_z * t42 * t56);
        const auto t97 = t88 * (t1_y * t54 + t1_z * t35 * t68 - t1_z * t48);
        const auto t98 = t48 * t56;
        const auto t99 = t30
            * (-t0_z - t14 * t68 + t1_y * t21 * t2_x * t43 * t56 - t2_x * t98);
        const auto t100 = t30 * (t10 * t68 + t21 * t68 * t74 - t2_y * t98);
        const auto t101 = t30 * (t0 * t94 + t0_x - t18 * t68 - t2_z * t98);
        const auto t102 = t54 * t56;
        const auto t103 = t30 * (t0_y + t102 * t2_x - t14 * t81 + t32 * t94);
        const auto t104 = t30 * (-t0_x + t1 * t94 + t10 * t81 + t102 * t2_y);
        const auto t105 =
            t30 * (t102 * t2_z - t18 * t81 + t21 * t2_z * t56 * t81);
        const auto t106 = t35 * t56;
        const auto t107 = t30 * t56;
        const auto t108 = t107 * (t10 * t2_x + t106 * t2_x * t2_y - t14 * t2_y);
        const auto t109 =
            t107 * (-t14 * t2_z - t18 * t2_x + 3 * t21 * t2_x * t2_z * t56);
        const auto t110 = t107 * (t10 * t2_z + t106 * t2_y * t2_z - t18 * t2_y);
        dA[0] = t31 * (-2 * t0_x * t2 + 3 * t21 * t3 * t7 - t21);
        dA[1] = t36;
        dA[2] = t39;
        dA[3] = t46;
        dA[4] = t51;
        dA[5] = t55;
        dA[6] = t58;
        dA[7] = t60;
        dA[8] = t61;
        dA[9] = t36;
        dA[10] = t31 * (2 * t0_y * t33 + t35 * t4 * t7 + t63);
        dA[11] = t65;
        dA[12] = t67;
        dA[13] = t70;
        dA[14] = t71;
        dA[15] = t73;
        dA[16] = t75;
        dA[17] = t76;
        dA[18] = t39;
        dA[19] = t65;
        dA[20] = t31 * (-2 * t0_z * t38 + 3 * t21 * t5 * t7 - t21);
        dA[21] = t79;
        dA[22] = t80;
        dA[23] = t82;
        dA[24] = t84;
        dA[25] = t85;
        dA[26] = t86;
        dA[27] = t46;
        dA[28] = t67;
        dA[29] = t79;
        dA[30] = t88 * (2 * t1_x * t42 + t22 * t87 + t63);
        dA[31] = t90;
        dA[32] = t91;
        dA[33] = t93;
        dA[34] = t95;
        dA[35] = t96;
        dA[36] = t51;
        dA[37] = t70;
        dA[38] = t80;
        dA[39] = t90;
        dA[40] = t88 * (-2 * t1_y * t48 + 3 * t21 * t23 * t43 - t21);
        dA[41] = t97;
        dA[42] = t99;
        dA[43] = t100;
        dA[44] = t101;
        dA[45] = t55;
        dA[46] = t71;
        dA[47] = t82;
        dA[48] = t91;
        dA[49] = t97;
        dA[50] = t88 * (2 * t1_z * t54 + t24 * t87 + t63);
        dA[51] = t103;
        dA[52] = t104;
        dA[53] = t105;
        dA[54] = t58;
        dA[55] = t73;
        dA[56] = t84;
        dA[57] = t93;
        dA[58] = t99;
        dA[59] = t103;
        dA[60] = t107 * (t106 * t26 - 3 * t15 + t62);
        dA[61] = t108;
        dA[62] = t109;
        dA[63] = t60;
        dA[64] = t75;
        dA[65] = t85;
        dA[66] = t95;
        dA[67] = t100;
        dA[68] = t104;
        dA[69] = t108;
        dA[70] = t107 * (3 * t10 * t2_y - t20 + 3 * t21 * t27 * t56);
        dA[71] = t110;
        dA[72] = t61;
        dA[73] = t76;
        dA[74] = t86;
        dA[75] = t96;
        dA[76] = t101;
        dA[77] = t105;
        dA[78] = t109;
        dA[79] = t110;
        dA[80] = t107 * (-t12 - t15 - 3 * t19 + 3 * t21 * t28 * t56);
    }

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
        double dA[9])
    {
        const auto t0 = -t2_y;
        const auto t1 = t0 + t1_y;
        const auto t2 = t0_x - t1_x;
        const auto t3 = t0 + t0_y;
        const auto t4 = -t2_x;
        const auto t5 = t0_x + t4;
        const auto t6 = t0_y - t1_y;
        const auto t7 = t2 * t3 - t5 * t6;
        const auto t8 = -t2_z;
        const auto t9 = t1_z + t8;
        const auto t10 = t0_z + t8;
        const auto t11 = t0_z - t1_z;
        const auto t12 = t10 * t2 - t11 * t5;
        const auto t13 = t10 * t6 - t11 * t3;
        const auto t14 = (1.0 / 2.0)
            / std::sqrt(std::pow(t12, 2) + std::pow(t13, 2) + std::pow(t7, 2));
        const auto t15 = t1_x + t4;
        const auto t16 = t10 * t2 - t11 * t5;
        dA[0] = t14 * (t1 * t7 + t12 * t9);
        dA[1] = -t14 * (-t13 * t9 + t15 * t7);
        dA[2] = -t14 * (t1 * t13 + t15 * t16);
        dA[3] = -t14 * (t10 * t16 + t3 * t7);
        dA[4] = t14 * (-t10 * t13 + t5 * t7);
        dA[5] = t14 * (t12 * t5 + t13 * t3);
        dA[6] = t14 * (t11 * t12 + t6 * t7);
        dA[7] = -t14 * (-t11 * t13 + t2 * t7);
        dA[8] = -t14 * (t13 * t6 + t16 * t2);
    }

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
        double dA[81])
    {
        const auto t0 = -t2_y;
        const auto t1 = t0 + t1_y;
        const auto t2 = std::pow(t1, 2);
        const auto t3 = -t2_z;
        const auto t4 = t1_z + t3;
        const auto t5 = std::pow(t4, 2);
        const auto t6 = t0_x - t1_x;
        const auto t7 = t0 + t0_y;
        const auto t8 = t6 * t7;
        const auto t9 = -t2_x;
        const auto t10 = t0_x + t9;
        const auto t11 = t0_y - t1_y;
        const auto t12 = t10 * t11;
        const auto t13 = -t12;
        const auto t14 = t13 + t8;
        const auto t15 = t1 * t14;
        const auto t16 = t0_z + t3;
        const auto t17 = t16 * t6;
        const auto t18 = t0_z - t1_z;
        const auto t19 = t10 * t18;
        const auto t20 = -t19;
        const auto t21 = t17 + t20;
        const auto t22 = t15 + t21 * t4;
        const auto t23 = t11 * t16;
        const auto t24 = t18 * t7;
        const auto t25 = -t24;
        const auto t26 = t23 + t25;
        const auto t27 = std::pow(t14, 2) + std::pow(t26, 2);
        const auto t28 = std::pow(t21, 2) + t27;
        const auto t29 = 1.0 / t28;
        const auto t30 = -t6;
        const auto t31 = -t16;
        const auto t32 = -t10;
        const auto t33 = -t18;
        const auto t34 = t30 * t31 - t32 * t33;
        const auto t35 = t15 + t34 * t4;
        const auto t36 = t29 * t35;
        const auto t37 = (1.0 / 2.0) / std::sqrt(t28);
        const auto t38 = t1_x + t9;
        const auto t39 = t1 * t38;
        const auto t40 = t14 * t38 - t26 * t4;
        const auto t41 = -t40;
        const auto t42 = t27 + std::pow(t34, 2);
        const auto t43 = 1.0 / t42;
        const auto t44 = -t7;
        const auto t45 = -t11;
        const auto t46 = t30 * t44 - t32 * t45;
        const auto t47 = -t34;
        const auto t48 = t1 * t46 - t4 * t47;
        const auto t49 = t43 * t48;
        const auto t50 = (1.0 / 2.0) / std::sqrt(t42);
        const auto t51 = t38 * t4;
        const auto t52 = t1 * t26 + t34 * t38;
        const auto t53 = t14 * t7 + t16 * t34;
        const auto t54 = t1 * t7;
        const auto t55 = t16 * t4;
        const auto t56 = t54 + t55;
        const auto t57 = t10 * t14 - t16 * t26;
        const auto t58 = t1 * t10 + t14;
        const auto t59 = t26 * t7;
        const auto t60 = t10 * t34 + t59;
        const auto t61 = t10 * t4 + t21;
        const auto t62 = t11 * t14;
        const auto t63 = t18 * t21 + t62;
        const auto t64 = t1 * t11;
        const auto t65 = t18 * t4;
        const auto t66 = t64 + t65;
        const auto t67 = t14 * t6 - t18 * t26;
        const auto t68 = -t67;
        const auto t69 = t1 * t6 + t14;
        const auto t70 = t11 * t26 + t34 * t6;
        const auto t71 = t21 + t4 * t6;
        const auto t72 = t31 * t45 - t33 * t44;
        const auto t73 = -t38 * t46 + t4 * t72;
        const auto t74 = t43 * t73;
        const auto t75 = std::pow(t38, 2);
        const auto t76 = t1 * t4;
        const auto t77 = t12 - t8;
        const auto t78 = t29 * t41;
        const auto t79 = t10 * t38;
        const auto t80 = t55 + t79;
        const auto t81 = t26 + t4 * t7;
        const auto t82 = t18 * t34 + t62;
        const auto t83 = t11 * t38 + t77;
        const auto t84 = t38 * t6;
        const auto t85 = t65 + t84;
        const auto t86 = t11 * t4 + t26;
        const auto t87 = t43 * (-t1 * t72 + t38 * t47);
        const auto t88 = -t17 + t19;
        const auto t89 = t16 * t38 + t88;
        const auto t90 = t10 * t21 + t59;
        const auto t91 = t54 + t79;
        const auto t92 = t18 * t38;
        const auto t93 = t1 * t18;
        const auto t94 = -t23 + t24;
        const auto t95 = t37 * (-t29 * t52 * t70 + t64 + t84);
        const auto t96 = t43 * (t16 * t47 + t44 * t46);
        const auto t97 = std::pow(t7, 2);
        const auto t98 = std::pow(t16, 2);
        const auto t99 = t10 * t7;
        const auto t100 = t10 * t16;
        const auto t101 = t11 * t7;
        const auto t102 = t16 * t18;
        const auto t103 = t101 + t102;
        const auto t104 = 2 * t17 + t20;
        const auto t105 = t10 * t46 + t31 * t72;
        const auto t106 = t105 * t43;
        const auto t107 = std::pow(t10, 2);
        const auto t108 = t16 * t7;
        const auto t109 = -2 * t10 * t11 + t8;
        const auto t110 = t10 * t6;
        const auto t111 = t102 + t110;
        const auto t112 = t32 * t47 + t7 * t72;
        const auto t113 = t112 * t43;
        const auto t114 = -2 * t10 * t18 + t17;
        const auto t115 = -2 * t18 * t7 + t23;
        const auto t116 = t101 + t110;
        const auto t117 = t29 * t82;
        const auto t118 = t11 * t46 + t33 * t47;
        const auto t119 = t118 * t43;
        const auto t120 = std::pow(t11, 2);
        const auto t121 = std::pow(t18, 2);
        const auto t122 = t11 * t6;
        const auto t123 = t18 * t6;
        const auto t124 = t18 * t72 + t30 * t46;
        const auto t125 = t124 * t43;
        const auto t126 = std::pow(t6, 2);
        const auto t127 = t11 * t18;
        const auto t128 = t43 * (t45 * t72 + t47 * t6);
        dA[0] = t37 * (t2 - t22 * t36 + t5);
        dA[1] = -t50 * (t39 + t41 * t49);
        dA[2] = t50 * (t43 * t48 * t52 - t51);
        dA[3] = t37 * (t29 * t35 * t53 - t56);
        dA[4] = t50 * (-t49 * t57 + t58);
        dA[5] = t50 * (-t49 * t60 + t61);
        dA[6] = t37 * (-t36 * t63 + t66);
        dA[7] = t50 * (-t49 * t68 - t69);
        dA[8] = t50 * (t43 * t48 * t70 - t71);
        dA[9] = -t50 * (t35 * t74 + t39);
        dA[10] = t50 * (-t41 * t74 + t5 + t75);
        dA[11] = t50 * (t43 * t52 * t73 - t76);
        dA[12] = t50 * (t38 * t7 + t53 * t74 + t77);
        dA[13] = -t37 * (t57 * t78 + t80);
        dA[14] = t50 * (-t60 * t74 + t81);
        dA[15] = t50 * (-t74 * t82 - t83);
        dA[16] = t37 * (t67 * t78 + t85);
        dA[17] = t50 * (t43 * t70 * t73 - t86);
        dA[18] = -t50 * (t35 * t87 + t51);
        dA[19] = -t50 * (t41 * t87 + t76);
        dA[20] = t37 * (t2 - t29 * std::pow(t52, 2) + t75);
        dA[21] = t50 * (t53 * t87 + t89);
        dA[22] = t50 * (t1 * t16 - t26 - t57 * t87);
        dA[23] = t37 * (t29 * t52 * t90 - t91);
        dA[24] = t50 * (-t82 * t87 - t88 - t92);
        dA[25] = t50 * (-t68 * t87 - t93 - t94);
        dA[26] = t95;
        dA[27] = -t50 * (t35 * t96 + t56);
        dA[28] = t50 * (-t14 + t38 * t7 - t41 * t96);
        dA[29] = t50 * (t52 * t96 + t89);
        dA[30] = t50 * (t53 * t96 + t97 + t98);
        dA[31] = -t50 * (t57 * t96 + t99);
        dA[32] = -t50 * (t100 + t60 * t96);
        dA[33] = -t50 * (t103 + t82 * t96);
        dA[34] = t50 * (-t12 + 2 * t6 * t7 - t68 * t96);
        dA[35] = t50 * (t104 + t70 * t96);
        dA[36] = t50 * (-t106 * t35 + t58);
        dA[37] = t37 * (t29 * t40 * t57 - t80);
        dA[38] = t50 * (t1 * t16 + t106 * t52 + t94);
        dA[39] = t50 * (t105 * t43 * t53 - t99);
        dA[40] = t37 * (t107 - t29 * std::pow(t57, 2) + t98);
        dA[41] = -t50 * (t106 * t60 + t108);
        dA[42] = t50 * (-t106 * t82 - t109);
        dA[43] = t37 * (-t111 + t29 * t57 * t67);
        dA[44] = t50 * (t106 * t70 + 2 * t23 + t25);
        dA[45] = t50 * (-t113 * t35 + t61);
        dA[46] = t50 * (-t113 * t41 + t81);
        dA[47] = t50 * (t112 * t43 * t52 - t91);
        dA[48] = t50 * (-t100 + t112 * t43 * t53);
        dA[49] = -t50 * (t108 + t113 * t57);
        dA[50] = t50 * (t107 - t113 * t60 + t97);
        dA[51] = t50 * (-t113 * t82 - t114);
        dA[52] = t50 * (-t113 * t68 - t115);
        dA[53] = t50 * (t112 * t43 * t70 - t116);
        dA[54] = t37 * (-t117 * t22 + t66);
        dA[55] = t50 * (-t119 * t41 - t83);
        dA[56] = t50 * (t119 * t52 + t21 - t92);
        dA[57] = t37 * (-t103 + t29 * t53 * t82);
        dA[58] = t50 * (-t109 - t119 * t57);
        dA[59] = t50 * (-t114 - t119 * t60);
        dA[60] = t37 * (-t117 * t63 + t120 + t121);
        dA[61] = -t50 * (t119 * t68 + t122);
        dA[62] = t50 * (t118 * t43 * t70 - t123);
        dA[63] = t50 * (-t125 * t35 - t69);
        dA[64] = t50 * (-t125 * t41 + t85);
        dA[65] = t50 * (t125 * t52 + t26 - t93);
        dA[66] = t50 * (t125 * t53 + t13 + 2 * t8);
        dA[67] = -t50 * (t111 + t125 * t57);
        dA[68] = t50 * (-t115 - t125 * t60);
        dA[69] = -t50 * (t122 + t125 * t82);
        dA[70] = t50 * (t121 - t125 * t68 + t126);
        dA[71] = t50 * (t124 * t43 * t70 - t127);
        dA[72] = t50 * (-t128 * t35 - t71);
        dA[73] = t50 * (-t128 * t41 - t86);
        dA[74] = t95;
        dA[75] = t50 * (t104 + t128 * t53);
        dA[76] = t50 * (2 * t11 * t16 - t128 * t57 - t24);
        dA[77] = t37 * (-t116 + t29 * t70 * t90);
        dA[78] = -t50 * (t123 + t128 * t82);
        dA[79] = -t50 * (t127 + t128 * t68);
        dA[80] = t37 * (t120 + t126 - t29 * std::pow(t70, 2));
    }


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
        double dA[12])
    {
        const auto t0 = t0_x - t1_x;
        const auto t1 = -t0;
        const auto t2 = -t3_y;
        const auto t3 = t0_y + t2;
        const auto t4 = -t3;
        const auto t5 = -t3_x;
        const auto t6 = t0_x + t5;
        const auto t7 = -t6;
        const auto t8 = t0_y - t1_y;
        const auto t9 = -t8;
        const auto t10 = t1 * t4 - t7 * t9;
        const auto t11 = -t3_z;
        const auto t12 = t0_z + t11;
        const auto t13 = t0_z - t1_z;
        const auto t14 = t0 * t12 - t13 * t6;
        const auto t15 = std::pow(t14, 2);
        const auto t16 = -t12;
        const auto t17 = -t13;
        const auto t18 = t16 * t9 - t17 * t4;
        const auto t19 = std::pow(t10, 2) + t15 + std::pow(t18, 2);
        const auto t20 = -t2_z;
        const auto t21 = t0_z + t20;
        const auto t22 = -t21;
        const auto t23 = -t2_x;
        const auto t24 = t0_x + t23;
        const auto t25 = -t24;
        const auto t26 = t1 * t22 - t17 * t25;
        const auto t27 = -t2_y;
        const auto t28 = t0_y + t27;
        const auto t29 = t0 * t28 - t24 * t8;
        const auto t30 = -t13 * t28 + t21 * t8;
        const auto t31 = std::pow(t29, 2) + std::pow(t30, 2);
        const auto t32 = std::pow(t26, 2) + t31;
        const auto t33 = std::pow(t19 * t32, -1.0 / 2.0);
        const auto t34 = t1_y + t27;
        const auto t35 = t1_y + t2;
        const auto t36 = t1_z + t20;
        const auto t37 = t11 + t1_z;
        const auto t38 = 1.0 / t19;
        const auto t39 = 1.0 / t32;
        const auto t40 = t10 * t29 + t14 * t26 + t18 * t30;
        const auto t41 = -t10;
        const auto t42 = t1 * t16 - t17 * t7;
        const auto t43 = -t28;
        const auto t44 = t1 * t43 - t25 * t9;
        const auto t45 = -t26;
        const auto t46 = -t17 * t43 + t22 * t9;
        const auto t47 = std::pow(t44, 2) + std::pow(t45, 2) + std::pow(t46, 2);
        const auto t48 = -t18;
        const auto t49 = std::pow(t41, 2) + std::pow(t42, 2) + std::pow(t48, 2);
        const auto t50 = t1_x + t23;
        const auto t51 = t1_x + t5;
        const auto t52 = t39 * t40;
        const auto t53 = t38 * t52;
        const auto t54 = t0 * t21 - t13 * t24;
        const auto t55 = t31 + std::pow(t54, 2);
        const auto t56 = t12 * t8 - t13 * t3;
        const auto t57 =
            t15 + std::pow(t56, 2) + std::pow(t0 * t3 - t6 * t8, 2);
        const auto t58 = std::pow(t55 * t57, -1.0 / 2.0);
        const auto t59 = t29 * t8;
        const auto t60 = 1.0 / t55;
        const auto t61 = t10 * t8 + t13 * t14;
        const auto t62 = t0 * t29;
        const auto t63 = t13 * t30;
        const auto t64 = t0 * t10 - t13 * t18;
        const auto t65 = t0 * t14;
        const auto t66 = t0 * t26 + t30 * t8;
        dA[0] = t33
            * (-t10 * t34 - t14 * t36 - t26 * t37 - t29 * t35
               + t38 * t39 * t40
                   * (t47 * (-t35 * t41 + t37 * t42)
                      + t49 * (t34 * t44 - t36 * t45)));
        dA[1] = t33
            * (t10 * t50 - t18 * t36 + t29 * t51 - t30 * t37
               + t53
                   * (t47 * (-t37 * t48 + t41 * t51)
                      + t49 * (t36 * t46 - t44 * t50)));
        dA[2] = t33
            * (t14 * t50 + t18 * t34 + t26 * t51 + t30 * t35
               + t53
                   * (t47 * (t35 * t48 - t42 * t51)
                      + t49 * (-t34 * t46 + t45 * t50)));
        dA[3] = t33
            * (t10 * t28 + t12 * t26 + t14 * t21 + t29 * t3
               + t53
                   * (t47 * (t16 * t42 + t3 * t41)
                      + t49 * (t21 * t45 + t43 * t44)));
        dA[4] = t33
            * (-t10 * t24 + t12 * t30 + t18 * t21 - t29 * t6
               + t53
                   * (t47 * (t12 * t48 + t41 * t7)
                      + t49 * (t22 * t46 + t24 * t44)));
        dA[5] = t33
            * (-t14 * t24 - t18 * t28 - t26 * t6 - t3 * t30
               + t38 * t39 * t40
                   * (t47 * (t4 * t48 + t42 * t6)
                      + t49 * (t25 * t45 + t28 * t46)));
        dA[6] = t58 * (t40 * t60 * (t13 * t54 + t59) - t61);
        dA[7] = t33 * (t52 * (-t62 + t63) + t64);
        dA[8] = t58 * (-t40 * t60 * t66 + t56 * t8 + t65);
        dA[9] = t33 * (-t13 * t26 + t38 * t40 * t61 - t59);
        dA[10] = t58 * (-t40 * t64 / t57 + t62 - t63);
        dA[11] = t33 * (-t38 * t40 * (t18 * t8 + t65) + t66);
    }

    // dA is (144×1) flattened in column-major order
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
        double dA[144])
    {
        const auto t0 = t0_x - t1_x;
        const auto t1 = -t2_z;
        const auto t2 = t0_z + t1;
        const auto t3 = t0 * t2;
        const auto t4 = -t2_x;
        const auto t5 = t0_x + t4;
        const auto t6 = t0_z - t1_z;
        const auto t7 = t5 * t6;
        const auto t8 = -t7;
        const auto t9 = t3 + t8;
        const auto t10 = -t2_y;
        const auto t11 = t0_y + t10;
        const auto t12 = t0 * t11;
        const auto t13 = t0_y - t1_y;
        const auto t14 = t13 * t5;
        const auto t15 = -t14;
        const auto t16 = t12 + t15;
        const auto t17 = t13 * t2;
        const auto t18 = t11 * t6;
        const auto t19 = -t18;
        const auto t20 = t17 + t19;
        const auto t21 = std::pow(t16, 2) + std::pow(t20, 2);
        const auto t22 = t21 + std::pow(t9, 2);
        const auto t23 = -t3_y;
        const auto t24 = t0_y + t23;
        const auto t25 = t0 * t24;
        const auto t26 = -t3_x;
        const auto t27 = t0_x + t26;
        const auto t28 = t13 * t27;
        const auto t29 = -t28;
        const auto t30 = t25 + t29;
        const auto t31 = -t3_z;
        const auto t32 = t0_z + t31;
        const auto t33 = t0 * t32;
        const auto t34 = t27 * t6;
        const auto t35 = -t34;
        const auto t36 = t33 + t35;
        const auto t37 = std::pow(t36, 2);
        const auto t38 = t13 * t32;
        const auto t39 = t24 * t6;
        const auto t40 = -t39;
        const auto t41 = t38 + t40;
        const auto t42 = std::pow(t30, 2) + t37 + std::pow(t41, 2);
        const auto t43 = std::pow(t22 * t42, -1.0 / 2.0);
        const auto t44 = t10 + t1_y;
        const auto t45 = t1_y + t23;
        const auto t46 = 2 * t44 * t45;
        const auto t47 = t1 + t1_z;
        const auto t48 = t1_z + t31;
        const auto t49 = 2 * t47 * t48;
        const auto t50 = 1.0 / t22;
        const auto t51 = 1.0 / t42;
        const auto t52 = -t0;
        const auto t53 = -t24;
        const auto t54 = t52 * t53;
        const auto t55 = -t27;
        const auto t56 = -t13;
        const auto t57 = t55 * t56;
        const auto t58 = t54 - t57;
        const auto t59 = -t2;
        const auto t60 = t52 * t59;
        const auto t61 = -t5;
        const auto t62 = -t6;
        const auto t63 = t61 * t62;
        const auto t64 = t60 - t63;
        const auto t65 = t16 * t45 + t36 * t47 + t44 * t58 + t48 * t64;
        const auto t66 = t16 * t44 + t47 * t64;
        const auto t67 = -t32;
        const auto t68 = t56 * t67;
        const auto t69 = -t53 * t62 + t68;
        const auto t70 = t37 + std::pow(t58, 2) + std::pow(t69, 2);
        const auto t71 = t36 * t48 + t45 * t58;
        const auto t72 = t21 + std::pow(t64, 2);
        const auto t73 = t66 * t70 + t71 * t72;
        const auto t74 = std::pow(t42, -2);
        const auto t75 = t16 * t58 + t20 * t69 + t36 * t64;
        const auto t76 = std::pow(t22, -2);
        const auto t77 = t75 * t76;
        const auto t78 = t74 * t77;
        const auto t79 = std::pow(t44, 2);
        const auto t80 = std::pow(t47, 2);
        const auto t81 = std::pow(t45, 2);
        const auto t82 = std::pow(t48, 2);
        const auto t83 = 2 * t66;
        const auto t84 = t51 * t77;
        const auto t85 = t83 * t84;
        const auto t86 = 2 * t71;
        const auto t87 = t50 * t74 * t75;
        const auto t88 = t86 * t87;
        const auto t89 = t1_x + t4;
        const auto t90 = -t16 * t89 + t20 * t47;
        const auto t91 = t1_x + t26;
        const auto t92 = t48 * t69 - t58 * t91;
        const auto t93 = t70 * t90 + t72 * t92;
        const auto t94 = t50 * t51;
        const auto t95 = t65 * t94;
        const auto t96 = t70 * t89;
        const auto t97 = t72 * t91;
        const auto t98 = t75 * t94;
        const auto t99 = t16 * t91 - t20 * t48;
        const auto t100 = t30 * t89 - t41 * t47 + t99;
        const auto t101 = t73 * t94;
        const auto t102 = t73 * t78;
        const auto t103 = -t100 * t101 - t102 * t93 + t44 * t91 + t45 * t89
            + t93 * t95
            + t98 * (-t44 * t96 - t45 * t97 + 2 * t66 * t92 + 2 * t71 * t90);
        const auto t104 = t20 * t44 + t64 * t89;
        const auto t105 = t36 * t91 + t45 * t69;
        const auto t106 = t104 * t70 + t105 * t72;
        const auto t107 = t20 * t45 + t36 * t89;
        const auto t108 = t107 + t41 * t44 + t9 * t91;
        const auto t109 = -t101 * t108 + t102 * t106 - t106 * t95 + t47 * t91
            + t48 * t89
            - t98 * (t104 * t86 + t105 * t83 + t47 * t96 + t48 * t97);
        const auto t110 = t11 * t16 + t2 * t64;
        const auto t111 = t24 * t58 + t32 * t36;
        const auto t112 = t110 * t70 + t111 * t72;
        const auto t113 = t16 * t24 + t2 * t36;
        const auto t114 = t11 * t30 + t113 + t32 * t9;
        const auto t115 = t11 * t44;
        const auto t116 = t2 * t47;
        const auto t117 = t24 * t45;
        const auto t118 = t32 * t48;
        const auto t119 = t2 * t48 + t32 * t47;
        const auto t120 = t11 * t45 + t24 * t44;
        const auto t121 = -t101 * t114 + t102 * t112 - t112 * t95 + t119 + t120
            - t98
                * (t110 * t86 + t111 * t83 + t70 * (t115 + t116)
                   + t72 * (t117 + t118));
        const auto t122 = t16 * t5 - t2 * t20;
        const auto t123 = t27 * t58 - t32 * t69;
        const auto t124 = t122 * t70 + t123 * t72;
        const auto t125 = t16 * t27 - t2 * t69 - t20 * t32 + t5 * t58;
        const auto t126 = t16 + t30;
        const auto t127 = t102 * t124 - t124 * t50 * t51 * t65
            - t125 * t50 * t51 * t73 + t126 + t27 * t44 + t45 * t5
            - t50 * t51 * t75
                * (t122 * t86 + t123 * t83 + t70 * (t16 + t44 * t5)
                   + t72 * (t27 * t45 + t30));
        const auto t128 = t11 * t20 + t5 * t64;
        const auto t129 = t24 * t69 + t27 * t36;
        const auto t130 = t128 * t70 + t129 * t72;
        const auto t131 = t11 * t69 + t20 * t24 + t27 * t64 + t36 * t5;
        const auto t132 = t102 * t130 - t130 * t50 * t51 * t65
            - t131 * t50 * t51 * t73 + t27 * t47 + t36 + t48 * t5
            - t50 * t51 * t75
                * (t128 * t86 + t129 * t83 + t70 * (t47 * t5 + t9)
                   + t72 * (t27 * t48 + t36))
            + t9;
        const auto t133 = std::pow(t70 * t72, -1.0 / 2.0);
        const auto t134 = t13 * t44;
        const auto t135 = t47 * t6;
        const auto t136 = t134 + t135;
        const auto t137 = 1.0 / t72;
        const auto t138 = t13 * t16;
        const auto t139 = t138 + t6 * t64;
        const auto t140 = t36 * t6;
        const auto t141 = t13 * t58 + t140;
        const auto t142 = 1.0 / t70;
        const auto t143 = -t45;
        const auto t144 = -t58;
        const auto t145 = t52 * t67 - t55 * t62;
        const auto t146 = t143 * t144 + t145 * t48;
        const auto t147 = -t11;
        const auto t148 = t147 * t52 - t56 * t61;
        const auto t149 = -t64;
        const auto t150 = t56 * t59;
        const auto t151 = t147 * t62;
        const auto t152 = -t151;
        const auto t153 = t150 + t152;
        const auto t154 =
            std::pow(t148, 2) + std::pow(t149, 2) + std::pow(t153, 2);
        const auto t155 = -t47;
        const auto t156 = t148 * t44 + t149 * t155;
        const auto t157 = -t69;
        const auto t158 =
            std::pow(t144, 2) + std::pow(t145, 2) + std::pow(t157, 2);
        const auto t159 = t146 * t154 + t156 * t158;
        const auto t160 = std::pow(t72, -2);
        const auto t161 = t139 * t160;
        const auto t162 = t156 * t75;
        const auto t163 = 2 * t162;
        const auto t164 = t142 * t75;
        const auto t165 = t159 * t164;
        const auto t166 = t13 * t45;
        const auto t167 = t48 * t6;
        const auto t168 = t166 + t167;
        const auto t169 = t0 * t16 - t20 * t6;
        const auto t170 = -t169;
        const auto t171 = t137 * t65;
        const auto t172 = t0 * t44 + t16;
        const auto t173 = -t172;
        const auto t174 = t137 * t75;
        const auto t175 = t0 * t58 - t6 * t69;
        const auto t176 = t142 * t175;
        const auto t177 = t137 * t159;
        const auto t178 = 2 * t170;
        const auto t179 = t160 * t178;
        const auto t180 = t160 * t165;
        const auto t181 = t0 * t45 + t30;
        const auto t182 = t0 * t47 + t9;
        const auto t183 = t13 * t20;
        const auto t184 = t0 * t64 + t183;
        const auto t185 = t160 * t184;
        const auto t186 = t0 * t36;
        const auto t187 = t13 * t69 + t186;
        const auto t188 = t142 * t177;
        const auto t189 = t0 * t48 + t36;
        const auto t190 = -t171 * t184 + t180 * t184 - t187 * t188 + t189;
        const auto t191 = 2 * t141;
        const auto t192 = std::pow(t70, -2);
        const auto t193 = t192 * t75;
        const auto t194 = t146 * t193;
        const auto t195 = t174 * t192;
        const auto t196 = t159 * t195;
        const auto t197 = -t175;
        const auto t198 = 2 * t197;
        const auto t199 = t142 * t65;
        const auto t200 = t137 * t169;
        const auto t201 = t142 * t200;
        const auto t202 = -t159 * t201 + t172 - t196 * t197 + t197 * t199;
        const auto t203 = -t189;
        const auto t204 = 2 * t187;
        const auto t205 = 2 * t92;
        const auto t206 = t205 * t87;
        const auto t207 = 2 * t90;
        const auto t208 = t207 * t84;
        const auto t209 = 2 * t89 * t91;
        const auto t210 = t100 * t94;
        const auto t211 = 2 * t93;
        const auto t212 = std::pow(t89, 2);
        const auto t213 = std::pow(t91, 2);
        const auto t214 = t93 * t94;
        const auto t215 = t78 * t93;
        const auto t216 = t106 * t210 + t106 * t215 - t108 * t214 + t44 * t48
            + t45 * t47
            - t98
                * (t104 * t205 + t105 * t207 + t44 * t47 * t70
                   + t45 * t48 * t72);
        const auto t217 = -t12;
        const auto t218 = t14 + t217;
        const auto t219 = -t25;
        const auto t220 = t219 + t28;
        const auto t221 = -t11 * t91 + t112 * t210 + t112 * t215 - t114 * t214
            + t126 - t24 * t89
            + t98
                * (-t110 * t205 - t111 * t207 + t70 * (t11 * t89 + t218)
                   + t72 * (t220 + t24 * t91));
        const auto t222 = t5 * t89;
        const auto t223 = t27 * t91;
        const auto t224 = t27 * t89 + t5 * t91;
        const auto t225 = t119 - t124 * t210 - t124 * t215 + t125 * t214 + t224
            + t98
                * (2 * t122 * t92 + 2 * t123 * t90 - t70 * (t116 + t222)
                   - t72 * (t118 + t223));
        const auto t226 = t20 + t41;
        const auto t227 = t11 * t48 + t130 * t210 + t130 * t215
            - t131 * t50 * t51 * t93 + t226 + t24 * t47
            - t50 * t51 * t75
                * (t128 * t205 + t129 * t207 + t70 * (t11 * t47 + t20)
                   + t72 * (t24 * t48 + t41));
        const auto t228 = t13 * t89;
        const auto t229 = -t89;
        const auto t230 = t148 * t229 + t153 * t47;
        const auto t231 = t230 * t75;
        const auto t232 = 2 * t231;
        const auto t233 = t13 * t91;
        const auto t234 = -t47 * t69 + t58 * t89 + t99;
        const auto t235 = t137 * t234;
        const auto t236 = t144 * t91 - t157 * t48;
        const auto t237 = t154 * t236 + t158 * t230;
        const auto t238 = t164 * t237;
        const auto t239 =
            -t137 * t141 * t142 * t237 + t139 * t235 + t161 * t238 - t233 + t30;
        const auto t240 = t0 * t89;
        const auto t241 = t135 + t240;
        const auto t242 = t137 * t237;
        const auto t243 = t160 * t170;
        const auto t244 = t0 * t91;
        const auto t245 = t167 + t244;
        const auto t246 = t13 * t47 + t20;
        const auto t247 = t142 * t242;
        const auto t248 = t13 * t48 + t41;
        const auto t249 = t184 * t235 + t185 * t238 - t187 * t247 + t248;
        const auto t250 = t142 * t234;
        const auto t251 = -t220 - t233;
        const auto t252 = t193 * t236;
        const auto t253 = t195 * t237;
        const auto t254 = t16 - t228;
        const auto t255 = -t248;
        const auto t256 = 2 * t105;
        const auto t257 = t256 * t87;
        const auto t258 = 2 * t104;
        const auto t259 = t258 * t84;
        const auto t260 = -t3 + t7;
        const auto t261 = -t33 + t34;
        const auto t262 = t106 * t78;
        const auto t263 = -t106 * t114 * t50 * t51 - t108 * t112 * t50 * t51
            + t112 * t262 + t2 * t91 + t260 + t261 + t32 * t89
            - t50 * t51 * t75
                * (t110 * t256 + t111 * t258 + t70 * (t2 * t89 + t260)
                   + t72 * (t261 + t32 * t91));
        const auto t264 = -t17;
        const auto t265 = t18 + t264;
        const auto t266 = -t38;
        const auto t267 = t266 + t39;
        const auto t268 = t108 * t94;
        const auto t269 = t106 * t94;
        const auto t270 = t124 * t262 - t124 * t268 - t125 * t269 - t2 * t45
            + t226 - t32 * t44
            + t98
                * (-t122 * t256 - t123 * t258 + t70 * (t2 * t44 + t265)
                   + t72 * (t267 + t32 * t45));
        const auto t271 = t120 + t130 * t262 - t130 * t268 - t131 * t269 + t224
            - t98
                * (t128 * t256 + t129 * t258 + t70 * (t115 + t222)
                   + t72 * (t117 + t223));
        const auto t272 = t6 * t89;
        const auto t273 = t260 + t272;
        const auto t274 = t149 * t89 - t153 * t44;
        const auto t275 = t274 * t75;
        const auto t276 = 2 * t275;
        const auto t277 = t6 * t91;
        const auto t278 = t107 + t44 * t69 + t64 * t91;
        const auto t279 = t137 * t278;
        const auto t280 = -t145 * t91 + t157 * t45;
        const auto t281 = t154 * t280 + t158 * t274;
        const auto t282 = t164 * t281;
        const auto t283 =
            -t137 * t141 * t142 * t281 + t139 * t279 + t161 * t282 - t277 + t36;
        const auto t284 = t45 * t6;
        const auto t285 = t44 * t6;
        const auto t286 = -t265 - t285;
        const auto t287 = t137 * t281;
        const auto t288 = t134 + t240;
        const auto t289 = t142 * t287;
        const auto t290 = t142 * t278;
        const auto t291 = -t261 - t277;
        const auto t292 = t193 * t280;
        const auto t293 = t195 * t281;
        const auto t294 = t267 + t284;
        const auto t295 = t197 * t290 + t197 * t293 + t20 + t201 * t281 - t285;
        const auto t296 = t166 + t244;
        const auto t297 = 2 * t111;
        const auto t298 = t297 * t87;
        const auto t299 = 2 * t110;
        const auto t300 = t299 * t84;
        const auto t301 = 2 * t11 * t24;
        const auto t302 = 2 * t2 * t32;
        const auto t303 = std::pow(t11, 2);
        const auto t304 = std::pow(t2, 2);
        const auto t305 = std::pow(t24, 2);
        const auto t306 = std::pow(t32, 2);
        const auto t307 = t114 * t94;
        const auto t308 = t112 * t94;
        const auto t309 = t5 * t70;
        const auto t310 = t27 * t72;
        const auto t311 = t112 * t78;
        const auto t312 = t11 * t27 - t124 * t307 + t124 * t311 - t125 * t308
            + t24 * t5
            - t98 * (t11 * t309 + t122 * t297 + t123 * t299 + t24 * t310);
        const auto t313 = -t130 * t307 + t130 * t311 - t131 * t308 + t2 * t27
            + t32 * t5
            - t98 * (t128 * t297 + t129 * t299 + t2 * t309 + t310 * t32);
        const auto t314 = -t32 * t6;
        const auto t315 = t11 * t13;
        const auto t316 = t2 * t6;
        const auto t317 = t315 + t316;
        const auto t318 = t11 * t58 + t113 + t32 * t64;
        const auto t319 = t137 * t318;
        const auto t320 = t147 * t148 + t149 * t2;
        const auto t321 = t320 * t75;
        const auto t322 = 2 * t321;
        const auto t323 = t144 * t24 + t145 * t67;
        const auto t324 = t154 * t323 + t158 * t320;
        const auto t325 = t164 * t324;
        const auto t326 = 2 * t12;
        const auto t327 = t15 + t326;
        const auto t328 = t137 * t324;
        const auto t329 = 2 * t25;
        const auto t330 = t29 + t329;
        const auto t331 = 2 * t3;
        const auto t332 = t331 + t8;
        const auto t333 = t142 * t328;
        const auto t334 = 2 * t33;
        const auto t335 = -t334 + t34;
        const auto t336 = -t2 * t6;
        const auto t337 = t13 * t24;
        const auto t338 = t32 * t6;
        const auto t339 = t337 + t338;
        const auto t340 = t142 * t318;
        const auto t341 = t193 * t323;
        const auto t342 = t195 * t324;
        const auto t343 = t334 + t35;
        const auto t344 = -t331 + t7;
        const auto t345 = 2 * t123;
        const auto t346 = t345 * t87;
        const auto t347 = 2 * t73;
        const auto t348 = t122 * t84;
        const auto t349 = 2 * t106;
        const auto t350 = 2 * t112;
        const auto t351 = 2 * t27 * t5;
        const auto t352 = std::pow(t5, 2);
        const auto t353 = std::pow(t27, 2);
        const auto t354 = 2 * t348;
        const auto t355 = t124 * t94;
        const auto t356 = t130 * t94;
        const auto t357 = t11 * t32 - t124 * t130 * t78 + t125 * t356
            + t131 * t355 + t2 * t24
            + t98
                * (-t11 * t2 * t70 + 2 * t122 * t129 + 2 * t123 * t128
                   - t24 * t32 * t72);
        const auto t358 = 2 * t28;
        const auto t359 = 2 * t14;
        const auto t360 = t12 - t359;
        const auto t361 = -t360;
        const auto t362 = -t125;
        const auto t363 = t137 * t362;
        const auto t364 = t148 * t5 + t153 * t59;
        const auto t365 = t364 * t75;
        const auto t366 = 2 * t365;
        const auto t367 = t144 * t55 + t157 * t32;
        const auto t368 = t154 * t367 + t158 * t364;
        const auto t369 = t164 * t368;
        const auto t370 = t0 * t5;
        const auto t371 = t316 + t370;
        const auto t372 = t137 * t368;
        const auto t373 = 2 * t17;
        const auto t374 = t19 + t373;
        const auto t375 = 2 * t38;
        const auto t376 = t142 * t372;
        const auto t377 = t184 * t363 + t185 * t369 - t187 * t376 - t375 + t39;
        const auto t378 = t25 - t358;
        const auto t379 = -t378;
        const auto t380 = t142 * t362;
        const auto t381 = t193 * t367;
        const auto t382 = t195 * t368;
        const auto t383 = t0 * t27;
        const auto t384 = t338 + t383;
        const auto t385 = t375 + t40;
        const auto t386 = 2 * t129;
        const auto t387 = t386 * t87;
        const auto t388 = t128 * t84;
        const auto t389 = 2 * t388;
        const auto t390 = t131 * t137;
        const auto t391 = t3 - 2 * t7;
        const auto t392 = -t391;
        const auto t393 = t141 * t142;
        const auto t394 = t11 * t153 + t149 * t61;
        const auto t395 = t145 * t27 + t157 * t53;
        const auto t396 = t154 * t395 + t158 * t394;
        const auto t397 = t137 * t396;
        const auto t398 = t394 * t75;
        const auto t399 = 2 * t398;
        const auto t400 = t164 * t396;
        const auto t401 = t33 - 2 * t34;
        const auto t402 = 2 * t18;
        const auto t403 = -t17 + t402;
        const auto t404 = 2 * t39;
        const auto t405 = t266 + t404;
        const auto t406 = t315 + t370;
        const auto t407 = t142 * t397;
        const auto t408 = t337 + t383;
        const auto t409 = t131 * t142;
        const auto t410 = -t401;
        const auto t411 = t193 * t395;
        const auto t412 = t195 * t396;
        const auto t413 =
            -t131 * t142 * t197 + t197 * t412 + t201 * t396 + t264 + t402;
        const auto t414 = t138 + t6 * t9;
        const auto t415 = t414 * t84;
        const auto t416 = t139 * t84;
        const auto t417 = t13 * t148 + t149 * t62;
        const auto t418 = 2 * t417;
        const auto t419 = t160 * t418;
        const auto t420 = t414 * t50;
        const auto t421 = t141 * t51;
        const auto t422 = t421 * t50;
        const auto t423 = 2 * t416;
        const auto t424 = std::pow(t139, 2);
        const auto t425 = t139 * t174;
        const auto t426 = std::pow(t13, 2);
        const auto t427 = std::pow(t6, 2);
        const auto t428 = t426 + t427;
        const auto t429 = -2 * t139 * t141 - t428 * t75;
        const auto t430 = t133 * t137;
        const auto t431 = t170 * t174;
        const auto t432 = t0 * t13;
        const auto t433 = t432 * t75;
        const auto t434 = t139 * t175 - t141 * t170 + t170 * t425 + t433;
        const auto t435 = t0 * t6;
        const auto t436 = t139 * t187 + t141 * t184 + t435 * t75;
        const auto t437 = -t137 * t139 * t184 * t75 + t436;
        const auto t438 = std::pow(t141, 2);
        const auto t439 =
            t133 * (t137 * t424 + t142 * t438 - t393 * t425 - t428);
        const auto t440 = t175 * t51;
        const auto t441 = t420 * t75;
        const auto t442 =
            t43 * (-t169 * t420 - t175 * t421 + t432 + t440 * t441);
        const auto t443 = t0 * t9 + t183;
        const auto t444 = t187 * t51;
        const auto t445 =
            t43 * (-t187 * t421 - t420 * t443 + t435 + t441 * t444);
        const auto t446 = t169 * t50;
        const auto t447 = t51 * (t0 * t30 - t41 * t6);
        const auto t448 = t447 * t50;
        const auto t449 = t169 * t84;
        const auto t450 = t178 * t84;
        const auto t451 = t148 * t52 + t153 * t6;
        const auto t452 = 2 * t451;
        const auto t453 = std::pow(t0, 2);
        const auto t454 = t427 + t453;
        const auto t455 = -t454 * t75;
        const auto t456 = t174 * t178;
        const auto t457 = t175 * t184;
        const auto t458 = t170 * t187;
        const auto t459 = t13 * t6;
        const auto t460 = t459 * t75;
        const auto t461 = -t460;
        const auto t462 = t174 * t184;
        const auto t463 = t13 * t30 + t140;
        const auto t464 = t463 * t51;
        const auto t465 = t464 * t75;
        const auto t466 =
            t43 * (-t139 * t446 + t432 + t446 * t465 - t447 * t463);
        const auto t467 = t142 * t174;
        const auto t468 =
            -t133 * (t170 * t197 * t467 + t170 * t200 + t176 * t197 + t454);
        const auto t469 = t169 * t187;
        const auto t470 = t43 * (t187 * t447 + t443 * t446 + t459 - t469 * t98);
        const auto t471 = t0 * t149 + t153 * t56;
        const auto t472 = 2 * t471;
        const auto t473 = t160 * t472;
        const auto t474 = t13 * t41 + t186;
        const auto t475 = 3 * t106;
        const auto t476 = t184 * t84;
        const auto t477 = t184 * t50;
        const auto t478 = t474 * t51;
        const auto t479 = t478 * t50;
        const auto t480 = 3 * t476;
        const auto t481 = std::pow(t184, 2);
        const auto t482 = t426 + t453;
        const auto t483 = t184 * t204 + t482 * t75;
        const auto t484 =
            t43 * (-t139 * t477 + t435 - t463 * t478 + t465 * t477);
        const auto t485 = t43 * (t175 * t478 + t184 * t446 - t457 * t98 + t459);
        const auto t486 = std::pow(t187, 2);
        const auto t487 =
            t133 * (t137 * t481 + t142 * t486 - t184 * t187 * t467 - t482);
        const auto t488 = t463 * t87;
        const auto t489 = t191 * t87;
        const auto t490 = t144 * t56 + t145 * t6;
        const auto t491 = t164 * t490;
        const auto t492 = t133 * t142;
        const auto t493 =
            -t139 * t197 + t141 * t164 * t197 + t141 * t169 + t433;
        const auto t494 = -t141 * t142 * t187 * t75 + t436;
        const auto t495 = t0 * t144 + t157 * t62;
        const auto t496 = 2 * t495;
        const auto t497 = t446 * t51;
        const auto t498 = t198 * t87;
        const auto t499 = t175 * t87;
        const auto t500 = t164 * t495;
        const auto t501 = t164 * t197;
        const auto t502 = t184 * t197;
        const auto t503 = t187 * t87;
        const auto t504 = 3 * t503;
        const auto t505 = t164 * (t13 * t157 + t145 * t52);
        dA[0] = t43
            * (-t46 - t49 + 2 * t50 * t51 * t65 * t73
               + t50 * t51 * t75
                   * (4 * t66 * t71 + t70 * (t79 + t80) + t72 * (t81 + t82))
               - std::pow(t73, 2) * t78 - t73 * t85 - t73 * t88);
        dA[1] = t43 * (t103 - t85 * t93 - t88 * t93);
        dA[2] = t43 * (t106 * t85 + t106 * t88 + t109);
        dA[3] = t43 * (t112 * t85 + t112 * t88 + t121);
        dA[4] = t43 * (-t124 * t85 - t124 * t88 - t127);
        dA[5] = t43 * (-t130 * t85 - t130 * t88 - t132);
        dA[6] = t133
            * (t136 * t137 * t75 + t137 * t139 * t65 + t137 * t141 * t142 * t159
               - t161 * t163 - t161 * t165 - t168);
        dA[7] = t133
            * (-t162 * t179 + t170 * t171 - t170 * t180 + t173 * t174
               - t176 * t177 + t181);
        dA[8] = t133 * (t163 * t185 - t174 * t182 + t190);
        dA[9] = t133
            * (-t136 + t137 * t139 * t142 * t159 + t141 * t142 * t65
               - t141 * t196 + t142 * t168 * t75 - t191 * t194);
        dA[10] = t133 * (-t164 * t181 - t194 * t198 + t202);
        dA[11] = t133
            * (t164 * t203 + t182 - t184 * t188 + t187 * t196 - t187 * t199
               + t194 * t204);
        dA[12] = t43 * (t103 - t206 * t73 - t208 * t73);
        dA[13] = t43
            * (-t206 * t93 - t208 * t93 - t209 - t210 * t211 - t49
               + t50 * t51 * t75
                   * (t70 * (t212 + t80) + t72 * (t213 + t82) + 4 * t90 * t92)
               - t78 * std::pow(t93, 2));
        dA[14] = t43 * (t106 * t206 + t106 * t208 + t216);
        dA[15] = t43 * (t112 * t206 + t112 * t208 + t221);
        dA[16] = t43 * (-t124 * t206 - t124 * t208 + t225);
        dA[17] = t43 * (-t130 * t206 - t130 * t208 - t227);
        dA[18] = t133 * (t137 * t75 * (-t218 - t228) - t161 * t232 - t239);
        dA[19] = t133
            * (t137 * t241 * t75 - t170 * t235 - t176 * t242 - t179 * t231
               - t238 * t243 - t245);
        dA[20] = t133 * (-t174 * t246 + t185 * t232 + t249);
        dA[21] = t133
            * (t137 * t139 * t142 * t237 - t141 * t250 - t141 * t253
               + t142 * t251 * t75 - t191 * t252 - t254);
        dA[22] = t133
            * (t142 * t245 * t75 - t197 * t250 - t197 * t253 - t198 * t252
               - t201 * t237 - t241);
        dA[23] = t133
            * (t164 * t255 - t184 * t247 + t187 * t250 + t187 * t253
               + t204 * t252 + t246);
        dA[24] = t43 * (t109 + t257 * t73 + t259 * t73);
        dA[25] = t43 * (t216 + t257 * t93 + t259 * t93);
        dA[26] = t43
            * (-std::pow(t106, 2) * t78 + 2 * t106 * t108 * t50 * t51
               - t106 * t257 - t106 * t259 - t209 - t46
               + t50 * t51 * t75
                   * (4 * t104 * t105 + t70 * (t212 + t79)
                      + t72 * (t213 + t81)));
        dA[27] = t43 * (-t112 * t257 - t112 * t259 - t263);
        dA[28] = t43 * (t124 * t257 + t124 * t259 + t270);
        dA[29] = t43 * (t130 * t257 + t130 * t259 + t271);
        dA[30] = t133 * (-t137 * t273 * t75 - t161 * t276 - t283);
        dA[31] = t133
            * (t137 * t286 * t75 - t170 * t279 - t176 * t287 - t179 * t275
               - t243 * t282 + t284 - t41);
        dA[32] = t133
            * (-t166 + t174 * t288 + t184 * t279 + t185 * t276 + t185 * t282
               - t187 * t289 - t244);
        dA[33] = t133
            * (t137 * t139 * t142 * t281 - t141 * t290 - t141 * t293
               + t142 * t291 * t75 - t191 * t292 + t272 - t9);
        dA[34] = t133 * (-t142 * t294 * t75 - t198 * t292 - t295);
        dA[35] = t133
            * (-t134 + t164 * t296 - t184 * t289 + t187 * t290 + t187 * t293
               + t204 * t292 - t240);
        dA[36] = t43 * (t121 + t298 * t73 + t300 * t73);
        dA[37] = t43 * (t221 + t298 * t93 + t300 * t93);
        dA[38] = t43 * (-t106 * t298 - t106 * t300 - t263);
        dA[39] = t43
            * (-std::pow(t112, 2) * t78 + 2 * t112 * t114 * t50 * t51
               - t112 * t298 - t112 * t300 - t301 - t302
               + t50 * t51 * t75
                   * (4 * t110 * t111 + t70 * (t303 + t304)
                      + t72 * (t305 + t306)));
        dA[40] = t43 * (t124 * t298 + t124 * t300 + t312);
        dA[41] = t43 * (t130 * t298 + t130 * t300 + t313);
        dA[42] = t133
            * (t13 * t24 + t137 * t141 * t142 * t324 - t139 * t319 - t161 * t322
               - t161 * t325 - t174 * t317 - t314);
        dA[43] = t133
            * (t137 * t327 * t75 - t170 * t319 - t176 * t328 - t179 * t321
               - t243 * t325 - t330);
        dA[44] = t133
            * (t174 * t332 + t184 * t319 + t185 * t322 + t185 * t325
               - t187 * t333 + t335);
        dA[45] = t133
            * (t11 * t13 + t137 * t139 * t142 * t324 - t141 * t340 - t141 * t342
               - t164 * t339 - t191 * t341 - t336);
        dA[46] = t133
            * (t142 * t330 * t75 - t197 * t340 - t197 * t342 - t198 * t341
               - t201 * t324 - t327);
        dA[47] = t133
            * (t164 * t343 - t184 * t333 + t187 * t340 + t187 * t342
               + t204 * t341 + t344);
        dA[48] = t43 * (-t127 - t346 * t73 - t347 * t348);
        dA[49] = t43 * (-t211 * t348 + t225 - t346 * t93);
        dA[50] = t43 * (t106 * t346 + t270 + t348 * t349);
        dA[51] = t43 * (t112 * t346 + t312 + t348 * t350);
        dA[52] = t43
            * (-std::pow(t124, 2) * t78 + 2 * t124 * t125 * t50 * t51
               - t124 * t346 - t124 * t354 - t302 - t351
               + t50 * t51 * t75
                   * (4 * t122 * t123 + t70 * (t304 + t352)
                      + t72 * (t306 + t353)));
        dA[53] = t43 * (-t130 * t346 - t130 * t354 + t357);
        dA[54] = t133
            * (t137 * t141 * t142 * t368 + t137 * t361 * t75 - t139 * t363
               - t161 * t366 - t161 * t369 - t219 - t358);
        dA[55] = t133
            * (t0 * t27 - t170 * t363 - t174 * t371 - t176 * t372 - t179 * t365
               - t243 * t369 - t314);
        dA[56] = t133 * (t174 * t374 + t185 * t366 + t377);
        dA[57] = t133
            * (t137 * t139 * t142 * t368 - t141 * t380 - t141 * t382
               + t142 * t379 * t75 - t191 * t381 - t217 - t359);
        dA[58] = t133
            * (t0 * t5 - t164 * t384 - t197 * t380 - t197 * t382 - t198 * t381
               - t201 * t368 - t336);
        dA[59] = t133
            * (t164 * t385 + t18 - t184 * t376 + t187 * t380 + t187 * t382
               + t204 * t381 - t373);
        dA[60] = t43 * (-t132 - t347 * t388 - t387 * t73);
        dA[61] = t43 * (-t211 * t388 - t227 - t387 * t93);
        dA[62] = t43 * (t106 * t387 + t271 + t349 * t388);
        dA[63] = t43 * (t112 * t387 + t313 + t350 * t388);
        dA[64] = t43 * (-t124 * t387 - t124 * t389 + t357);
        dA[65] = t43
            * (-std::pow(t130, 2) * t78 + 2 * t130 * t131 * t50 * t51
               - t130 * t387 - t130 * t389 - t301 - t351
               + t50 * t51 * t75
                   * (4 * t128 * t129 + t70 * (t303 + t352)
                      + t72 * (t305 + t353)));
        dA[66] = t133
            * (t139 * t390 - t161 * t399 - t161 * t400 + t174 * t392
               + t393 * t397 + t401);
        dA[67] = t133
            * (t131 * t137 * t170 + t137 * t403 * t75 - t176 * t397
               - t179 * t398 - t243 * t400 - t405);
        dA[68] = t133
            * (-t174 * t406 - t184 * t390 + t185 * t399 + t185 * t400
               - t187 * t407 + t408);
        dA[69] = t133
            * (t139 * t407 + t141 * t409 - t141 * t412 + t164 * t410
               - t191 * t411 + t391);
        dA[70] = t133 * (t142 * t75 * (-t38 + t404) - t198 * t411 - t413);
        dA[71] = t133
            * (-t164 * t408 - t184 * t407 - t187 * t409 + t187 * t412
               + t204 * t411 + t406);
        dA[72] = t43
            * (t141 * t50 * t51 * t73 - t168 - t347 * t416 + t414 * t50 * t65
               - t415 * t73 + t50 * t51 * t75 * (t136 * t70 + t139 * t86));
        dA[73] = t133
            * (t137 * t142 * t75 * (t158 * (t13 * t229 + t148) + t236 * t418)
               - t238 * t419 - t239);
        dA[74] = t133
            * (t137 * t142 * t75 * (t158 * (t62 * t89 + t64) + t280 * t418)
               - t282 * t419 - t283);
        dA[75] = t43
            * (t112 * t415 - t112 * t422 - t114 * t420 + t339 + t350 * t416
               - t98 * (t139 * t297 + t317 * t70));
        dA[76] = t43
            * (-t124 * t415 + t124 * t422 - t124 * t423 + t125 * t420 + t378
               + t98 * (t139 * t345 + t361 * t70));
        dA[77] = t43
            * (-t130 * t415 + t130 * t422 - t130 * t423 + t131 * t420 + t401
               + t98 * (t139 * t386 + t392 * t70));
        dA[78] = t430 * (-t137 * t424 * t75 - t418 * t425 - t429);
        dA[79] = t430 * (-t418 * t431 - t434);
        dA[80] = t430 * (2 * t137 * t184 * t417 * t75 - t437);
        dA[81] = t439;
        dA[82] = t442;
        dA[83] = t445;
        dA[84] = t43
            * (t181 - t446 * t65 - t448 * t73 + t449 * t73 - t450 * t73
               + t98 * (t170 * t86 + t173 * t70));
        dA[85] = t43
            * (t100 * t169 * t50 + t169 * t51 * t75 * t76 * t93 - t245
               - t448 * t93 - t450 * t93
               + t50 * t51 * t75 * (t170 * t205 + t241 * t70));
        dA[86] = t43
            * (t106 * t448 - t106 * t449 + t106 * t450 + t108 * t446 + t294
               + t98 * (-t170 * t256 + t286 * t70));
        dA[87] = t43
            * (t112 * t448 - t112 * t449 + t112 * t450 + t114 * t446 + t28
               - t329 + t98 * (-t170 * t297 + t327 * t70));
        dA[88] = t43
            * (-t124 * t448 + t124 * t449 - t124 * t450 - t125 * t446 + t384
               + t98 * (2 * t123 * t170 - t371 * t70));
        dA[89] = t43
            * (t130 * t169 * t51 * t75 * t76 - t130 * t448 - t130 * t450
               - t131 * t446 - t405
               + t50 * t51 * t75 * (t170 * t386 + t403 * t70));
        dA[90] = t430 * (-t425 * t452 - t434);
        dA[91] = t430
            * (-std::pow(t170, 2) * t174 - t175 * t178 - t451 * t456 - t455);
        dA[92] = t430 * (t184 * t431 + t452 * t462 + t457 - t458 + t461);
        dA[93] = t466;
        dA[94] = t468;
        dA[95] = t470;
        dA[96] = t133
            * (-t180 * t472 + t190
               + t467 * (t146 * t472 + t158 * (t0 * t155 - t60 + t63)));
        dA[97] = t133
            * (-t238 * t473 + t249
               + t467 * (t158 * (-t150 + t151 + t47 * t56) + t236 * t472));
        dA[98] = t43
            * (t106 * t474 * t50 * t51 + t108 * t184 * t50 - t296 - t475 * t476
               + t50 * t51 * t75 * (t184 * t256 + t288 * t70));
        dA[99] = t43
            * (t112 * t479 - t112 * t480 + t114 * t477 + t335
               + t98 * (t184 * t297 + t332 * t70));
        dA[100] = t133
            * (-t369 * t473 + t377
               + t467 * (t158 * (2 * t150 + t152) + t367 * t472));
        dA[101] = t43
            * (-t130 * t479 + t130 * t480 - t131 * t477 + t408
               - t98 * (t184 * t386 + t406 * t70));
        dA[102] = t430 * (-t425 * t472 - t437);
        dA[103] = t430
            * (t137 * t170 * t184 * t75 + t175 * t184 - t456 * t471 - t458
               - t460);
        dA[104] = t430 * (-t137 * t481 * t75 + t462 * t472 + t483);
        dA[105] = t484;
        dA[106] = t485;
        dA[107] = t487;
        dA[108] = t43
            * (-t136 + t139 * t50 * t51 * t73 + t463 * t51 * t65 - t488 * t73
               - t489 * t73 + t50 * t51 * t75 * (t141 * t83 + t168 * t72));
        dA[109] = t43
            * (-t100 * t464 + t139 * t50 * t51 * t93 - t254 - t488 * t93
               - t489 * t93 + t50 * t51 * t75 * (t141 * t207 + t251 * t72));
        dA[110] = t43
            * (t106 * t488 + t106 * t489 - t108 * t464 - t139 * t269 + t273
               + t98 * (-t141 * t258 + t291 * t72));
        dA[111] = t43
            * (t112 * t488 + t112 * t489 - t114 * t464 - t139 * t308 + t317
               - t98 * (t141 * t299 + t339 * t72));
        dA[112] = t43
            * (-t124 * t488 - t124 * t489 + t125 * t464 + t139 * t355 + t360
               + t98 * (t122 * t191 + t379 * t72));
        dA[113] = t43
            * (-t130 * t488 - t130 * t489 + t131 * t464 + t139 * t356 + t391
               + t98 * (t128 * t191 + t410 * t72));
        dA[114] = t439;
        dA[115] = t466;
        dA[116] = t484;
        dA[117] = t492 * (-t142 * t438 * t75 - t191 * t491 - t429);
        dA[118] = t492 * (-t198 * t491 - t493);
        dA[119] = t492 * (2 * t142 * t187 * t490 * t75 - t494);
        dA[120] = t133
            * (-t196 * t496 + t202
               + t467 * (t154 * (t0 * t143 - t54 + t57) + t156 * t496));
        dA[121] = t43
            * (t100 * t175 * t51 + t175 * t50 * t74 * t75 * t93 - t241
               - t497 * t93 - t498 * t93
               + t50 * t51 * t75 * (t197 * t207 + t245 * t72));
        dA[122] = t133
            * (t137 * t142 * t75 * (t154 * (t45 * t62 + t69) + t274 * t496)
               - t293 * t496 - t295);
        dA[123] = t43
            * (t112 * t497 + t112 * t498 - t112 * t499 + t114 * t440 + t14
               - t326 + t98 * (-t197 * t299 + t330 * t72));
        dA[124] = t43
            * (-t124 * t497 - t124 * t498 + t124 * t499 - t125 * t440 + t371
               + t98 * (2 * t122 * t197 - t384 * t72));
        dA[125] = t133
            * (t137 * t142 * t75 * (t154 * (2 * t53 * t62 - t68) + t394 * t496)
               - t412 * t496 - t413);
        dA[126] = t442;
        dA[127] = t468;
        dA[128] = t485;
        dA[129] = t492 * (-t191 * t500 - t493);
        dA[130] = t492
            * (-t164 * std::pow(t197, 2) - t169 * t198 - t455 - t496 * t501);
        dA[131] = t492 * (t187 * t501 + t204 * t500 + t461 + t469 - t502);
        dA[132] = t43
            * (-t101 * t443 + t182 - t444 * t65 + t504 * t73
               + t98 * (-t187 * t83 + t203 * t72));
        dA[133] = t43
            * (t100 * t444 - t214 * t443 + t246 + t504 * t93
               + t98 * (-t187 * t207 + t255 * t72));
        dA[134] = t43
            * (t106 * t443 * t50 * t51 + t108 * t187 * t51 - t288 - t475 * t503
               + t50 * t51 * t75 * (t187 * t258 + t296 * t72));
        dA[135] = t43
            * (-t112 * t504 + t114 * t444 + t308 * t443 + t344
               + t98 * (t187 * t299 + t343 * t72));
        dA[136] = t43
            * (3 * t124 * t187 * t50 * t74 * t75 - t125 * t444 - t355 * t443
               - t374 + t50 * t51 * t75 * (-t122 * t204 + t385 * t72));
        dA[137] = t43
            * (t130 * t504 - t131 * t444 - t356 * t443 + t406
               - t98 * (t128 * t204 + t408 * t72));
        dA[138] = t445;
        dA[139] = t470;
        dA[140] = t487;
        dA[141] = t492 * (-t191 * t505 - t494);
        dA[142] = t492
            * (t142 * t187 * t197 * t75 + t169 * t187 - t198 * t505 - t460
               - t502);
        dA[143] = t492 * (-t142 * t486 * t75 + t204 * t505 + t483);
    }


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
        double dA[12])
    {
        const auto t0 = t0_x - t1_x;
        const auto t1 = std::pow(t0, 2);
        const auto t2 = t0_y - t1_y;
        const auto t3 = std::pow(t2, 2);
        const auto t4 = t0_z - t1_z;
        const auto t5 = std::pow(t4, 2);
        const auto t6 = t1 + t3 + t5;
        const auto t7 = -t0;
        const auto t8 = -t2_y;
        const auto t9 = t0_y + t8;
        const auto t10 = -t9;
        const auto t11 = -t2_x;
        const auto t12 = t0_x + t11;
        const auto t13 = -t12;
        const auto t14 = -t2;
        const auto t15 = t10 * t7 - t13 * t14;
        const auto t16 = -t2_z;
        const auto t17 = t0_z + t16;
        const auto t18 = t0 * t17 - t12 * t4;
        const auto t19 = -t17;
        const auto t20 = -t4;
        const auto t21 = -t10 * t20 + t14 * t19;
        const auto t22 = std::pow(t15, 2) + std::pow(t18, 2) + std::pow(t21, 2);
        const auto t23 = -t3_z;
        const auto t24 = t0_z + t23;
        const auto t25 = -t24;
        const auto t26 = -t3_x;
        const auto t27 = t0_x + t26;
        const auto t28 = -t27;
        const auto t29 = -t20 * t28 + t25 * t7;
        const auto t30 = -t3_y;
        const auto t31 = t0_y + t30;
        const auto t32 = t0 * t31 - t2 * t27;
        const auto t33 = t2 * t24 - t31 * t4;
        const auto t34 = std::pow(t29, 2) + std::pow(t32, 2) + std::pow(t33, 2);
        const auto t35 = std::pow(t22 * t34 * t6, -1.0 / 2.0);
        const auto t36 = t18 * t32;
        const auto t37 = t15 * t29;
        const auto t38 = t1_y + t8;
        const auto t39 = t1_y + t30;
        const auto t40 = t16 + t1_z;
        const auto t41 = t1_z + t23;
        const auto t42 = -t38;
        const auto t43 = -t29;
        const auto t44 = -t41;
        const auto t45 = -t15;
        const auto t46 = -t13 * t20 + t19 * t7;
        const auto t47 = -t31;
        const auto t48 = -t14 * t28 + t47 * t7;
        const auto t49 = t14 * t25 - t20 * t47;
        const auto t50 = std::pow(t43, 2) + std::pow(t48, 2) + std::pow(t49, 2);
        const auto t51 = std::pow(t14, 2) + std::pow(t20, 2) + std::pow(t7, 2);
        const auto t52 = t50 * t51;
        const auto t53 = -t21;
        const auto t54 = std::pow(t45, 2) + std::pow(t46, 2) + std::pow(t53, 2);
        const auto t55 = t51 * t54;
        const auto t56 = t50 * t54;
        const auto t57 = 1.0 / t34;
        const auto t58 = t15 * t33;
        const auto t59 = t21 * t32;
        const auto t60 = t0 * (t43 * t45 - t46 * t48) - t2 * (-t58 + t59)
            + t4 * (-t43 * t53 + t46 * t49);
        const auto t61 = t60 / t22;
        const auto t62 = t57 * t61 / t6;
        const auto t63 = t11 + t1_x;
        const auto t64 = t1_x + t26;
        const auto t65 = -t64;
        const auto t66 = -t40;
        const auto t67 = -t58 + t59;
        const auto t68 = -t63;
        const auto t69 = -t39;
        const auto t70 = -t18 * t33 + t21 * t29;
        const auto t71 = t18 * t31;
        const auto t72 = t17 * t32;
        const auto t73 = t12 * t33 - t21 * t27;
        const auto t74 = t57 * t60;
        dA[0] = t35
            * (t0 * (-t39 * t46 - t40 * t48 + t42 * t43 + t44 * t45)
               - t2 * (t21 * t39 - t33 * t38) - t36 + t37
               - t4 * (t21 * t41 - t33 * t40)
               - t62
                   * (t0 * t56 + t52 * (t40 * t46 + t42 * t45)
                      + t55 * (t39 * t48 + t43 * t44)));
        dA[1] = t35
            * (-t0 * (-t18 * t64 + t29 * t63)
               + t2 * (-t41 * t45 + t48 * t66 - t49 * t63 + t53 * t65)
               - t4 * (-t18 * t41 + t29 * t40)
               - t62
                   * (t2 * t56 + t52 * (t45 * t63 + t53 * t66)
                      + t55 * (t41 * t49 + t48 * t65))
               - t67);
        dA[2] = t35
            * (-t0 * (t15 * t64 - t32 * t63) - t2 * (t15 * t39 - t32 * t38)
               + t4 * (-t38 * t43 + t46 * t69 + t49 * t68 - t53 * t64)
               - t62
                   * (t4 * t56 + t52 * (t38 * t53 + t46 * t68)
                      + t55 * (t43 * t64 + t49 * t69))
               - t70);
        dA[3] = t35
            * (-t0 * (t15 * t24 + t29 * t9 - t71 - t72)
               + t2 * (t47 * t53 - t49 * t9) + t36 - t37
               + t4 * (t19 * t49 - t24 * t53)
               - t62
                   * (t52 * (t19 * t46 + t45 * t9)
                      + t55 * (t24 * t43 + t47 * t48) + t56 * t7));
        dA[4] = t35
            * (t0 * (t13 * t43 - t27 * t46) - t2 * (t15 * t24 - t72 - t73)
               + t4 * (-t17 * t43 + t25 * t46)
               - t62
                   * (t14 * t56 + t52 * (t13 * t45 + t17 * t53)
                      + t55 * (t25 * t49 + t27 * t48))
               + t67);
        dA[5] = t35
            * (t0 * (-t12 * t48 + t28 * t45) + t2 * (t10 * t48 - t31 * t45)
               - t4 * (t29 * t9 - t71 - t73)
               - t62
                   * (t20 * t56 + t52 * (t10 * t53 + t12 * t46)
                      + t55 * (t28 * t43 + t31 * t49))
               + t70);
        dA[6] = t35
            * (t0 * (t14 * t43 - t4 * t48) + t3 * t33 + t33 * t5
               - t61 * (t15 * t2 + t18 * t4));
        dA[7] = -t35
            * (t1 * t29 + t2 * (t0 * t33 + t32 * t4) + t29 * t5
               + t61 * (-t0 * t15 + t21 * t4));
        dA[8] = t35
            * (t1 * t32 + t3 * t32 + t4 * (-t2 * t43 + t49 * t7)
               + t61 * (t0 * t18 + t2 * t21));
        dA[9] = -t35
            * (t0 * (-t15 * t4 + t18 * t2) + t21 * t3 + t21 * t5
               + t74 * (t2 * t32 + t29 * t4));
        dA[10] = t35
            * (t1 * t18 + t18 * t5 + t2 * (-t4 * t45 + t53 * t7)
               - t74 * (-t0 * t32 + t33 * t4));
        dA[11] = t35
            * (-t1 * t15 - t15 * t3 - t4 * (-t0 * t21 + t18 * t2)
               + t57 * t60 * (t0 * t29 + t2 * t33));
    }

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
        double dA[144])
    {
        const auto t0 = t0_x - t1_x;
        const auto t1 = std::pow(t0, 2);
        const auto t2 = t0_y - t1_y;
        const auto t3 = std::pow(t2, 2);
        const auto t4 = t0_z - t1_z;
        const auto t5 = std::pow(t4, 2);
        const auto t6 = t3 + t5;
        const auto t7 = t1 + t6;
        const auto t8 = -t2_y;
        const auto t9 = t0_y + t8;
        const auto t10 = t0 * t9;
        const auto t11 = -t2_x;
        const auto t12 = t0_x + t11;
        const auto t13 = t12 * t2;
        const auto t14 = -t13;
        const auto t15 = t10 + t14;
        const auto t16 = -t2_z;
        const auto t17 = t0_z + t16;
        const auto t18 = t0 * t17;
        const auto t19 = t12 * t4;
        const auto t20 = -t19;
        const auto t21 = t18 + t20;
        const auto t22 = std::pow(t21, 2);
        const auto t23 = t17 * t2;
        const auto t24 = t4 * t9;
        const auto t25 = -t24;
        const auto t26 = t23 + t25;
        const auto t27 = std::pow(t15, 2) + t22 + std::pow(t26, 2);
        const auto t28 = -t3_z;
        const auto t29 = t0_z + t28;
        const auto t30 = t0 * t29;
        const auto t31 = -t3_x;
        const auto t32 = t0_x + t31;
        const auto t33 = t32 * t4;
        const auto t34 = -t33;
        const auto t35 = t30 + t34;
        const auto t36 = -t3_y;
        const auto t37 = t0_y + t36;
        const auto t38 = t0 * t37;
        const auto t39 = t2 * t32;
        const auto t40 = -t39;
        const auto t41 = t38 + t40;
        const auto t42 = t2 * t29;
        const auto t43 = t37 * t4;
        const auto t44 = -t43;
        const auto t45 = t42 + t44;
        const auto t46 = std::pow(t41, 2) + std::pow(t45, 2);
        const auto t47 = std::pow(t35, 2) + t46;
        const auto t48 = std::pow(t27 * t47 * t7, -1.0 / 2.0);
        const auto t49 = t1_y + t8;
        const auto t50 = t1_z + t28;
        const auto t51 = t1_y + t36;
        const auto t52 = t16 + t1_z;
        const auto t53 = t49 * t50 - t51 * t52;
        const auto t54 = 2 * t0;
        const auto t55 = -t0;
        const auto t56 = -t9;
        const auto t57 = t55 * t56;
        const auto t58 = -t12;
        const auto t59 = -t2;
        const auto t60 = t58 * t59;
        const auto t61 = t57 - t60;
        const auto t62 = t21 * t52 + t49 * t61;
        const auto t63 = -t29;
        const auto t64 = t55 * t63;
        const auto t65 = -t32;
        const auto t66 = -t4;
        const auto t67 = t65 * t66;
        const auto t68 = t64 - t67;
        const auto t69 = t46 + std::pow(t68, 2);
        const auto t70 = t69 * t7;
        const auto t71 = t41 * t51 + t50 * t68;
        const auto t72 = -t17;
        const auto t73 = t59 * t72;
        const auto t74 = t56 * t66;
        const auto t75 = t73 - t74;
        const auto t76 = t22 + std::pow(t61, 2) + std::pow(t75, 2);
        const auto t77 = t7 * t76;
        const auto t78 = t69 * t76;
        const auto t79 = t0 * t78;
        const auto t80 = t62 * t70 + t71 * t77 + t79;
        const auto t81 = std::pow(t7, -2);
        const auto t82 = std::pow(t47, -2);
        const auto t83 = std::pow(t27, -2);
        const auto t84 = t61 * t68;
        const auto t85 = t21 * t41;
        const auto t86 = t45 * t61;
        const auto t87 = t41 * t75;
        const auto t88 = t86 - t87;
        const auto t89 = -t21 * t45;
        const auto t90 = t68 * t75 + t89;
        const auto t91 = t0 * (t84 - t85) + t2 * t88 - t4 * t90;
        const auto t92 = t83 * t91;
        const auto t93 = t81 * t82 * t92;
        const auto t94 = std::pow(t49, 2);
        const auto t95 = std::pow(t52, 2);
        const auto t96 = std::pow(t51, 2);
        const auto t97 = std::pow(t50, 2);
        const auto t98 = t62 * t7;
        const auto t99 = 4 * t0;
        const auto t100 = t62 * t69;
        const auto t101 = t71 * t76;
        const auto t102 = 1.0 / t47;
        const auto t103 = 1.0 / t27;
        const auto t104 = 1.0 / t7;
        const auto t105 = t103 * t104;
        const auto t106 = t102 * t105;
        const auto t107 = t106 * t91;
        const auto t108 = t45 * t49;
        const auto t109 = t51 * t75;
        const auto t110 = t108 - t109;
        const auto t111 = t45 * t52;
        const auto t112 = t50 * t75;
        const auto t113 = t111 - t112;
        const auto t114 = -t49;
        const auto t115 = -t68;
        const auto t116 = -t50;
        const auto t117 = -t61;
        const auto t118 = t55 * t72;
        const auto t119 = t58 * t66;
        const auto t120 = t118 - t119;
        const auto t121 = -t37;
        const auto t122 = t121 * t55;
        const auto t123 = t59 * t65;
        const auto t124 = t122 - t123;
        const auto t125 = -t84 + t85;
        const auto t126 =
            -t0 * (t114 * t115 + t116 * t117 - t120 * t51 - t124 * t52)
            - t110 * t2 - t113 * t4 + t125;
        const auto t127 = t106 * t80;
        const auto t128 = 2 * t71;
        const auto t129 = t105 * t82 * t91;
        const auto t130 = t128 * t129;
        const auto t131 = t102 * t80;
        const auto t132 = t104 * t92;
        const auto t133 = t132 * t62;
        const auto t134 = 2 * t133;
        const auto t135 = t54 * t81;
        const auto t136 = t103 * t91;
        const auto t137 = t135 * t136;
        const auto t138 = t41 * t52;
        const auto t139 = -2 * t138 + 2 * t15 * t50;
        const auto t140 = -2 * t21 * t51 + 2 * t35 * t49;
        const auto t141 = t2 * t78;
        const auto t142 = t11 + t1_x;
        const auto t143 = -t142 * t61 + t52 * t75;
        const auto t144 = t1_x + t31;
        const auto t145 = -t144 * t41 + t45 * t50;
        const auto t146 = t141 + t143 * t70 + t145 * t77;
        const auto t147 = t102 * t146;
        const auto t148 = t144 * t21;
        const auto t149 = t142 * t50 - t144 * t52;
        const auto t150 = t142 * t68;
        const auto t151 = -t148 + t150;
        const auto t152 = t21 * t50;
        const auto t153 = -t152;
        const auto t154 = t153 + t52 * t68;
        const auto t155 = -t144;
        const auto t156 = -t75;
        const auto t157 = -t52;
        const auto t158 = t59 * t63;
        const auto t159 = t121 * t66;
        const auto t160 = t158 - t159;
        const auto t161 = -t86 + t87;
        const auto t162 = t0 * t151 + t154 * t4 + t161
            - t2 * (-t117 * t50 + t124 * t157 - t142 * t160 + t155 * t156);
        const auto t163 = t106 * t126;
        const auto t164 = 2 * t2;
        const auto t165 = t100 * t164;
        const auto t166 = t101 * t164;
        const auto t167 = 2 * t98;
        const auto t168 = t143 * t7;
        const auto t169 = t142 * t70;
        const auto t170 = t144 * t77;
        const auto t171 = t54 * t69;
        const auto t172 = t54 * t76;
        const auto t173 = t143 * t171 + t145 * t172;
        const auto t174 = t80 * t93;
        const auto t175 = -t0 * t149
            - t107
                * (t128 * t168 + t145 * t167 + t165 + t166 - t169 * t49
                   - t170 * t51 + t173)
            + t110 + t127 * t162 + t146 * t163 + t146 * t174 + t148 - t150
            + t2 * t53;
        const auto t176 = t142 * t21 + t49 * t75;
        const auto t177 = t144 * t68 + t45 * t51;
        const auto t178 = -t4 * t78;
        const auto t179 = -t176 * t70 - t177 * t77 - t178;
        const auto t180 = 2 * t179;
        const auto t181 = t102 * t180;
        const auto t182 = t102 * t137;
        const auto t183 = -t51;
        const auto t184 = t142 * t51 - t144 * t49;
        const auto t185 = t142 * t41;
        const auto t186 = -t144 * t61 + t185;
        const auto t187 = t41 * t49;
        const auto t188 = t187 - t51 * t61;
        const auto t189 = -t142;
        const auto t190 = -t0 * t186 - t188 * t2
            - t4 * (-t115 * t49 + t120 * t183 - t144 * t156 + t160 * t189)
            + t90;
        const auto t191 = t128 * t7;
        const auto t192 = 2 * t4;
        const auto t193 = -t100 * t192 - t101 * t192;
        const auto t194 = t171 * t176 + t172 * t177;
        const auto t195 = t0 * t184
            - t107
                * (-t167 * t177 - t169 * t52 - t170 * t50 - t176 * t191 - t193
                   - t194)
            + t113 + t127 * t190 + t163 * t179 + t174 * t179 + t186
            + t4 * (-t116 * t49 + t183 * t52);
        const auto t196 = t17 * t21 + t61 * t9;
        const auto t197 = t29 * t68 + t37 * t41;
        const auto t198 = t196 * t70 + t197 * t77 + t79;
        const auto t199 = t182 * t198;
        const auto t200 = t45 * t9;
        const auto t201 = t37 * t75;
        const auto t202 = t200 - t201;
        const auto t203 = t17 * t45;
        const auto t204 = t203 - t29 * t75;
        const auto t205 = t21 * t37;
        const auto t206 = -t205;
        const auto t207 = t17 * t41;
        const auto t208 = -t207;
        const auto t209 = t206 + t208;
        const auto t210 = -t0 * (t15 * t29 + t209 + t35 * t9) - t15 * t35
            - t2 * t202 - t204 * t4 + t85;
        const auto t211 = t49 * t9;
        const auto t212 = t17 * t52;
        const auto t213 = t37 * t51;
        const auto t214 = t29 * t50;
        const auto t215 = t102 * t134;
        const auto t216 = t29 * t49;
        const auto t217 = t17 * t51;
        const auto t218 = t216 - t217;
        const auto t219 = t50 * t9;
        const auto t220 = t37 * t52;
        const auto t221 = t219 - t220;
        const auto t222 = t68 * t9;
        const auto t223 = t29 * t61;
        const auto t224 = t209 + t222 + t223;
        const auto t225 = -t21 * t51 + t49 * t68;
        const auto t226 = t50 * t61;
        const auto t227 = t226 - t41 * t52;
        const auto t228 = t0 * (t218 + t221) + t224 + t225 + t227;
        const auto t229 = -t116 * t17 + t52 * t63;
        const auto t230 = t32 * t52;
        const auto t231 = t12 * t50;
        const auto t232 = -t30 + t33;
        const auto t233 = t21 + t232;
        const auto t234 = t0 * (t230 - t231 + t233);
        const auto t235 = -t42 + t43;
        const auto t236 = t235 + t26;
        const auto t237 = t2 * (t218 + t236);
        const auto t238 = t12 * t61 - t17 * t75;
        const auto t239 = -t29 * t45 + t32 * t41;
        const auto t240 = -t141 + t238 * t70 + t239 * t77;
        const auto t241 = t32 * t51;
        const auto t242 = t12 * t68;
        const auto t243 = t21 * t32;
        const auto t244 = t242 - t243;
        const auto t245 = t17 * t68;
        const auto t246 = t21 * t29;
        const auto t247 = t245 - t246;
        const auto t248 = t12 * t45;
        const auto t249 = t32 * t75;
        const auto t250 = t248 - t249;
        const auto t251 = t207 - t223 + t250;
        const auto t252 =
            t0 * t244 - t15 * t45 + t2 * t251 + t247 * t4 + t26 * t41;
        const auto t253 = t32 * t49;
        const auto t254 = t12 * t51;
        const auto t255 = -t254;
        const auto t256 = -t38 + t39;
        const auto t257 = t15 + t256;
        const auto t258 = -t37 * t49 + t51 * t9;
        const auto t259 = t12 * t21 + t75 * t9;
        const auto t260 = t32 * t68 + t37 * t45;
        const auto t261 = t178 + t259 * t70 + t260 * t77;
        const auto t262 = t12 * t52;
        const auto t263 = t12 * t41;
        const auto t264 = t32 * t61;
        const auto t265 = t263 - t264;
        const auto t266 = t41 * t9;
        const auto t267 = t37 * t61;
        const auto t268 = t205 - t222 + t250;
        const auto t269 =
            -t0 * t265 + t2 * (-t266 + t267) + t26 * t35 + t268 * t4 + t89;
        const auto t270 = -t111 - t263 - t4 * (t221 + t236);
        const auto t271 = std::pow(t7 * t78, -1.0 / 2.0);
        const auto t272 = t4 * t51;
        const auto t273 = -t272;
        const auto t274 = t2 * t49;
        const auto t275 = t4 * t52;
        const auto t276 = t274 + t275;
        const auto t277 = t0 * (t115 * t117 - t120 * t124) + t2 * t88
            + t4 * (-t115 * t156 + t120 * t160);
        const auto t278 = 1.0 / t76;
        const auto t279 = t277 * t278;
        const auto t280 = t2 * t61;
        const auto t281 = t21 * t4;
        const auto t282 = t280 + t281;
        const auto t283 = -t126;
        const auto t284 = t278 * t283;
        const auto t285 = 2 * t282;
        const auto t286 = t114 * t117 + t120 * t52;
        const auto t287 = std::pow(t76, -2);
        const auto t288 = t277 * t287;
        const auto t289 = t286 * t288;
        const auto t290 = t3 * t45 + t45 * t5;
        const auto t291 = t0 * (t115 * t59 - t124 * t4) + t290;
        const auto t292 = 1.0 / t69;
        const auto t293 = t291 * t292;
        const auto t294 =
            std::pow(t115, 2) + std::pow(t124, 2) + std::pow(t160, 2);
        const auto t295 = std::pow(t55, 2);
        const auto t296 = std::pow(t59, 2);
        const auto t297 = std::pow(t66, 2);
        const auto t298 = t295 + t296 + t297;
        const auto t299 = t294 * t298;
        const auto t300 = t115 * t116 + t124 * t51;
        const auto t301 =
            std::pow(t117, 2) + std::pow(t120, 2) + std::pow(t156, 2);
        const auto t302 = t298 * t301;
        const auto t303 = t294 * t301;
        const auto t304 = t0 * t303 + t286 * t299 + t300 * t302;
        const auto t305 = t104 * t304;
        const auto t306 = t278 * t305;
        const auto t307 = t277 * t292;
        const auto t308 = t287 * t307;
        const auto t309 = t305 * t308;
        const auto t310 = t2 * t68;
        const auto t311 = t4 * t41;
        const auto t312 = -t311;
        const auto t313 = t310 + t312;
        const auto t314 = t0 * t68;
        const auto t315 = t0 * t61;
        const auto t316 = t4 * t75;
        const auto t317 = -t316;
        const auto t318 = t315 + t317;
        const auto t319 = -t318;
        const auto t320 = t0 * t49;
        const auto t321 = -t15 - t320;
        const auto t322 = t0 * t45;
        const auto t323 = t311 + t322;
        const auto t324 = t1 * t68 + t2 * t323 + t5 * t68;
        const auto t325 = t5 * t50;
        const auto t326 = -t158 + t159;
        const auto t327 = -t2 * (t326 + t51 * t66) + t325;
        const auto t328 = t2 * t50;
        const auto t329 = t3 * t51;
        const auto t330 = -t329;
        const auto t331 = t0 * t41;
        const auto t332 = 2 * t331;
        const auto t333 = t0 * t52;
        const auto t334 = t0 * t21;
        const auto t335 = t2 * t75;
        const auto t336 = t334 + t335;
        const auto t337 = 2 * t336;
        const auto t338 = t1 * t41 + t3 * t41;
        const auto t339 = t338 + t4 * (-t115 * t2 + t160 * t55);
        const auto t340 = t292 * t339;
        const auto t341 = -t278 * t283 * t336 + t306 * t340 + t309 * t336;
        const auto t342 = t2 * t52;
        const auto t343 = t4 * t49;
        const auto t344 = t0 * (t342 - t343);
        const auto t345 = t2 * t51;
        const auto t346 = t4 * t50;
        const auto t347 = t345 + t346;
        const auto t348 = t2 * t41;
        const auto t349 = t4 * t68;
        const auto t350 = t348 + t349;
        const auto t351 = t283 * t292;
        const auto t352 = std::pow(t69, -2);
        const auto t353 = t2 * t21;
        const auto t354 = t4 * t61;
        const auto t355 = t353 - t354;
        const auto t356 = t0 * t355 + t3 * t75 + t5 * t75;
        const auto t357 = t5 * t52;
        const auto t358 = 2 * t334;
        const auto t359 = t0 * t51;
        const auto t360 = t331 - t4 * t45;
        const auto t361 = -t360;
        const auto t362 = t351 * t361;
        const auto t363 = 2 * t361;
        const auto t364 = t277 * t352;
        const auto t365 = t300 * t364;
        const auto t366 = t1 * t21 + t21 * t5;
        const auto t367 = t2 * (-t117 * t4 + t156 * t55) + t366;
        const auto t368 = t278 * t367;
        const auto t369 = t292 * t305;
        const auto t370 = t368 * t369;
        const auto t371 = t279 * t361;
        const auto t372 = t305 * t352;
        const auto t373 = t0 * t50;
        const auto t374 = -t35 - t373;
        const auto t375 = t2 * t45;
        const auto t376 = t314 + t375;
        const auto t377 = 2 * t376;
        const auto t378 = t0 * t75;
        const auto t379 = -t353;
        const auto t380 = t378 + t379;
        const auto t381 = t1 * t61 + t3 * t61 - t380 * t4;
        const auto t382 = t279 * t372;
        const auto t383 = t3 * t49;
        const auto t384 = t383 - t4 * (t52 * t59 + t75);
        const auto t385 = 2 * t145;
        const auto t386 = t129 * t385;
        const auto t387 = t132 * t143;
        const auto t388 = 2 * t387;
        const auto t389 = t136 * t81;
        const auto t390 = t164 * t389;
        const auto t391 = -t149;
        const auto t392 = std::pow(t142, 2);
        const auto t393 = std::pow(t144, 2);
        const auto t394 = 4 * t2;
        const auto t395 = t143 * t69;
        const auto t396 = t145 * t76;
        const auto t397 = t106 * t146;
        const auto t398 = t142 * t45;
        const auto t399 = 2 * t144 * t26 - 2 * t398;
        const auto t400 = t102 * t390;
        const auto t401 = t35 * t52;
        const auto t402 = t106 * t162;
        const auto t403 = t176 * t7;
        const auto t404 = 2 * t168;
        const auto t405 = -t192 * t395 - t192 * t396;
        const auto t406 = t164 * t69;
        const auto t407 = t164 * t76;
        const auto t408 = t176 * t406 + t177 * t407;
        const auto t409 = t146 * t93;
        const auto t410 = -t107
                * (-t177 * t404 - t385 * t403 - t405 - t408 - t49 * t52 * t70
                   - t50 * t51 * t77)
            - t15 * t51 + t152 + t179 * t402 + t179 * t409 + t184 * t2 + t187
            + t190 * t397 + t391 * t4 - t401;
        const auto t411 = t142 * t29;
        const auto t412 = t144 * t17;
        const auto t413 = t411 - t412;
        const auto t414 = -t219;
        const auto t415 = t102 * t388;
        const auto t416 = t142 * t9;
        const auto t417 = -t10 + t13;
        const auto t418 = t385 * t7;
        const auto t419 = t240 * t400;
        const auto t420 = -t230 + t231;
        const auto t421 = t2 * (-t413 - t420);
        const auto t422 = t144 * t75;
        const auto t423 = t12 * t142;
        const auto t424 = t144 * t32;
        const auto t425 = -t266;
        const auto t426 = t144 * t9;
        const auto t427 = t142 * t37;
        const auto t428 = -t427;
        const auto t429 = t37 * t50;
        const auto t430 = 2 * t375;
        const auto t431 = -t162;
        const auto t432 = t278 * t431;
        const auto t433 = t142 * t2;
        const auto t434 = -t417 - t433;
        const auto t435 = t117 * t142 + t156 * t157;
        const auto t436 = t288 * t435;
        const auto t437 = t124 * t155 + t160 * t50;
        const auto t438 = t2 * t303 + t299 * t435 + t302 * t437;
        const auto t439 = t104 * t438;
        const auto t440 = t278 * t439;
        const auto t441 = t308 * t439;
        const auto t442 = t144 * t4;
        const auto t443 = -t0 * (-t35 - t442) + t325;
        const auto t444 = t0 * t142;
        const auto t445 = t275 + t444;
        const auto t446 = t2 * (t373 - t442) + t323;
        const auto t447 = t1 * t144;
        const auto t448 = -t26 - t342;
        const auto t449 = -t118 + t119;
        const auto t450 = t144 * t2;
        const auto t451 = -t256 - t450;
        const auto t452 = t292 * t431;
        const auto t453 = t142 * t4;
        const auto t454 = -t453;
        const auto t455 = t0 * t144;
        const auto t456 = t346 + t455;
        const auto t457 = t364 * t437;
        const auto t458 = t292 * t439;
        const auto t459 = t352 * t439;
        const auto t460 = t354 + t378;
        const auto t461 = t1 * t142;
        const auto t462 = 2 * t280;
        const auto t463 = -t18 + t19;
        const auto t464 = -t328 - t45;
        const auto t465 = t279 * t459;
        const auto t466 = t376 * t465;
        const auto t467 = 2 * t177;
        const auto t468 = t129 * t467;
        const auto t469 = t132 * t176;
        const auto t470 = 2 * t469;
        const auto t471 = t192 * t389;
        const auto t472 = 4 * t4;
        const auto t473 = t106 * t190;
        const auto t474 = t102 * t471;
        const auto t475 = -t426 + t427;
        const auto t476 = t0 * (t257 + t475);
        const auto t477 = -t216;
        const auto t478 = t106 * t179;
        const auto t479 = t144 * t29;
        const auto t480 = t192 * t69;
        const auto t481 = t192 * t76;
        const auto t482 = t196 * t7;
        const auto t483 = 2 * t403;
        const auto t484 = t17 * t49;
        const auto t485 = -t23 + t24;
        const auto t486 = t467 * t7;
        const auto t487 = t102 * t470;
        const auto t488 = -t253 + t254;
        const auto t489 = -t0 * (t144 * t58 - t189 * t32) + t188
            + t2 * (t257 + t488) - t245 + t246 + t4 * (t233 - t411 + t412);
        const auto t490 = t261 * t474;
        const auto t491 = -t142 * t45 - t248 + t249 + t422;
        const auto t492 = t206 + t222 + t225 + t4 * (t475 + t488) + t491;
        const auto t493 = -t122 + t123;
        const auto t494 = -t453 - t463;
        const auto t495 = -t190;
        const auto t496 = t278 * t495;
        const auto t497 = t120 * t189 + t156 * t49;
        const auto t498 = t115 * t144 + t160 * t183;
        const auto t499 = t299 * t497 + t302 * t498 + t303 * t4;
        const auto t500 = t104 * t499;
        const auto t501 = t278 * t500;
        const auto t502 = -2 * t349;
        const auto t503 = -t343 - t485;
        const auto t504 = 2 * t319;
        const auto t505 = t288 * t497;
        const auto t506 = t292 * t324;
        const auto t507 = t308 * t500;
        const auto t508 = -t319 * t496 + t319 * t507 + t501 * t506;
        const auto t509 = -t450;
        const auto t510 = t274 + t444;
        const auto t511 = -t310;
        const auto t512 = t322 + t511;
        const auto t513 = -2 * t316;
        const auto t514 = t292 * t495;
        const auto t515 = -t232 - t442;
        const auto t516 = 2 * t350;
        const auto t517 = t364 * t498;
        const auto t518 = t278 * t356;
        const auto t519 = t292 * t500;
        const auto t520 = t279 * t352 * t500;
        const auto t521 = -t0 * (-t15 - t433) + t383;
        const auto t522 = -t235 - t272;
        const auto t523 = t345 + t455;
        const auto t524 = t353 - t378 + t4 * (t320 - t433);
        const auto t525 = t115 * t29 + t121 * t124;
        const auto t526 = 2 * t298;
        const auto t527 = t525 * t526;
        const auto t528 = t117 * t9 + t120 * t72;
        const auto t529 = t526 * t528;
        const auto t530 = 2 * t55;
        const auto t531 = t294 * t530;
        const auto t532 = t301 * t530;
        const auto t533 = t301 * t525;
        const auto t534 = t294 * t528;
        const auto t535 = -t303;
        const auto t536 = t104 * t292;
        const auto t537 = t279 * t536;
        const auto t538 = -t0 * t224 + t125 + t2 * (t121 * t156 - t160 * t9)
            + t4 * (-t156 * t29 + t160 * t72);
        const auto t539 = t278 * t538;
        const auto t540 = t299 * t528 + t302 * t525 + t303 * t55;
        const auto t541 = t536 * t540;
        const auto t542 = t304 * t81;
        const auto t543 = t279 * t292;
        const auto t544 = t542 * t543;
        const auto t545 = -t64 + t67;
        const auto t546 = t120 + t545;
        const auto t547 = t326 + t75;
        const auto t548 = -t57 + t60;
        const auto t549 = t352 * t540;
        const auto t550 = t288 * t549 * t81;
        const auto t551 = t135 * t543;
        const auto t552 = 2 * t528;
        const auto t553 = 2 * t525;
        const auto t554 = t17 * t37 - t29 * t9;
        const auto t555 = 2 * t205;
        const auto t556 = 2 * t207;
        const auto t557 = 2 * t106;
        const auto t558 = 2 * t198;
        const auto t559 = t102 * t132;
        const auto t560 = std::pow(t9, 2);
        const auto t561 = std::pow(t17, 2);
        const auto t562 = std::pow(t37, 2);
        const auto t563 = std::pow(t29, 2);
        const auto t564 = t117 * t58 + t156 * t17;
        const auto t565 = t124 * t32 + t160 * t63;
        const auto t566 = t299 * t564 + t302 * t565 + t303 * t59;
        const auto t567 = t104 * t308;
        const auto t568 = t566 * t567;
        const auto t569 = t104 * t279 * t352 * t566;
        const auto t570 = 2 * t59;
        const auto t571 = t0 * (t115 * t58 - t120 * t32) + t161 + t2 * t251
            + t4 * (-t115 * t17 + t120 * t63);
        const auto t572 = t278 * t571;
        const auto t573 = t536 * t539;
        const auto t574 = -t242 + t243;
        const auto t575 = t0 * (t29 * t58 - t32 * t72)
            + t2 * (t121 * t17 - t63 * t9) + t202
            + t4 * (-t17 * t29 + t63 * t72)
            - t537
                * (t121 * t302 * t32 + t299 * t58 * t9 + t527 * t564
                   + t529 * t565 + t531 * t564 + t532 * t565 + t533 * t570
                   + t534 * t570)
            - t541 * t572 + t550 * t566 - t566 * t573 + t574;
        const auto t576 = t115 * t65 + t160 * t37;
        const auto t577 = t12 * t120 + t156 * t56;
        const auto t578 = t299 * t577 + t302 * t576 + t303 * t66;
        const auto t579 = t567 * t578;
        const auto t580 = t352 * t578;
        const auto t581 = t104 * t279 * t580;
        const auto t582 = 2 * t66;
        const auto t583 = t0 * (t117 * t65 - t12 * t124)
            + t2 * (-t117 * t37 + t124 * t56) + t268 * t4 + t90;
        const auto t584 = t278 * t583;
        const auto t585 = t0 * (-t12 * t121 + t65 * t9)
            + t2 * (t121 * t56 - t37 * t9) + t204 + t265 - t4 * t554
            - t537
                * (t12 * t299 * t72 + t29 * t302 * t65 + t527 * t577
                   + t529 * t576 + t531 * t577 + t532 * t576 + t533 * t582
                   + t534 * t582)
            - t541 * t584 + t550 * t578 - t573 * t578;
        const auto t586 = t2 * t9;
        const auto t587 = t17 * t4;
        const auto t588 = t586 + t587;
        const auto t589 = t288 * t528;
        const auto t590 = t104 * t278;
        const auto t591 = t540 * t590;
        const auto t592 = t540 * t567;
        const auto t593 = t0 * t115;
        const auto t594 = 2 * t10 + t14;
        const auto t595 = 2 * t18 + t20;
        const auto t596 = t3 * t37 + t37 * t5;
        const auto t597 = -t0 * t26;
        const auto t598 = t2 * t37;
        const auto t599 = t29 * t4;
        const auto t600 = t598 + t599;
        const auto t601 = t292 * t538;
        const auto t602 = t364 * t525;
        const auto t603 = t104 * t279 * t549;
        const auto t604 = 2 * t38 + t40;
        const auto t605 = t17 * t3 + t17 * t5;
        const auto t606 = t0 * t117;
        const auto t607 = 2 * t30 + t34;
        const auto t608 = t526 * t564;
        const auto t609 = t526 * t565;
        const auto t610 = t294 * t564;
        const auto t611 = t301 * t565;
        const auto t612 = t294 * t570;
        const auto t613 = t301 * t570;
        const auto t614 = t536 * t566;
        const auto t615 = t543 * t81;
        const auto t616 = t164 * t615;
        const auto t617 = -t73 + t74;
        const auto t618 = t12 * t29 - t17 * t32;
        const auto t619 = std::pow(t12, 2);
        const auto t620 = std::pow(t32, 2);
        const auto t621 = 4 * t7;
        const auto t622 = 2 * t248 - 2 * t32 * t75;
        const auto t623 = -t12 * t37 + t32 * t9;
        const auto t624 = t536 * t578;
        const auto t625 = t267 + t425;
        const auto t626 = -t0 * (-t12 * t32 + t58 * t65) + t2 * t623 + t247
            - t277 * t287 * t352 * t566 * t578 * t81 + t4 * t618
            + t537
                * (t17 * t299 * t56 + t302 * t37 * t63 + t576 * t608
                   + t576 * t613 + t577 * t609 + t577 * t612 + t582 * t610
                   + t582 * t611)
            + t572 * t624 + t584 * t614 + t625;
        const auto t627 = t29 * t3;
        const auto t628 = -t10 + 2 * t12 * t2;
        const auto t629 = t566 * t590;
        const auto t630 = t29 * t5;
        const auto t631 = t1 * t29 + t630;
        const auto t632 = t0 * t12;
        const auto t633 = t587 + t632;
        const auto t634 = t288 * t564;
        const auto t635 = t124 * t2;
        const auto t636 = 2 * t23 + t25;
        const auto t637 = t295 * t72;
        const auto t638 = t156 * t2;
        const auto t639 = 2 * t2 * t32 - t38;
        const auto t640 = t292 * t571;
        const auto t641 = t350 * t640;
        const auto t642 = t0 * t32;
        const auto t643 = t599 + t642;
        const auto t644 = -t2 * (t17 * t55 - t4 * t58) + t460;
        const auto t645 = t12 * t3;
        const auto t646 = 2 * t42 + t44;
        const auto t647 = t1 * t12;
        const auto t648 = t12 * t5 + t647;
        const auto t649 = t493 + t61;
        const auto t650 = t526 * t576;
        const auto t651 = t526 * t577;
        const auto t652 = t301 * t576;
        const auto t653 = t294 * t577;
        const auto t654 = t294 * t582;
        const auto t655 = t301 * t582;
        const auto t656 = t288 * t580;
        const auto t657 = 2 * t577;
        const auto t658 = 2 * t576;
        const auto t659 = t192 * t615;
        const auto t660 = 2 * t12 * t4 - t18;
        const auto t661 = t160 * t4;
        const auto t662 = t578 * t590;
        const auto t663 = -t104 * t277 * t282 * t287 * t292 * t578 + t121 * t295
            + t282 * t584 + t293 * t662 - t296 * t37 + t661;
        const auto t664 = t3 * t32;
        const auto t665 = -t23 + 2 * t4 * t9;
        const auto t666 = t1 * t32;
        const auto t667 = t32 * t5 + t666;
        const auto t668 = t586 + t632;
        const auto t669 = t1 * t9;
        const auto t670 = t292 * t583;
        const auto t671 = -t30 + 2 * t32 * t4;
        const auto t672 = t3 * t9;
        const auto t673 = t5 * t9 + t672;
        const auto t674 = t120 * t4;
        const auto t675 = 2 * t37 * t4 - t42;
        const auto t676 = t598 + t642;
        const auto t677 = -t15 * t4 + t380;
        const auto t678 = t2 * t35;
        const auto t679 = t15 * t2 + t281;
        const auto t680 = t103 * t679;
        const auto t681 = t102 * (t0 * t313 + t290);
        const auto t682 = t105 * t681;
        const auto t683 = t285 * t7;
        const auto t684 = t171 * t282;
        const auto t685 = t132 * t679;
        const auto t686 = t132 * t285;
        const auto t687 = t282 * t406;
        const auto t688 = t102 * t685;
        const auto t689 = t285 * t559;
        const auto t690 = -t678;
        const auto t691 = t117 * t59;
        const auto t692 = t674 + t691;
        const auto t693 = -t277 * t6;
        const auto t694 = t279 * t692;
        const auto t695 = t271 * t278;
        const auto t696 = t0 * t277;
        const auto t697 = t2 * t696;
        const auto t698 = t279 * t282;
        const auto t699 = t282 * t324 - t291 * t319 + t319 * t698 + t697;
        const auto t700 = t282 * t339;
        const auto t701 = -t0 * t277 * t4;
        const auto t702 = t336 * t698;
        const auto t703 = t348 + t35 * t4;
        const auto t704 = t102 * t703;
        const auto t705 = t680 * t91;
        const auto t706 = t48 * (t356 * t680 - t681 * t703 + t704 * t705);
        const auto t707 = t2 * t460 + t366;
        const auto t708 = t102 * t360;
        const auto t709 =
            t48 * (t360 * t681 + t4 * t7 - t680 * t707 - t705 * t708);
        const auto t710 = t0 * t55;
        const auto t711 = t4 * t66;
        const auto t712 = t271
            * (t278 * t282 * t381 + t291 * t292 * t376 - t292 * t376 * t698
               - t59 * (-t296 + t710 + t711));
        const auto t713 = t103 * t318;
        const auto t714 = t504 * t7;
        const auto t715 = t171 * t319;
        const auto t716 = t132 * t318;
        const auto t717 = t319 * t406;
        const auto t718 = t102 * t716;
        const auto t719 = t115 * t66;
        const auto t720 = t156 * t66 + t606;
        const auto t721 = t102 * t324;
        const auto t722 = t105 * t721;
        const auto t723 = t504 * t559;
        const auto t724 = t279 * t720;
        const auto t725 = -t277 * (t1 + t5);
        const auto t726 = t279 * t319;
        const auto t727 =
            -t2 * t277 * t4 + t319 * t339 + t324 * t336 + t336 * t726;
        const auto t728 = t2 * t59;
        const auto t729 = t271
            * (t292 * t350 * t726 + t319 * t518 + t350 * t506
               - t66 * (-t297 + t710 + t728));
        const auto t730 = t713 * t91;
        const auto t731 = t48 * (-t360 * t721 + t707 * t713 + t708 * t730);
        const auto t732 = t102 * t376;
        const auto t733 =
            t48 * (t0 * t7 - t376 * t721 - t381 * t713 + t730 * t732);
        const auto t734 = t124 * t55;
        const auto t735 = t493 + t51 * t55;
        const auto t736 = t449 + t52 * t55;
        const auto t737 = t120 * t55;
        const auto t738 = t638 + t737;
        const auto t739 = -t348;
        const auto t740 = t103 * t336;
        const auto t741 = t338 - t4 * t512;
        const auto t742 = t102 * t741;
        const auto t743 = t105 * t742;
        const auto t744 = t336 * t406;
        const auto t745 = t337 * t7;
        const auto t746 = 3 * t336;
        const auto t747 = -t336 * t480;
        const auto t748 = t559 * t746;
        const auto t749 = t4 * t696;
        const auto t750 = t279 * t738;
        const auto t751 = t277 * (t1 + t3);
        const auto t752 = t740 * t91;
        const auto t753 =
            t48 * (t2 * t7 - t356 * t740 - t703 * t742 - t704 * t752);
        const auto t754 = t271
            * (t278 * t336 * t367 - t292 * t336 * t371 - t340 * t361
               - t55 * (-t295 + t711 + t728));
        const auto t755 = t48 * (t376 * t742 - t381 * t740 + t732 * t752);
        const auto t756 = t172 * t350;
        const auto t757 = t129 * t703;
        const auto t758 = t129 * t516;
        const auto t759 = t354 + t379;
        const auto t760 = t350 * t481;
        const auto t761 = t106 * t356;
        const auto t762 = t635 + t719;
        const auto t763 = t259 * t7;
        const auto t764 = t307 * t762;
        const auto t765 = t271 * t292;
        const auto t766 = t307 * t350;
        const auto t767 = -t350 * t367 + t356 * t361 + t361 * t766 + t697;
        const auto t768 = t356 * t376;
        const auto t769 = t376 * t766;
        const auto t770 = t661 + t734;
        const auto t771 = t361 * t407;
        const auto t772 = t129 * t363;
        const auto t773 = t129 * t360;
        const auto t774 = -t281;
        const auto t775 = t361 * t481;
        const auto t776 = t106 * t707;
        const auto t777 = t307 * t770;
        const auto t778 =
            t2 * t277 * t4 - t307 * t361 * t376 + t361 * t381 + t367 * t376;
        const auto t779 = t172 * t376;
        const auto t780 = 3 * t129 * t376;
        const auto t781 = t160 * t59 + t593;
        const auto t782 = -t376 * t481;
        const auto t783 = t307 * t781;
        dA[0] = t48
            * (-t107
                   * (t100 * t99 + t101 * t99 + t70 * (t94 + t95)
                      + 4 * t71 * t98 + t77 * (t96 + t97) + t78)
               + 2 * t126 * t127 + t130 * t80 + t131 * t134 + t131 * t137 + t139
               + t140 + t53 * t54 + std::pow(t80, 2) * t93);
        dA[1] = t48 * (t130 * t146 + t134 * t147 + t137 * t147 + t175);
        dA[2] = t48 * (t130 * t179 + t133 * t181 + t179 * t182 + t195);
        dA[3] = t48
            * (t102 * t103 * t104 * t91
                   * (t100 * t54 + t101 * t54 + t167 * t197 + t171 * t196
                      + t172 * t197 + t191 * t196 + t70 * (t211 + t212)
                      + t77 * (t213 + t214) + t78)
               - t127 * t210 - t130 * t198 - t163 * t198 - t174 * t198
               - t198 * t215 - t199 - t228);
        dA[4] = t48
            * (-t107
                   * (-t165 - t166 + t167 * t239 + t171 * t238 + t172 * t239
                      + t191 * t238 + t70 * (t12 * t49 + t15)
                      + t77 * (t241 + t41))
               - t108 + t109 - t127 * t252 + t130 * t240 + t163 * t240
               + t174 * t240 + t182 * t240 + t215 * t240 + t229 * t4 - t234
               - t237 + t244);
        dA[5] = t48
            * (t0 * (t253 + t255 + t257)
               - t107
                   * (t167 * t260 + t171 * t259 + t172 * t260 + t191 * t259
                      + t193 + t70 * (t21 + t262) + t77 * (t32 * t50 + t35))
               - t127 * t269 + t130 * t261 + t15 * t32 + t163 * t261
               + t174 * t261 + t182 * t261 - t2 * t258 + t215 * t261 + t26 * t50
               + t270);
        dA[6] = t271
            * (t0 * (t116 * t59 + t273) - t276 * t279 - t282 * t284
               + t282 * t309 + t285 * t289 - t293 * t306 + t313);
        dA[7] = t271
            * (-t1 * t50 + t104 * t277 * t287 * t292 * t304 * t319
               + t104 * t278 * t292 * t304 * t324
               + 2 * t277 * t286 * t287 * t319 - t279 * t321 - t284 * t319
               - 2 * t314 - t327);
        dA[8] = t271
            * (t1 * t51 - t279 * (-t21 - t333) - t289 * t337 - t330 + t332
               - t341 - t4 * (-t235 - t328));
        dA[9] = t271
            * (t104 * t277 * t278 * t304 * t350 * t352
               + t104 * t278 * t292 * t304 * t356
               + 2 * t277 * t300 * t350 * t352 - t307 * t347 - t344
               - t350 * t351 - t355);
        dA[10] = t271
            * (t1 * t52 - t2 * (-t26 - t343) - t307 * (-t359 - t41) + t357
               + t358 - t362 + t363 * t365 - t370 + t371 * t372);
        dA[11] = t271
            * (-t1 * t49 + t104 * t278 * t292 * t304 * t381 + t283 * t292 * t376
               - t307 * t374 - 2 * t315 - t365 * t377 - t376 * t382 - t384);
        dA[12] = t48 * (t131 * t388 + t131 * t390 + t175 + t386 * t80);
        dA[13] = t48
            * (-t107
                   * (4 * t145 * t168 + t394 * t395 + t394 * t396
                      + t70 * (t392 + t95) + t77 * (t393 + t97) + t78)
               + t139 + std::pow(t146, 2) * t93 + t146 * t386 + t147 * t388
               + t147 * t390 + 2 * t162 * t397 + t164 * t391 + t399);
        dA[14] = t48 * (t179 * t386 + t179 * t400 + t181 * t387 + t410);
        dA[15] = t48
            * (t0 * (t233 + t413)
               - t107
                   * (-t173 - t196 * t406 - t196 * t418 - t197 * t404
                      - t197 * t407 + t69 * t7 * (t416 + t417)
                      + t7 * t76 * (t144 * t37 + t256))
               + t142 * t35 - t148 - t198 * t386 - t198 * t400 - t198 * t402
               - t198 * t409 - t198 * t415 + t2 * (t220 + t236 + t414) - t200
               - t210 * t397 + t26 * t37 + t4 * (-t17 * t50 + t29 * t52));
        dA[16] = t48
            * (-t107
                   * (2 * t143 * t239 * t7 + 2 * t145 * t238 * t7 - t164 * t395
                      - t164 * t396 + 2 * t2 * t238 * t69 + 2 * t2 * t239 * t76
                      - t70 * (t212 + t423) - t77 * (t214 + t424) - t78)
               + t138 - t226 + t240 * t386 + t240 * t402 + t240 * t409
               + t240 * t415 + t251 - t252 * t397 + t398 + t419 - t421 - t422);
        dA[17] = t48
            * (t0 * (t12 * t144 - t142 * t32)
               - t107
                   * (t259 * t406 + t259 * t418 + t260 * t404 + t260 * t407
                      + t405 + t70 * (t26 + t52 * t9) + t77 * (t429 + t45))
               + t15 * t37 + t153 + t2 * (t257 + t426 + t428) + t261 * t386
               + t261 * t400 + t261 * t402 + t261 * t409 + t261 * t415
               - t269 * t397 + t4 * (t233 + t420) + t401 + t425);
        dA[18] = t271
            * (-t279 * t434 - t282 * t432 + t282 * t441 + t285 * t436
               - t293 * t440 + t3 * t50 + t430 + t443);
        dA[19] = t271
            * (t104 * t277 * t287 * t292 * t319 * t438
               + t104 * t278 * t292 * t324 * t438
               + 2 * t277 * t287 * t319 * t435 - t279 * t445 - t319 * t432
               - t446);
        dA[20] = t271
            * (-t144 * t3 + 2 * t2 * t41 + t278 * t336 * t431 - t279 * t448
               - t336 * t441 - t337 * t436 - t340 * t440
               + t4 * (t50 * t55 + t68) - t447);
        dA[21] = t271
            * (t0 * (t142 * t66 + t449)
               + t104 * t277 * t278 * t350 * t352 * t438
               + t104 * t278 * t292 * t356 * t438
               + 2 * t277 * t350 * t352 * t437 - t3 * t52 - t307 * t451
               - 2 * t335 - t350 * t452 - t357);
        dA[22] = t271
            * (t2 * (t157 * t55 + t454) - t307 * t456 - t361 * t452
               + t363 * t457 - t368 * t458 + t371 * t459 + t460);
        dA[23] = t271
            * (t104 * t278 * t292 * t381 * t438 + t142 * t3 + t292 * t376 * t431
               - t307 * t464 - t377 * t457 - t4 * (-t333 - t463) + t461 - t462
               - t466);
        dA[24] = t48 * (-t131 * t470 + t131 * t471 + t195 - t468 * t80);
        dA[25] = t48 * (-t146 * t468 - t147 * t470 + t147 * t471 + t410);
        dA[26] = t48
            * (-t107
                   * (-t176 * t472 * t69 + 4 * t177 * t403 - t177 * t472 * t76
                      + t70 * (t392 + t94) + t77 * (t393 + t96) + t78)
               + t140 + std::pow(t179, 2) * t93 - t179 * t468 + t179 * t474
               + t180 * t473 - t181 * t469 + t184 * t192 + t399);
        dA[27] = t48
            * (2 * t102 * t104 * t176 * t198 * t83 * t91
               + 2 * t103 * t104 * t177 * t198 * t82 * t91
               - t107
                   * (t194 - t196 * t480 - t197 * t481 + t197 * t483
                      + t467 * t482 + t70 * (t142 * t17 + t463)
                      + t77 * (t232 + t479))
               + t144 * t15 - t179 * t198 * t93 - t185 - t198 * t473
               - t198 * t474 + t2 * t258 - t203 - t210 * t478 + t26 * t29
               + t4 * (t217 + t236 + t477) - t476);
        dA[28] = t48
            * (t102 * t103 * t104 * t190 * t240
               + 2 * t102 * t103 * t240 * t4 * t81 * t91
               - t107
                   * (t238 * t480 - t238 * t486 + t239 * t481 - t239 * t483
                      + t408 + t70 * (t484 + t485) + t77 * (t235 + t29 * t51))
               + t179 * t240 * t81 * t82 * t83 * t91 - t240 * t468 - t240 * t487
               - t252 * t478 - t489);
        dA[29] = t48
            * (t102 * t103 * t104 * t190 * t261
               - t107
                   * (2 * t176 * t4 * t69 + 2 * t177 * t4 * t76
                      + 2 * t259 * t4 * t69 - t259 * t486 + 2 * t260 * t4 * t76
                      - t260 * t483 - t70 * (t211 + t423) - t77 * (t213 + t424)
                      - t78)
               + t179 * t261 * t81 * t82 * t83 * t91 - t261 * t468 - t261 * t487
               - t269 * t478 + t490 - t492);
        dA[30] = t271
            * (t0 * (t144 * t59 + t493)
               + t104 * t277 * t282 * t287 * t292 * t499
               + 2 * t277 * t282 * t287 * t497 - t279 * t494 - t282 * t496
               - t293 * t501 - t329 + 2 * t4 * t45 - t5 * t51);
        dA[31] = t271
            * (t144 * t5 - t2 * (-t256 - t359) - t279 * t503 + t447 + t502
               + t504 * t505 + t508);
        dA[32] = t271
            * (t278 * t336 * t495 - t279 * t510 - t336 * t507 - t337 * t505
               - t340 * t501 + t4 * (t183 * t55 + t509) - t512);
        dA[33] = t271
            * (-t307 * t515 - t350 * t514 + t350 * t520 + t49 * t5 + t513
               + t516 * t517 + t518 * t519 + t521);
        dA[34] = t271
            * (t104 * t277 * t278 * t352 * t361 * t499 - t142 * t5
               + t2 * (t49 * t55 + t61) + 2 * t21 * t4
               + 2 * t277 * t352 * t361 * t498 - t307 * t522 - t361 * t514
               - t368 * t519 - t461);
        dA[35] = t271
            * (t104 * t278 * t292 * t381 * t499 + t292 * t376 * t495
               - t307 * t523 - t376 * t520 - t377 * t517 - t524);
        dA[36] = t271
            * (2 * t104 * t277 * t278 * t304 * t352 * t525
               + 2 * t104 * t277 * t287 * t292 * t304 * t528 - t228
               + t277 * t287 * t304 * t352 * t540 * t81 - t284 * t541
               - t369 * t539
               - t537
                   * (t286 * t527 + t286 * t531 + t299 * (t114 * t9 + t52 * t72)
                      + t300 * t529 + t300 * t532
                      + t302 * (t116 * t29 + t121 * t51) + t533 * t54
                      + t534 * t54 + t535)
               - t54 * t544);
        dA[37] = t271
            * (t0 * (-t155 * t72 + t411 + t546) + t151
               + t2 * (t121 * t157 + t414 + t547) - t200 + t201
               + t4 * (-t157 * t29 + t50 * t72) - t432 * t541 + t438 * t550
               - t438 * t551 + t441 * t552 - t458 * t539 + t465 * t553
               - t537
                   * (t164 * t533 + t164 * t534 + t299 * (t416 + t548)
                      + t302 * (t121 * t155 + t493) + t435 * t527 + t435 * t531
                      + t437 * t529 + t437 * t532));
        dA[38] = t271
            * (2 * t104 * t277 * t278 * t352 * t499 * t525
               + 2 * t104 * t277 * t287 * t292 * t499 * t528 - t186
               + t2 * (t121 * t49 - t183 * t9) - t204
               + t277 * t287 * t352 * t499 * t540 * t81
               + t4 * (t183 * t72 + t477 + t547) - t476 - t496 * t541
               - t499 * t551 - t519 * t539
               - t537
                   * (t192 * t533 + t192 * t534 + t299 * (t189 * t72 + t449)
                      + t302 * (t479 + t545) + t497 * t527 + t497 * t531
                      + t498 * t529 + t498 * t532));
        dA[39] = t48
            * (-t107
                   * (t196 * t69 * t99 + 4 * t197 * t482 + t197 * t76 * t99
                      + t70 * (t560 + t561) + t77 * (t562 + t563) + t78)
               + t129 * t197 * t558 + t196 * t558 * t559
               + std::pow(t198, 2) * t93 + t198 * t210 * t557 + t199 + 2 * t222
               + 2 * t223 - t54 * t554 - t555 - t556);
        dA[40] = t271 * (-t551 * t566 + t552 * t568 + t553 * t569 + t575);
        dA[41] = t271 * (-t551 * t578 + t552 * t579 + t553 * t581 + t585);
        dA[42] = t271
            * (t0 * (-t121 * t4 + t29 * t59) + t279 * t588 - t282 * t539
               + t282 * t592 + t285 * t589 - t293 * t591 + t311 + t511);
        dA[43] = t271
            * (t104 * t277 * t287 * t292 * t319 * t540
               + t104 * t278 * t292 * t324 * t540 + t115 * t55
               + 2 * t277 * t287 * t319 * t528 - t279 * t594 + t29 * t297
               - t296 * t63 - t30 * t55 - t319 * t539 - t593);
        dA[44] = t271
            * (-t1 * t37 + t278 * t336 * t538 - t279 * t595 - t332 - t336 * t592
               - t337 * t589 - t340 * t591 - t596);
        dA[45] = t271
            * (t307 * t600 - t350 * t601 + t350 * t603 + t355 + t516 * t602
               + t518 * t541 - t597);
        dA[46] = t271
            * (-t1 * t17 + t104 * t277 * t278 * t352 * t361 * t540
               + 2 * t277 * t352 * t361 * t525 - t307 * t604 - t358
               - t361 * t601 - t368 * t541 - t605);
        dA[47] = t271
            * (-t10 * t55 + t104 * t278 * t292 * t381 * t540 + t117 * t55
               + t292 * t376 * t538 + t296 * t9 - t297 * t56 - t307 * t607
               - t376 * t603 - t377 * t602 - t606);
        dA[48] = t271
            * (2 * t104 * t277 * t278 * t304 * t352 * t565
               + 2 * t104 * t277 * t287 * t292 * t304 * t564 - t110
               - t164 * t544 + t229 * t4 - t234 - t237
               + t277 * t287 * t304 * t352 * t566 * t81 - t284 * t614
               - t369 * t572
               - t537
                   * (t286 * t609 + t286 * t612 + t299 * (t114 * t58 + t61)
                      + t300 * t608 + t300 * t613 + t302 * (t124 + t241)
                      + t54 * t610 + t54 * t611)
               - t574);
        dA[49] = t271
            * (2 * t104 * t277 * t278 * t352 * t438 * t565
               + 2 * t104 * t277 * t287 * t292 * t438 * t564 - t208 - t223
               - t227 + t277 * t287 * t352 * t438 * t566 * t81 - t421
               - t432 * t614 - t438 * t616 - t458 * t572 - t491
               - t537
                   * (t164 * t610 + t164 * t611
                      + t299 * (t142 * t58 + t157 * t17)
                      + t302 * (t155 * t32 + t50 * t63) + t435 * t609
                      + t435 * t612 + t437 * t608 + t437 * t613 + t535));
        dA[50] = t271
            * (2 * t104 * t277 * t278 * t352 * t499 * t565
               + 2 * t104 * t277 * t287 * t292 * t499 * t564
               + t277 * t287 * t352 * t499 * t566 * t81 - t489 - t496 * t614
               - t499 * t616 - t519 * t572
               - t537
                   * (t192 * t610 + t192 * t611 + t299 * (t484 + t617)
                      + t302 * (t183 * t63 + t326) + t497 * t609 + t497 * t612
                      + t498 * t608 + t498 * t613));
        dA[51] =
            t271 * (-t540 * t616 + 2 * t564 * t592 + 2 * t565 * t603 + t575);
        dA[52] = t48
            * (2 * t102 * t104 * t238 * t240 * t83 * t91
               + 2 * t103 * t104 * t239 * t240 * t82 * t91
               - t107
                   * (t238 * t239 * t621 - t238 * t394 * t69 - t239 * t394 * t76
                      + t70 * (t561 + t619) + t77 * (t563 + t620) + t78)
               - t164 * t618 + std::pow(t240, 2) * t81 * t82 * t83 * t91
               - t240 * t252 * t557 + 2 * t29 * t61 - t419 - t556 - t622);
        dA[53] = t271
            * (2 * t104 * t277 * t278 * t352 * t565 * t578
               + 2 * t104 * t277 * t287 * t292 * t564 * t578 - t578 * t616
               - t626);
        dA[54] = t271
            * (t104 * t277 * t282 * t287 * t292 * t566
               + 2 * t277 * t282 * t287 * t564 - t279 * t628 - t282 * t572
               - t293 * t629 - t430 - t627 - t631);
        dA[55] = t271
            * (t2 * t35 + t279 * t633 + t319 * t568 - t319 * t572 + t323
               + t504 * t634 + t506 * t629);
        dA[56] = t271
            * (t124 * t59 + t278 * t336 * t571 - t279 * t636 + t295 * t32
               - t297 * t65 - t336 * t568 - t337 * t634 - t340 * t629
               - t39 * t59 - t635);
        dA[57] = t271
            * (t104 * t277 * t278 * t350 * t352 * t566
               + t104 * t278 * t292 * t356 * t566 + t156 * t59 + t17 * t297
               - t23 * t59 + 2 * t277 * t350 * t352 * t565 - t307 * t639 - t637
               - t638 - t641);
        dA[58] = t271
            * (t104 * t277 * t278 * t352 * t361 * t566 + t277 * t292 * t643
               + 2 * t277 * t352 * t361 * t565 - t361 * t640 - t368 * t614
               - t644);
        dA[59] = t271
            * (t104 * t278 * t292 * t381 * t566 + t292 * t376 * t571
               - t307 * t646 - t364 * t377 * t565 - t376 * t569 + t462 - t645
               - t648);
        dA[60] = t271
            * (t0 * (t114 * t65 + t255 + t649) + t112 - t192 * t544
               + t2 * (-t114 * t37 + t51 * t56) + t264 + t270 - t284 * t624
               + t309 * t657 - t369 * t584 + t382 * t658
               - t537
                   * (t286 * t650 + t286 * t654 + t299 * (t120 + t262)
                      + t300 * t651 + t300 * t655 + t302 * (t116 * t65 + t68)
                      + t54 * t652 + t54 * t653)
               + t542 * t656);
        dA[61] = t271
            * (t0 * (-t12 * t155 + t142 * t65) + t154
               + t2 * (t155 * t56 + t428 + t649)
               + t4 * (-t157 * t65 + t231 + t546) - t432 * t624
               + t438 * t656 * t81 - t438 * t659 + t441 * t657 - t458 * t584
               + t465 * t658
               - t537
                   * (t164 * t652 + t164 * t653 + t299 * (t157 * t56 + t75)
                      + t302 * (t160 + t429) + t435 * t650 + t435 * t654
                      + t437 * t651 + t437 * t655)
               + t625);
        dA[62] = t271
            * (2 * t104 * t277 * t278 * t352 * t499 * t576
               + 2 * t104 * t277 * t287 * t292 * t499 * t577
               + t277 * t287 * t352 * t499 * t578 * t81 - t492 - t496 * t624
               - t499 * t659 - t519 * t584
               - t537
                   * (t192 * t652 + t192 * t653
                      + t299 * (t12 * t189 + t49 * t56)
                      + t302 * (t144 * t65 + t183 * t37) + t497 * t650
                      + t497 * t654 + t498 * t651 + t498 * t655 + t535));
        dA[63] = t271 * (-t540 * t659 + t585 + t592 * t657 + t603 * t658);
        dA[64] = t271
            * (2 * t104 * t277 * t278 * t352 * t566 * t576
               + 2 * t104 * t277 * t287 * t292 * t566 * t577 - t566 * t659
               - t626);
        dA[65] = t48
            * (2 * t102 * t104 * t259 * t261 * t83 * t91
               + 2 * t103 * t104 * t260 * t261 * t82 * t91
               - t107
                   * (t259 * t260 * t621 - t259 * t472 * t69 - t260 * t472 * t76
                      + t70 * (t560 + t619) + t77 * (t562 + t620) + t78)
               - t192 * t623 + std::pow(t261, 2) * t81 * t82 * t83 * t91
               - t261 * t269 * t557 - t490 - t555 - t622 + 2 * t68 * t9);
        dA[66] = t271
            * (t160 * t66 + 2 * t277 * t282 * t287 * t577 - t279 * t660
               - t43 * t66 - t663);
        dA[67] = t271
            * (t104 * t277 * t287 * t292 * t319 * t578
               + t104 * t278 * t292 * t324 * t578
               + 2 * t277 * t287 * t319 * t577 - t279 * t665 - t319 * t584
               - t502 - t664 - t667);
        dA[68] = t271
            * (t279 * t668 - t288 * t337 * t577 - t336 * t579 + t336 * t584
               - t340 * t662 + t4 * (-t2 * t65 + t37 * t55) + t512);
        dA[69] = t271
            * (t104 * t277 * t278 * t350 * t352 * t578
               + t104 * t278 * t292 * t356 * t578
               + 2 * t277 * t350 * t352 * t576 - t307 * t671 - t350 * t670
               - t513 - t669 - t673);
        dA[70] = t271
            * (t104 * t277 * t278 * t352 * t361 * t578 + t12 * t295 + t120 * t66
               - t19 * t66 + 2 * t277 * t352 * t361 * t576 - t296 * t58
               - t307 * t675 - t361 * t670 - t368 * t624 - t674);
        dA[71] = t271
            * (t104 * t278 * t292 * t381 * t578 + t277 * t292 * t676
               + t292 * t376 * t583 - t364 * t377 * t576 - t376 * t581 - t677);
        dA[72] = t48
            * (t0 * (t273 + t328) - t107 * (t276 * t70 + t683 * t71 + t684)
               + t126 * t680 + t131 * t685 + t131 * t686 + t312 + t678
               - t682 * t80);
        dA[73] = t48
            * (-t107 * (t145 * t683 + t434 * t70 + t687) - t146 * t682
               + t146 * t688 + t147 * t686 + t162 * t680 - t2 * t464 + t375
               + t443);
        dA[74] = t48
            * (t0 * (-t41 - t450)
               - t107 * (-t177 * t683 + t282 * t480 + t494 * t70) - t179 * t682
               + t179 * t688 + t179 * t689 + t190 * t680 + t330 + t4 * t45
               + t4 * t522);
        dA[75] = t48
            * (-t0 * t45 + t107 * (t197 * t683 + t588 * t70 + t684)
               + t198 * t682 - t198 * t688 - t198 * t689 - t210 * t680 + t311
               + t690);
        dA[76] = t48
            * (2 * t102 * t104 * t240 * t282 * t83 * t91
               + t102 * t104 * t240 * t679 * t83 * t91
               - t107 * (t239 * t683 + t628 * t70 - t687) - t2 * t646
               - t240 * t682 - t252 * t680 - t375 - t631);
        dA[77] = t271
            * (2 * t104 * t277 * t287 * t292 * t578 * t692
               - t537 * (t299 * (t19 + t449) + t650 * t692 + t654 * t692)
               + t66 * (-t326 - t43) - t663);
        dA[78] = t695
            * (t279 * std::pow(t282, 2) - t285 * t291 + t285 * t694 + t693);
        dA[79] = t695 * (t504 * t694 + t699);
        dA[80] = t695 * (t291 * t336 - t337 * t694 - t700 - t701 - t702);
        dA[81] = t706;
        dA[82] = t709;
        dA[83] = t712;
        dA[84] = t48
            * (t0 * (t0 * t116 + t545) + t102 * t103 * t104 * t324 * t80
               + 2 * t102 * t104 * t319 * t80 * t83 * t91
               - t107 * (t321 * t70 + t71 * t714 + t715) - t126 * t713
               - t131 * t716 - t314 - t327);
        dA[85] = t48
            * (t102 * t103 * t104 * t146 * t324
               + 2 * t102 * t104 * t146 * t319 * t83 * t91
               - t107 * (t145 * t714 + t445 * t70 + t717) - t146 * t718
               - t162 * t713 - t446);
        dA[86] = t271
            * (-t455 * t55 + 2 * t507 * t720 + t508
               - t537
                   * (t192 * t294 * t720 + t299 * (t49 * t66 + t75)
                      + t498 * t526 * t720)
               + t59 * (t0 * t183 + t124) + t66 * (t144 * t66 + t68) - t719);
        dA[87] = t48
            * (t0 * t35 + t0 * t607
               - t107 * (-t197 * t714 + t594 * t69 * t7 - t715) + t198 * t718
               - t198 * t722 - t198 * t723 + t210 * t713 + t627 + t630);
        dA[88] = t48
            * (-t107 * (2 * t239 * t319 * t7 - t633 * t70 - t717) - t240 * t718
               + t240 * t722 + t240 * t723 + t252 * t713 + t323 + t678);
        dA[89] = t48
            * (-t107 * (t260 * t714 - t319 * t480 + t665 * t70) - t261 * t718
               + t261 * t722 + t261 * t723 + t269 * t713 + t349
               + t4 * (t64 - 2 * t67) - t664 - t666);
        dA[90] = t695 * (t285 * t724 + t699);
        dA[91] = t695
            * (t279 * std::pow(t319, 2) + t324 * t504 + t504 * t724 + t725);
        dA[92] = t695 * (-t337 * t724 - t727);
        dA[93] = t729;
        dA[94] = t731;
        dA[95] = t733;
        dA[96] = t271
            * (2 * t104 * t277 * t287 * t292 * t304 * t738 - t341 - t345 * t59
               - t537 * (t294 * t54 * t738 + t299 * t736 + t300 * t526 * t738)
               + t55 * t735 + t66 * (t116 * t2 + t160) - t734);
        dA[97] = t48
            * (-t107 * (-t145 * t745 + t448 * t69 * t7 - t744)
               - t132 * t147 * t746 - t146 * t743 - t162 * t740 + t2 * t451
               + t4 * (-t232 - t373) - t447 - t739);
        dA[98] = t48
            * (-t107 * (t177 * t745 + t510 * t70 + t747) - t179 * t743
               - t179 * t748 - t190 * t740 - t322 + t4 * (t359 + t509) - t690);
        dA[99] = t48
            * (-t0 * t604 + t102 * t103 * t104 * t198 * t741
               + 3 * t102 * t104 * t198 * t336 * t83 * t91 + t103 * t210 * t336
               - t107 * (t171 * t336 + t197 * t745 + t595 * t70) - t331 - t596);
        dA[100] = t48
            * (-t107 * (-t239 * t745 + t636 * t70 + t744) + t2 * t639
               - t240 * t743 - t240 * t748 + t252 * t740 + t667 + t739);
        dA[101] = t48
            * (t103 * t269 * t336 - t107 * (-t260 * t745 - t668 * t70 - t747)
               - t261 * t743 - t261 * t748 + t322 - t4 * t41 - t678);
        dA[102] = t695 * (t285 * t750 + t291 * t336 - t700 - t702 + t749);
        dA[103] = t695 * (2 * t277 * t278 * t319 * t738 - t727);
        dA[104] = t695
            * (t277 * t278 * std::pow(t336, 2) + 2 * t336 * t339 - t337 * t750
               - t751);
        dA[105] = t753;
        dA[106] = t754;
        dA[107] = t755;
        dA[108] = t48
            * (-t107 * (t347 * t77 + t516 * t98 + t756) + t126 * t704
               + t127 * t356 - t344 + t757 * t80 + t758 * t80 + t759);
        dA[109] = t48
            * (t0 * (-t21 - t453)
               - t107 * (t168 * t516 + t350 * t407 + t451 * t77) + t146 * t757
               + t146 * t758 + t162 * t704 - t2 * t26 + t2 * t448 + t356 * t397
               - t357);
        dA[110] = t48
            * (-t107 * (-t403 * t516 + t515 * t77 + t760) + t179 * t757
               + t179 * t758 + t190 * t704 + t317 + t356 * t478 - t4 * t503
               + t521);
        dA[111] = t48
            * (t102 * t103 * t104 * t91 * (t482 * t516 + t600 * t77 + t756)
               - t198 * t757 - t198 * t758 - t198 * t761 - t210 * t704 - t597
               - t759);
        dA[112] = t271
            * (t17 * t297 + t350 * t569 + t518 * t614
               - t537 * (t302 * (t39 + t493) + t608 * t762 + t613 * t762)
               + 2 * t569 * t762 + t59 * (-t23 - t75) - t637 - t638 - t641);
        dA[113] = t48
            * (-t107 * (t516 * t763 + t671 * t77 - t760) + t261 * t757
               + t261 * t758 + t261 * t761 - t269 * t704 + t316
               + t4 * (t73 - 2 * t74) - t669 - t672);
        dA[114] = t706;
        dA[115] = t729;
        dA[116] = t753;
        dA[117] = t765
            * (t307 * std::pow(t350, 2) + t356 * t516 + t516 * t764 + t693);
        dA[118] = t765 * (t363 * t764 + t767);
        dA[119] = t765 * (t350 * t381 - t377 * t764 - t701 - t768 - t769);
        dA[120] = t271
            * (t104 * t277 * t278 * t304 * t352 * t361
               + 2 * t104 * t277 * t278 * t304 * t352 * t770 - t275 * t66 - t362
               - t370
               - t537 * (t286 * t526 * t770 + t301 * t54 * t770 + t302 * t735)
               + t55 * t736 + t59 * (t114 * t4 + t617) - t737);
        dA[121] = t48
            * (t0 * t26 - t107 * (t168 * t363 + t456 * t77 + t771) + t146 * t772
               - t146 * t773 + t15 * t4 - t162 * t708 + t2 * (t333 + t454)
               - t397 * t707);
        dA[122] = t48
            * (2 * t103 * t104 * t179 * t361 * t82 * t91
               - t107 * (-t363 * t403 + t522 * t77 + t775) - t179 * t773
               - t190 * t708 + t2 * (-t320 - t417) + t4 * t494 - t461
               - t478 * t707 - t774);
        dA[123] = t48
            * (-t0 * t595 + t102 * t103 * t104 * t198 * t707
               + t102 * t210 * t360 + t103 * t104 * t198 * t360 * t82 * t91
               - t107 * (-t172 * t361 - t363 * t482 + t604 * t7 * t76)
               - t198 * t772 - t334 - t605);
        dA[124] = t48
            * (t102 * t252 * t360 + 2 * t103 * t104 * t240 * t361 * t82 * t91
               - t107 * (2 * t238 * t361 * t7 - t643 * t77 - t771) - t240 * t773
               - t240 * t776 - t644);
        dA[125] = t48
            * (-t107 * (t363 * t763 + t675 * t77 - t775) + t261 * t772
               - t261 * t773 - t261 * t776 + t269 * t708 + t4 * t660 + t645
               + t647 + t774);
        dA[126] = t709;
        dA[127] = t731;
        dA[128] = t754;
        dA[129] = t765 * (t516 * t777 + t767);
        dA[130] = t765
            * (t307 * std::pow(t361, 2) - t363 * t367 + t363 * t777 + t725);
        dA[131] = t765 * (-t377 * t777 + t778);
        dA[132] = t48
            * (t0 * (t0 * t114 + t548) + t102 * t103 * t104 * t381 * t80
               - t107 * (t374 * t7 * t76 - t377 * t98 - t779) - t126 * t732
               - t315 - t384 - t780 * t80);
        dA[133] = t271
            * (t278 * t381 * t458 + t376 * t452 - t444 * t55 + 2 * t465 * t781
               - t466
               - t537
                   * (t164 * t301 * t781 + t302 * (t326 + t50 * t59)
                      + t435 * t526 * t781)
               + t59 * (t142 * t59 + t61) + t66 * (t0 * t157 + t120) - t691);
        dA[134] = t48
            * (t102 * t103 * t104 * t179 * t381
               - t107 * (t377 * t403 + t523 * t77 + t782) - t179 * t780
               - t190 * t732 - t524);
        dA[135] = t48
            * (t0 * t15 + t0 * t594 - t106 * t198 * t381
               - t107 * (t377 * t482 + t607 * t77 + t779) + t198 * t780
               + t210 * t732 + t673);
        dA[136] = t48
            * (t102 * t103 * t104 * t240 * t381 + t102 * t252 * t376
               - t107 * (-t238 * t377 * t7 + t376 * t407 + t646 * t77)
               + t2 * t61 + t2 * (t57 - 2 * t60) - t240 * t780 - t648);
        dA[137] = t48
            * (t102 * t103 * t104 * t261 * t381 + t102 * t269 * t376
               - t107 * (-t377 * t763 - t676 * t77 - t782) - t261 * t780
               - t677);
        dA[138] = t712;
        dA[139] = t733;
        dA[140] = t755;
        dA[141] = t765 * (t350 * t381 + t516 * t783 + t749 - t768 - t769);
        dA[142] = t765 * (t363 * t783 + t778);
        dA[143] = t765
            * (t277 * t292 * std::pow(t376, 2) - t377 * t381 - t377 * t783
               - t751);
    }

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
        double dA[9])
    {
        const auto t0 = -t2_y;
        const auto t1 = t0 + t1_y;
        const auto t2 = t0_x - t1_x;
        const auto t3 = t0 + t0_y;
        const auto t4 = -t2_x;
        const auto t5 = t0_x + t4;
        const auto t6 = t0_y - t1_y;
        const auto t7 = t2 * t3 - t5 * t6;
        const auto t8 = -t2_z;
        const auto t9 = t1_z + t8;
        const auto t10 = t0_z + t8;
        const auto t11 = t0_z - t1_z;
        const auto t12 = t10 * t2 - t11 * t5;
        const auto t13 =
            1.0 / (std::pow(t11, 2) + std::pow(t2, 2) + std::pow(t6, 2));
        const auto t14 = t10 * t6 - t11 * t3;
        const auto t15 = std::pow(t14, 2) + std::pow(t7, 2);
        const auto t16 = std::pow(t12, 2) + t15;
        const auto t17 = t10 * t2 - t11 * t5;
        const auto t18 = t15 + std::pow(t17, 2);
        const auto t19 = t13 * t18;
        const auto t20 = std::sqrt(t19) / t18;
        const auto t21 = t1_x + t4;
        const auto t22 = t13 * t16;
        dA[0] = -t20 * (-t1 * t7 - t12 * t9 + t13 * t16 * t2);
        dA[1] = -t20 * (-t14 * t9 + t21 * t7 + t22 * t6);
        dA[2] = -t20 * (t1 * t14 + t11 * t22 + t12 * t21);
        dA[3] = t20 * (-t10 * t12 + t13 * t18 * t2 - t3 * t7);
        dA[4] = t20 * (-t10 * t14 + t19 * t6 + t5 * t7);
        dA[5] = t20 * (t11 * t19 + t14 * t3 + t17 * t5);
        dA[6] = t20 * (t11 * t17 + t6 * t7);
        dA[7] = -t20 * (-t11 * t14 + t2 * t7);
        dA[8] = -t20 * (t12 * t2 + t14 * t6);
    }

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
        double dA[81])
    {
        const auto t0 = -t2_y;
        const auto t1 = t0 + t1_y;
        const auto t2 = std::pow(t1, 2);
        const auto t3 = t0_x - t1_x;
        const auto t4 = -t2_z;
        const auto t5 = t0_z + t4;
        const auto t6 = t3 * t5;
        const auto t7 = -t2_x;
        const auto t8 = t0_x + t7;
        const auto t9 = t0_z - t1_z;
        const auto t10 = t8 * t9;
        const auto t11 = -t10;
        const auto t12 = t11 + t6;
        const auto t13 = t0 + t0_y;
        const auto t14 = t13 * t3;
        const auto t15 = t0_y - t1_y;
        const auto t16 = t15 * t8;
        const auto t17 = t14 - t16;
        const auto t18 = t15 * t5;
        const auto t19 = t13 * t9;
        const auto t20 = -t19;
        const auto t21 = t18 + t20;
        const auto t22 = std::pow(t17, 2) + std::pow(t21, 2);
        const auto t23 = std::pow(t12, 2) + t22;
        const auto t24 = 1.0 / t23;
        const auto t25 = std::pow(t9, 2);
        const auto t26 = std::pow(t3, 2);
        const auto t27 = std::pow(t15, 2);
        const auto t28 = t26 + t27;
        const auto t29 = t25 + t28;
        const auto t30 = 1.0 / t29;
        const auto t31 = -t3;
        const auto t32 = -t5;
        const auto t33 = -t8;
        const auto t34 = -t9;
        const auto t35 = t31 * t32 - t33 * t34;
        const auto t36 = t22 + std::pow(t35, 2);
        const auto t37 = -t3 * t30 * t36;
        const auto t38 = t1 * t17;
        const auto t39 = t1_z + t4;
        const auto t40 = t35 * t39 + t38;
        const auto t41 = t37 + t40;
        const auto t42 = -t41;
        const auto t43 = t12 * t39 + t38;
        const auto t44 = t3 * t30;
        const auto t45 = 2 * t44;
        const auto t46 = t24 * t42;
        const auto t47 = 2 * t40;
        const auto t48 = 4 / std::pow(t29, 2);
        const auto t49 = t23 * t48;
        const auto t50 = t26 * t49;
        const auto t51 = t23 * t30;
        const auto t52 = -t51;
        const auto t53 = std::pow(t39, 2) + t52;
        const auto t54 = t24 * std::sqrt(t51);
        const auto t55 = t30 * t36;
        const auto t56 = t15 * t55;
        const auto t57 = t1_x + t7;
        const auto t58 = t17 * t57 - t21 * t39;
        const auto t59 = t56 + t58;
        const auto t60 = t30 * t47;
        const auto t61 = t15 * t60;
        const auto t62 = -t58;
        const auto t63 = t15 * t3;
        const auto t64 = t36 * t48;
        const auto t65 = t63 * t64;
        const auto t66 = -t65;
        const auto t67 = t45 * t62 + t66;
        const auto t68 = t1 * t57 - t24 * t42 * t59 + t61 + t67;
        const auto t69 = t39 * t57;
        const auto t70 = t55 * t9;
        const auto t71 = t1 * t21 + t35 * t57;
        const auto t72 = t70 + t71;
        const auto t73 = t45 * t72;
        const auto t74 = t45 * t71;
        const auto t75 = t24 * t72;
        const auto t76 = t60 * t9;
        const auto t77 = t3 * t9;
        const auto t78 = t64 * t77;
        const auto t79 = -t76 + t78;
        const auto t80 = t13 * t17 + t35 * t5;
        const auto t81 = t23 * t3 * t30 - t80;
        const auto t82 = t24 * t81;
        const auto t83 = t39 * t5;
        const auto t84 = t52 + t83;
        const auto t85 = t1 * t13;
        const auto t86 = -2 * t3 * t30 * t80 + t85;
        const auto t87 = t17 * t8 - t21 * t5;
        const auto t88 = t15 * t51 + t87;
        const auto t89 = t24 * t88;
        const auto t90 = t45 * t87;
        const auto t91 = -t14;
        const auto t92 = t16 + t91;
        const auto t93 = -t1 * t8 - t61 + t65 + t90 + t92;
        const auto t94 = t13 * t21;
        const auto t95 = t12 * t8 + t94;
        const auto t96 = t51 * t9 + t95;
        const auto t97 = t24 * t96;
        const auto t98 = t35 * t8 + t94;
        const auto t99 = t45 * t98;
        const auto t100 = -t6;
        const auto t101 = t10 + t100;
        const auto t102 = t101 - t39 * t8 + t79 + t99;
        const auto t103 = t15 * t17;
        const auto t104 = t103 + t12 * t9;
        const auto t105 = t24 * t47;
        const auto t106 = t1 * t15;
        const auto t107 = t39 * t9;
        const auto t108 = -t104 * t46 + t106 + t107;
        const auto t109 = t1 * t3;
        const auto t110 = t17 * t3 - t21 * t9;
        const auto t111 = t3 * t39;
        const auto t112 = t15 * t21 + t3 * t35;
        const auto t113 = t112 * t46;
        const auto t114 = t15 * t30;
        const auto t115 = 2 * t114;
        const auto t116 = std::pow(t57, 2);
        const auto t117 = t24 * t59;
        const auto t118 = 2 * t62;
        const auto t119 = t27 * t49;
        const auto t120 = 4 * t114;
        const auto t121 = t1 * t39;
        const auto t122 = t30 * t9;
        const auto t123 = t118 * t122;
        const auto t124 = t15 * t9;
        const auto t125 = t124 * t64;
        const auto t126 = -t123 + t125;
        const auto t127 = t115 * t71;
        const auto t128 = -t115 * t72 + t127;
        const auto t129 = t115 * t80;
        const auto t130 = t129 + t13 * t57 + t67 + t92;
        const auto t131 = t57 * t8;
        const auto t132 = t115 * t87 + t131;
        const auto t133 = t115 * t98;
        const auto t134 = -t18 + t19;
        const auto t135 = t126 - t13 * t39 + t133 + t134;
        const auto t136 = t118 * t24;
        const auto t137 = t104 * t117 + t15 * t57 + t92;
        const auto t138 = t3 * t57;
        const auto t139 = t107 + t110 * t117 + t138;
        const auto t140 = t15 * t39;
        const auto t141 = t112 * t117;
        const auto t142 = 2 * t122;
        const auto t143 = 2 * t71;
        const auto t144 = -t78;
        const auto t145 = t144 - t74;
        const auto t146 = -t142 * t72;
        const auto t147 = 4 * t122;
        const auto t148 = t25 * t49 + t52;
        const auto t149 = t142 * t80;
        const auto t150 = t101 + t145 + t149 + t5 * t57;
        const auto t151 = t125 + t142 * t87;
        const auto t152 = -t1 * t5 + t151 + t21;
        const auto t153 = t131 + t142 * t71 + t85;
        const auto t154 = t101 + t104 * t75 + t57 * t9;
        const auto t155 = t1 * t9;
        const auto t156 = t143 * t24;
        const auto t157 = t112 * t75;
        const auto t158 = t106 + t138;
        const auto t159 = t26 * t64;
        const auto t160 = 1.0 / t36;
        const auto t161 = -t37 - t80;
        const auto t162 = -t13;
        const auto t163 = -t15;
        const auto t164 = t162 * t31 - t163 * t33;
        const auto t165 = -t35;
        const auto t166 = 2 * t162 * t164 + 2 * t165 * t5;
        const auto t167 = t160 * t166;
        const auto t168 = -t55;
        const auto t169 = t168 + t83;
        const auto t170 = t160 * std::sqrt(t55);
        const auto t171 = -t59;
        const auto t172 = t160 * t161;
        const auto t173 = std::pow(t13, 2);
        const auto t174 = t168 + std::pow(t5, 2);
        const auto t175 = t56 + t87;
        const auto t176 = t160 * t175;
        const auto t177 = t129 + t13 * t8 - t160 * t161 * t175 + t66 - t90;
        const auto t178 = t70 + t98;
        const auto t179 = t160 * t178;
        const auto t180 = t144 + t149 - t160 * t161 * t178 + t5 * t8 - t99;
        const auto t181 = t13 * t15;
        const auto t182 = t5 * t9;
        const auto t183 = -t104 * t24 * t81 + t181 + t182;
        const auto t184 = 2 * t24;
        const auto t185 = t184 * t80;
        const auto t186 = t110 * t82 - 2 * t13 * t3 + t16;
        const auto t187 = t112 * t82;
        const auto t188 = -t162 * t34 + t163 * t32;
        const auto t189 = t164 * t8 + t188 * t32;
        const auto t190 = 2 * t189;
        const auto t191 = t160 * t190;
        const auto t192 = t27 * t64;
        const auto t193 = std::pow(t8, 2);
        const auto t194 = -t13 * t5 + t133 + t151 + t176 * t178;
        const auto t195 = t184 * t87;
        const auto t196 = t3 * t8;
        const auto t197 = t110 * t89 + t182 + t196;
        const auto t198 = t112 * t89;
        const auto t199 = 2 * t18 + t20;
        const auto t200 = t13 * t188 + t165 * t33;
        const auto t201 = 2 * t200;
        const auto t202 = t160 * t201;
        const auto t203 = t168 + t25 * t64;
        const auto t204 = t110 * t97 - 2 * t13 * t9 + t18;
        const auto t205 = t112 * t97;
        const auto t206 = t181 + t196;
        const auto t207 = t103 + t35 * t9;
        const auto t208 = 2 * t207;
        const auto t209 = -t110;
        const auto t210 = t160 * t209;
        const auto t211 = t15 * t164 + t165 * t34;
        const auto t212 = -t160 * t207 * t209 + t63;
        const auto t213 = t160 * t207;
        const auto t214 = t112 * t213 + t77;
        const auto t215 = 2 * t209;
        const auto t216 = t164 * t31 + t188 * t9;
        const auto t217 = t112 * t210 + t124;
        const auto t218 = -t112 * t45;
        const auto t219 = -t112 * t115;
        const auto t220 = t112 * t142;
        const auto t221 = 2 * t163 * t188 + 2 * t165 * t3;
        dA[0] = t54
            * (t2 + t24 * std::pow(t42, 2) - t42 * t45 - 4 * t43 * t44
               + t46 * t47 + t50 + t53);
        dA[1] = t54 * (2 * t24 * t40 * t59 - t45 * t59 - t68);
        dA[2] = t54 * (t46 * t72 + t47 * t75 - t69 - t73 + t74 + t79);
        dA[3] = t54
            * (2 * t3 * t30 * t43 + 2 * t3 * t30 * t81 - t46 * t81 - t47 * t82
               - t50 - t84 - t86);
        dA[4] = t54 * (2 * t3 * t30 * t88 - t46 * t88 - t47 * t89 - t93);
        dA[5] = t54 * (-t102 + 2 * t3 * t30 * t96 - t46 * t96 - t47 * t97);
        dA[6] = t54 * (-t104 * t105 + t108);
        dA[7] = t54 * (t105 * t110 - t109 + t110 * t46 + t92);
        dA[8] = t54 * (t101 + t105 * t112 - t111 + t113);
        dA[9] = t54 * (-t115 * t42 + 2 * t24 * t42 * t62 - t68);
        dA[10] = t54
            * (-t115 * t59 + t116 + t117 * t118 + t119 + t120 * t58
               + t24 * std::pow(t59, 2) + t53);
        dA[11] = t54 * (t117 * t72 + t118 * t75 - t121 + t126 + t128);
        dA[12] = t54 * (t115 * t81 - t117 * t81 - t118 * t82 + t130);
        dA[13] = t54
            * (-t115 * t58 - t117 * t88 - t118 * t89 - t119 - t132
               + 2 * t15 * t30 * t88 - t84);
        dA[14] = t54 * (-t117 * t96 - t118 * t97 - t135 + 2 * t15 * t30 * t96);
        dA[15] = -t54 * (t104 * t136 + t137);
        dA[16] = t54 * (t110 * t136 + t139);
        dA[17] = t54 * (t112 * t136 + t134 - t140 + t141);
        dA[18] = t54
            * (-t142 * t42 - t143 * t46 - t145 + t24 * t42 * t72 - t69 - t76);
        dA[19] = t54
            * (-t117 * t143 - t121 - t123 + t125 + t127 - t142 * t59
               + t24 * t59 * t72);
        dA[20] = t54
            * (t116 - t143 * t75 + t146 + t147 * t71 + t148 + t2
               + t24 * std::pow(t72, 2));
        dA[21] = t54 * (t142 * t81 + t143 * t82 + t150 - t75 * t81);
        dA[22] = t54
            * (-t127 - t152 + 2 * t24 * t71 * t88 + 2 * t30 * t88 * t9
               - t75 * t88);
        dA[23] = t54
            * (-t142 * t95 - t148 - t153 + 2 * t24 * t71 * t96
               + 2 * t30 * t9 * t96 - t75 * t96);
        dA[24] = t54 * (2 * t104 * t24 * t71 - t154);
        dA[25] = t54 * (-t110 * t156 + t110 * t24 * t72 - t134 - t155);
        dA[26] = t54 * (-t112 * t156 + t157 + t158);
        dA[27] = t170
            * (-t159 + t160 * t161 * t41 - t167 * t41 - t169
               + 2 * t3 * t30 * t40 - t41 * t45 - t86);
        dA[28] = t170 * (t130 - t167 * t171 + t171 * t172 - t171 * t45);
        dA[29] = t170 * (t150 + t167 * t72 - t172 * t72 + t73);
        dA[30] = t170
            * (t159 + t160 * std::pow(t161, 2) - t161 * t45 - t166 * t172 + t173
               + t174 - 4 * t44 * t80);
        dA[31] = t170 * (-t166 * t176 - t175 * t45 - t177);
        dA[32] = t170 * (-t166 * t179 - t178 * t45 - t180);
        dA[33] = t54 * (2 * t104 * t24 * t80 - t183);
        dA[34] = t54 * (-t110 * t185 - t186);
        dA[35] = t54 * (-t10 - t112 * t185 - t187 + 2 * t3 * t5);
        dA[36] = t170 * (-t115 * t41 + t160 * t175 * t41 - t191 * t41 - t93);
        dA[37] = t170
            * (-t115 * t171 - t132 + 2 * t15 * t30 * t62 + t160 * t171 * t175
               - t169 - t171 * t191 - t192);
        dA[38] = t170 * (-t128 - t152 + 2 * t160 * t189 * t72 - t176 * t72);
        dA[39] = t170 * (-t115 * t161 - t172 * t190 - t177);
        dA[40] = t170
            * (-t115 * t175 + t120 * t87 + t160 * std::pow(t175, 2) + t174
               - t176 * t190 + t192 + t193);
        dA[41] = t170 * (-t115 * t178 - t179 * t190 + t194);
        dA[42] = t54 * (-t104 * t195 + t104 * t24 * t88 - t14 + 2 * t15 * t8);
        dA[43] = t54 * (2 * t110 * t24 * t87 - t197);
        dA[44] = t54 * (t112 * t195 - t198 + t199);
        dA[45] = t170 * (-t102 - t142 * t41 + t160 * t178 * t41 - t202 * t41);
        dA[46] =
            t170 * (-t135 - t142 * t171 + t160 * t171 * t178 - t171 * t202);
        dA[47] = t170
            * (-t142 * t98 - t146 - t153 + 2 * t160 * t200 * t72 - t179 * t72
               - t203);
        dA[48] = t170 * (-t142 * t161 - t172 * t201 - t180);
        dA[49] = t170 * (-t142 * t175 - t176 * t201 + t194);
        dA[50] = t170
            * (-t142 * t178 + t147 * t98 + t160 * std::pow(t178, 2) + t173
               - t179 * t201 + t193 + t203);
        dA[51] =
            t54 * (-t104 * t184 * t98 + t104 * t24 * t96 - t6 + 2 * t8 * t9);
        dA[52] = t54 * (2 * t110 * t24 * t98 - t204);
        dA[53] = t54 * (2 * t112 * t24 * t98 - t205 - t206);
        dA[54] = t54 * (t108 - t207 * t45 + t208 * t46);
        dA[55] = t54 * (-t115 * t207 - t137 + 2 * t207 * t24 * t59);
        dA[56] = t54 * (-t142 * t207 - t154 + 2 * t207 * t24 * t72);
        dA[57] = t54 * (2 * t104 * t3 * t30 - t183 - t208 * t82);
        dA[58] = t54 * (t104 * t115 + t104 * t89 + 2 * t16 - t208 * t89 + t91);
        dA[59] = t54 * (2 * t10 + t100 + t104 * t142 + t104 * t97 - t208 * t97);
        dA[60] =
            t54 * (std::pow(t104, 2) * t24 - t104 * t184 * t207 + t25 + t27);
        dA[61] = t170 * (-2 * t210 * t211 - t212);
        dA[62] = t170 * (2 * t112 * t160 * t211 - t214);
        dA[63] = t54
            * (-t109 + t110 * t24 * t42 - t17 + 2 * t209 * t24 * t42
               - t209 * t45);
        dA[64] = t54 * (-t115 * t209 + t117 * t215 + t139);
        dA[65] = t54 * (t110 * t75 - t142 * t209 - t155 + t21 + t215 * t75);
        dA[66] = t54 * (-t110 * t45 - t186 - t215 * t82);
        dA[67] = -t54 * (t110 * t115 + t197 + t215 * t89);
        dA[68] = t54 * (-t110 * t142 - t204 - t215 * t97);
        dA[69] = t170 * (-t212 - 2 * t213 * t216);
        dA[70] =
            t54 * (std::pow(t110, 2) * t24 + t110 * t184 * t209 + t25 + t26);
        dA[71] = t170 * (2 * t112 * t160 * t216 - t217);
        dA[72] = t54 * (-t111 - t113 - t12 - t218);
        dA[73] = t54 * (-t140 - t141 - t21 - t219);
        dA[74] = t54 * (-t157 + t158 + t220);
        dA[75] = t54 * (t11 + t187 + t218 + 2 * t6);
        dA[76] = t54 * (t198 + t199 + t219);
        dA[77] = t54 * (t205 - t206 - t220);
        dA[78] = -t170 * (t213 * t221 + t214);
        dA[79] = -t170 * (t210 * t221 + t217);
        dA[80] = t54 * (-std::pow(t112, 2) * t24 + t28);
    }

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
        double grad[9])
    {
        const auto t0 = -p2x;
        const auto t1 = p1x + t0;
        const auto t2 = -p2y;
        const auto t3 = p1y + t2;
        const auto t4 = -p2z;
        const auto t5 = p1z + t4;
        const auto t6 = std::pow(t1, 2) + std::pow(t3, 2) + std::pow(t5, 2);
        const auto t7 = p2x - p3x;
        const auto t8 = p2y - p3y;
        const auto t9 = p2z - p3z;
        const auto t10 = std::pow(t7, 2) + std::pow(t8, 2) + std::pow(t9, 2);
        const auto t11 = std::pow(t10 * t6, -1.0 / 2.0);
        const auto t12 = 1.0 / t6;
        const auto t13 = t1 * t7 + t3 * t8 + t5 * t9;
        const auto t14 = t12 * t13;
        const auto t15 = 1.0 / t10;
        const auto t16 = std::pow(t7, 2) + std::pow(t8, 2) + std::pow(t9, 2);
        const auto t17 = t13 * t15;
        grad[0] = t11 * (p3x + t0 + t1 * t14);
        grad[1] = t11 * (p3y + t14 * t3 + t2);
        grad[2] = t11 * (p3z + t14 * t5 + t4);
        grad[3] = t11
            * (-p1x + 2 * p2x - p3x + t12 * t13 * t15 * (-t1 * t16 + t6 * t7));
        grad[4] = t11
            * (-p1y + 2 * p2y - p3y + t12 * t13 * t15 * (-t16 * t3 + t6 * t8));
        grad[5] = t11
            * (-p1z + 2 * p2z - p3z + t12 * t13 * t15 * (-t16 * t5 + t6 * t9));
        grad[6] = t11 * (p1x - p2x - t17 * t7);
        grad[7] = t11 * (p1y - p2y - t17 * t8);
        grad[8] = t11 * (p1z - p2z - t17 * t9);
    }

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
        double hess[81])
    {
        const auto t0 = p1x - p2x;
        const auto t1 = std::pow(t0, 2);
        const auto t2 = p1y - p2y;
        const auto t3 = std::pow(t2, 2);
        const auto t4 = p1z - p2z;
        const auto t5 = std::pow(t4, 2);
        const auto t6 = t1 + t3 + t5;
        const auto t7 = 1.0 / t6;
        const auto t8 = t1 * t7;
        const auto t9 = p2x - p3x;
        const auto t10 = t0 * t9;
        const auto t11 = p2y - p3y;
        const auto t12 = t11 * t2;
        const auto t13 = p2z - p3z;
        const auto t14 = t13 * t4;
        const auto t15 = t12 + t14;
        const auto t16 = t10 + t15;
        const auto t17 = 3 * t16;
        const auto t18 = 3 * t10 + t15;
        const auto t19 = std::pow(t9, 2);
        const auto t20 = std::pow(t11, 2);
        const auto t21 = std::pow(t13, 2);
        const auto t22 = t19 + t20 + t21;
        const auto t23 = std::pow(t22 * t6, -1.0 / 2.0);
        const auto t24 = t23 * t7;
        const auto t25 = t16 * t7;
        const auto t26 = t0 * t25;
        const auto t27 = 3 * t2;
        const auto t28 = t0 * t11;
        const auto t29 = t2 * t9;
        const auto t30 = t28 + t29;
        const auto t31 = t24 * (-t26 * t27 + t30);
        const auto t32 = t26 * t4;
        const auto t33 = t0 * t13;
        const auto t34 = t4 * t9;
        const auto t35 = t33 + t34;
        const auto t36 = t24 * (-3 * t32 + t35);
        const auto t37 = p1x - 2 * p2x + p3x;
        const auto t38 = 1.0 / t22;
        const auto t39 = -t0 * t22 + t6 * t9;
        const auto t40 = -2 * t10;
        const auto t41 = t25 * t38;
        const auto t42 = std::pow(t6, -2);
        const auto t43 = t16 * t38;
        const auto t44 = t42 * t43;
        const auto t45 = t0 * t44;
        const auto t46 = 2 * t43;
        const auto t47 = std::pow(t11, 2) + std::pow(t13, 2) + std::pow(t9, 2);
        const auto t48 = t11 * t6 - t2 * t47;
        const auto t49 = t41 * t48;
        const auto t50 = t0 * t49;
        const auto t51 = p1y - 2 * p2y + p3y;
        const auto t52 = t38 * t9;
        const auto t53 = t0 * t51 + t48 * t52;
        const auto t54 = t13 * t6 - t4 * t47;
        const auto t55 = t41 * t54;
        const auto t56 = t0 * t55;
        const auto t57 = p1z - 2 * p2z + p3z;
        const auto t58 = t0 * t57 + t52 * t54;
        const auto t59 = t19 * t38;
        const auto t60 = t23 * (t0 * t16 * t38 * t7 * t9 - t59 - t8 + 1);
        const auto t61 = t0 * t7;
        const auto t62 = t11 * t52 + t2 * t61;
        const auto t63 = t23 * (t0 * t11 * t16 * t38 * t7 - t62);
        const auto t64 = t13 * t52 + t4 * t61;
        const auto t65 = t23 * (t0 * t13 * t16 * t38 * t7 - t64);
        const auto t66 = t3 * t7;
        const auto t67 = t10 + 3 * t12 + t14;
        const auto t68 = t25 * t4;
        const auto t69 = t13 * t2;
        const auto t70 = t11 * t4;
        const auto t71 = t69 + t70;
        const auto t72 = t24 * (-t27 * t68 + t71);
        const auto t73 = -t0 * t47 + t6 * t9;
        const auto t74 = t41 * t73;
        const auto t75 = t11 * t38;
        const auto t76 = t73 * t75;
        const auto t77 = t2 * t37 + t76;
        const auto t78 = t11 * t6 - t2 * t22;
        const auto t79 = -2 * t12;
        const auto t80 = t2 * t57 + t54 * t75;
        const auto t81 = t23 * (t16 * t2 * t38 * t7 * t9 - t62);
        const auto t82 = t20 * t38;
        const auto t83 = t23 * (t11 * t16 * t2 * t38 * t7 - t66 - t82 + 1);
        const auto t84 = t2 * t7;
        const auto t85 = t13 * t75 + t4 * t84;
        const auto t86 = t23 * (t13 * t16 * t2 * t38 * t7 - t85);
        const auto t87 = t5 * t7;
        const auto t88 = t10 + t12 + 3 * t14;
        const auto t89 = t4 * t74;
        const auto t90 = t13 * t38;
        const auto t91 = t73 * t90;
        const auto t92 = t37 * t4 + t91;
        const auto t93 = t4 * t49;
        const auto t94 = t48 * t90;
        const auto t95 = t4 * t51 + t94;
        const auto t96 = t13 * t6 - t22 * t4;
        const auto t97 = -2 * t14;
        const auto t98 = t4 * t44;
        const auto t99 = t23 * (t16 * t38 * t4 * t7 * t9 - t64);
        const auto t100 = t23 * (t11 * t16 * t38 * t4 * t7 - t85);
        const auto t101 = t21 * t38;
        const auto t102 = t23 * (-t101 + t13 * t16 * t38 * t4 * t7 - t87 + 1);
        const auto t103 = 2 * t16;
        const auto t104 = t103 * t42;
        const auto t105 = -t25 - 1;
        const auto t106 = 2 * t26;
        const auto t107 = t106 * t2;
        const auto t108 = 2 * t32;
        const auto t109 = 2 * t39;
        const auto t110 = t38 * t7;
        const auto t111 = t22 + t6;
        const auto t112 = std::pow(t22, -2);
        const auto t113 = t112 * t16 * t42;
        const auto t114 = t42 * t46;
        const auto t115 = t112 * t25;
        const auto t116 = t46 * t9;
        const auto t117 = t39 * t41;
        const auto t118 = -t103 * t30 - t117 * t78 + t37 * t78 + t39 * t51;
        const auto t119 = t24 * t38;
        const auto t120 = -t103 * t35 - t117 * t96 + t37 * t96 + t39 * t57;
        const auto t121 = t37 * t52 + t38 * t61 * t73 + 1;
        const auto t122 = t11 * t37;
        const auto t123 = -2 * t11 * t16 * t38 * t9;
        const auto t124 = t73 * t84;
        const auto t125 = t23 * t38;
        const auto t126 = t13 * t37;
        const auto t127 = -2 * t13 * t16 * t38 * t9;
        const auto t128 = t4 * t7;
        const auto t129 = t128 * t73;
        const auto t130 = 2 * t2;
        const auto t131 = t130 * t68;
        const auto t132 = t11 * t46;
        const auto t133 = 2 * t78;
        const auto t134 = -t103 * t71 - t41 * t78 * t96 + t51 * t96 + t57 * t78;
        const auto t135 = t48 * t61 + t51 * t9;
        const auto t136 = t38 * t48 * t84 + t51 * t75 + 1;
        const auto t137 = t13 * t51;
        const auto t138 = -2 * t11 * t13 * t16 * t38;
        const auto t139 = t128 * t48;
        const auto t140 = t13 * t46;
        const auto t141 = 2 * t96;
        const auto t142 = t54 * t61 + t57 * t9;
        const auto t143 = t11 * t57 + t54 * t84;
        const auto t144 = t128 * t38 * t54 + t57 * t90 + 1;
        const auto t145 = 3 * t43 * t9;
        const auto t146 = t125 * (-t11 * t145 + t30);
        const auto t147 = t125 * (-t13 * t145 + t35);
        const auto t148 = 2 * t25;
        const auto t149 = t125 * (-3 * t11 * t13 * t43 + t71);
        hess[0] = t24 * (-t17 * t8 + t18);
        hess[1] = t31;
        hess[2] = t36;
        hess[3] = t23
            * (t0 * t37 * t7 + t38 * t39 * t7 * t9 - 3 * t39 * t45
               - t41 * (t22 + t40) - 1);
        hess[4] = t24 * (t28 * t46 - 3 * t50 + t53);
        hess[5] = t24 * (t33 * t46 - 3 * t56 + t58);
        hess[6] = t60;
        hess[7] = t63;
        hess[8] = t65;
        hess[9] = t31;
        hess[10] = t24 * (-t17 * t66 + t67);
        hess[11] = t72;
        hess[12] = t24 * (-t27 * t74 + t29 * t46 + t77);
        hess[13] = t23
            * (t11 * t38 * t7 * t78 + t2 * t51 * t7 - t27 * t44 * t78
               - t41 * (t22 + t79) - 1);
        hess[14] = t24 * (-t27 * t55 + t46 * t69 + t80);
        hess[15] = t81;
        hess[16] = t83;
        hess[17] = t86;
        hess[18] = t36;
        hess[19] = t72;
        hess[20] = t24 * (-t17 * t87 + t88);
        hess[21] = t24 * (t34 * t46 - 3 * t89 + t92);
        hess[22] = t24 * (t46 * t70 - 3 * t93 + t95);
        hess[23] = t23
            * (t13 * t38 * t7 * t96 + t4 * t57 * t7 - t41 * (t22 + t97)
               - 3 * t96 * t98 - 1);
        hess[24] = t99;
        hess[25] = t100;
        hess[26] = t102;
        hess[27] =
            t23 * (t1 * t104 + t105 + t37 * t61 - t45 * t73 + t52 * t7 * t73);
        hess[28] = t24 * (t107 - t2 * t74 + t77);
        hess[29] = t24 * (t108 - t89 + t92);
        hess[30] = t23
            * (t0 * t114 * t39 + t109 * t110 * t37 - t109 * t115 * t9
               - t113 * std::pow(t39, 2) + t41 * (-4 * t10 + t111) + 2);
        hess[31] = t119 * (t106 * t78 - t116 * t78 + t118);
        hess[32] = t119 * (t106 * t96 - t116 * t96 + t120);
        hess[33] = t23
            * (2 * t112 * t16 * t19 + t112 * t16 * t7 * t73 * t9 - t121 - t43);
        hess[34] = t125 * (t11 * t16 * t38 * t7 * t73 - t122 - t123 - t124);
        hess[35] = t125 * (-t126 - t127 - t129 + t13 * t16 * t38 * t7 * t73);
        hess[36] = t24 * (t107 - t50 + t53);
        hess[37] = t23
            * (t104 * t3 + t105 - t2 * t44 * t48 + t48 * t7 * t75 + t51 * t84);
        hess[38] = t24 * (t131 - t93 + t95);
        hess[39] = t119 * (t109 * t2 * t25 + t118 - t132 * t39);
        hess[40] = t23
            * (-t11 * t115 * t133 + t110 * t133 * t51 - t113 * std::pow(t78, 2)
               + t114 * t2 * t78 + t41 * (t111 - 4 * t12) + 2);
        hess[41] = t119 * (t130 * t25 * t96 - t132 * t96 + t134);
        hess[42] = t125 * (-t123 - t135 + t16 * t38 * t48 * t7 * t9);
        hess[43] = t23
            * (t11 * t112 * t16 * t48 * t7 + 2 * t112 * t16 * t20 - t136 - t43);
        hess[44] = t125 * (t13 * t16 * t38 * t48 * t7 - t137 - t138 - t139);
        hess[45] = t24 * (t108 - t56 + t58);
        hess[46] = t24 * (t131 - t2 * t55 + t80);
        hess[47] =
            t23 * (t104 * t5 + t105 + t128 * t57 + t54 * t7 * t90 - t54 * t98);
        hess[48] = t119 * (t109 * t68 + t120 - t140 * t39);
        hess[49] = t119 * (t133 * t68 + t134 - t140 * t78);
        hess[50] = t23
            * (t110 * t141 * t57 - t113 * std::pow(t96, 2) + t114 * t4 * t96
               - t115 * t13 * t141 + t41 * (t111 - 4 * t14) + 2);
        hess[51] = t125 * (-t127 - t142 + t16 * t38 * t54 * t7 * t9);
        hess[52] = t125 * (t11 * t16 * t38 * t54 * t7 - t138 - t143);
        hess[53] = t23
            * (t112 * t13 * t16 * t54 * t7 + 2 * t112 * t16 * t21 - t144 - t43);
        hess[54] = t60;
        hess[55] = t81;
        hess[56] = t99;
        hess[57] =
            t23 * (3 * t112 * t16 * t7 * t73 * t9 - t121 - t41 * (t40 + t6));
        hess[58] = t125
            * (-t135 + 2 * t16 * t2 * t7 * t9 + 3 * t16 * t38 * t48 * t7 * t9);
        hess[59] = t125
            * (-t142 + 3 * t16 * t38 * t54 * t7 * t9 + 2 * t16 * t4 * t7 * t9);
        hess[60] = t125 * (-t17 * t59 + t18);
        hess[61] = t146;
        hess[62] = t147;
        hess[63] = t63;
        hess[64] = t83;
        hess[65] = t100;
        hess[66] = t125 * (-t122 - t124 + t148 * t28 + 3 * t25 * t76);
        hess[67] =
            t23 * (3 * t11 * t112 * t16 * t48 * t7 - t136 - t41 * (t6 + t79));
        hess[68] = t125
            * (3 * t11 * t16 * t38 * t54 * t7 + 2 * t11 * t16 * t4 * t7 - t143);
        hess[69] = t146;
        hess[70] = t125 * (-t17 * t82 + t67);
        hess[71] = t149;
        hess[72] = t65;
        hess[73] = t86;
        hess[74] = t102;
        hess[75] = t125 * (-t126 - t129 + t148 * t33 + 3 * t25 * t91);
        hess[76] = t125 * (-t137 - t139 + t148 * t69 + 3 * t25 * t94);
        hess[77] =
            t23 * (3 * t112 * t13 * t16 * t54 * t7 - t144 - t41 * (t6 + t97));
        hess[78] = t147;
        hess[79] = t149;
        hess[80] = t125 * (-t101 * t17 + t88);
    }


void curve_twist_gradient(double t0_x, double t0_y, double t0_z, double t1_x, double t1_y, double t1_z, double t2_x, double t2_y, double t2_z, double t3_x, double t3_y, double t3_z, double dA[12]){
const auto t0 = t1_x - t2_x;
const auto t1 = -t0;
const auto t2 = std::pow(t1, 2);
const auto t3 = t1_y - t2_y;
const auto t4 = -t3;
const auto t5 = std::pow(t4, 2);
const auto t6 = t1_z - t2_z;
const auto t7 = -t6;
const auto t8 = std::pow(t7, 2);
const auto t9 = t2 + t5 + t8;
const auto t10 = 1.0/t9;
const auto t11 = t0_x - t1_x;
const auto t12 = -t1*t11;
const auto t13 = t0_y - t1_y;
const auto t14 = -t13*t4;
const auto t15 = t0_z - t1_z;
const auto t16 = -t15*t7;
const auto t17 = t14 + t16;
const auto t18 = t12 + t17;
const auto t19 = t10*t18;
const auto t20 = t1*t19;
const auto t21 = t11 + t20;
const auto t22 = t19*t4;
const auto t23 = t13 + t22;
const auto t24 = t19*t7;
const auto t25 = t15 + t24;
const auto t26 = std::pow(t21, 2) + std::pow(t23, 2) + std::pow(t25, 2);
const auto t27 = t2_x - t3_x;
const auto t28 = t2_y - t3_y;
const auto t29 = t2_z - t3_z;
const auto t30 = -t1*t27 - t28*t4 - t29*t7;
const auto t31 = t10*t30;
const auto t32 = t1*t31;
const auto t33 = t27 + t32;
const auto t34 = t31*t4;
const auto t35 = t28 + t34;
const auto t36 = t31*t7;
const auto t37 = t29 + t36;
const auto t38 = std::pow(t33, 2) + std::pow(t35, 2) + std::pow(t37, 2);
const auto t39 = std::pow(t26*t38, -1.0/2.0);
const auto t40 = -t35;
const auto t41 = t10*t4;
const auto t42 = t0*t41;
const auto t43 = -t37;
const auto t44 = t10*t7;
const auto t45 = t0*t44;
const auto t46 = t1*t10;
const auto t47 = t0*t46 + 1;
const auto t48 = -t33;
const auto t49 = -t23;
const auto t50 = -t25;
const auto t51 = -t21;
const auto t52 = 1.0/t26;
const auto t53 = t21*t33 + t23*t35 + t25*t37;
const auto t54 = t52*t53;
const auto t55 = t3*t46;
const auto t56 = t3*t44;
const auto t57 = t3*t41 + 1;
const auto t58 = t46*t6;
const auto t59 = t41*t6;
const auto t60 = t44*t6 + 1;
const auto t61 = t27 + 2*t32;
const auto t62 = t23*t41;
const auto t63 = t25*t44;
const auto t64 = 2*t20;
const auto t65 = t0_x - 2*t1_x + t2_x;
const auto t66 = t64 + t65;
const auto t67 = t41*t66;
const auto t68 = t44*t66;
const auto t69 = -t1*t27 - 2*t10*t2*t30 + t30;
const auto t70 = std::pow(t9, -2);
const auto t71 = t18*t70;
const auto t72 = -t19 - 1;
const auto t73 = 2*t2*t71 + t46*t65 + t72;
const auto t74 = 1.0/t38;
const auto t75 = t4*t40;
const auto t76 = t43*t7;
const auto t77 = std::pow(t49, 2) + std::pow(t50, 2) + std::pow(t51, 2);
const auto t78 = t10*t77;
const auto t79 = std::pow(t40, 2) + std::pow(t43, 2) + std::pow(t48, 2);
const auto t80 = t28 + 2*t34;
const auto t81 = t21*t46;
const auto t82 = 2*t22;
const auto t83 = t0_y - 2*t1_y + t2_y;
const auto t84 = t82 + t83;
const auto t85 = t46*t84;
const auto t86 = t44*t84;
const auto t87 = -2*t10*t30*t5 - t28*t4 + t30;
const auto t88 = t41*t83 + 2*t5*t71 + t72;
const auto t89 = t1*t48;
const auto t90 = t29 + 2*t36;
const auto t91 = 2*t24;
const auto t92 = t0_z - 2*t1_z + t2_z;
const auto t93 = t91 + t92;
const auto t94 = t46*t93;
const auto t95 = t41*t93;
const auto t96 = -2*t10*t30*t8 - t29*t7 + t30;
const auto t97 = t44*t92 + 2*t71*t8 + t72;
const auto t98 = 2*t0;
const auto t99 = -t0_x + t19*t98 + t1_x;
const auto t100 = t35*t41;
const auto t101 = t37*t44;
const auto t102 = t1_x - 2*t2_x + t3_x;
const auto t103 = t102 + t31*t98;
const auto t104 = t0*t64 + 2*t12 + t17;
const auto t105 = t30*t70;
const auto t106 = t31 + 1;
const auto t107 = t1*t105*t98 + t102*t46 + t106;
const auto t108 = t4*t49;
const auto t109 = t50*t7;
const auto t110 = t10*t79;
const auto t111 = t40*t41;
const auto t112 = t43*t44;
const auto t113 = t54*t74;
const auto t114 = 2*t3;
const auto t115 = -t0_y + t114*t19 + t1_y;
const auto t116 = t33*t46;
const auto t117 = t1_y - 2*t2_y + t3_y;
const auto t118 = t114*t31 + t117;
const auto t119 = t12 + 2*t14 + t16 + t3*t82;
const auto t120 = t105*t114*t4 + t106 + t117*t41;
const auto t121 = t1*t51;
const auto t122 = t46*t48;
const auto t123 = 2*t6;
const auto t124 = -t0_z + t123*t19 + t1_z;
const auto t125 = t1_z - 2*t2_z + t3_z;
const auto t126 = t123*t31 + t125;
const auto t127 = t12 + t14 + 2*t16 + t6*t91;
const auto t128 = t105*t123*t7 + t106 + t125*t44;
const auto t129 = -t10*t2 + 1;
const auto t130 = t53*t74;
const auto t131 = t46*t51;
const auto t132 = -t10*t5 + 1;
const auto t133 = -t10*t8 + 1;
dA[0] = t39*(t40*t42 + t43*t45 + t47*t48 - t54*(t42*t49 + t45*t50 + t47*t51));
dA[1] = t39*(t40*t57 + t43*t56 + t48*t55 - t54*(t49*t57 + t50*t56 + t51*t55));
dA[2] = t39*(t40*t59 + t43*t60 + t48*t58 - t54*(t49*t59 + t50*t60 + t51*t58));
dA[3] = t39*(t10*t21*t69 - t33*t73 - t35*t67 - t37*t68 + t52*t53*t74*(t78*(t48*t69 - t61*t75 - t61*t76) + t79*(-t49*t67 - t50*t68 - t51*t73)) - t61*t62 - t61*t63);
dA[4] = t39*(t10*t23*t87 - t33*t85 - t35*t88 - t37*t86 + t52*t53*t74*(t78*(t40*t87 - t76*t80 - t80*t89) + t79*(-t49*t88 - t50*t86 - t51*t85)) - t63*t80 - t80*t81);
dA[5] = t39*(t10*t25*t96 - t33*t94 - t35*t95 - t37*t97 + t52*t53*t74*(t78*(t43*t96 - t75*t90 - t89*t90) + t79*(-t49*t95 - t50*t97 - t51*t94)) - t62*t90 - t81*t90);
dA[6] = -t39*(t10*t104*t33 + t100*t99 + t101*t99 + t103*t62 + t103*t63 + t107*t21 + t113*(t110*(t104*t51 + t108*t99 + t109*t99) + t77*(t103*t111 + t103*t112 + t107*t48)));
dA[7] = -t39*(t10*t119*t35 + t101*t115 + t113*(t110*(t109*t115 + t115*t121 + t119*t49) + t77*(t112*t118 + t118*t122 + t120*t40)) + t115*t116 + t118*t63 + t118*t81 + t120*t23);
dA[8] = -t39*(t10*t127*t37 + t100*t124 + t113*(t110*(t108*t124 + t121*t124 + t127*t50) + t77*(t111*t126 + t122*t126 + t128*t43)) + t116*t124 + t126*t62 + t126*t81 + t128*t25);
dA[9] = t39*(t108*t46 + t109*t46 - t129*t51 - t130*(-t129*t48 + t46*t75 + t46*t76));
dA[10] = t39*(t109*t41 - t130*(t122*t4 - t132*t40 + t41*t76) + t131*t4 - t132*t49);
dA[11] = t39*(-t130*(t111*t7 + t122*t7 - t133*t43) + t131*t7 - t133*t50 + t41*t49*t7);
}


// dA is (144×1) flattened in column-major order
void curve_twist_hessian(double t0_x, double t0_y, double t0_z, double t1_x, double t1_y, double t1_z, double t2_x, double t2_y, double t2_z, double t3_x, double t3_y, double t3_z, double dA[144]){
const auto t0 = -t2_x;
const auto t1 = t0 + t1_x;
const auto t2 = std::pow(t1, 2);
const auto t3 = -t2_y;
const auto t4 = t1_y + t3;
const auto t5 = std::pow(t4, 2);
const auto t6 = -t2_z;
const auto t7 = t1_z + t6;
const auto t8 = std::pow(t7, 2);
const auto t9 = t2 + t5 + t8;
const auto t10 = 1.0/t9;
const auto t11 = t10*t2;
const auto t12 = t11 - 1;
const auto t13 = -t12;
const auto t14 = std::pow(t9, -2);
const auto t15 = t14*t2;
const auto t16 = t15*t5;
const auto t17 = t15*t8;
const auto t18 = t16 + t17;
const auto t19 = -t1_x;
const auto t20 = t0_x + t19;
const auto t21 = t1*t20;
const auto t22 = -t1_y;
const auto t23 = t0_y + t22;
const auto t24 = t23*t4;
const auto t25 = -t1_z;
const auto t26 = t0_z + t25;
const auto t27 = t26*t7;
const auto t28 = t24 + t27;
const auto t29 = t21 + t28;
const auto t30 = t10*t29;
const auto t31 = t1*t30;
const auto t32 = -t0_x;
const auto t33 = t1_x + t32;
const auto t34 = -t31 - t33;
const auto t35 = -t3_x;
const auto t36 = t2_x + t35;
const auto t37 = t1*t36;
const auto t38 = -t3_y;
const auto t39 = t2_y + t38;
const auto t40 = t39*t4;
const auto t41 = -t3_z;
const auto t42 = t2_z + t41;
const auto t43 = t42*t7;
const auto t44 = t40 + t43;
const auto t45 = t37 + t44;
const auto t46 = t10*t45;
const auto t47 = t1*t46;
const auto t48 = t0 + t3_x;
const auto t49 = -t47 - t48;
const auto t50 = t30*t4;
const auto t51 = -t0_y;
const auto t52 = t1_y + t51;
const auto t53 = -t50 - t52;
const auto t54 = t4*t46;
const auto t55 = t3 + t3_y;
const auto t56 = -t54 - t55;
const auto t57 = t30*t7;
const auto t58 = -t0_z;
const auto t59 = t1_z + t58;
const auto t60 = -t57 - t59;
const auto t61 = t46*t7;
const auto t62 = t3_z + t6;
const auto t63 = -t61 - t62;
const auto t64 = t34*t49 + t53*t56 + t60*t63;
const auto t65 = -t1;
const auto t66 = std::pow(t65, 2);
const auto t67 = -t4;
const auto t68 = std::pow(t67, 2);
const auto t69 = -t7;
const auto t70 = std::pow(t69, 2);
const auto t71 = t66 + t68 + t70;
const auto t72 = 1.0/t71;
const auto t73 = t65*t72;
const auto t74 = t1*t73;
const auto t75 = t74 + 1;
const auto t76 = -t36*t65 - t39*t67 - t42*t69;
const auto t77 = t72*t76;
const auto t78 = t65*t77;
const auto t79 = t36 + t78;
const auto t80 = t75*t79;
const auto t81 = t67*t77;
const auto t82 = t39 + t81;
const auto t83 = t10*t4;
const auto t84 = t82*t83;
const auto t85 = t69*t77;
const auto t86 = t42 + t85;
const auto t87 = t10*t7;
const auto t88 = t86*t87;
const auto t89 = t1*t84 + t1*t88;
const auto t90 = -t80 + t89;
const auto t91 = -t20;
const auto t92 = t65*t91;
const auto t93 = -t23;
const auto t94 = t67*t93;
const auto t95 = -t26;
const auto t96 = t69*t95;
const auto t97 = t94 + t96;
const auto t98 = t92 + t97;
const auto t99 = t72*t98;
const auto t100 = t65*t99;
const auto t101 = t100 + t20;
const auto t102 = t101*t75;
const auto t103 = t67*t99;
const auto t104 = t103 + t23;
const auto t105 = t104*t83;
const auto t106 = t69*t99;
const auto t107 = t106 + t26;
const auto t108 = t107*t87;
const auto t109 = t1*t105 + t1*t108;
const auto t110 = -t102 + t109;
const auto t111 = 2*t110;
const auto t112 = std::pow(t34, 2) + std::pow(t53, 2) + std::pow(t60, 2);
const auto t113 = 1.0/t112;
const auto t114 = t113*t64;
const auto t115 = t1*t83;
const auto t116 = t1*t87;
const auto t117 = std::pow(t49, 2) + std::pow(t56, 2) + std::pow(t63, 2);
const auto t118 = std::pow(t112*t117, -1.0/2.0);
const auto t119 = t113*t118;
const auto t120 = t101*t79 + t104*t82 + t107*t86;
const auto t121 = -t101;
const auto t122 = t4*t73;
const auto t123 = -t107;
const auto t124 = t69*t72;
const auto t125 = t124*t4;
const auto t126 = t67*t72;
const auto t127 = t126*t4;
const auto t128 = t127 + 1;
const auto t129 = -t104;
const auto t130 = t121*t122 + t123*t125 + t128*t129;
const auto t131 = std::pow(t101, 2) + std::pow(t104, 2) + std::pow(t107, 2);
const auto t132 = 1.0/t131;
const auto t133 = t130*t132;
const auto t134 = t120*t133;
const auto t135 = -t79;
const auto t136 = -t86;
const auto t137 = -t82;
const auto t138 = t128*t137;
const auto t139 = t122*t135 + t125*t136 + t138;
const auto t140 = t1*t126;
const auto t141 = t1*t124;
const auto t142 = t121*t75 + t123*t141 + t129*t140;
const auto t143 = t135*t75;
const auto t144 = t136*t141 + t137*t140 + t143;
const auto t145 = t70*t72;
const auto t146 = t1*t145;
const auto t147 = t65*t75;
const auto t148 = t128*t67;
const auto t149 = t120*t72;
const auto t150 = t132*t142;
const auto t151 = t120*t150;
const auto t152 = t130*t144 - t130*t151 + t139*t142 + t149*(t1*t148 + t146*t4 + t147*t4);
const auto t153 = std::pow(t79, 2) + std::pow(t82, 2) + std::pow(t86, 2);
const auto t154 = std::pow(t131*t153, -1.0/2.0);
const auto t155 = t132*t154;
const auto t156 = t7*t73;
const auto t157 = t126*t7;
const auto t158 = t124*t7;
const auto t159 = t158 + 1;
const auto t160 = t121*t156 + t123*t159 + t129*t157;
const auto t161 = t132*t160;
const auto t162 = t120*t161;
const auto t163 = t136*t159;
const auto t164 = t135*t156 + t137*t157 + t163;
const auto t165 = t68*t72;
const auto t166 = t1*t165;
const auto t167 = t159*t69;
const auto t168 = t142*t164 + t144*t160 + t149*(t1*t167 + t147*t7 + t166*t7) - t151*t160;
const auto t169 = 2*t78;
const auto t170 = t169 + t36;
const auto t171 = t115*t67;
const auto t172 = t116*t69;
const auto t173 = t66*t72;
const auto t174 = t1*t173 + t19 + t2_x;
const auto t175 = 2*t79;
const auto t176 = t174*t175;
const auto t177 = 2*t74;
const auto t178 = t177 + 1;
const auto t179 = t67*t82;
const auto t180 = t69*t86;
const auto t181 = -t36*t65 - 2*t66*t72*t76 + t76;
const auto t182 = -t181;
const auto t183 = t182*t75;
const auto t184 = t104*t126;
const auto t185 = t107*t124;
const auto t186 = 2*t100;
const auto t187 = 2*t1_x;
const auto t188 = t0_x - t187 + t2_x;
const auto t189 = t186 + t188;
const auto t190 = t126*t189;
const auto t191 = t124*t189;
const auto t192 = t182*t72;
const auto t193 = t188*t73;
const auto t194 = std::pow(t71, -2);
const auto t195 = t194*t66;
const auto t196 = 2*t98;
const auto t197 = -t99 - 1;
const auto t198 = t193 + t195*t196 + t197;
const auto t199 = t101*t192 + t170*t184 + t170*t185 + t190*t82 + t191*t86 + t198*t79;
const auto t200 = t137*t67;
const auto t201 = t136*t69;
const auto t202 = t135*t181 - t170*t200 - t170*t201;
const auto t203 = std::pow(t121, 2) + std::pow(t123, 2) + std::pow(t129, 2);
const auto t204 = t203*t72;
const auto t205 = std::pow(t135, 2) + std::pow(t136, 2) + std::pow(t137, 2);
const auto t206 = -t198;
const auto t207 = t121*t206 - t123*t191 - t129*t190;
const auto t208 = t202*t204 + t205*t207;
const auto t209 = 1.0/t153;
const auto t210 = t144*t209;
const auto t211 = t132*t210;
const auto t212 = std::pow(t131, -2);
const auto t213 = t120*t212;
const auto t214 = t209*t213;
const auto t215 = t111*t214;
const auto t216 = t142*t214;
const auto t217 = 2*t142;
const auto t218 = t202*t72;
const auto t219 = t194*t68;
const auto t220 = t1*t189;
const auto t221 = t194*t70;
const auto t222 = 2*t121;
const auto t223 = t222*t72;
const auto t224 = t126*t129;
const auto t225 = t123*t124;
const auto t226 = t132*t209;
const auto t227 = t120*t226;
const auto t228 = 2*t81;
const auto t229 = t228 + t39;
const auto t230 = 2*t135;
const auto t231 = t67*t74;
const auto t232 = 2*t1;
const auto t233 = t126*t201;
const auto t234 = 2*t165;
const auto t235 = t234 - 1;
const auto t236 = -t235;
const auto t237 = t137*t236;
const auto t238 = t147*t229;
const auto t239 = t39*t67;
const auto t240 = 2*t77;
const auto t241 = -t239 - t240*t68 + t76;
const auto t242 = t126*t241;
const auto t243 = t101*t73;
const auto t244 = 2*t103;
const auto t245 = 2*t1_y;
const auto t246 = t0_y - t245 + t2_y;
const auto t247 = t244 + t246;
const auto t248 = t247*t73;
const auto t249 = t124*t86;
const auto t250 = -t241;
const auto t251 = t250*t72;
const auto t252 = t126*t246;
const auto t253 = t196*t219 + t197 + t252;
const auto t254 = t104*t251 + t185*t229 + t229*t243 + t247*t249 + t248*t79 + t253*t82;
const auto t255 = t135*t65;
const auto t256 = t137*t241 - t201*t229 - t229*t255;
const auto t257 = -t253;
const auto t258 = -t121*t248 + t129*t257 - t225*t247;
const auto t259 = t204*t256 + t205*t258;
const auto t260 = t123*t69;
const auto t261 = t126*t260;
const auto t262 = t257*t67;
const auto t263 = t149*t226;
const auto t264 = 2*t85;
const auto t265 = t264 + t42;
const auto t266 = t69*t74;
const auto t267 = 2*t137;
const auto t268 = t140*t267;
const auto t269 = 2*t145;
const auto t270 = t269 - 1;
const auto t271 = -t270;
const auto t272 = t136*t271;
const auto t273 = t147*t265;
const auto t274 = t42*t69;
const auto t275 = -t240*t70 - t274 + t76;
const auto t276 = t124*t275;
const auto t277 = 2*t106;
const auto t278 = 2*t1_z;
const auto t279 = t0_z - t278 + t2_z;
const auto t280 = t277 + t279;
const auto t281 = t280*t73;
const auto t282 = t126*t82;
const auto t283 = -t275;
const auto t284 = t283*t72;
const auto t285 = t124*t279;
const auto t286 = t196*t221 + t197 + t285;
const auto t287 = t107*t284 + t184*t265 + t243*t265 + t280*t282 + t281*t79 + t286*t86;
const auto t288 = t136*t275 - t200*t265 - t255*t265;
const auto t289 = -t286;
const auto t290 = -t121*t281 + t123*t289 - t224*t280;
const auto t291 = t204*t288 + t205*t290;
const auto t292 = 2*t129;
const auto t293 = t140*t292;
const auto t294 = t289*t69;
const auto t295 = t1*t99;
const auto t296 = 2*t295;
const auto t297 = t296 + t33;
const auto t298 = t1*t77;
const auto t299 = 2*t298;
const auto t300 = 2*t2_x;
const auto t301 = t1_x - t300 + t3_x;
const auto t302 = t299 + t301;
const auto t303 = 2*t92;
const auto t304 = t100*t232 + t303 + t97;
const auto t305 = t304*t72;
const auto t306 = t301*t73;
const auto t307 = t194*t76;
const auto t308 = t307*t65;
const auto t309 = t77 + 1;
const auto t310 = t232*t308 + t306 + t309;
const auto t311 = t101*t310 + t184*t302 + t185*t302 + t249*t297 + t282*t297 + t305*t79;
const auto t312 = 2*t72;
const auto t313 = t2*t312 - 1;
const auto t314 = t129*t67;
const auto t315 = t121*t304 + t260*t297 + t297*t314;
const auto t316 = t205*t72;
const auto t317 = t126*t137;
const auto t318 = t124*t136;
const auto t319 = t135*t310 + t302*t317 + t302*t318;
const auto t320 = t203*t319 + t315*t316;
const auto t321 = -t313;
const auto t322 = t1 + t2*t73;
const auto t323 = t304*t75;
const auto t324 = t175*t72;
const auto t325 = t1*t10;
const auto t326 = t302*t325;
const auto t327 = t127*t326;
const auto t328 = t158*t326;
const auto t329 = t310*t75 + t322*t324 - t327 - t328;
const auto t330 = t240*t4;
const auto t331 = 2*t2_y;
const auto t332 = t1_y - t331 + t3_y;
const auto t333 = t330 + t332;
const auto t334 = t4*t74;
const auto t335 = 2*t136;
const auto t336 = t141*t335;
const auto t337 = 2*t127;
const auto t338 = t337 + 1;
const auto t339 = t137*t338;
const auto t340 = t147*t333;
const auto t341 = t126*t332;
const auto t342 = 2*t4;
const auto t343 = t307*t67;
const auto t344 = t309 + t341 + t342*t343;
const auto t345 = t344*t67;
const auto t346 = 2*t99;
const auto t347 = t346*t4;
const auto t348 = t347 + t52;
const auto t349 = t73*t79;
const auto t350 = 2*t94;
const auto t351 = t103*t342 + t350 + t92 + t96;
const auto t352 = t351*t72;
const auto t353 = t104*t344 + t185*t333 + t243*t333 + t249*t348 + t348*t349 + t352*t82;
const auto t354 = t121*t65;
const auto t355 = t129*t351 + t260*t348 + t348*t354;
const auto t356 = t135*t73;
const auto t357 = t137*t344 + t318*t333 + t333*t356;
const auto t358 = t203*t357 + t316*t355;
const auto t359 = 2*t123;
const auto t360 = t141*t359;
const auto t361 = t360*t4;
const auto t362 = t129*t338;
const auto t363 = t240*t7;
const auto t364 = 2*t2_z;
const auto t365 = t1_z - t364 + t3_z;
const auto t366 = t363 + t365;
const auto t367 = t7*t74;
const auto t368 = 2*t158;
const auto t369 = t368 + 1;
const auto t370 = t136*t369;
const auto t371 = t147*t366;
const auto t372 = t124*t365;
const auto t373 = 2*t7;
const auto t374 = t307*t69;
const auto t375 = t309 + t372 + t373*t374;
const auto t376 = t375*t69;
const auto t377 = t346*t7;
const auto t378 = t377 + t59;
const auto t379 = 2*t96;
const auto t380 = t106*t373 + t379 + t92 + t94;
const auto t381 = t380*t72;
const auto t382 = t107*t375 + t184*t366 + t243*t366 + t282*t378 + t349*t378 + t381*t86;
const auto t383 = t123*t380 + t314*t378 + t354*t378;
const auto t384 = t136*t375 + t317*t366 + t356*t366;
const auto t385 = t203*t384 + t316*t383;
const auto t386 = t293*t7;
const auto t387 = t123*t369;
const auto t388 = t173 - 1;
const auto t389 = t314*t73;
const auto t390 = t260*t73;
const auto t391 = -t388;
const auto t392 = -t121*t391 + t389 + t390;
const auto t393 = t200*t73;
const auto t394 = t201*t73;
const auto t395 = t135*t391;
const auto t396 = t393 + t394 - t395;
const auto t397 = t209*t396;
const auto t398 = t154*(t150*t392 - t151*t397 + t18 + t210*t396 - t388*t75);
const auto t399 = t165 - 1;
const auto t400 = -t399;
const auto t401 = t121*t73;
const auto t402 = -t129*t400 + t261 + t401*t67;
const auto t403 = t137*t400;
const auto t404 = t233 + t356*t67 - t403;
const auto t405 = t209*t404;
const auto t406 = t154*(t126*(t1*t400 - t146 - t147) + t150*t402 - t151*t405 + t210*t404);
const auto t407 = t145 - 1;
const auto t408 = -t407;
const auto t409 = -t123*t408 + t224*t69 + t401*t69;
const auto t410 = t136*t408;
const auto t411 = t317*t69 + t356*t69 - t410;
const auto t412 = t209*t411;
const auto t413 = t154*(t124*(t1*t408 - t147 - t166) + t150*t409 - t151*t412 + t210*t411);
const auto t414 = t104*t128;
const auto t415 = t101*t83;
const auto t416 = t107*t83;
const auto t417 = t1*t415 + t416*t7;
const auto t418 = -t414 + t417;
const auto t419 = 2*t418;
const auto t420 = t10*t5;
const auto t421 = t420 - 1;
const auto t422 = -t421;
const auto t423 = t14*t5;
const auto t424 = t423*t8;
const auto t425 = t16 + t424;
const auto t426 = t128*t82;
const auto t427 = t79*t83;
const auto t428 = t83*t86;
const auto t429 = t1*t427 + t428*t7;
const auto t430 = -t426 + t429;
const auto t431 = t7*t83;
const auto t432 = t173*t4;
const auto t433 = t130*t164 - t134*t160 + t139*t160 + t149*(t148*t7 + t167*t4 + t432*t7);
const auto t434 = t145*t4;
const auto t435 = 2*t173;
const auto t436 = t435 - 1;
const auto t437 = -t436;
const auto t438 = t135*t437;
const auto t439 = t148*t170;
const auto t440 = t181*t73;
const auto t441 = t139*t209;
const auto t442 = t132*t441;
const auto t443 = 2*t130;
const auto t444 = t206*t65;
const auto t445 = t214*t419;
const auto t446 = t130*t214;
const auto t447 = t115*t65;
const auto t448 = t431*t69;
const auto t449 = t165*t4 + t22 + t2_y;
const auto t450 = t449*t82;
const auto t451 = t65*t79;
const auto t452 = t128*t250;
const auto t453 = t247*t4;
const auto t454 = t129*t312;
const auto t455 = t148*t265;
const auto t456 = t123*t271;
const auto t457 = t122*t222;
const auto t458 = t148*t280;
const auto t459 = t148*t302;
const auto t460 = t310*t65;
const auto t461 = t148*t297;
const auto t462 = t121*t178;
const auto t463 = t304*t73;
const auto t464 = t312*t5 - 1;
const auto t465 = -t464;
const auto t466 = t126*t5 + t4;
const auto t467 = t128*t351;
const auto t468 = t312*t82;
const auto t469 = t333*t83;
const auto t470 = t469*t74;
const auto t471 = t158*t469;
const auto t472 = t128*t344 + t466*t468 - t470 - t471;
const auto t473 = -2*t135*t4*t65*t7*t72;
const auto t474 = t148*t366;
const auto t475 = -t457*t7;
const auto t476 = t148*t378;
const auto t477 = t124*t380;
const auto t478 = t154*(t133*t392 - t134*t397 + t396*t441 + t73*(-t148 + t391*t4 - t434));
const auto t479 = t154*(-t128*t399 + t133*t402 - t134*t405 + t404*t441 + t425);
const auto t480 = t154*(t124*(-t148 + t4*t408 - t432) + t133*t409 - t134*t412 + t411*t441);
const auto t481 = t107*t159;
const auto t482 = t101*t87;
const auto t483 = t1*t482 + t105*t7;
const auto t484 = -t481 + t483;
const auto t485 = 2*t484;
const auto t486 = t10*t8;
const auto t487 = t486 - 1;
const auto t488 = -t487;
const auto t489 = t17 + t424;
const auto t490 = t159*t86;
const auto t491 = t79*t87;
const auto t492 = t1*t491 + t7*t84;
const auto t493 = -t490 + t492;
const auto t494 = t165*t7;
const auto t495 = t167*t170;
const auto t496 = t164*t209;
const auto t497 = t132*t496;
const auto t498 = 2*t160;
const auto t499 = t214*t485;
const auto t500 = t160*t214;
const auto t501 = t173*t7;
const auto t502 = t167*t229;
const auto t503 = t129*t236;
const auto t504 = t156*t222;
const auto t505 = t167*t247;
const auto t506 = t116*t65;
const auto t507 = t431*t67;
const auto t508 = t145*t7 + t25 + t2_z;
const auto t509 = t508*t86;
const auto t510 = t159*t283;
const auto t511 = t280*t7;
const auto t512 = t123*t312;
const auto t513 = t167*t302;
const auto t514 = t167*t297;
const auto t515 = t167*t333;
const auto t516 = t167*t348;
const auto t517 = t126*t351;
const auto t518 = t312*t8 - 1;
const auto t519 = -t518;
const auto t520 = t124*t8 + t7;
const auto t521 = t159*t380;
const auto t522 = t312*t86;
const auto t523 = t366*t87;
const auto t524 = t523*t74;
const auto t525 = t127*t523;
const auto t526 = t159*t375 + t520*t522 - t524 - t525;
const auto t527 = t120*t397;
const auto t528 = t154*(t161*t392 - t161*t527 + t396*t496 + t73*(-t167 + t391*t7 - t494));
const auto t529 = t120*t405;
const auto t530 = t154*(t126*(-t167 + t400*t7 - t501) + t161*t402 - t161*t529 + t404*t496);
const auto t531 = t154*(-t159*t407 + t161*t409 - t162*t412 + t411*t496 + t489);
const auto t532 = t101*t312;
const auto t533 = t189*t325;
const auto t534 = t127*t533;
const auto t535 = t158*t533;
const auto t536 = -t105;
const auto t537 = -t108;
const auto t538 = t15*t342;
const auto t539 = t104*t538;
const auto t540 = t15*t373;
const auto t541 = t107*t540;
const auto t542 = t536 + t537 + t539 + t541;
const auto t543 = 2*t47;
const auto t544 = t48 + t543;
const auto t545 = t53*t83;
const auto t546 = t60*t87;
const auto t547 = 2*t31;
const auto t548 = -t188 + t547;
const auto t549 = t56*t83;
const auto t550 = t63*t87;
const auto t551 = t30 + 1;
const auto t552 = t10*t34*(2*t10*t2*t45 - 2*t37 - t44) + t49*(2*t14*t2*t29 - t188*t325 - t551) + t544*t545 + t544*t546 + t548*t549 + t548*t550;
const auto t553 = t113*t552;
const auto t554 = t170*t179 + t170*t180 + t182*t79;
const auto t555 = t131*t72;
const auto t556 = t101*t198 + t184*t189 + t185*t189;
const auto t557 = t153*t556 + t554*t555;
const auto t558 = 1.0/t117;
const auto t559 = t113*t558;
const auto t560 = t557*t559;
const auto t561 = std::pow(t112, -2);
const auto t562 = t170*t325;
const auto t563 = t127*t562;
const auto t564 = t158*t562;
const auto t565 = -t84;
const auto t566 = -t88;
const auto t567 = t538*t82;
const auto t568 = t540*t86;
const auto t569 = t565 + t566 + t567 + t568;
const auto t570 = t115*t198;
const auto t571 = t158*t83;
const auto t572 = t189*t571;
const auto t573 = t101*t538;
const auto t574 = t232*t423;
const auto t575 = t104*t574;
const auto t576 = t14*t4*t7;
const auto t577 = t232*t576;
const auto t578 = t107*t577;
const auto t579 = t573 + t575 + t578;
const auto t580 = -t415 + t579;
const auto t581 = t115*t192;
const auto t582 = t170*t571;
const auto t583 = t15*t175;
const auto t584 = t4*t583;
const auto t585 = t574*t82;
const auto t586 = t577*t86;
const auto t587 = t584 + t585 + t586;
const auto t588 = -t427 + t587;
const auto t589 = t116*t198;
const auto t590 = t127*t87;
const auto t591 = t189*t590;
const auto t592 = t101*t540;
const auto t593 = t14*t8;
const auto t594 = t232*t593;
const auto t595 = t107*t594;
const auto t596 = t104*t577;
const auto t597 = t592 + t595 + t596;
const auto t598 = -t482 + t597;
const auto t599 = t116*t192;
const auto t600 = t170*t590;
const auto t601 = t583*t7;
const auto t602 = t594*t86;
const auto t603 = t577*t82;
const auto t604 = t601 + t602 + t603;
const auto t605 = -t491 + t604;
const auto t606 = 2*t219;
const auto t607 = t170*t189;
const auto t608 = 2*t221;
const auto t609 = t104*t67;
const auto t610 = t36*t65;
const auto t611 = 2*t610;
const auto t612 = 4*t77;
const auto t613 = -t611 - t612*t66 + t76;
const auto t614 = 2*t194;
const auto t615 = -t613*t614;
const auto t616 = t107*t69;
const auto t617 = 4*t98;
const auto t618 = t195*t617;
const auto t619 = 2*t193 + t197 + t618;
const auto t620 = 2*t619;
const auto t621 = std::pow(t1, 3);
const auto t622 = 2*t11;
const auto t623 = 4*t14;
const auto t624 = t0_x - 3*t1_x + t300;
const auto t625 = std::pow(t153, -2);
const auto t626 = t213*t625;
const auto t627 = 2*t202;
const auto t628 = t132*t625;
const auto t629 = t149*t628;
const auto t630 = t627*t629;
const auto t631 = 2*t207;
const auto t632 = t214*t631;
const auto t633 = std::pow(t170, 2);
const auto t634 = t126*t267;
const auto t635 = t124*t335;
const auto t636 = std::pow(t65, 3);
const auto t637 = t194*t636;
const auto t638 = 4*t76;
const auto t639 = std::pow(t189, 2)*t194;
const auto t640 = -t619;
const auto t641 = t126*t292;
const auto t642 = t124*t359;
const auto t643 = t145*t247;
const auto t644 = t145*t229;
const auto t645 = t39*t65;
const auto t646 = t36*t67;
const auto t647 = 4*t67;
const auto t648 = t645 + t646 + t647*t78;
const auto t649 = 2*t185;
const auto t650 = t246*t65;
const auto t651 = t188*t67;
const auto t652 = t100*t647 + t650 + t651;
const auto t653 = 2*t249;
const auto t654 = t198*t65;
const auto t655 = t253*t67;
const auto t656 = 8*t76;
const auto t657 = t195*t67;
const auto t658 = -t228 + t55;
const auto t659 = t126*t611 + t39*t435 + t656*t657 + t658;
const auto t660 = 2*t239;
const auto t661 = t219*t65;
const auto t662 = -t169 + t48;
const auto t663 = t234*t36 + t656*t661 + t660*t73 + t662;
const auto t664 = 2*t73;
const auto t665 = 8*t98;
const auto t666 = -t244 + t245 + t3 + t51;
const auto t667 = t246*t435 + t651*t664 + t657*t665 + t666;
const auto t668 = 2*t126;
const auto t669 = t0 - t186 + t187 + t32;
const auto t670 = t188*t234 + t650*t668 + t661*t665 + t669;
const auto t671 = t208*t626;
const auto t672 = -t120*t132*t209*t72*(t203*(-t135*t659 - t137*t663 + t170*t229*t70*t72 - t170*t242 - t229*t440 - t635*t648) + t205*(-t121*t667 - t129*t670 + t189*t247*t70*t72 - t189*t262 - t247*t444 - t642*t652) + t256*t631 + t258*t627) - t132*t199*t209*t259 - t132*t208*t209*t254 + t259*t671 + t72*(t101*t659 + t104*t663 + t170*t643 + t170*t655 + t182*t248 + t189*t644 + t190*t250 + t229*t654 + t648*t649 + t652*t653 + t667*t79 + t670*t82);
const auto t673 = t165*t280;
const auto t674 = t165*t265;
const auto t675 = t42*t65;
const auto t676 = t36*t69;
const auto t677 = 4*t69;
const auto t678 = t675 + t676 + t677*t78;
const auto t679 = 2*t184;
const auto t680 = t279*t65;
const auto t681 = t188*t69;
const auto t682 = t100*t677 + t680 + t681;
const auto t683 = 2*t282;
const auto t684 = t286*t69;
const auto t685 = t195*t69;
const auto t686 = -t264 + t62;
const auto t687 = t124*t611 + t42*t435 + t656*t685 + t686;
const auto t688 = 2*t274;
const auto t689 = t221*t65;
const auto t690 = t269*t36 + t656*t689 + t662 + t688*t73;
const auto t691 = -t277 + t278 + t58 + t6;
const auto t692 = t279*t435 + t664*t681 + t665*t685 + t691;
const auto t693 = 2*t124;
const auto t694 = t188*t269 + t665*t689 + t669 + t680*t693;
const auto t695 = -t120*t132*t209*t72*(t203*(-t135*t687 - t136*t690 + t170*t265*t68*t72 - t170*t276 - t265*t440 - t634*t678) + t205*(-t121*t692 - t123*t694 + t189*t280*t68*t72 - t189*t294 - t280*t444 - t641*t682) + t288*t631 + t290*t627) - t132*t199*t209*t291 - t132*t208*t209*t287 + t291*t671 + t72*(t101*t687 + t107*t690 + t170*t673 + t170*t684 + t182*t281 + t189*t674 + t191*t283 + t265*t654 + t678*t679 + t682*t683 + t692*t79 + t694*t86);
const auto t696 = t19 + t300 + t35;
const auto t697 = t1*t195*t638 + t169 + t173*t301 - t298 + t37*t73 + t696;
const auto t698 = t1*t618 + t173*t91 + t188*t74 + t189 - t295;
const auto t699 = t170*t297;
const auto t700 = t189*t302;
const auto t701 = 8*t308;
const auto t702 = t1*t701;
const auto t703 = t240 + 1;
const auto t704 = 2*t306 + t312*t37 + t702 + t703;
const auto t705 = t303*t72;
const auto t706 = t1*t312;
const auto t707 = t1*t665;
const auto t708 = t194*t65;
const auto t709 = t707*t708;
const auto t710 = t346 + 1;
const auto t711 = t188*t706 + t705 + t709 + t710;
const auto t712 = t199*t226;
const auto t713 = t194*t627;
const auto t714 = t165*t189;
const auto t715 = t145*t189;
const auto t716 = t165*t170;
const auto t717 = t145*t170;
const auto t718 = -t120*t132*t209*(t203*t72*(-t181*t310 - t200*t704 - t201*t704 - t230*t697 + t302*t716 + t302*t717) + t205*t72*(-t206*t304 - t222*t698 - t260*t711 + t297*t714 + t297*t715 - t314*t711) - t315*t713 - t319*t631) - t120*t208*t212*t320*t625 - t132*t208*t209*t311 + t182*t194*t304 + t184*t704 + t185*t704 + t198*t310 + t219*t699 + t219*t700 + t221*t699 + t221*t700 + t249*t711 + t282*t711 + t320*t712 + t324*t698 + t532*t697;
const auto t719 = t145*t348;
const auto t720 = t145*t333;
const auto t721 = t332*t65;
const auto t722 = 4*t4;
const auto t723 = t36*t4 + t721 + t722*t78;
const auto t724 = t65*t93;
const auto t725 = t188*t4;
const auto t726 = t100*t722 + t724 + t725;
const auto t727 = t182*t73;
const auto t728 = t195*t4;
const auto t729 = t23 - t347;
const auto t730 = t435*t93 + t664*t725 + t665*t728 + t729;
const auto t731 = t312*t4;
const auto t732 = t22 + t331 + t38;
const auto t733 = -t330 + t732;
const auto t734 = t332*t435 + t610*t731 + t656*t728 + t733;
const auto t735 = t170 + t4*t67*t701 + t646*t731 + t668*t721;
const auto t736 = t4*t665;
const auto t737 = t189 + t350*t73 + t651*t731 + t67*t708*t736;
const auto t738 = t208*t226;
const auto t739 = t227*(t203*t72*(-t135*t734 - t137*t735 + t170*t333*t70*t72 + t170*t344*t67 - t333*t440 - t635*t723) + t205*t72*(-t121*t730 - t129*t737 + t189*t348*t70*t72 + t189*t351*t67*t72 - t348*t444 - t642*t726) - t355*t713 - t357*t631) + t353*t738 + t358*t671 - t358*t712 - t72*(t101*t734 + t104*t735 + t170*t517 + t170*t719 + t189*t345 + t189*t720 + t333*t654 + t348*t727 + t649*t723 + t653*t726 + t730*t79 + t737*t82);
const auto t740 = t165*t378;
const auto t741 = t165*t366;
const auto t742 = t365*t65;
const auto t743 = 4*t7;
const auto t744 = t36*t7 + t742 + t743*t78;
const auto t745 = t65*t95;
const auto t746 = t188*t7;
const auto t747 = t100*t743 + t745 + t746;
const auto t748 = t195*t7;
const auto t749 = t26 - t377;
const auto t750 = t435*t95 + t664*t746 + t665*t748 + t749;
const auto t751 = t312*t7;
const auto t752 = t25 + t364 + t41;
const auto t753 = -t363 + t752;
const auto t754 = t365*t435 + t610*t751 + t656*t748 + t753;
const auto t755 = t69*t7;
const auto t756 = t170 + t676*t751 + t693*t742 + t701*t755;
const auto t757 = t665*t7;
const auto t758 = t69*t757;
const auto t759 = t189 + t379*t73 + t681*t751 + t708*t758;
const auto t760 = t227*(t203*t72*(-t135*t754 - t136*t756 + t170*t366*t68*t72 + t170*t375*t69 - t366*t440 - t634*t744) + t205*t72*(-t121*t750 - t123*t759 + t189*t378*t68*t72 + t189*t380*t69*t72 - t378*t444 - t641*t747) - t383*t713 - t384*t631) + t382*t738 + t385*t671 - t385*t712 - t72*(t101*t754 + t107*t756 + t170*t477 + t170*t740 + t189*t376 + t189*t741 + t366*t654 + t378*t727 + t679*t744 + t683*t747 + t750*t79 + t759*t86);
const auto t761 = t1 + t636*t72;
const auto t762 = -t567 - t568 + t84 + t88;
const auto t763 = t558*t64;
const auto t764 = t388*t79 + t89;
const auto t765 = t101*t388;
const auto t766 = t109 + t765;
const auto t767 = std::pow(t117, -2);
const auto t768 = t114*t767;
const auto t769 = t557*t768;
const auto t770 = t64*t767;
const auto t771 = t312*t770;
const auto t772 = t554*t771;
const auto t773 = t198*t388 + t534 + t535;
const auto t774 = t126*t399;
const auto t775 = -t584 - t585 - t586;
const auto t776 = t399*t82 + t429;
const auto t777 = t104*t399;
const auto t778 = t417 + t777;
const auto t779 = -t573 - t575 - t578;
const auto t780 = t124*t407;
const auto t781 = -t601 - t602 - t603;
const auto t782 = t407*t86 + t492;
const auto t783 = t107*t407;
const auto t784 = t483 + t783;
const auto t785 = -t592 - t595 - t596;
const auto t786 = t115*t253;
const auto t787 = t158*t325;
const auto t788 = t247*t787;
const auto t789 = t104*t325;
const auto t790 = t579 - t789;
const auto t791 = 2*t54;
const auto t792 = t55 + t791;
const auto t793 = t325*t34;
const auto t794 = 2*t50;
const auto t795 = -t246 + t794;
const auto t796 = t325*t49;
const auto t797 = t10*t53*(2*t10*t45*t5 - t37 - 2*t40 - t43) + t546*t792 + t550*t795 + t56*(2*t14*t29*t5 - t246*t83 - t551) + t792*t793 + t795*t796;
const auto t798 = t113*t797;
const auto t799 = t180*t229 + t229*t451 + t250*t82;
const auto t800 = t104*t253 + t185*t247 + t243*t247;
const auto t801 = t153*t800 + t555*t799;
const auto t802 = t559*t801;
const auto t803 = t115*t251;
const auto t804 = t229*t787;
const auto t805 = t325*t82;
const auto t806 = t587 - t805;
const auto t807 = t104*t312;
const auto t808 = t101*t325;
const auto t809 = t101*t574;
const auto t810 = t373*t423;
const auto t811 = t107*t810;
const auto t812 = t74*t83;
const auto t813 = t247*t571 + t247*t812;
const auto t814 = t108 + t808 - t809 - t811 + t813;
const auto t815 = t229*t812;
const auto t816 = t229*t571;
const auto t817 = t325*t79;
const auto t818 = -t817;
const auto t819 = t1*t175;
const auto t820 = t423*t819;
const auto t821 = t810*t86;
const auto t822 = t566 + t818 + t820 + t821;
const auto t823 = t253*t431;
const auto t824 = t74*t87;
const auto t825 = t247*t824;
const auto t826 = t104*t87;
const auto t827 = t104*t810;
const auto t828 = t342*t593;
const auto t829 = t107*t828;
const auto t830 = t101*t577;
const auto t831 = t827 + t829 + t830;
const auto t832 = -t826 + t831;
const auto t833 = t251*t431;
const auto t834 = t229*t824;
const auto t835 = t82*t87;
const auto t836 = t810*t82;
const auto t837 = t828*t86;
const auto t838 = t576*t819;
const auto t839 = t836 + t837 + t838;
const auto t840 = -t835 + t839;
const auto t841 = 2*t258;
const auto t842 = t214*t841;
const auto t843 = 2*t629;
const auto t844 = t256*t843;
const auto t845 = t229*t247;
const auto t846 = 2*t195;
const auto t847 = t101*t65;
const auto t848 = -t612*t68 - t660 + t76;
const auto t849 = -t614*t848;
const auto t850 = t219*t617;
const auto t851 = t197 + 2*t252 + t850;
const auto t852 = t175*t73;
const auto t853 = std::pow(t4, 3);
const auto t854 = 2*t420;
const auto t855 = t0_y - 3*t1_y + t331;
const auto t856 = 4*t72;
const auto t857 = std::pow(t229, 2);
const auto t858 = t230*t73;
const auto t859 = std::pow(t67, 3);
const auto t860 = t194*t859;
const auto t861 = t194*std::pow(t247, 2);
const auto t862 = -t851;
const auto t863 = t222*t73;
const auto t864 = t173*t280;
const auto t865 = t173*t265;
const auto t866 = t42*t67;
const auto t867 = t39*t69;
const auto t868 = t677*t81 + t866 + t867;
const auto t869 = 2*t243;
const auto t870 = t279*t67;
const auto t871 = t246*t69;
const auto t872 = t103*t677 + t870 + t871;
const auto t873 = t126*t250;
const auto t874 = t124*t283;
const auto t875 = t219*t69;
const auto t876 = t124*t660 + t234*t42 + t656*t875 + t686;
const auto t877 = t221*t67;
const auto t878 = t126*t688 + t269*t39 + t656*t877 + t658;
const auto t879 = t234*t279 + t665*t875 + t668*t871 + t691;
const auto t880 = t246*t269 + t665*t877 + t666 + t693*t870;
const auto t881 = t259*t626;
const auto t882 = 2*t290;
const auto t883 = -t120*t132*t209*t72*(t203*(-t136*t878 - t137*t876 - t229*t276 + t229*t865 - t242*t265 - t858*t868) + t205*(-t123*t880 - t129*t879 - t247*t294 + t247*t864 - t262*t280 - t863*t872) + t256*t882 + t288*t841) - t132*t209*t254*t291 - t132*t209*t259*t287 + t291*t881 + t72*(t104*t876 + t107*t878 + t229*t684 + t229*t864 + t247*t865 + t247*t874 + t265*t655 + t280*t873 + t82*t879 + t852*t872 + t86*t880 + t868*t869);
const auto t884 = t301*t67;
const auto t885 = 4*t1;
const auto t886 = t1*t39 + t81*t885 + t884;
const auto t887 = t67*t91;
const auto t888 = t1*t246;
const auto t889 = t103*t885 + t887 + t888;
const auto t890 = t20 - t296;
const auto t891 = t219*t707 + t234*t91 + t668*t888 + t890;
const auto t892 = t219*t656;
const auto t893 = -t299 + t696;
const auto t894 = t1*t892 + t234*t301 + t239*t706 + t893;
const auto t895 = t229 + t645*t706 + t664*t884 + t67*t702;
const auto t896 = t126*t303 + t247 + t650*t706 + t67*t709;
const auto t897 = t226*t259;
const auto t898 = 2*t315;
const auto t899 = t194*t256;
const auto t900 = t226*t254;
const auto t901 = t227*(t203*t72*(-t135*t895 - t137*t894 + t229*t460 - t242*t302 + t302*t644 - t635*t886) + t205*t72*(-t121*t896 - t129*t891 + t247*t463 - t262*t297 + t297*t643 - t642*t889) - t319*t841 - t898*t899) + t311*t897 + t320*t881 - t320*t900 - t72*(t101*t895 + t104*t894 + t229*t463 + t247*t460 + t297*t644 + t297*t873 + t302*t643 + t302*t655 + t649*t886 + t653*t889 + t79*t896 + t82*t891);
const auto t902 = t4*t77;
const auto t903 = t126*t40 + t165*t332 + t219*t4*t638 + t228 + t732 - t902;
const auto t904 = t4*t99;
const auto t905 = t127*t246 + t165*t93 + t247 + t4*t850 - t904;
const auto t906 = t229*t348;
const auto t907 = t247*t333;
const auto t908 = 8*t343;
const auto t909 = t4*t908;
const auto t910 = t312*t40 + 2*t341 + t703 + t909;
const auto t911 = t312*t94;
const auto t912 = t194*t67;
const auto t913 = t736*t912;
const auto t914 = t246*t731 + t710 + t911 + t913;
const auto t915 = 2*t355;
const auto t916 = t173*t247;
const auto t917 = t173*t229;
const auto t918 = -t120*t132*t209*(t203*t72*(-t201*t910 - t241*t344 - t255*t910 - t267*t903 + t333*t644 + t333*t917) + t205*t72*(-t257*t351 - t260*t914 - t292*t905 + t348*t643 + t348*t916 - t354*t914) - t357*t841 - t899*t915) - t120*t212*t259*t358*t625 - t132*t209*t259*t353 + t185*t910 + t194*t250*t351 + t195*t906 + t195*t907 + t221*t906 + t221*t907 + t243*t910 + t249*t914 + t253*t344 + t349*t914 + t358*t900 + t468*t905 + t807*t903;
const auto t919 = t173*t378;
const auto t920 = t173*t366;
const auto t921 = t365*t67;
const auto t922 = t39*t7 + t743*t81 + t921;
const auto t923 = t67*t95;
const auto t924 = t246*t7;
const auto t925 = t103*t743 + t923 + t924;
const auto t926 = t219*t757 + t234*t95 + t668*t924 + t749;
const auto t927 = t234*t365 + t239*t751 + t7*t892 + t753;
const auto t928 = t229 + t693*t921 + t751*t867 + t755*t908;
const auto t929 = t126*t379 + t247 + t751*t871 + t758*t912;
const auto t930 = 2*t383;
const auto t931 = t227*(t203*t72*(-t136*t928 - t137*t927 + t229*t376 + t229*t920 - t242*t366 - t858*t922) + t205*t72*(-t123*t929 - t129*t926 + t247*t477 + t247*t919 - t262*t378 - t863*t925) - t384*t841 - t899*t930) + t382*t897 + t385*t881 - t385*t900 - t72*(t104*t927 + t107*t928 + t229*t477 + t229*t919 + t247*t376 + t247*t920 + t366*t655 + t378*t873 + t82*t926 + t852*t925 + t86*t929 + t869*t922);
const auto t932 = t388*t73;
const auto t933 = t768*t801;
const auto t934 = t771*t799;
const auto t935 = t253*t399;
const auto t936 = t4 + t72*t859;
const auto t937 = t817 - t820 - t821 + t88;
const auto t938 = -t836 - t837 - t838;
const auto t939 = -t827 - t829 - t830;
const auto t940 = t116*t286;
const auto t941 = t127*t325;
const auto t942 = t280*t941;
const auto t943 = t107*t325;
const auto t944 = t597 - t943;
const auto t945 = 2*t61;
const auto t946 = t62 + t945;
const auto t947 = 2*t57;
const auto t948 = -t279 + t947;
const auto t949 = t10*t60*(2*t10*t45*t8 - t37 - t40 - 2*t43) + t545*t946 + t549*t948 + t63*(2*t14*t29*t8 - t279*t87 - t551) + t793*t946 + t796*t948;
const auto t950 = t113*t949;
const auto t951 = t179*t265 + t265*t451 + t283*t86;
const auto t952 = t107*t286 + t184*t280 + t243*t280;
const auto t953 = t153*t952 + t555*t951;
const auto t954 = t559*t953;
const auto t955 = t116*t284;
const auto t956 = t265*t941;
const auto t957 = t325*t86;
const auto t958 = t604 - t957;
const auto t959 = t286*t431;
const auto t960 = t280*t812;
const auto t961 = -t416 + t831;
const auto t962 = t284*t431;
const auto t963 = t265*t812;
const auto t964 = -t428 + t839;
const auto t965 = t107*t312;
const auto t966 = t101*t594;
const auto t967 = t104*t828;
const auto t968 = t280*t590 + t280*t824;
const auto t969 = t105 + t808 - t966 - t967 + t968;
const auto t970 = t265*t824;
const auto t971 = t265*t590;
const auto t972 = t593*t819;
const auto t973 = t82*t828;
const auto t974 = t565 + t818 + t972 + t973;
const auto t975 = t214*t882;
const auto t976 = t288*t843;
const auto t977 = t265*t280;
const auto t978 = -t612*t70 - t688 + t76;
const auto t979 = -t614*t978;
const auto t980 = t221*t617;
const auto t981 = t197 + 2*t285 + t980;
const auto t982 = std::pow(t7, 3);
const auto t983 = 2*t486;
const auto t984 = t0_z - 3*t1_z + t364;
const auto t985 = std::pow(t265, 2);
const auto t986 = std::pow(t69, 3);
const auto t987 = t194*t986;
const auto t988 = t194*std::pow(t280, 2);
const auto t989 = -t981;
const auto t990 = t301*t69;
const auto t991 = t1*t42 + t85*t885 + t990;
const auto t992 = t69*t91;
const auto t993 = t1*t279;
const auto t994 = t106*t885 + t992 + t993;
const auto t995 = t221*t707 + t269*t91 + t693*t993 + t890;
const auto t996 = t221*t656;
const auto t997 = t1*t996 + t269*t301 + t274*t706 + t893;
const auto t998 = t265 + t664*t990 + t675*t706 + t69*t702;
const auto t999 = t124*t303 + t280 + t680*t706 + t69*t709;
const auto t1000 = t226*t291;
const auto t1001 = t194*t288;
const auto t1002 = t226*t287;
const auto t1003 = t291*t626;
const auto t1004 = t1000*t311 - t1002*t320 + t1003*t320 + t227*(-t1001*t898 + t203*t72*(-t135*t998 - t136*t997 + t265*t460 - t276*t302 + t302*t674 - t634*t991) + t205*t72*(-t121*t999 - t123*t995 + t280*t463 - t294*t297 + t297*t673 - t641*t994) - t319*t882) - t72*(t101*t998 + t107*t997 + t265*t463 + t280*t460 + t297*t674 + t297*t874 + t302*t673 + t302*t684 + t679*t991 + t683*t994 + t79*t999 + t86*t995);
const auto t1005 = t332*t69;
const auto t1006 = t1005 + t4*t42 + t722*t85;
const auto t1007 = t69*t93;
const auto t1008 = t279*t4;
const auto t1009 = t1007 + t1008 + t106*t722;
const auto t1010 = t1008*t693 + t221*t736 + t269*t93 + t729;
const auto t1011 = t269*t332 + t274*t731 + t4*t996 + t733;
const auto t1012 = t1005*t668 + t265 + t69*t909 + t731*t866;
const auto t1013 = t124*t350 + t280 + t69*t913 + t731*t870;
const auto t1014 = t1000*t353 - t1002*t358 + t1003*t358 + t227*(-t1001*t915 + t203*t72*(-t1006*t858 - t1011*t136 - t1012*t137 + t265*t345 - t276*t333 + t333*t865) + t205*t72*(-t1009*t863 - t1010*t123 - t1013*t129 + t280*t517 - t294*t348 + t348*t864) - t357*t882) - t72*(t1006*t869 + t1009*t852 + t1010*t86 + t1011*t107 + t1012*t104 + t1013*t82 + t265*t517 + t280*t345 + t333*t684 + t333*t864 + t348*t865 + t348*t874);
const auto t1015 = t124*t43 + t145*t365 + t221*t638*t7 + t264 - t7*t77 + t752;
const auto t1016 = t145*t95 + t158*t279 + t280 + t7*t980 - t7*t99;
const auto t1017 = t265*t378;
const auto t1018 = t280*t366;
const auto t1019 = 8*t374*t7;
const auto t1020 = t1019 + t312*t43 + 2*t372 + t703;
const auto t1021 = t312*t96;
const auto t1022 = t194*t69;
const auto t1023 = t1021 + t1022*t757 + t279*t751 + t710;
const auto t1024 = t1002*t385 + t1015*t965 + t1016*t522 + t1017*t195 + t1017*t219 + t1018*t195 + t1018*t219 + t1020*t184 + t1020*t243 + t1023*t282 + t1023*t349 - t120*t132*t209*(-t1001*t930 + t203*t72*(-t1015*t335 - t1020*t200 - t1020*t255 - t275*t375 + t366*t674 + t366*t865) + t205*t72*(-t1016*t359 - t1023*t314 - t1023*t354 - t289*t380 + t378*t673 + t378*t864) - t384*t882) - t120*t212*t291*t385*t625 - t132*t209*t291*t382 + t194*t283*t380 + t286*t375;
const auto t1025 = t768*t953;
const auto t1026 = t771*t951;
const auto t1027 = t286*t407;
const auto t1028 = t7 + t72*t986;
const auto t1029 = t817 + t84 - t972 - t973;
const auto t1030 = t297*t787 + t297*t941 + t542;
const auto t1031 = -t33 - t547;
const auto t1032 = t301 + t543;
const auto t1033 = 2*t29;
const auto t1034 = 2*t45;
const auto t1035 = -t10*t45 - 1;
const auto t1036 = t10*t49*(-t1033*t11 + 2*t21 + t28) + t1031*t549 + t1031*t550 - t1032*t545 - t1032*t546 + t34*(-t1034*t15 - t1035 - t301*t325);
const auto t1037 = t1036*t113;
const auto t1038 = t101*t304 + t297*t609 + t297*t616;
const auto t1039 = t153*t72;
const auto t1040 = t249*t302 + t282*t302 + t310*t79;
const auto t1041 = t1038*t1039 + t1040*t131;
const auto t1042 = t1041*t559;
const auto t1043 = t115*t305 + t297*t571 + t580;
const auto t1044 = t561*t763;
const auto t1045 = t1041*t1044;
const auto t1046 = t561*t64;
const auto t1047 = t1038*t1046*t312;
const auto t1048 = t115*t310 + t302*t571 + t588;
const auto t1049 = t116*t305 + t297*t590 + t598;
const auto t1050 = t116*t310 + t302*t590 + t605;
const auto t1051 = 2*t319;
const auto t1052 = t120*t628;
const auto t1053 = t1051*t1052;
const auto t1054 = t149*t209*t212;
const auto t1055 = t1054*t898;
const auto t1056 = t165*t297;
const auto t1057 = t145*t297;
const auto t1058 = -2*t1*t91 - 4*t2*t72*t98 + t98;
const auto t1059 = -t1058;
const auto t1060 = t194*t638;
const auto t1061 = t1060*t2;
const auto t1062 = -t77 - 1;
const auto t1063 = t1061 + t1062 + t301*t706;
const auto t1064 = t1*t705 - t100 + t2*t617*t708 + t297;
const auto t1065 = t1061*t65 + t177*t301 + t187 + t299 - 3*t2_x + t3_x - t78;
const auto t1066 = t226*t320;
const auto t1067 = std::pow(t297, 2);
const auto t1068 = t194*std::pow(t302, 2);
const auto t1069 = -t1063;
const auto t1070 = t1*t93 + t295*t722 + t4*t91;
const auto t1071 = t1*t332;
const auto t1072 = t301*t4;
const auto t1073 = t1071 + t1072 + t298*t722;
const auto t1074 = t348 + t4*t705 + t4*t709 + t706*t724;
const auto t1075 = t297 + t4*t707*t912 + t706*t94 + t731*t887;
const auto t1076 = t1072*t664 + t333 + t4*t702 + t706*t721;
const auto t1077 = t1*t909 + t1071*t668 + t302 + t731*t884;
const auto t1078 = t226*t311;
const auto t1079 = t320*t626;
const auto t1080 = t1066*t353 + t1078*t358 + t1079*t358 - t120*t132*t209*t72*(t1051*t355 + t203*(-t1073*t635 - t1076*t135 - t1077*t137 + t302*t345 + t302*t720 + t333*t460) + t205*(-t1070*t642 - t1074*t121 - t1075*t129 + t297*t517 + t297*t719 + t348*t463) + t357*t898) + t72*(t101*t1076 + t104*t1077 + t1070*t653 + t1073*t649 + t1074*t79 + t1075*t82 + t297*t345 + t297*t720 + t302*t517 + t302*t719 + t333*t463 + t348*t460);
const auto t1081 = t1*t95 + t295*t743 + t7*t91;
const auto t1082 = t1*t365;
const auto t1083 = t301*t7;
const auto t1084 = t1082 + t1083 + t298*t743;
const auto t1085 = t378 + t7*t705 + t7*t709 + t706*t745;
const auto t1086 = t1022*t7;
const auto t1087 = t1086*t707 + t297 + t706*t96 + t751*t992;
const auto t1088 = t1083*t664 + t366 + t7*t702 + t706*t742;
const auto t1089 = t1*t1019 + t1082*t693 + t302 + t751*t990;
const auto t1090 = t1066*t382 + t1078*t385 + t1079*t385 - t120*t132*t209*t72*(t1051*t383 + t203*(-t1084*t634 - t1088*t135 - t1089*t136 + t302*t376 + t302*t741 + t366*t460) + t205*(-t1081*t641 - t1085*t121 - t1087*t123 + t297*t477 + t297*t740 + t378*t463) + t384*t898) + t72*(t101*t1088 + t107*t1089 + t1081*t683 + t1084*t679 + t1085*t79 + t1087*t86 + t297*t376 + t297*t741 + t302*t477 + t302*t740 + t366*t463 + t378*t460);
const auto t1091 = t304*t388;
const auto t1092 = 2*t764;
const auto t1093 = t1040*t770;
const auto t1094 = t1041*t768;
const auto t1095 = 2*t776;
const auto t1096 = 2*t782;
const auto t1097 = t115*t352 + t348*t787 + t790;
const auto t1098 = -t52 - t794;
const auto t1099 = t332 + t791;
const auto t1100 = t10*t56*(-t1033*t420 + t21 + 2*t24 + t27) + t1098*t550 + t1098*t796 - t1099*t546 - t1099*t793 + t53*(-t1034*t423 - t1035 - t332*t83);
const auto t1101 = t1100*t113;
const auto t1102 = t104*t351 + t348*t616 + t348*t847;
const auto t1103 = t249*t333 + t333*t349 + t344*t82;
const auto t1104 = t1039*t1102 + t1103*t131;
const auto t1105 = t1104*t559;
const auto t1106 = t1044*t1104;
const auto t1107 = t111*t72;
const auto t1108 = t1046*t1102;
const auto t1109 = t115*t344 + t333*t787 + t806;
const auto t1110 = -t808;
const auto t1111 = t1110 + t348*t571 + t348*t812 + t537 + t809 + t811;
const auto t1112 = t348*t824 + t352*t431 + t832;
const auto t1113 = t333*t824 + t344*t431 + t840;
const auto t1114 = 2*t1052;
const auto t1115 = t1114*t357;
const auto t1116 = t1054*t915;
const auto t1117 = t173*t348;
const auto t1118 = -2*t4*t93 - 4*t5*t72*t98 + t98;
const auto t1119 = -t1118;
const auto t1120 = t1060*t5;
const auto t1121 = t1062 + t1120 + t332*t731;
const auto t1122 = -t103 + t348 + t4*t911 + t5*t617*t912;
const auto t1123 = t1120*t67 + t245 - 3*t2_y + t330 + t332*t337 + t3_y - t81;
const auto t1124 = t226*t358;
const auto t1125 = std::pow(t348, 2);
const auto t1126 = t194*std::pow(t333, 2);
const auto t1127 = -t1121;
const auto t1128 = t4*t95 + t7*t93 + t743*t904;
const auto t1129 = t365*t4;
const auto t1130 = t332*t7;
const auto t1131 = t1129 + t1130 + t743*t902;
const auto t1132 = t378 + t7*t911 + t7*t913 + t731*t923;
const auto t1133 = t1007*t751 + t1086*t736 + t348 + t731*t96;
const auto t1134 = t1130*t668 + t366 + t7*t909 + t731*t921;
const auto t1135 = t1005*t751 + t1019*t4 + t1129*t693 + t333;
const auto t1136 = t226*t385;
const auto t1137 = t1124*t382 + t1136*t353 - t120*t132*t209*t72*(t203*(-t1131*t858 - t1134*t137 - t1135*t136 + t333*t376 + t333*t920 + t345*t366) + t205*(-t1128*t863 - t1132*t129 - t1133*t123 + t348*t477 + t348*t919 + t378*t517) + t357*t930 + t384*t915) + t358*t385*t626 + t72*(t104*t1134 + t107*t1135 + t1128*t852 + t1131*t869 + t1132*t82 + t1133*t86 + t333*t477 + t333*t919 + t345*t378 + t348*t376 + t348*t920 + t366*t517);
const auto t1138 = t1103*t770;
const auto t1139 = t1104*t768;
const auto t1140 = t351*t399;
const auto t1141 = t116*t381 + t378*t941 + t944;
const auto t1142 = -t59 - t947;
const auto t1143 = t365 + t945;
const auto t1144 = t10*t63*(-t1033*t486 + t21 + t24 + 2*t27) + t1142*t549 + t1142*t796 - t1143*t545 - t1143*t793 + t60*(-t1034*t593 - t1035 - t365*t87);
const auto t1145 = t113*t1144;
const auto t1146 = t107*t380 + t378*t609 + t378*t847;
const auto t1147 = t282*t366 + t349*t366 + t375*t86;
const auto t1148 = t1039*t1146 + t1147*t131;
const auto t1149 = t1148*t559;
const auto t1150 = t1044*t1148;
const auto t1151 = t1046*t1146;
const auto t1152 = t116*t375 + t366*t941 + t958;
const auto t1153 = t378*t812 + t381*t431 + t961;
const auto t1154 = t366*t812 + t375*t431 + t964;
const auto t1155 = t1110 + t378*t590 + t378*t824 + t536 + t966 + t967;
const auto t1156 = t1114*t384;
const auto t1157 = t1054*t930;
const auto t1158 = -2*t7*t95 - 4*t72*t8*t98 + t98;
const auto t1159 = -t1158;
const auto t1160 = t1060*t8;
const auto t1161 = t1062 + t1160 + t365*t751;
const auto t1162 = t1021*t7 + t1022*t617*t8 - t106 + t378;
const auto t1163 = t1160*t69 + t278 - 3*t2_z + t363 + t365*t368 + t3_z - t85;
const auto t1164 = std::pow(t378, 2);
const auto t1165 = t194*std::pow(t366, 2);
const auto t1166 = -t1161;
const auto t1167 = t1147*t770;
const auto t1168 = t1148*t768;
const auto t1169 = t380*t407;
const auto t1170 = -t396;
const auto t1171 = 2*t65;
const auto t1172 = t1052*t1092;
const auto t1173 = 2*t261 + t262 - t503 + t67*t863;
const auto t1174 = -t137*t236 + 2*t233 + t242 + t67*t858;
const auto t1175 = t203*t73;
const auto t1176 = t1052*t396;
const auto t1177 = t294 - t456 + t641*t69 + t69*t863;
const auto t1178 = -t136*t271 + t276 + t634*t69 + t69*t858;
const auto t1179 = t125*t359 + t362 + t457 - t517;
const auto t1180 = t122*t230 + t125*t335 + t339 - t344*t67;
const auto t1181 = t203*t65;
const auto t1182 = t157*t292 + t387 - t477 + t504;
const auto t1183 = t156*t230 + t157*t267 + t370 - t375*t69;
const auto t1184 = 3*t763;
const auto t1185 = t118*t558;
const auto t1186 = t120*(-t145 - t165 - t173 + 2);
const auto t1187 = t1186*t73;
const auto t1188 = -t1187*t67 + t392*t404 + t396*t402 - t404*t527;
const auto t1189 = t154*t209;
const auto t1190 = t120*t412;
const auto t1191 = -t1187*t69 + t392*t411 + t396*t409 - t411*t527;
const auto t1192 = -t121*t437 + 2*t389 + 2*t390 + t444;
const auto t1193 = -t404;
const auto t1194 = -t135*t437 + 2*t393 + 2*t394 + t440;
const auto t1195 = t126*t203;
const auto t1196 = t1052*t1095;
const auto t1197 = t1052*t404;
const auto t1198 = 2*t67;
const auto t1199 = t293 + t360 + t462 - t463;
const auto t1200 = t135*t178 + t268 - t310*t65 + t336;
const auto t1201 = t203*t67;
const auto t1202 = -t1186*t126*t69 + t402*t411 + t404*t409 - t411*t529;
const auto t1203 = -t411;
const auto t1204 = t124*t203;
const auto t1205 = t1052*t1096;
const auto t1206 = t1052*t411;
const auto t1207 = 2*t69;
const auto t1208 = t203*t69;
dA[0] = t119*(-std::pow(t110, 2)*t114 - t111*t114*(t115*t53 + t116*t60 - t13*t34) + t111*t90 + t64*(std::pow(t13, 2) + t18));
dA[1] = t155*(-t111*t134 + t152);
dA[2] = t155*(-t111*t162 + t168);
dA[3] = t154*(-t150*t199 - t208*t211 + t208*t215 + t208*t216 + t227*(t205*(-t174*t223 - t178*t224 - t178*t225 - t206*t75 + t219*t220 + t220*t221) - t217*t218) - t72*(-t170*t171 - t170*t172 + t176 + t178*t179 + t178*t180 + t183));
dA[4] = t154*(-t150*t254 - t211*t259 + t215*t259 + t216*t259 + t263*(t205*(t1*t129*t236 + t1*t247*t70*t72 - t1*t262 - t222*t231 - t232*t261 + t247*t65*t75) - t217*t256) + t72*(-t1*t237 + t1*t242 - t146*t229 + t230*t231 + t232*t233 - t238));
dA[5] = t154*(-t150*t287 - t211*t291 + t215*t291 + t216*t291 + t263*(t205*(t1*t123*t271 + t1*t280*t68*t72 - t1*t294 - t222*t266 + t280*t65*t75 - t293*t69) - t217*t288) + t72*(-t1*t272 + t1*t276 - t166*t265 + t230*t266 + t268*t69 - t273));
dA[6] = t154*(t120*t132*t209*(t217*t319 + t316*(t146*t297 + t166*t297 - t222*t322 + t260*t321 + t314*t321 + t323)) + t132*t144*t209*t320 - t150*t311 - t215*t320 - t216*t320 - t249*t313 - t282*t313 - t329);
dA[7] = t154*(-t150*t353 + t211*t358 - t215*t358 - t216*t358 + t227*(t217*t357 + t316*(t1*t348*t70*t72 + t1*t351*t67*t72 - t1*t362 - t222*t334 + t348*t65*t75 - t361)) + t72*(t1*t339 - t1*t345 - t146*t333 + t230*t334 + t336*t4 - t340));
dA[8] = t154*(-t150*t382 + t211*t385 - t215*t385 - t216*t385 + t227*(t217*t384 + t316*(t1*t378*t68*t72 + t1*t380*t69*t72 - t1*t387 - t222*t367 + t378*t65*t75 - t386)) + t72*(t1*t370 - t1*t376 - t166*t366 + t230*t367 + t268*t7 - t371));
dA[9] = t398;
dA[10] = t406;
dA[11] = t413;
dA[12] = t155*(-t151*t419 + t152);
dA[13] = t119*(-t114*std::pow(t418, 2) - t114*t419*(t115*t34 - t422*t53 + t431*t60) + t419*t430 + t64*(std::pow(t422, 2) + t425));
dA[14] = t155*(-t162*t419 + t433);
dA[15] = t154*(-t133*t199 - t208*t442 + t208*t445 + t208*t446 + t263*(-t202*t443 + t205*(t121*t4*t437 + t128*t189*t67 + t189*t4*t70*t72 - t342*t389 - t342*t390 - t4*t444)) + t72*(-t170*t434 + t342*t393 + t342*t394 - t4*t438 + t4*t440 - t439));
dA[16] = t154*(-t133*t254 + t227*(-t130*t256*t312 + t205*(-t128*t257 + t195*t453 + t221*t453 - t225*t338 - t338*t401 - t449*t454)) - t259*t442 + t259*t445 + t259*t446 - t72*(t180*t338 - t229*t447 - t229*t448 + t338*t451 + 2*t450 + t452));
dA[17] = t154*(-t133*t287 + t263*(t205*(-t127*t292*t69 + t280*t432 - t294*t4 + t4*t456 - t457*t69 + t458) - t288*t443) - t291*t442 + t291*t445 + t291*t446 + t72*(2*t135*t4*t65*t69*t72 + 2*t137*t4*t67*t69*t72 - t265*t432 - t272*t4 + t275*t4*t69*t72 - t455));
dA[18] = t154*(-t133*t311 + t227*(t316*(-t1*t127*t292 + t297*t434 - t361 - t4*t462 + t4*t463 + t461) + t319*t443) + t320*t442 - t320*t445 - t320*t446 + t72*(2*t1*t136*t4*t69*t72 + 2*t1*t137*t4*t67*t72 + t135*t178*t4 - t302*t434 - t4*t460 - t459));
dA[19] = t154*(t120*t132*t209*(t316*(t260*t465 - t292*t466 + t348*t432 + t348*t434 + t354*t465 + t467) + t357*t443) + t132*t139*t209*t358 - t133*t353 - t249*t464 - t349*t464 - t358*t445 - t358*t446 - t472);
dA[20] = t154*(-t133*t382 + t227*(t316*(-t127*t292*t7 + t378*t432 - t387*t4 + t4*t477 + t475 + t476) + t384*t443) + t385*t442 - t385*t445 - t385*t446 + t72*(t136*t369*t4 + 2*t137*t4*t67*t7*t72 - t366*t432 - t376*t4 - t473 - t474));
dA[21] = t478;
dA[22] = t479;
dA[23] = t480;
dA[24] = t155*(-t151*t485 + t168);
dA[25] = t155*(-t134*t485 + t433);
dA[26] = t119*(-t114*std::pow(t484, 2) - t114*t485*(t116*t34 + t431*t53 - t488*t60) + t485*t493 + t64*(std::pow(t488, 2) + t489));
dA[27] = t154*(-t161*t199 - t208*t497 + t208*t499 + t208*t500 + t263*(-t202*t498 + t205*(t121*t437*t7 + t159*t189*t69 + t189*t68*t7*t72 - t373*t389 - t373*t390 - t444*t7)) + t72*(-t170*t494 + t373*t393 + t373*t394 - t438*t7 + t440*t7 - t495));
dA[28] = t154*(-t161*t254 - t259*t497 + t259*t499 + t259*t500 + t263*(t205*(t247*t501 - t261*t373 - t262*t7 + t503*t7 - t504*t67 + t505) - t256*t498) + t72*(2*t135*t65*t67*t7*t72 + 2*t136*t67*t69*t7*t72 - t229*t501 - t237*t7 + t241*t67*t7*t72 - t502));
dA[29] = t154*(-t161*t287 + t227*(-t160*t288*t312 + t205*(-t159*t289 + t195*t511 + t219*t511 - t224*t369 - t369*t401 - t508*t512)) - t291*t497 + t291*t499 + t291*t500 - t72*(t179*t369 - t265*t506 - t265*t507 + t369*t451 + 2*t509 + t510));
dA[30] = t154*(-t161*t311 + t227*(t316*(-t1*t158*t359 + t297*t494 - t386 - t462*t7 + t463*t7 + t514) + t319*t498) + t320*t497 - t320*t499 - t320*t500 + t72*(2*t1*t136*t69*t7*t72 + 2*t1*t137*t67*t7*t72 + t135*t178*t7 - t302*t494 - t460*t7 - t513));
dA[31] = t154*(-t161*t353 + t227*(t316*(-t158*t359*t4 + t348*t501 - t362*t7 + t475 + t516 + t517*t7) + t357*t498) + t358*t497 - t358*t499 - t358*t500 + t72*(2*t136*t4*t69*t7*t72 + t137*t338*t7 - t333*t501 - t345*t7 - t473 - t515));
dA[32] = t154*(t120*t132*t209*(t316*(t314*t519 + t354*t519 - t359*t520 + t378*t494 + t378*t501 + t521) + t384*t498) + t132*t164*t209*t385 - t161*t382 - t282*t518 - t349*t518 - t385*t499 - t385*t500 - t526);
dA[33] = t528;
dA[34] = t530;
dA[35] = t531;
dA[36] = t118*(-t110*t553 + 2*t110*t556*t561*t64 + t110*t557*t558*t561*t64 - t114*(-t174*t532 - t198*t75 + t534 + t535 - t542) - t176*t72 - t183*t72 - t560*t90 + t563 + t564 - t569);
dA[37] = t118*(-t114*(-t148*t189*t72 + t570 + t572 - t580) - t418*t553 + 2*t418*t556*t561*t64 + t418*t557*t558*t561*t64 - t430*t560 - t439*t72 + t581 + t582 - t588);
dA[38] = t118*(-t114*(-t167*t189*t72 + t589 + t591 - t598) - t484*t553 + 2*t484*t556*t561*t64 + t484*t557*t558*t561*t64 - t493*t560 - t495*t72 + t599 + t600 - t605);
dA[39] = t154*(2*t10*t101*(4*t14*t45*t621 - t36*t622 - 3*t47 - t48) + 2*t10*t79*(-t188*t622 + t29*t621*t623 - 3*t31 + t624) + t120*t132*t209*(t204*(t145*t633 + t165*t633 + std::pow(t181, 2)*t72 + t230*(-t36*t435 - t48 - t637*t638 + 3*t65*t72*t76) + t613*t634 + t613*t635) + t205*(std::pow(t206, 2) + t223*(3*t100 - t188*t435 - t617*t637 + t624) + t639*t68 + t639*t70 + t640*t641 + t640*t642) + 4*t207*t218) + 2*t132*t199*t208*t209 - 2*t192*t198 - std::pow(t208, 2)*t626 - t208*t630 - t208*t632 - t249*t620 - t282*t620 - t606*t607 - t607*t608 - t609*t615 - t615*t616);
dA[40] = t154*(-t259*t630 - t259*t632 - t672);
dA[41] = t154*(-t291*t630 - t291*t632 - t695);
dA[42] = t154*(2*t120*t132*t202*t320*t625*t72 + 2*t120*t207*t209*t212*t320 - t718);
dA[43] = t154*(t358*t630 + t358*t632 + t739);
dA[44] = t154*(t385*t630 + t385*t632 + t760);
dA[45] = t118*(-t105 - t108 + t113*t557*t558*t766 - t532*t761 + t539 + t541 + t552*t558*t764 - t763*(-t192*t388 - t324*t761 - t563 - t564 - t762) - t764*t769 - t764*t772 - t773);
dA[46] = t118*(t113*t557*t558*t778 - t190*t399 - t415 + t552*t558*t776 - t570 - t572 - t763*(-t170*t774 - t427 - t581 - t582 - t775) - t769*t776 - t772*t776 - t779);
dA[47] = t118*(t113*t557*t558*t784 - t191*t407 - t482 + t552*t558*t782 - t589 - t591 - t763*(-t170*t780 - t491 - t599 - t600 - t781) - t769*t782 - t772*t782 - t785);
dA[48] = t118*(t110*t558*t561*t64*t801 + 2*t110*t561*t64*t800 - t110*t798 - t114*(-t147*t247*t72 + t786 + t788 - t790) - t238*t72 - t802*t90 + t803 + t804 - t806);
dA[49] = t118*(-t114*(-t128*t253 - t449*t807 + t814) - t312*t450 + t418*t558*t561*t64*t801 + 2*t418*t561*t64*t800 - t418*t798 - t430*t802 - t452*t72 + t815 + t816 - t822);
dA[50] = t118*(-t114*(-t505*t72 + t823 + t825 - t832) + t484*t558*t561*t64*t801 + 2*t484*t561*t64*t800 - t484*t798 - t493*t802 - t502*t72 + t833 + t834 - t840);
dA[51] = t154*(-t208*t842 - t208*t844 - t672);
dA[52] = t154*(2*t10*t104*(4*t14*t45*t853 - t39*t854 - 3*t54 - t55) + 2*t10*t82*(-t246*t854 + t29*t623*t853 - 3*t50 + t855) + t120*t132*t209*(t204*(t145*t857 + t173*t857 + std::pow(t241, 2)*t72 + t267*(-t234*t39 - t55 - t638*t860 + 3*t67*t72*t76) + t635*t848 + t848*t858) + t205*(std::pow(t257, 2) + t454*(3*t103 - t234*t246 - t617*t860 + t855) + t642*t862 + t66*t861 + t70*t861 + t862*t863) + t256*t258*t856) + 2*t132*t209*t254*t259 - 2*t251*t253 - std::pow(t259, 2)*t626 - t259*t842 - t259*t844 - t608*t845 - t616*t849 - t653*t851 - t845*t846 - t847*t849 - t851*t852);
dA[53] = t154*(-t291*t842 - t291*t844 - t883);
dA[54] = t154*(t320*t842 + t320*t844 + t901);
dA[55] = t154*(2*t120*t132*t256*t358*t625*t72 + 2*t120*t209*t212*t258*t358 - t918);
dA[56] = t154*(t385*t842 + t385*t844 + t931);
dA[57] = t118*(t113*t558*t766*t801 - t248*t388 + t558*t764*t797 - t763*(-t229*t932 - t775 - t803 - t804 - t805) - t764*t933 - t764*t934 - t779 - t786 - t788 - t789);
dA[58] = t118*(t113*t558*t778*t801 + t558*t776*t797 - t763*(-t251*t399 - t468*t936 - t815 - t816 - t937) - t776*t933 - t776*t934 - t807*t936 - t814 - t935);
dA[59] = t118*(t113*t558*t784*t801 - t247*t780 + t558*t782*t797 - t763*(-t229*t780 - t833 - t834 - t835 - t938) - t782*t933 - t782*t934 - t823 - t825 - t826 - t939);
dA[60] = t118*(t110*t558*t561*t64*t953 + 2*t110*t561*t64*t952 - t110*t950 - t114*(-t147*t280*t72 + t940 + t942 - t944) - t273*t72 - t90*t954 + t955 + t956 - t958);
dA[61] = t118*(-t114*(-t458*t72 + t959 + t960 - t961) + t418*t558*t561*t64*t953 + 2*t418*t561*t64*t952 - t418*t950 - t430*t954 - t455*t72 + t962 + t963 - t964);
dA[62] = t118*(-t114*(-t159*t286 - t508*t965 + t969) - t312*t509 + t484*t558*t561*t64*t953 + 2*t484*t561*t64*t952 - t484*t950 - t493*t954 - t510*t72 + t970 + t971 - t974);
dA[63] = t154*(-t208*t975 - t208*t976 - t695);
dA[64] = t154*(-t259*t975 - t259*t976 - t883);
dA[65] = t154*(2*t10*t107*(4*t14*t45*t982 - t42*t983 - 3*t61 - t62) + 2*t10*t86*(-t279*t983 + t29*t623*t982 - 3*t57 + t984) + t120*t132*t209*(t204*(t165*t985 + t173*t985 + std::pow(t275, 2)*t72 + t335*(-t269*t42 - t62 - t638*t987 + 3*t69*t72*t76) + t634*t978 + t858*t978) + t205*(std::pow(t289, 2) + t512*(3*t106 - t269*t279 - t617*t987 + t984) + t641*t989 + t66*t988 + t68*t988 + t863*t989) + t288*t290*t856) + 2*t132*t209*t287*t291 - 2*t284*t286 - std::pow(t291, 2)*t626 - t291*t975 - t291*t976 - t606*t977 - t609*t979 - t683*t981 - t846*t977 - t847*t979 - t852*t981);
dA[66] = t154*(t1004 + t320*t975 + t320*t976);
dA[67] = t154*(t1014 + t358*t975 + t358*t976);
dA[68] = t154*(-t1024 + 2*t120*t132*t288*t385*t625*t72 + 2*t120*t209*t212*t290*t385);
dA[69] = t118*(-t1025*t764 - t1026*t764 + t113*t558*t766*t953 - t281*t388 + t558*t764*t949 - t763*(-t265*t932 - t781 - t955 - t956 - t957) - t785 - t940 - t942 - t943);
dA[70] = t118*(-t1025*t776 - t1026*t776 + t113*t558*t778*t953 - t280*t774 - t416 + t558*t776*t949 - t763*(-t265*t774 - t428 - t938 - t962 - t963) - t939 - t959 - t960);
dA[71] = t118*(-t1025*t782 - t1026*t782 - t1027 - t1028*t965 + t113*t558*t784*t953 + t558*t782*t949 - t763*(-t1028*t522 - t1029 - t284*t407 - t970 - t971) - t969);
dA[72] = t118*(-t1037*t110 + 2*t1038*t110*t561*t64*t72 + t1041*t110*t558*t561*t64 - t1042*t90 - t114*(t1030 - t322*t532 - t323*t72) - t329 - t762);
dA[73] = t118*(-t1037*t418 - t1042*t430 + t1045*t418 + t1047*t418 + t1048 - t114*(t1043 - t461*t72) - t459*t72);
dA[74] = t118*(-t1037*t484 - t1042*t493 + t1045*t484 + t1047*t484 + t1050 - t114*(t1049 - t514*t72) - t513*t72);
dA[75] = t154*(2*t120*t132*t208*t319*t625 + 2*t120*t208*t209*t212*t315*t72 - t718);
dA[76] = t154*(t1053*t259 + t1055*t259 + t901);
dA[77] = t154*(t1004 + t1053*t291 + t1055*t291);
dA[78] = t154*(-t1053*t320 - t1055*t320 - 2*t1066*t311 + t120*t132*t209*(t203*(-t1065*t135*t312 + t1068*t68 + t1068*t70 + t1069*t634 + t1069*t635 + std::pow(t310, 2)) + t315*t319*t856 + t316*(t1058*t641 + t1058*t642 - t1064*t222 + t1067*t145 + t1067*t165 + std::pow(t304, 2)*t72)) - t312*(t101*t1065 + t1056*t302 + t1057*t302 + t1059*t249 + t1059*t282 + t1063*t609 + t1063*t616 + t1064*t79 + t304*t310) - std::pow(t320, 2)*t626);
dA[79] = t154*(-t1053*t358 - t1055*t358 - t1080);
dA[80] = t154*(-t1053*t385 - t1055*t385 - t1090);
dA[81] = t118*(-t102*t664 - t1030 + t1036*t558*t764 + t1041*t113*t558*t766 - t1091*t72 - t1092*t1093 - t1094*t764 - t763*(-t310*t388 - t327 - t328 - t569 - t664*t80));
dA[82] = t118*(t1036*t558*t776 + t1041*t113*t558*t778 - t1043 - t1093*t1095 - t1094*t776 - t297*t774 - t763*(-t1048 - t302*t774));
dA[83] = t118*(t1036*t558*t782 + t1041*t113*t558*t784 - t1049 - t1093*t1096 - t1094*t782 - t297*t780 - t763*(-t1050 - t302*t780));
dA[84] = t118*(-t110*t1101 + t110*t1106 - t1105*t90 + t1107*t1108 + t1109 - t114*(t1097 - t147*t348*t72) - t340*t72);
dA[85] = t118*(-t1101*t418 + 2*t1102*t418*t561*t64*t72 + t1104*t418*t558*t561*t64 - t1105*t430 - t114*(t1111 - t466*t807 - t467*t72) - t472 - t937);
dA[86] = t118*(-t1101*t484 - t1105*t493 + t1106*t484 + t1108*t312*t484 + t1113 - t114*(t1112 - t516*t72) - t515*t72);
dA[87] = t154*(t1115*t208 + t1116*t208 + t739);
dA[88] = t154*(2*t120*t132*t259*t357*t625 + 2*t120*t209*t212*t259*t355*t72 - t918);
dA[89] = t154*(t1014 + t1115*t291 + t1116*t291);
dA[90] = t154*(-t1080 - t1115*t320 - t1116*t320);
dA[91] = t154*(-t1115*t358 - t1116*t358 - 2*t1124*t353 + t120*t132*t209*(t203*(-t1123*t137*t312 + t1126*t66 + t1126*t70 + t1127*t635 + t1127*t858 + std::pow(t344, 2)) + t316*(t1118*t642 + t1118*t863 - t1122*t292 + t1125*t145 + t1125*t173 + std::pow(t351, 2)*t72) + t355*t357*t856) - t312*(t104*t1123 + t1117*t333 + t1119*t249 + t1119*t349 + t1121*t616 + t1121*t847 + t1122*t82 + t333*t719 + t344*t351) - std::pow(t358, 2)*t626);
dA[92] = t154*(-t1115*t385 - t1116*t385 - t1137);
dA[93] = t118*(-t1092*t1138 - t1097 + t1100*t558*t764 + t1104*t113*t558*t766 - t1139*t764 - t348*t932 - t763*(-t1109 - t333*t932));
dA[94] = t118*(-t1095*t1138 + t1100*t558*t776 + t1104*t113*t558*t778 - t1111 - t1139*t776 - t1140*t72 - t414*t668 - t763*(-t344*t399 - t426*t668 - t470 - t471 - t822));
dA[95] = t118*(-t1096*t1138 + t1100*t558*t782 + t1104*t113*t558*t784 - t1112 - t1139*t782 - t348*t780 - t763*(-t1113 - t333*t780));
dA[96] = t118*(-t110*t1145 + t110*t1150 + t1107*t1151 - t114*(t1141 - t147*t378*t72) - t1149*t90 + t1152 - t371*t72);
dA[97] = t118*(-t114*(t1153 - t476*t72) - t1145*t418 - t1149*t430 + t1150*t418 + t1151*t312*t418 + t1154 - t474*t72);
dA[98] = t118*(-t1029 - t114*(t1155 - t520*t965 - t521*t72) - t1145*t484 + 2*t1146*t484*t561*t64*t72 + t1148*t484*t558*t561*t64 - t1149*t493 - t526);
dA[99] = t154*(t1156*t208 + t1157*t208 + t760);
dA[100] = t154*(t1156*t259 + t1157*t259 + t931);
dA[101] = t154*(-t1024 + 2*t120*t132*t291*t384*t625 + 2*t120*t209*t212*t291*t383*t72);
dA[102] = t154*(-t1090 - t1156*t320 - t1157*t320);
dA[103] = t154*(-t1137 - t1156*t358 - t1157*t358);
dA[104] = t154*(-2*t1136*t382 - t1156*t385 - t1157*t385 + t120*t132*t209*(t203*(-t1163*t136*t312 + t1165*t66 + t1165*t68 + t1166*t634 + t1166*t858 + std::pow(t375, 2)) + t316*(t1158*t641 + t1158*t863 - t1162*t359 + t1164*t165 + t1164*t173 + std::pow(t380, 2)*t72) + t383*t384*t856) - t312*(t107*t1163 + t1159*t282 + t1159*t349 + t1161*t609 + t1161*t847 + t1162*t86 + t366*t740 + t366*t919 + t375*t380) - std::pow(t385, 2)*t626);
dA[105] = t118*(-t1092*t1167 + t113*t1148*t558*t766 - t1141 + t1144*t558*t764 - t1168*t764 - t378*t932 - t763*(-t1152 - t366*t932));
dA[106] = t118*(-t1095*t1167 + t113*t1148*t558*t778 + t1144*t558*t776 - t1153 - t1168*t776 - t378*t774 - t763*(-t1154 - t366*t774));
dA[107] = t118*(-t1096*t1167 + t113*t1148*t558*t784 + t1144*t558*t782 - t1155 - t1168*t782 - t1169*t72 - t481*t693 - t763*(-t375*t407 - t490*t693 - t524 - t525 - t974));
dA[108] = t398;
dA[109] = t478;
dA[110] = t528;
dA[111] = t154*(-t1172*t208 + t120*t132*t208*t396*t625 + t120*t132*t209*(t1170*t631 + t204*(t1171*t395 + t181*t391 + t200*t437 + t201*t437 + t65*t716 + t65*t717)) - t184*t436 - t185*t436 - t199*t397 - t392*t738 - t664*t765 - t773);
dA[112] = t154*(-t1172*t259 + t1176*t259 + t227*(t1170*t841 + t1175*(-t1174 - t229*t391 + t229*t70*t72)) - t254*t397 - t392*t897 + t73*(t1173 + t247*t391 - t643));
dA[113] = t154*(-t1000*t392 - t1172*t291 + t1176*t291 + t227*(t1170*t882 + t1175*(-t1178 - t265*t391 + t265*t68*t72)) - t287*t397 + t73*(t1177 + t280*t391 - t673));
dA[114] = t154*(t1066*t392 + t1172*t320 - t1176*t320 + t227*(-t1170*t312*t315 + t203*(-t143*t664 - t178*t317 - t178*t318 + t194*t302*t65*t68 + t194*t302*t65*t70 - t310*t391)) - t311*t397 - t72*(t102*t1171 + t1091 + t171*t297 + t172*t297 + t178*t609 + t178*t616));
dA[115] = t154*(t1124*t392 + t1172*t358 - t1176*t358 + t263*(-t1170*t915 + t1181*(-t1180 - t333*t391 + t333*t70*t72)) - t353*t397 + t73*(t1179 + t348*t391 - t719));
dA[116] = t154*(t1136*t392 + t1172*t385 - t1176*t385 + t263*(-t1170*t930 + t1181*(-t1183 - t366*t391 + t366*t68*t72)) - t382*t397 + t73*(t1182 + t378*t391 - t740));
dA[117] = t1185*(t1092*t766 - t1184*std::pow(t764, 2) + t64*(std::pow(t12, 2) + t18));
dA[118] = t1189*(t1092*t529 + t1188);
dA[119] = t1189*(t1092*t1190 + t1191);
dA[120] = t406;
dA[121] = t479;
dA[122] = t530;
dA[123] = t154*(-t1196*t208 + t1197*t208 + t126*(t1192 + t189*t400 - t715) - t199*t405 + t227*(t1193*t631 + t1195*(-t1194 - t170*t400 + t170*t70*t72)) - t402*t738);
dA[124] = t154*(-t1196*t259 + t120*t132*t209*(t1193*t841 + t204*(t1198*t403 + t201*t236 + t236*t255 + t241*t400 + t644*t67 + t67*t917)) + t120*t132*t259*t404*t625 - t185*t235 - t235*t243 - t254*t405 - t402*t897 - t668*t777 - t813 - t935);
dA[125] = t154*(-t1000*t402 - t1196*t291 + t1197*t291 + t126*(t1177 + t280*t400 - t864) + t227*(t1193*t882 + t1195*(-t1178 - t265*t400 + t265*t66*t72)) - t287*t405);
dA[126] = t154*(t1066*t402 + t1196*t320 - t1197*t320 + t126*(-t1057 + t1199 + t297*t400) + t263*(-t1193*t898 + t1201*(-t1200 - t302*t400 + t302*t70*t72)) - t311*t405);
dA[127] = t154*(t1124*t402 + t1196*t358 - t1197*t358 + t227*(-t1193*t312*t355 + t203*(-t138*t668 + t194*t333*t66*t67 + t194*t333*t67*t70 - t318*t338 - t338*t356 - t344*t400)) - t353*t405 - t72*(t1140 + t1198*t414 + t338*t616 + t338*t847 + t348*t447 + t348*t448));
dA[128] = t154*(t1136*t402 + t1196*t385 - t1197*t385 + t126*(t1182 + t378*t400 - t919) + t263*(-t1193*t930 + t1201*(-t1183 - t366*t400 + t366*t66*t72)) - t382*t405);
dA[129] = t1189*(t1095*t527 + t1188);
dA[130] = t1185*(t1095*t778 - t1184*std::pow(t776, 2) + t64*(std::pow(t421, 2) + t425));
dA[131] = t1189*(t1095*t1190 + t1202);
dA[132] = t413;
dA[133] = t480;
dA[134] = t531;
dA[135] = t154*(-t1205*t208 + t1206*t208 + t124*(t1192 + t189*t408 - t714) - t199*t412 + t227*(t1203*t631 + t1204*(-t1194 - t170*t408 + t170*t68*t72)) - t409*t738);
dA[136] = t154*(-t1205*t259 + t1206*t259 + t124*(t1173 + t247*t408 - t916) + t227*(t1203*t841 + t1204*(-t1174 - t229*t408 + t229*t66*t72)) - t254*t412 - t409*t897);
dA[137] = t154*(-t1000*t409 - t1027 + t120*t132*t209*(t1203*t882 + t204*(t1207*t410 + t200*t271 + t255*t271 + t275*t408 + t674*t69 + t69*t865)) + t120*t132*t291*t411*t625 - t1205*t291 - t184*t270 - t243*t270 - t287*t412 - t693*t783 - t968);
dA[138] = t154*(t1066*t409 + t1205*t320 - t1206*t320 + t124*(-t1056 + t1199 + t297*t408) + t263*(-t1203*t898 + t1208*(-t1200 - t302*t408 + t302*t68*t72)) - t311*t412);
dA[139] = t154*(t1124*t409 + t1205*t358 - t1206*t358 + t124*(-t1117 + t1179 + t348*t408) + t263*(-t1203*t915 + t1208*(-t1180 - t333*t408 + t333*t66*t72)) - t353*t412);
dA[140] = t154*(t1136*t409 + t1205*t385 - t1206*t385 + t227*(-t1203*t312*t383 + t203*(-t163*t693 + t194*t366*t66*t69 + t194*t366*t68*t69 - t317*t369 - t356*t369 - t375*t408)) - t382*t412 - t72*(t1169 + t1207*t481 + t369*t609 + t369*t847 + t378*t506 + t378*t507));
dA[141] = t1189*(t1096*t527 + t1191);
dA[142] = t1189*(t1096*t529 + t1202);
dA[143] = t1185*(t1096*t784 - t1184*std::pow(t782, 2) + t64*(std::pow(t487, 2) + t489));
}

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
        double dA[9])
    {
        const auto t0 = t0_x - t2_x;
        const auto t1 = t0_x - t1_x;
        const auto t2 = t0_y - t1_y;
        const auto t3 = t0_z - t1_z;
        const auto t4 = std::pow(t1, 2) + std::pow(t2, 2) + std::pow(t3, 2);
        const auto t5 = t0_y - t2_y;
        const auto t6 = t0_z - t2_z;
        const auto t7 =
            1.0 / (std::pow(t0, 2) + std::pow(t5, 2) + std::pow(t6, 2));
        const auto t8 = t4 * t7;
        const auto t9 = std::sqrt(t8);
        const auto t10 = t9 / t4;
        const auto t11 = t7 * t9;
        dA[0] = t10 * (-t0 * t8 + t0_x - t1_x);
        dA[1] = t10 * (t0_y - t1_y - t5 * t8);
        dA[2] = t10 * (t0_z - t1_z - t6 * t8);
        dA[3] = -t1 * t10;
        dA[4] = -t10 * t2;
        dA[5] = -t10 * t3;
        dA[6] = t0 * t11;
        dA[7] = t11 * t5;
        dA[8] = t11 * t6;
    }

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
        double dA[81])
    {
        const auto t0 = t0_x - t1_x;
        const auto t1 = std::pow(t0, 2);
        const auto t2 = t0_y - t1_y;
        const auto t3 = std::pow(t2, 2);
        const auto t4 = t0_z - t1_z;
        const auto t5 = std::pow(t4, 2);
        const auto t6 = t1 + t3 + t5;
        const auto t7 = 1.0 / t6;
        const auto t8 = t0_x - t2_x;
        const auto t9 = std::pow(t8, 2);
        const auto t10 = t0_y - t2_y;
        const auto t11 = std::pow(t10, 2);
        const auto t12 = t0_z - t2_z;
        const auto t13 = std::pow(t12, 2);
        const auto t14 = t11 + t13 + t9;
        const auto t15 = 1.0 / t14;
        const auto t16 = t15 * t6;
        const auto t17 = t16 * t8;
        const auto t18 = -t0_x + t1_x;
        const auto t19 = -t17 - t18;
        const auto t20 = t0 * t8;
        const auto t21 = 4 * t15;
        const auto t22 = t19 * t7;
        const auto t23 = t0 * t22;
        const auto t24 = t19 * t8;
        const auto t25 = 2 * t15;
        const auto t26 = 4 * t9;
        const auto t27 = std::pow(t14, -2);
        const auto t28 = t27 * t6;
        const auto t29 = 1 - t16;
        const auto t30 = std::sqrt(t16);
        const auto t31 = t30 * t7;
        const auto t32 = t10 * t16;
        const auto t33 = -t0_y + t1_y;
        const auto t34 = -t32 - t33;
        const auto t35 = t34 * t7;
        const auto t36 = 2 * t0;
        const auto t37 = t0 * t10;
        const auto t38 = t2 * t8;
        const auto t39 = 2 * t17;
        const auto t40 = -t19 * t34 * t7 + t25 * (-t10 * t39 + t37 + t38);
        const auto t41 = t12 * t16;
        const auto t42 = -t0_z + t1_z;
        const auto t43 = -t41 - t42;
        const auto t44 = t43 * t7;
        const auto t45 = t0 * t12;
        const auto t46 = t4 * t8;
        const auto t47 = -t19 * t43 * t7 + t25 * (-t12 * t39 + t45 + t46);
        const auto t48 = t30 / std::pow(t6, 2);
        const auto t49 = t48 * (t0 + t17);
        const auto t50 = t15 * t30;
        const auto t51 = t25 * t8;
        const auto t52 = t50 * (t19 * t7 - t51);
        const auto t53 = 2 * t2;
        const auto t54 = t10 * t2;
        const auto t55 = t2 * t35;
        const auto t56 = t10 * t34;
        const auto t57 = 4 * t28;
        const auto t58 = t12 * t2;
        const auto t59 = t10 * t4;
        const auto t60 = t25 * (-2 * t12 * t32 + t58 + t59) - t34 * t43 * t7;
        const auto t61 = t48 * (t2 + t32);
        const auto t62 = t10 * t25;
        const auto t63 = t50 * (t34 * t7 - t62);
        const auto t64 = 2 * t4;
        const auto t65 = t12 * t4;
        const auto t66 = t4 * t44;
        const auto t67 = t12 * t43;
        const auto t68 = t48 * (t4 + t41);
        const auto t69 = t12 * t25;
        const auto t70 = t50 * (t43 * t7 - t69);
        const auto t71 = t35 + t62;
        const auto t72 = t0 * t31;
        const auto t73 = t44 + t69;
        const auto t74 = t0 * t48;
        const auto t75 = -t2 * t74;
        const auto t76 = -t4 * t74;
        const auto t77 = t15 * t31;
        const auto t78 = -t20 * t77;
        const auto t79 = -t37 * t77;
        const auto t80 = -t45 * t77;
        const auto t81 = t22 + t51;
        const auto t82 = t2 * t31;
        const auto t83 = -t2 * t4 * t48;
        const auto t84 = -t38 * t77;
        const auto t85 = -t54 * t77;
        const auto t86 = -t58 * t77;
        const auto t87 = t31 * t4;
        const auto t88 = -t46 * t77;
        const auto t89 = -t59 * t77;
        const auto t90 = -t65 * t77;
        const auto t91 = -3 * t32 - t33;
        const auto t92 = t77 * t8;
        const auto t93 = -3 * t41 - t42;
        const auto t94 = 3 * t15;
        const auto t95 = 3 * t27 * t30 * t8;
        const auto t96 = t10 * t95;
        const auto t97 = t12 * t95;
        const auto t98 = -3 * t17 - t18;
        const auto t99 = t10 * t77;
        const auto t100 = 4 * t16;
        const auto t101 = 3 * t10 * t12 * t27 * t30;
        const auto t102 = t12 * t77;
        dA[0] = t31
            * (std::pow(t19, 2) * t7 - t20 * t21 - 2 * t23 + t24 * t25
               + t26 * t28 + t29);
        dA[1] = t31 * (2 * t15 * t34 * t8 - t35 * t36 - t40);
        dA[2] = t31 * (2 * t15 * t43 * t8 - t36 * t44 - t47);
        dA[3] = t31 * (2 * t1 * t7 - t23 - 1);
        dA[4] = t2 * t49;
        dA[5] = t4 * t49;
        dA[6] = t50 * (t22 * t8 - t25 * t9 + 1);
        dA[7] = t10 * t52;
        dA[8] = t12 * t52;
        dA[9] = t31 * (2 * t10 * t15 * t19 - t22 * t53 - t40);
        dA[10] = t31
            * (t11 * t57 - t21 * t54 + t25 * t56 + t29 + std::pow(t34, 2) * t7
               - 2 * t55);
        dA[11] = t31 * (2 * t10 * t15 * t43 - t44 * t53 - t60);
        dA[12] = t0 * t61;
        dA[13] = t31 * (2 * t3 * t7 - t55 - 1);
        dA[14] = t4 * t61;
        dA[15] = t63 * t8;
        dA[16] = t50 * (t10 * t35 - t11 * t25 + 1);
        dA[17] = t12 * t63;
        dA[18] = t31 * (2 * t12 * t15 * t19 - t22 * t64 - t47);
        dA[19] = t31 * (2 * t12 * t15 * t34 - t35 * t64 - t60);
        dA[20] = t31
            * (t13 * t57 - t21 * t65 + t25 * t67 + t29 + std::pow(t43, 2) * t7
               - 2 * t66);
        dA[21] = t0 * t68;
        dA[22] = t2 * t68;
        dA[23] = t31 * (2 * t5 * t7 - t66 - 1);
        dA[24] = t70 * t8;
        dA[25] = t10 * t70;
        dA[26] = t50 * (t12 * t44 - t13 * t25 + 1);
        dA[27] = t31 * (t0 * t51 + t23 - 1);
        dA[28] = t71 * t72;
        dA[29] = t72 * t73;
        dA[30] = t31 * (-t1 * t7 + 1);
        dA[31] = t75;
        dA[32] = t76;
        dA[33] = t78;
        dA[34] = t79;
        dA[35] = t80;
        dA[36] = t81 * t82;
        dA[37] = t31 * (t2 * t62 + t55 - 1);
        dA[38] = t73 * t82;
        dA[39] = t75;
        dA[40] = t31 * (-t3 * t7 + 1);
        dA[41] = t83;
        dA[42] = t84;
        dA[43] = t85;
        dA[44] = t86;
        dA[45] = t81 * t87;
        dA[46] = t71 * t87;
        dA[47] = t31 * (t4 * t69 + t66 - 1);
        dA[48] = t76;
        dA[49] = t83;
        dA[50] = t31 * (-t5 * t7 + 1);
        dA[51] = t88;
        dA[52] = t89;
        dA[53] = t90;
        dA[54] = t77 * (-t16 * t26 + 2 * t20 - t24 + t6);
        dA[55] = t91 * t92;
        dA[56] = t92 * t93;
        dA[57] = t78;
        dA[58] = t84;
        dA[59] = t88;
        dA[60] = t50 * (t9 * t94 - 1);
        dA[61] = t96;
        dA[62] = t97;
        dA[63] = t98 * t99;
        dA[64] = t77 * (t10 * t53 - t100 * t11 - t56 + t6);
        dA[65] = t93 * t99;
        dA[66] = t79;
        dA[67] = t85;
        dA[68] = t89;
        dA[69] = t96;
        dA[70] = t50 * (t11 * t94 - 1);
        dA[71] = t101;
        dA[72] = t102 * t98;
        dA[73] = t102 * t91;
        dA[74] = t77 * (-t100 * t13 + t12 * t64 + t6 - t67);
        dA[75] = t80;
        dA[76] = t86;
        dA[77] = t90;
        dA[78] = t97;
        dA[79] = t101;
        dA[80] = t50 * (t13 * t94 - 1);
    }

        void line_projection_uv_gradient(
            double t, double vx, double vy, double vz, double dA[4],
            double a0x, double a0y, double a0z,
            double a1x, double a1y, double a1z,
            double b0x, double b0y, double b0z,
            double b1x, double b1y, double b1z)
        {
            const auto t0 = -b0x;
            const auto t1 = b0x - b1x;
            const auto t2 = a0x - a1x;
            const auto t3 = a0x - t * t2;
            const auto t4 = t * t1 + t0 + t3;
            const auto t5 = -b0y;
            const auto t6 = b0y - b1y;
            const auto t7 = a0y - a1y;
            const auto t8 = a0y - t * t7;
            const auto t9 = t * t6 + t5 + t8;
            const auto t10 = -b0z;
            const auto t11 = b0z - b1z;
            const auto t12 = a0z - a1z;
            const auto t13 = a0z - t * t12;
            const auto t14 = t * t11 + t10 + t13;
            const auto t15 =
                1.0 / (std::pow(t14, 2) + std::pow(t4, 2) + std::pow(t9, 2));
            const auto t16 = b1x + t0 + t2;
            const auto t17 = t3 - vx;
            const auto t18 = b1y + t5 + t7;
            const auto t19 = t8 - vy;
            const auto t20 = b1z + t10 + t12;
            const auto t21 = t13 - vz;
            const auto t22 = a0x + t * t1 - t * t2 + t0;
            const auto t23 = a0y + t * t6 - t * t7 + t5;
            const auto t24 = a0z + t * t11 - t * t12 + t10;
            const auto t25 =
                1.0 / (std::pow(t22, 2) + std::pow(t23, 2) + std::pow(t24, 2));
            dA[0] = -t15
                * (t12 * t14
                + 2 * t15 * (-t14 * t20 - t16 * t4 - t18 * t9)
                    * (t14 * t21 + t17 * t4 + t19 * t9)
                + t16 * t17 + t18 * t19 + t2 * t4 + t20 * t21 + t7 * t9);
            dA[1] = -t22 * t25;
            dA[2] = -t23 * t25;
            dA[3] = -t24 * t25;
        }

        // dA is (16×1) flattened in column-major order
        void line_projection_uv_hessian(
            double t, double vx, double vy, double vz, double dA[16],
            double a0x, double a0y, double a0z,
            double a1x, double a1y, double a1z,
            double b0x, double b0y, double b0z,
            double b1x, double b1y, double b1z)
        {
            const auto t0 = a0x - a1x;
            const auto t1 = -b0x;
            const auto t2 = b1x + t0 + t1;
            const auto t3 = a0y - a1y;
            const auto t4 = -b0y;
            const auto t5 = b1y + t3 + t4;
            const auto t6 = a0z - a1z;
            const auto t7 = -b0z;
            const auto t8 = b1z + t6 + t7;
            const auto t9 = b0x - b1x;
            const auto t10 = t * t0;
            const auto t11 = a0x + t * t9 + t1 - t10;
            const auto t12 = b0y - b1y;
            const auto t13 = t * t3;
            const auto t14 = a0y + t * t12 - t13 + t4;
            const auto t15 = b0z - b1z;
            const auto t16 = t * t6;
            const auto t17 = a0z + t * t15 - t16 + t7;
            const auto t18 = std::pow(t11, 2) + std::pow(t14, 2) + std::pow(t17, 2);
            const auto t19 = 1.0 / t18;
            const auto t20 = -a0x;
            const auto t21 = -a0y;
            const auto t22 = -a0z;
            const auto t23 = t11 * (-t10 - t20 - vx) + t14 * (-t13 - t21 - vy)
                + t17 * (-t16 - t22 - vz);
            const auto t24 = a0x - t * t0;
            const auto t25 = t * t9 + t1 + t24;
            const auto t26 = a0y - t * t3;
            const auto t27 = t * t12 + t26 + t4;
            const auto t28 = a0z - t * t6;
            const auto t29 = t * t15 + t28 + t7;
            const auto t30 = t2 * t25 + t27 * t5 + t29 * t8;
            const auto t31 = 2 * t19;
            const auto t32 = t30 * t31;
            const auto t33 = t19 * (-a1x - t11 * t32 - t20 - t9);
            const auto t34 = t19 * (-a1y - t12 - t14 * t32 - t21);
            const auto t35 = t19 * (-a1z - t15 - t17 * t32 - t22);
            dA[0] = t31
                * (t0 * t2
                - t19 * t23
                    * (std::pow(t2, 2) + std::pow(t5, 2) + std::pow(t8, 2))
                + t3 * t5
                - t32
                    * (t0 * t25 + t2 * (t24 - vx) + t27 * t3 + t29 * t6
                        + t5 * (t26 - vy) + t8 * (t28 - vz))
                + t6 * t8 + 4 * t23 * std::pow(t30, 2) / std::pow(t18, 2));
            dA[1] = t33;
            dA[2] = t34;
            dA[3] = t35;
            dA[4] = t33;
            dA[5] = 0;
            dA[6] = 0;
            dA[7] = 0;
            dA[8] = t34;
            dA[9] = 0;
            dA[10] = 0;
            dA[11] = 0;
            dA[12] = t35;
            dA[13] = 0;
            dA[14] = 0;
            dA[15] = 0;
        }

}
