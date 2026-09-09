#include "auto_derivatives.hpp"
#include <cmath>

namespace polyfem::autogen {

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
        double grad[12])
    {
        const auto t0 = -p2y;
        const auto t1 = p1y + t0;
        const auto t2 = -p2z;
        const auto t3 = p1z + t2;
        const auto t4 = -t3;
        const auto t5 = -p2x;
        const auto t6 = p1x + t5;
        const auto t7 = -t6;
        const auto t8 = std::pow(t7, 2);
        const auto t9 = -t1;
        const auto t10 = std::pow(t9, 2);
        const auto t11 = std::pow(t4, 2);
        const auto t12 = t10 + t11 + t8;
        const auto t13 = 1.0 / t12;
        const auto t14 = p2x - p3x;
        const auto t15 = p2y - p3y;
        const auto t16 = p2z - p3z;
        const auto t17 = -t14 * t7 - t15 * t9 - t16 * t4;
        const auto t18 = t13 * t17;
        const auto t19 = t18 * t4;
        const auto t20 = t16 + t19;
        const auto t21 = t1 * t20;
        const auto t22 = t18 * t9;
        const auto t23 = t15 + t22;
        const auto t24 = t23 * t3;
        const auto t25 = std::pow(t6, 2);
        const auto t26 = std::pow(t1, 2);
        const auto t27 = std::pow(t3, 2);
        const auto t28 = t25 + t26 + t27;
        const auto t29 = 1.0 / t28;
        const auto t30 = t25 * t29;
        const auto t31 = t18 * t7;
        const auto t32 = t14 + t31;
        const auto t33 = t3 * t32;
        const auto t34 = t29 * t6;
        const auto t35 = t33 * t34;
        const auto t36 = t13 * t7;
        const auto t37 = t1 * t15 + t14 * t6 + t16 * t3;
        const auto t38 = t29 * t37;
        const auto t39 = -p3x - t38 * t6 - t5;
        const auto t40 = t1 * t29;
        const auto t41 = t40 * t6;
        const auto t42 = 1 - t30;
        const auto t43 = t1 * t38;
        const auto t44 = -p3y - t0 - t43;
        const auto t45 = p0y - p1y;
        const auto t46 = t1 * t45;
        const auto t47 = p0x - p1x;
        const auto t48 = p0z - p1z;
        const auto t49 = t3 * t48 + t47 * t6;
        const auto t50 = t46 + t49;
        const auto t51 = t29 * t50;
        const auto t52 = t1 * t51;
        const auto t53 = -p0y + p1y;
        const auto t54 = -t52 - t53;
        const auto t55 = -p0z + p1z;
        const auto t56 = -t3 * t51 - t55;
        const auto t57 = t3 * t56;
        const auto t58 = -p0x + p1x;
        const auto t59 = -t51 * t6 - t58;
        const auto t60 = std::pow(t54, 2) + std::pow(t56, 2) + std::pow(t59, 2);
        const auto t61 = 1.0 / t60;
        const auto t62 = -t47 * t7;
        const auto t63 = -t45 * t9;
        const auto t64 = -t4 * t48;
        const auto t65 = t63 + t64;
        const auto t66 = t62 + t65;
        const auto t67 = t13 * t66;
        const auto t68 = t67 * t9;
        const auto t69 = t45 + t68;
        const auto t70 = t20 * t69;
        const auto t71 = t4 * t67;
        const auto t72 = t48 + t71;
        const auto t73 = t23 * t72;
        const auto t74 = t70 - t73;
        const auto t75 = t67 * t7;
        const auto t76 = t47 + t75;
        const auto t77 = t23 * t76;
        const auto t78 = t32 * t69;
        const auto t79 = t77 - t78;
        const auto t80 = -t76;
        const auto t81 = -t20;
        const auto t82 = -t72;
        const auto t83 = -t32;
        const auto t84 = -t1 * (t80 * t81 - t82 * t83) + t3 * t79 + t6 * t74;
        const auto t85 = t61 * t84;
        const auto t86 = std::pow(t28, -1.0 / 2.0);
        const auto t87 = -p3z - t2 - t3 * t38;
        const auto t88 = std::pow(t39, 2) + std::pow(t44, 2) + std::pow(t87, 2);
        const auto t89 = t86 / (std::sqrt(t60) * std::sqrt(t88));
        const auto t90 = t20 * t6;
        const auto t91 = t26 * t29;
        const auto t92 = t24 * t40;
        const auto t93 = t13 * t9;
        const auto t94 = t1 * t93 + 1;
        const auto t95 = -t23;
        const auto t96 = t23 * t6;
        const auto t97 = t1 * t32;
        const auto t98 = t27 * t29;
        const auto t99 = t29 * t3;
        const auto t100 = t21 * t99;
        const auto t101 = t13 * t4;
        const auto t102 = t3 * t34;
        const auto t103 = 1 - t98;
        const auto t104 = t14 + 2 * t31;
        const auto t105 = t101 * t76;
        const auto t106 = 2 * t75;
        const auto t107 = p0x - 2 * p1x + p2x;
        const auto t108 = t106 + t107;
        const auto t109 = 2 * t8;
        const auto t110 = -t109 * t18 - t14 * t7 + t17;
        const auto t111 = std::pow(t12, -2);
        const auto t112 = t111 * t66;
        const auto t113 = -t13 * t66 - 1;
        const auto t114 = t107 * t36 + t109 * t112 + t113;
        const auto t115 = t80 * t93;
        const auto t116 = t108 * t93;
        const auto t117 = -t69;
        const auto t118 = -t114;
        const auto t119 = -t84;
        const auto t120 = t119 * t29;
        const auto t121 = t104 * t9;
        const auto t122 = t23 * t4;
        const auto t123 = t13 * t6;
        const auto t124 =
            std::pow(t69, 2) + std::pow(t72, 2) + std::pow(t76, 2);
        const auto t125 = t119 / t124;
        const auto t126 =
            std::pow(t20, 2) + std::pow(t23, 2) + std::pow(t32, 2);
        const auto t127 = t119 * t13 / t126;
        const auto t128 = t86 / (std::sqrt(t124) * std::sqrt(t126));
        const auto t129 = t15 + 2 * t22;
        const auto t130 = t129 * t7;
        const auto t131 = t129 * t4;
        const auto t132 = 2 * t68;
        const auto t133 = p0y - 2 * p1y + p2y;
        const auto t134 = t132 + t133;
        const auto t135 = 2 * t10;
        const auto t136 = -t135 * t18 - t15 * t9 + t17;
        const auto t137 = t112 * t135 + t113 + t133 * t93;
        const auto t138 = t101 * t134;
        const auto t139 = -t137;
        const auto t140 = t16 + 2 * t19;
        const auto t141 = t72 * t93;
        const auto t142 = 2 * t71;
        const auto t143 = p0z - 2 * p1z + p2z;
        const auto t144 = t142 + t143;
        const auto t145 = 2 * t11;
        const auto t146 = -t145 * t18 - t16 * t4 + t17;
        const auto t147 = t101 * t143 + t112 * t145 + t113;
        const auto t148 = t36 * t82;
        const auto t149 = t144 * t36;
        const auto t150 = -t147;
        const auto t151 = t140 * t7;
        const auto t152 = t32 * t9;
        const auto t153 = t13 * t3;
        const auto t154 = 2 * t6;
        const auto t155 = t154 * t67 + t58;
        const auto t156 = p1x - 2 * p2x + p3x;
        const auto t157 = t154 * t18 + t156;
        const auto t158 = t106 * t6 + 2 * t62 + t65;
        const auto t159 = t13 * t158;
        const auto t160 = t111 * t17;
        const auto t161 = t18 + 1;
        const auto t162 = t154 * t160 * t7 + t156 * t36 + t161;
        const auto t163 = t29 * t84;
        const auto t164 = t69 * t9;
        const auto t165 = t4 * t72;
        const auto t166 = 1.0 / t88;
        const auto t167 = t23 * t93;
        const auto t168 = t101 * t20;
        const auto t169 = 2 * t1;
        const auto t170 = t169 * t67 + t53;
        const auto t171 = t170 * t7;
        const auto t172 = p1y - 2 * p2y + p3y;
        const auto t173 = t169 * t18 + t172;
        const auto t174 = t1 * t132 + t62 + 2 * t63 + t64;
        const auto t175 = t160 * t169 * t9 + t161 + t172 * t93;
        const auto t176 = t32 * t36;
        const auto t177 = t166 * t84;
        const auto t178 = 2 * t3;
        const auto t179 = t178 * t67 + t55;
        const auto t180 = p1z - 2 * p2z + p3z;
        const auto t181 = t178 * t18 + t180;
        const auto t182 = t142 * t3 + t62 + t63 + 2 * t64;
        const auto t183 = t13 * t182;
        const auto t184 = t101 * t180 + t160 * t178 * t4 + t161;
        const auto t185 = t1 * t72;
        const auto t186 = t13 * t8 - 1;
        const auto t187 = t10 * t13 - 1;
        const auto t188 = t6 * t69;
        const auto t189 = t11 * t13 - 1;
        grad[0] = t89
            * (t1 * (t20 * (t36 * t6 + 1) + t35) - t3 * (t39 * t41 + t42 * t44)
               + t30 * (t21 - t24) - t85 * (t34 * t57 + t41 * t54 - t42 * t59));
        grad[1] = -t89
            * (-t3 * (t1 * t36 * t95 - t83 * t94) + t6 * (t20 * t94 + t92)
               + t85 * (t40 * t57 + t41 * t59 - t54 * (1 - t91))
               + t91 * (-t33 + t90));
        grad[2] = t89
            * (-t1 * (t102 * t87 + t103 * t39)
               + t6 * (t100 + t23 * (t101 * t3 + 1))
               - t85 * (t102 * t59 - t103 * t56 + t3 * t40 * t54)
               + t98 * (t96 - t97));
        grad[3] = t128
            * (-t1
                   * (-t104 * t105 + t108 * t13 * t32 * t4 - t110 * t13 * t72
                      - t114 * t20)
               - t120 * t6
               - t123
                   * (t104 * t4 * t69 - t108 * t122 + t108 * t20 * t9
                      - t121 * t72)
               - t125 * (-t101 * t108 * t82 - t116 * t117 + t118 * t80)
               - t127 * (-t104 * t4 * t81 + t110 * t83 - t121 * t95)
               + t3
                   * (t104 * t115 + t110 * t117 * t13 - t116 * t83 - t118 * t95)
               - t74);
        grad[4] = t128
            * (-t1 * t120
               + t1 * t13
                   * (t130 * t82 - t131 * t80 + t134 * t4 * t83
                      - t134 * t7 * t81)
               - t125 * (t117 * t139 - t134 * t36 * t80 - t138 * t82)
               - t127 * (-t130 * t83 - t131 * t81 + t136 * t95) + t20 * t76
               - t3
                   * (-t129 * t36 * t69 + t13 * t134 * t23 * t7
                      - t13 * t136 * t76 - t137 * t32)
               - t32 * t72
               + t6
                   * (t101 * t117 * t129 + t13 * t136 * t82 - t138 * t95
                      - t139 * t81));
        grad[5] = t128
            * (t1 * (t13 * t146 * t80 + t140 * t148 - t149 * t81 - t150 * t83)
               - t120 * t3
               - t125 * (-t117 * t144 * t93 - t149 * t80 + t150 * t82)
               - t127 * (-t140 * t9 * t95 + t146 * t81 - t151 * t83)
               - t153
                   * (t140 * t76 * t9 - t144 * t152 + t144 * t23 * t7
                      - t151 * t69)
               - t6
                   * (t13 * t144 * t20 * t9 - t13 * t146 * t69 - t140 * t141
                      - t147 * t23)
               - t79);
        grad[6] = t89
            * (-t1 * (t101 * t155 * t32 - t105 * t157 - t159 * t20 + t162 * t72)
               - t123
                   * (-t122 * t155 + t155 * t20 * t9 + t157 * t4 * t69
                      - t157 * t72 * t9)
               + t13 * t61 * t84 * (t155 * t164 + t155 * t165 + t158 * t76)
               - t163 * t6
               + t166 * t84 * (t157 * t167 + t157 * t168 + t162 * t32)
               + t3
                   * (t115 * t157 - t117 * t162 - t155 * t83 * t93 + t159 * t95)
               + t70 - t73);
        grad[7] = t89
            * (t1 * t13
                   * (-t170 * t32 * t4 + t171 * t20 + t173 * t4 * t76
                      - t173 * t7 * t72)
               - t1 * t163 + t13 * t85 * (t165 * t170 + t171 * t76 + t174 * t69)
               + t177 * (t168 * t173 + t173 * t176 + t175 * t23)
               - t3
                   * (-t29 * t39 * (2 * t46 + t49 - 2 * t50 * t91)
                      + t34 * t44 * (-2 * t52 - t53)
                      + t34 * t54 * (t172 + 2 * t43)
                      + t59
                          * (-t172 * t40 - 2 * t26 * t37 / std::pow(t28, 2)
                             + t29 * t37 + 1))
               + t39 * t56 - t59 * t87
               + t6
                   * (t101 * t170 * t23 - t101 * t173 * t69 - t13 * t174 * t20
                      + t175 * t72));
        grad[8] = t89
            * (t1 * (t148 * t181 - t179 * t36 * t81 + t183 * t83 - t184 * t80)
               + t13 * t61 * t84 * (t164 * t179 + t179 * t7 * t76 + t182 * t72)
               - t153
                   * (-t152 * t179 + t179 * t23 * t7 - t181 * t69 * t7
                      + t181 * t76 * t9)
               - t163 * t3
               + t166 * t84 * (t167 * t181 + t176 * t181 + t184 * t20)
               - t6
                   * (-t141 * t181 + t179 * t20 * t93 - t183 * t23 + t184 * t69)
               + t77 - t78);
        grad[9] = t89
            * (-t1 * (t186 * t72 - t3 * t34 * t76)
               + t166 * t84 * (t186 * t32 + t40 * t96 + t90 * t99)
               + t3 * (-t117 * t186 + t36 * t80 * t9)
               - t30 * (-t185 + t3 * t69));
        grad[10] = t89
            * (t177 * (t100 + t187 * t23 + t34 * t97)
               + t3 * (-t187 * t76 + t188 * t40)
               + t6 * (t187 * t72 - t3 * t40 * t69)
               + t91 * (t3 * t76 - t6 * t72));
        grad[11] = t89
            * (t1 * (t148 * t4 - t189 * t80)
               + t166 * t84 * (t189 * t20 + t35 + t92)
               - t6 * (-t185 * t99 + t189 * t69) - t98 * (t1 * t76 - t188));
    }

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
        double hess[144])
    {
        const auto t0 = -p2y;
        const auto t1 = p1y + t0;
        const auto t2 = -p2x;
        const auto t3 = p1x + t2;
        const auto t4 = std::pow(t3, 2);
        const auto t5 = std::pow(t1, 2);
        const auto t6 = -p2z;
        const auto t7 = p1z + t6;
        const auto t8 = std::pow(t7, 2);
        const auto t9 = t4 + t5 + t8;
        const auto t10 = 1.0 / t9;
        const auto t11 = -p1y;
        const auto t12 = p0y + t11;
        const auto t13 = t1 * t12;
        const auto t14 = -p1x;
        const auto t15 = p0x + t14;
        const auto t16 = t15 * t3;
        const auto t17 = -p1z;
        const auto t18 = p0z + t17;
        const auto t19 = t18 * t7;
        const auto t20 = t16 + t19;
        const auto t21 = t13 + t20;
        const auto t22 = t10 * t21;
        const auto t23 = t1 * t22;
        const auto t24 = -p0y;
        const auto t25 = p1y + t24;
        const auto t26 = t23 + t25;
        const auto t27 = -t26;
        const auto t28 = t1 * t27;
        const auto t29 = t10 * t3;
        const auto t30 = t28 * t29;
        const auto t31 = t22 * t7;
        const auto t32 = -p0z;
        const auto t33 = p1z + t32;
        const auto t34 = t31 + t33;
        const auto t35 = -t34;
        const auto t36 = t35 * t7;
        const auto t37 = t29 * t36;
        const auto t38 = t10 * t4;
        const auto t39 = t38 - 1;
        const auto t40 = -t39;
        const auto t41 = t22 * t3;
        const auto t42 = -p0x;
        const auto t43 = p1x + t42;
        const auto t44 = t41 + t43;
        const auto t45 = -t44;
        const auto t46 = t40 * t45;
        const auto t47 = t30 + t37 - t46;
        const auto t48 = -t7;
        const auto t49 = -t3;
        const auto t50 = std::pow(t49, 2);
        const auto t51 = -t1;
        const auto t52 = std::pow(t51, 2);
        const auto t53 = std::pow(t48, 2);
        const auto t54 = t50 + t52 + t53;
        const auto t55 = 1.0 / t54;
        const auto t56 = -p3x;
        const auto t57 = p2x + t56;
        const auto t58 = -p3y;
        const auto t59 = p2y + t58;
        const auto t60 = -p3z;
        const auto t61 = p2z + t60;
        const auto t62 = -t48 * t61 - t49 * t57 - t51 * t59;
        const auto t63 = t55 * t62;
        const auto t64 = t48 * t63;
        const auto t65 = t61 + t64;
        const auto t66 = t1 * t65;
        const auto t67 = t51 * t63;
        const auto t68 = t59 + t67;
        const auto t69 = t68 * t7;
        const auto t70 = t49 * t63;
        const auto t71 = t57 + t70;
        const auto t72 = t7 * t71;
        const auto t73 = t29 * t72;
        const auto t74 = t3 * t55;
        const auto t75 = t49 * t74;
        const auto t76 = t75 + 1;
        const auto t77 = t65 * t76;
        const auto t78 = t73 + t77;
        const auto t79 = t1 * t59;
        const auto t80 = t3 * t57;
        const auto t81 = t61 * t7;
        const auto t82 = t80 + t81;
        const auto t83 = t79 + t82;
        const auto t84 = t10 * t83;
        const auto t85 = t3 * t84;
        const auto t86 = p3x + t2;
        const auto t87 = -t85 - t86;
        const auto t88 = t1 * t87;
        const auto t89 = t1 * t84;
        const auto t90 = p3y + t0;
        const auto t91 = -t89 - t90;
        const auto t92 = t40 * t91;
        const auto t93 = t29 * t88 + t92;
        const auto t94 = t1 * t78 + t38 * (t66 - t69) - t7 * t93;
        const auto t95 = std::pow(t9, -2);
        const auto t96 = t4 * t95;
        const auto t97 = t5 * t96;
        const auto t98 = t8 * t96;
        const auto t99 = t97 + t98;
        const auto t100 = -t15;
        const auto t101 = t100 * t49;
        const auto t102 = -t12;
        const auto t103 = t102 * t51;
        const auto t104 = -t18;
        const auto t105 = t104 * t48;
        const auto t106 = t103 + t105;
        const auto t107 = t101 + t106;
        const auto t108 = t107 * t55;
        const auto t109 = t108 * t51;
        const auto t110 = t109 + t12;
        const auto t111 = t110 * t65;
        const auto t112 = t108 * t48;
        const auto t113 = t112 + t18;
        const auto t114 = t113 * t68;
        const auto t115 = t111 - t114;
        const auto t116 = t108 * t49;
        const auto t117 = t116 + t15;
        const auto t118 = t117 * t68;
        const auto t119 = t110 * t71;
        const auto t120 = t118 - t119;
        const auto t121 = -t117;
        const auto t122 = -t65;
        const auto t123 = -t113;
        const auto t124 = -t71;
        const auto t125 =
            -t1 * (t121 * t122 - t123 * t124) + t115 * t3 + t120 * t7;
        const auto t126 =
            std::pow(t27, 2) + std::pow(t35, 2) + std::pow(t45, 2);
        const auto t127 = 1.0 / t126;
        const auto t128 = t125 * t127;
        const auto t129 = 3 * t128;
        const auto t130 = std::pow(t9, -1.0 / 2.0);
        const auto t131 = t7 * t84;
        const auto t132 = p3z + t6;
        const auto t133 = -t131 - t132;
        const auto t134 =
            std::pow(t133, 2) + std::pow(t87, 2) + std::pow(t91, 2);
        const auto t135 = t130 / std::sqrt(t134);
        const auto t136 = t135 / std::pow(t126, 3.0 / 2.0);
        const auto t137 = t3 * t45;
        const auto t138 = t1 * t10;
        const auto t139 = t137 * t138;
        const auto t140 = t138 * t36;
        const auto t141 = t10 * t5;
        const auto t142 = t141 - 1;
        const auto t143 = -t142;
        const auto t144 = t143 * t27;
        const auto t145 = t139 + t140 - t144;
        const auto t146 = t3 * t65;
        const auto t147 = t146 - t72;
        const auto t148 = t138 * t69;
        const auto t149 = t1 * t55;
        const auto t150 = t149 * t51;
        const auto t151 = t150 + 1;
        const auto t152 = t148 + t151 * t65;
        const auto t153 = -t68;
        const auto t154 = t153 * t49;
        const auto t155 =
            t141 * t147 + t152 * t3 - t7 * (-t124 * t151 + t149 * t154);
        const auto t156 = t1 * t3;
        const auto t157 = t10 * t8;
        const auto t158 = t157 - 2;
        const auto t159 = t10 * t125;
        const auto t160 = t159 * (-t141 - t158 - t38);
        const auto t161 = t127 * t47;
        const auto t162 = 3 * t125;
        const auto t163 = t161 * t162;
        const auto t164 =
            t136 * (-t145 * t163 + t145 * t94 - t155 * t47 - t156 * t160);
        const auto t165 = t3 * t68;
        const auto t166 = t1 * t71;
        const auto t167 = t165 - t166;
        const auto t168 = t10 * t7;
        const auto t169 = t168 * t66;
        const auto t170 = t55 * t7;
        const auto t171 = t170 * t48;
        const auto t172 = t171 + 1;
        const auto t173 = t172 * t68;
        const auto t174 = t169 + t173;
        const auto t175 = t133 * t3;
        const auto t176 = t157 - 1;
        const auto t177 = -t176;
        const auto t178 = t177 * t87;
        const auto t179 = t168 * t175 + t178;
        const auto t180 = -t1 * t179 + t157 * t167 + t174 * t3;
        const auto t181 = t137 * t168;
        const auto t182 = t168 * t28;
        const auto t183 = t177 * t35;
        const auto t184 = t181 + t182 - t183;
        const auto t185 = t3 * t7;
        const auto t186 =
            t136 * (-t160 * t185 - t163 * t184 + t180 * t47 + t184 * t94);
        const auto t187 = t50 * t55;
        const auto t188 = p2x + t14 + t187 * t3;
        const auto t189 = 2 * t68;
        const auto t190 = t188 * t189;
        const auto t191 = 2 * t70;
        const auto t192 = t191 + t57;
        const auto t193 = t192 * t51;
        const auto t194 = t193 * t76;
        const auto t195 = 2 * t75;
        const auto t196 = t195 + 1;
        const auto t197 = t51 * t71;
        const auto t198 = t196 * t197;
        const auto t199 = t49 * t57;
        const auto t200 = 2 * t63;
        const auto t201 = -t199 - t200 * t50 + t62;
        const auto t202 = -t201;
        const auto t203 = t138 * t3;
        const auto t204 = std::pow(t54, -2);
        const auto t205 = t204 * t52;
        const auto t206 = 2 * t116;
        const auto t207 = 2 * p1x;
        const auto t208 = p0x + p2x - t207;
        const auto t209 = t206 + t208;
        const auto t210 = t209 * t3;
        const auto t211 = t204 * t53;
        const auto t212 = -t188;
        const auto t213 = t121 * t55;
        const auto t214 = -t110;
        const auto t215 = t214 * t51;
        const auto t216 = t215 * t55;
        const auto t217 = t123 * t48;
        const auto t218 = t217 * t55;
        const auto t219 = t49 * t55;
        const auto t220 = t208 * t219;
        const auto t221 = t204 * t50;
        const auto t222 = 2 * t107;
        const auto t223 = -t108 - 1;
        const auto t224 = t220 + t221 * t222 + t223;
        const auto t225 = -t224;
        const auto t226 = -t125;
        const auto t227 =
            std::pow(t110, 2) + std::pow(t113, 2) + std::pow(t117, 2);
        const auto t228 = 1.0 / t227;
        const auto t229 = t226 * t228;
        const auto t230 = t122 * t51;
        const auto t231 = t153 * t48;
        const auto t232 = t124 * t48;
        const auto t233 = 2 * t122;
        const auto t234 = t192 * t48;
        const auto t235 = t201 * t48 * t74 + t212 * t233 + t234 * t76;
        const auto t236 = t121 * t225 - t209 * t216 - t209 * t218;
        const auto t237 = t230 - t231;
        const auto t238 = t4 * t55;
        const auto t239 = t166 * t29;
        const auto t240 = t239 + t68 * t76;
        const auto t241 =
            t1 * (-t122 * t76 + t232 * t74) + t237 * t238 - t240 * t7;
        const auto t242 = t228 * t241;
        const auto t243 = t110 * t3;
        const auto t244 = t138 * t243;
        const auto t245 = t113 * t3;
        const auto t246 = t168 * t245;
        const auto t247 = -t117 * t76 + t244 + t246;
        const auto t248 = t117 * t55;
        const auto t249 = t234 * t248;
        const auto t250 = t48 * t55;
        const auto t251 = t250 * t71;
        const auto t252 = t209 * t251;
        const auto t253 = t202 * t55;
        const auto t254 = t113 * t253;
        const auto t255 = t224 * t65;
        const auto t256 = t249 - t252 - t254 + t255;
        const auto t257 = t121 * t51;
        const auto t258 = t257 * t55;
        const auto t259 = t124 * t51;
        const auto t260 = t259 * t55;
        const auto t261 = t201 * t55;
        const auto t262 = t48 * t68;
        const auto t263 =
            -t110 * t192 * t48 + t113 * t193 + t209 * t262 - t209 * t51 * t65;
        const auto t264 = -t1 * t256 + t115 - t263 * t74
            - t7 * (-t153 * t225 + t192 * t258 - t209 * t260 + t214 * t261);
        const auto t265 = -t264;
        const auto t266 = -t122 * t234 + t124 * t201 - t153 * t193;
        const auto t267 =
            std::pow(t65, 2) + std::pow(t68, 2) + std::pow(t71, 2);
        const auto t268 = 1.0 / t267;
        const auto t269 = t268 * t55;
        const auto t270 = t241 * t269;
        const auto t271 = 3 * t226;
        const auto t272 = t271 / std::pow(t227, 2);
        const auto t273 = t247 * t272;
        const auto t274 = t228 * t247;
        const auto t275 = t226 * t268;
        const auto t276 = t275 * t55;
        const auto t277 = t274 * t276;
        const auto t278 = t10 * t241;
        const auto t279 = t29 * t69;
        const auto t280 = t29 * t66;
        const auto t281 = t10 * t226;
        const auto t282 = t274 * t281;
        const auto t283 = t278 * t3 + t279 - t280 + t282 * t3;
        const auto t284 = -t228 * t247 * t265 + t236 * t242 + t236 * t273
            + t266 * t270 + t266 * t277 + t283;
        const auto t285 = t130 / (std::sqrt(t227) * std::sqrt(t267));
        const auto t286 = 2 * t1;
        const auto t287 = t286 * t95;
        const auto t288 = t287 * t69;
        const auto t289 = 2 * t67;
        const auto t290 = t289 + t59;
        const auto t291 = t290 * t48;
        const auto t292 = t149 * t291;
        const auto t293 = t52 * t55;
        const auto t294 = 2 * t293;
        const auto t295 = t294 - 1;
        const auto t296 = -t51 * t59 - 2 * t52 * t55 * t62 + t62;
        const auto t297 = -t296;
        const auto t298 = t290 * t51;
        const auto t299 = -t295;
        const auto t300 = t124 * t299;
        const auto t301 = 2 * t74;
        const auto t302 = t295 * t55;
        const auto t303 = 2 * p1y;
        const auto t304 = p0y + p2y - t303;
        const auto t305 = t51 * t55;
        const auto t306 = t304 * t305;
        const auto t307 = t205 * t222 + t223 + t306;
        const auto t308 = t3 * t307;
        const auto t309 = -t138 * t308;
        const auto t310 = t49 * t76;
        const auto t311 = 2 * t109;
        const auto t312 = t304 + t311;
        const auto t313 = t312 * t55;
        const auto t314 = t1 * t113;
        const auto t315 = 2 * t3;
        const auto t316 = t315 * t95;
        const auto t317 = t316 * t7;
        const auto t318 = t314 * t317;
        const auto t319 = t1 * t117;
        const auto t320 = 2 * t96;
        const auto t321 = t318 + t319 * t320;
        const auto t322 =
            -t10 * t3 * t312 * t48 * t55 * t7 + t309 + t310 * t313 + t321;
        const auto t323 = t121 * t49;
        const auto t324 = t323 * t55;
        const auto t325 = -t307;
        const auto t326 = t214 * t325 - t218 * t312 - t312 * t324;
        const auto t327 = t195 * t230;
        const auto t328 = t117 * t65;
        const auto t329 = t113 * t71;
        const auto t330 = t123 * t49;
        const auto t331 = t121 * t48;
        const auto t332 = t122 * t49;
        const auto t333 = t110 * t219;
        const auto t334 = -t117 * t297 * t55 + t290 * t333 + t307 * t71
            - t312 * t49 * t55 * t68;
        const auto t335 = t214 * t48;
        const auto t336 = t335 * t55;
        const auto t337 = t123 * t55;
        const auto t338 =
            t149 * (t232 * t312 + t290 * t330 - t290 * t331 - t312 * t332)
            + t3 * (-t122 * t325 - t231 * t313 + t290 * t336 + t296 * t337)
            + t328 - t329 + t334 * t7;
        const auto t339 = t124 * t49;
        const auto t340 = -t122 * t291 + t153 * t296 - t290 * t339;
        const auto t341 = t1 * t278 + t1 * t282
            - t1 * t55
                * (2 * t124 * t3 * t48 * t51 * t55 + t290 * t48 * t76
                   - t291 * t75 - t327)
            - t228 * t247 * t338 + t242 * t326 + t270 * t340 + t273 * t326
            + t277 * t340 - t73 - t77;
        const auto t342 = 2 * t65;
        const auto t343 = t7 * t96;
        const auto t344 = t342 * t343;
        const auto t345 = 2 * t64;
        const auto t346 = t345 + t61;
        const auto t347 = t346 * t75;
        const auto t348 = t168 * t347;
        const auto t349 = t53 * t55;
        const auto t350 = 2 * t349;
        const auto t351 = t350 - 1;
        const auto t352 = -t48 * t61 - 2 * t53 * t55 * t62 + t62;
        const auto t353 = -t352;
        const auto t354 = t353 * t55;
        const auto t355 = -t351;
        const auto t356 = t153 * t355;
        const auto t357 = t305 * t346;
        const auto t358 = 2 * t250;
        const auto t359 = t230 * t358 + t305 * t352 + t357 * t48;
        const auto t360 = t351 * t55;
        const auto t361 = t117 * t7;
        const auto t362 = t320 * t361;
        const auto t363 = 2 * t112;
        const auto t364 = 2 * p1z;
        const auto t365 = p0z + p2z - t364;
        const auto t366 = t363 + t365;
        const auto t367 = t366 * t55;
        const auto t368 = t250 * t365;
        const auto t369 = t211 * t222 + t223 + t368;
        const auto t370 = t110 * t7;
        const auto t371 = t287 * t3;
        const auto t372 = t370 * t371;
        const auto t373 = t366 * t51;
        const auto t374 = t138 * t74;
        const auto t375 = t372 - t373 * t374;
        const auto t376 = -t10 * t3 * t369 * t7 + t310 * t367 + t362 + t375;
        const auto t377 = t371 * t72;
        const auto t378 = -t377;
        const auto t379 = t138 * t347 + t378;
        const auto t380 = -t369;
        const auto t381 = t123 * t380 - t216 * t366 - t324 * t366;
        const auto t382 = t113 * t357;
        const auto t383 = t51 * t65;
        const auto t384 = t367 * t383;
        const auto t385 = t110 * t354;
        const auto t386 = t369 * t68;
        const auto t387 = t382 - t384 - t385 + t386;
        const auto t388 = t330 * t55;
        const auto t389 = t122 * t219;
        const auto t390 = t110 * t49;
        const auto t391 =
            -t117 * t346 * t51 + t197 * t366 + t346 * t390 - t366 * t49 * t68;
        const auto t392 =
            -t1 * (-t124 * t380 + t213 * t352 + t346 * t388 - t366 * t389)
            + t120 - t170 * t391 - t3 * t387;
        const auto t393 = -t392;
        const auto t394 = t346 * t51;
        const auto t395 = t122 * t352 - t153 * t394 - t339 * t346;
        const auto t396 = -t228 * t247 * t393 + t240 + t242 * t381 + t270 * t395
            + t273 * t381 + t277 * t395 + t278 * t7 + t282 * t7
            + t7 * (t320 * t69 + t357 * t76 + t379);
        const auto t397 = 2 * t238 - 1;
        const auto t398 = t197 * t55;
        const auto t399 = t219 * t4 + t3;
        const auto t400 = t189 * t55;
        const auto t401 = t3 * t63;
        const auto t402 = 2 * t401;
        const auto t403 = 2 * p2x;
        const auto t404 = p1x + p3x - t403;
        const auto t405 = t402 + t404;
        const auto t406 = t305 * t405;
        const auto t407 = t219 * t404;
        const auto t408 = t204 * t62;
        const auto t409 = t408 * t49;
        const auto t410 = t63 + 1;
        const auto t411 = t315 * t409 + t407 + t410;
        const auto t412 = t138 * t411;
        const auto t413 = t3 * t412;
        const auto t414 = -t397;
        const auto t415 = -t399;
        const auto t416 = t411 * t48;
        const auto t417 = t108 * t3;
        const auto t418 = 2 * t417;
        const auto t419 = t418 + t43;
        const auto t420 = t383 * t419;
        const auto t421 = t262 * t419;
        const auto t422 = t405 * t48;
        const auto t423 = t110 * t422;
        const auto t424 = t405 * t51;
        const auto t425 = t113 * t424;
        const auto t426 = t420 - t421 + t423 - t425;
        const auto t427 = t250 * t405;
        const auto t428 = 2 * t101;
        const auto t429 = t106 + t116 * t315 + t428;
        const auto t430 = t55 * t65;
        const auto t431 = t113 * t411 - t117 * t427 + t251 * t419 - t429 * t430;
        const auto t432 = t153 * t55;
        const auto t433 = t1 * t431 - t111 + t114 + t426 * t74
            - t7 * (-t214 * t411 + t258 * t405 - t260 * t419 + t429 * t432);
        const auto t434 = -t433;
        const auto t435 = t122 * t427 + t124 * t411 + t153 * t406;
        const auto t436 = t268 * t435;
        const auto t437 = t121 * t429 + t215 * t419 + t217 * t419;
        const auto t438 = t437 * t55;
        const auto t439 = t3 * t419;
        const auto t440 = 2 * t121;
        const auto t441 = t229 * t55;
        const auto t442 = t286 * t63;
        const auto t443 = 2 * p2y;
        const auto t444 = p1y + p3y - t443;
        const auto t445 = t442 + t444;
        const auto t446 = t445 * t48;
        const auto t447 = t149 * t446;
        const auto t448 = 2 * t150;
        const auto t449 = t448 + 1;
        const auto t450 = t305 * t444;
        const auto t451 = t408 * t51;
        const auto t452 = t286 * t451 + t410 + t450;
        const auto t453 = t168 * t452;
        const auto t454 = t445 * t49;
        const auto t455 = t204 * t3;
        const auto t456 = t455 * t51;
        const auto t457 = t124 * t3;
        const auto t458 = t452 * t76;
        const auto t459 = 2 * t141;
        const auto t460 = 2 * t13 + t20 - t21 * t459;
        const auto t461 = t138 * t460;
        const auto t462 = 2 * t139 + 2 * t140 + t461;
        const auto t463 = -t27 * (1 - t459) + t462;
        const auto t464 = 2 * t23;
        const auto t465 = -t25 - t464;
        const auto t466 = t157 * t465;
        const auto t467 = -t40 * t465 + t466;
        const auto t468 = t127 * t159;
        const auto t469 = t3 * t468;
        const auto t470 = t219 * t71;
        const auto t471 = t430 * t446 + t445 * t470 + t452 * t68;
        const auto t472 = 1.0 / t134;
        const auto t473 = t472 * t94;
        const auto t474 = t35 * t87;
        const auto t475 = t133 * t45;
        const auto t476 = t108 * t286;
        const auto t477 = t25 + t476;
        const auto t478 = t477 * t49;
        const auto t479 = t478 * t65;
        const auto t480 = t117 * t446;
        const auto t481 = t477 * t48;
        const auto t482 = t481 * t71;
        const auto t483 = t113 * t454;
        const auto t484 = t479 + t480 - t482 - t483;
        const auto t485 = t29 * t91;
        const auto t486 = 2 * t89;
        const auto t487 = t444 + t486;
        const auto t488 = t27 * t3;
        const auto t489 = t10 * t488;
        const auto t490 = t10 * t87;
        const auto t491 = t5 * t95;
        const auto t492 = 2 * t83;
        const auto t493 = -t10 * t83 - 1;
        const auto t494 = t262 * t55;
        const auto t495 = t477 * t494;
        const auto t496 = t250 * t445;
        const auto t497 = t110 * t496;
        const auto t498 = 2 * t103;
        const auto t499 = t101 + t105 + t109 * t286 + t498;
        const auto t500 = t430 * t499;
        const auto t501 = t113 * t452;
        const auto t502 = t149 * t484 + t3 * (t495 - t497 - t500 + t501) + t474
            - t475
            - t7
                * (t45 * (-t138 * t444 - t491 * t492 - t493) - t460 * t490
                   + t465 * t485 + t487 * t489);
        const auto t503 = t110 * t499 + t113 * t481 + t117 * t478;
        const auto t504 = t127 * t55;
        const auto t505 = t504 * t94;
        const auto t506 = t159 * t161;
        const auto t507 = std::pow(t126, -2);
        const auto t508 = -t1 * t10 * t94 + t1 * t506
            - t1 * t55
                * (2 * t1 * t124 * t3 * t48 * t55 - t1 * t233 * t75
                   + t445 * t48 * t76 - t446 * t75)
            - t125 * t127 * t47 * t471 * t472
            - 3 * t125 * t47 * t503 * t507 * t55 - t127 * t47 * t502
            + t471 * t473 + t503 * t505 + t78;
        const auto t509 = std::pow(t126, -1.0 / 2.0);
        const auto t510 = t135 * t509;
        const auto t511 = 2 * t131;
        const auto t512 = 2 * p2z;
        const auto t513 = p1z + p3z - t512;
        const auto t514 = t511 + t513;
        const auto t515 = 2 * t38;
        const auto t516 = t1 * t40;
        const auto t517 = t168
            * (2 * t1 * t10 * t3 * t7 * t87 - t1 * t38 * t514 - t514 * t516
               - t515 * t7 * t91);
        const auto t518 = 2 * t7;
        const auto t519 = 2 * t157;
        const auto t520 = 1 - t519;
        const auto t521 = t8 * t95;
        const auto t522 = -t168 * t513 - t492 * t521 - t493;
        const auto t523 = t518 * t95;
        const auto t524 = t200 * t7;
        const auto t525 = t513 + t524;
        const auto t526 = t51 * t525;
        const auto t527 = t170 * t526;
        const auto t528 = 2 * t171;
        const auto t529 = t528 + 1;
        const auto t530 = t55 * t68;
        const auto t531 = t250 * t513;
        const auto t532 = t408 * t48;
        const auto t533 = t410 + t518 * t532 + t531;
        const auto t534 = t1 * t533;
        const auto t535 = 2 * t108;
        const auto t536 = t535 * t7;
        const auto t537 = t33 + t536;
        const auto t538 = t49 * t537;
        const auto t539 = t110 * t51;
        const auto t540 = 2 * t105;
        const auto t541 = t101 + t103 + t112 * t518 + t540;
        const auto t542 = t113 * t541 + t117 * t538 + t537 * t539;
        const auto t543 = t505 * t542;
        const auto t544 = t305 * t525;
        const auto t545 = t470 * t525 + t533 * t65 + t544 * t68;
        const auto t546 = t473 * t545;
        const auto t547 = t506 * t7;
        const auto t548 = t13 + t16 + 2 * t19 - t21 * t519;
        const auto t549 = t168 * t548 + 2 * t181 + 2 * t182;
        const auto t550 = -t35 * t520 + t549;
        const auto t551 = 2 * t31;
        const auto t552 = -t33 - t551;
        const auto t553 = t141 * t552;
        const auto t554 = -t40 * t552 + t553;
        const auto t555 = t197 * t537;
        const auto t556 = t390 * t525;
        const auto t557 = t49 * t68;
        const auto t558 = t537 * t557;
        const auto t559 = t117 * t526;
        const auto t560 = t555 + t556 - t558 - t559;
        const auto t561 = t383 * t55;
        const auto t562 = t110 * t533 - t113 * t544 - t530 * t541 + t537 * t561;
        const auto t563 = t124 * t55;
        const auto t564 =
            -t1 * (-t121 * t533 + t388 * t525 - t389 * t537 + t541 * t563)
            - t118 + t119 - t170 * t560 + t3 * t562;
        const auto t565 = t161 * t564;
        const auto t566 = t187 - 1;
        const auto t567 = -t566;
        const auto t568 = t138 * t165;
        const auto t569 = t146 * t168;
        const auto t570 = t566 * t71 + t568 + t569;
        const auto t571 = t29 * t361;
        const auto t572 = t113 * t566;
        const auto t573 = t571 - t572;
        const auto t574 = t219 * t257;
        const auto t575 = t214 * t567;
        const auto t576 =
            -t1 * t573 + t38 * (-t314 + t370) - t7 * (t574 + t575);
        const auto t577 = t510
            * (t125 * t127 * t47 * t472 * t570 - t161 * t576 - t473 * t570
               - t7 * (-t203 * (t187 + t75) + t51 * t55 * (t3 * t567 + t310)));
        const auto t578 = t205 * t49;
        const auto t579 = t293 - 1;
        const auto t580 = -t579;
        const auto t581 = -t141 + t293 - 1;
        const auto t582 = t38 + t76;
        const auto t583 = t169 + t239 + t579 * t68;
        const auto t584 = t245 - t361;
        const auto t585 = t138 * t370;
        const auto t586 = t113 * t579;
        const auto t587 = t585 - t586;
        const auto t588 = t117 * t579;
        const auto t589 = t244 - t588;
        const auto t590 = -t141 * t584 - t3 * t587 + t589 * t7;
        const auto t591 = t510
            * (t125 * t127 * t47 * t472 * t583 + t127 * t47 * t590 - t473 * t583
               - t7 * (t10 * t4 * t581 - t141 * t582 - t3 * t578 - t580 * t76));
        const auto t592 = t349 - 1;
        const auto t593 = -t157 + t349 - 1;
        const auto t594 = t148 + t592 * t65 + t73;
        const auto t595 = t243 - t319;
        const auto t596 = t168 * t314;
        const auto t597 = t110 * t592;
        const auto t598 = t596 - t597;
        const auto t599 = t250 * t330;
        const auto t600 = -t592;
        const auto t601 = t121 * t600;
        const auto t602 = -t1 * (t599 + t601) - t157 * t595 - t3 * t598;
        const auto t603 = t125 * t472;
        const auto t604 = t161 * t603;
        const auto t605 = t510
            * (t1 * (-t157 * t582 + t38 * t593 + t592 * t76 + t98) - t161 * t602
               - t473 * t594 + t594 * t604);
        const auto t606 = t491 * t8;
        const auto t607 = t606 + t97;
        const auto t608 = t1 * t7;
        const auto t609 = t127 * t145;
        const auto t610 = t136
            * (t145 * t180 - t155 * t184 - t160 * t608 - t162 * t184 * t609);
        const auto t611 = 2 * t71;
        const auto t612 = t3 * t611;
        const auto t613 = t491 * t612;
        const auto t614 = t193 * t74;
        const auto t615 = t138 * t614;
        const auto t616 = 2 * t187;
        const auto t617 = t616 - 1;
        const auto t618 = -t617;
        const auto t619 = t122 * t618;
        const auto t620 = t219 * t234;
        const auto t621 = 2 * t219;
        const auto t622 = t232 * t621 + t261 * t48 + t620;
        const auto t623 = t55 * t617;
        const auto t624 = t209 * t48;
        const auto t625 = t149 * t168;
        const auto t626 = -t624 * t625;
        const auto t627 = 2 * t491;
        const auto t628 = t243 * t627;
        const auto t629 = t151 * t305;
        const auto t630 = -t1 * t10 * t224 * t3 + t209 * t629 + t626 + t628;
        const auto t631 = t234 * t55;
        const auto t632 = t288 * t3;
        const auto t633 = -t632;
        const auto t634 = t150 * t168;
        const auto t635 = t192 * t634 + t633;
        const auto t636 = -t155;
        const auto t637 = t228 * t636;
        const auto t638 = t29 * t319;
        const auto t639 = -t110 * t151 + t596 + t638;
        const auto t640 = t269 * t636;
        const auto t641 = t272 * t639;
        const auto t642 = t228 * t639;
        const auto t643 = t281 * t642;
        const auto t644 = t266 * t269;
        const auto t645 = t229 * t639;
        const auto t646 = t152 - t228 * t265 * t639 + t236 * t637 + t236 * t641
            + t266 * t640 + t29 * t636 + t3 * t643
            + t3 * (t146 * t627 + t151 * t631 + t635) + t644 * t645;
        const auto t647 = p2y + t1 * t293 + t11;
        const auto t648 = t151 * t291;
        const auto t649 = t262 * t449;
        const auto t650 = t1 * t168;
        const auto t651 = t290 * t49;
        const auto t652 = t49 * t65;
        const auto t653 = t1 * t312;
        const auto t654 = -t647;
        const auto t655 = 2 * t214;
        const auto t656 = 2 * t124;
        const auto t657 = t149 * t49;
        const auto t658 = t151 * t290 * t49 + t296 * t657 + t654 * t656;
        const auto t659 = t269 * t340;
        const auto t660 = t138 * t72;
        const auto t661 = t280 - t660;
        const auto t662 = t1 * t643 + t138 * t636 + t661;
        const auto t663 = -t228 * t338 * t639 + t326 * t637 + t326 * t641
            + t340 * t640 + t645 * t659 + t662;
        const auto t664 = t146 * t523;
        const auto t665 = t346 * t49;
        const auto t666 = t170 * t665;
        const auto t667 = t346 * t48;
        const auto t668 = 2 * t149;
        const auto t669 = t1 * t369;
        const auto t670 = -t168 * t669;
        const auto t671 = t361 * t371;
        const auto t672 = t370 * t627;
        const auto t673 = t671 + t672;
        const auto t674 =
            -t1 * t10 * t3 * t366 * t49 * t55 + t366 * t629 + t670 + t673;
        const auto t675 = t151 * t71;
        const auto t676 = t269 * t395;
        const auto t677 = t168 * t636 - t228 * t393 * t639 + t381 * t637
            + t381 * t641 + t395 * t640
            - t55 * t7
                * (2 * t1 * t153 * t48 * t49 * t55 - t150 * t665
                   + t151 * t346 * t49 - t259 * t48 * t668)
            - t568 + t643 * t7 + t645 * t676 - t675;
        const auto t678 = t422 * t74;
        const auto t679 = t411 * t7;
        const auto t680 = -t613;
        const auto t681 = t424 * t74;
        const auto t682 = t138 * t681;
        const auto t683 = t151 * t411 + t680 + t682;
        const auto t684 = t13 + 2 * t16 + t19 - t21 * t515;
        const auto t685 = t29 * t684 + 2 * t30 + 2 * t37;
        const auto t686 = -t45 * (1 - t515) + t685;
        const auto t687 = -2 * t41 - t43;
        const auto t688 = t157 * t687;
        const auto t689 = -t143 * t687 + t688;
        const auto t690 = t1 * t468;
        const auto t691 = t406 * t68 + t411 * t71 + t422 * t430;
        const auto t692 = t155 * t472;
        const auto t693 = t404 + 2 * t85;
        const auto t694 = t693 * t7;
        const auto t695 = t419 * t48;
        const auto t696 = t113 * t695 + t117 * t429 + t419 * t539;
        const auto t697 = t155 * t504;
        const auto t698 = t603 * t609;
        const auto t699 = t159 * t609;
        const auto t700 = t162 * t507;
        const auto t701 = t55 * t700;
        const auto t702 = t145 * t701;
        const auto t703 = t133 * t143 - t155 * t29
            - t29
                * (2 * t1 * t10 * t3 * t7 * t91 - t141 * t694 - t143 * t694
                   - t175 * t459)
            - t3 * t699 - t433 * t609 + t650 * t91 + t691 * t692 + t691 * t698
            + t696 * t697 + t696 * t702;
        const auto t704 = 2 * t5 * t55;
        const auto t705 = t704 - 1;
        const auto t706 = t1 + t305 * t5;
        const auto t707 = 2 * t430;
        const auto t708 = t1 * t453;
        const auto t709 = t151 * t496 + t706 * t707 + t708;
        const auto t710 = -t705;
        const auto t711 = -t706;
        const auto t712 = t452 * t49;
        const auto t713 = t1 * t712;
        const auto t714 = t219 * t68;
        const auto t715 = t55 * t71;
        const auto t716 = t117 * t452 - t333 * t445 + t477 * t714 - t499 * t715;
        const auto t717 = t122 * t55;
        const auto t718 = t149
                * (t123 * t445 * t49 + t124 * t477 * t48 - t331 * t445
                   - t332 * t477)
            + t3
                * (-t123 * t452 - t231 * t477 * t55 + t336 * t445 + t499 * t717)
            - t328 + t329 - t7 * t716;
        const auto t719 = t124 * t219;
        const auto t720 = t122 * t496 + t153 * t452 + t445 * t719;
        const auto t721 = t268 * t720;
        const auto t722 = t214 * t499 + t217 * t477 + t323 * t477;
        const auto t723 = t55 * t722;
        const auto t724 = t1 * t477;
        const auto t725 = t229 * t721;
        const auto t726 = t49 * t525;
        const auto t727 = t1 * t204;
        const auto t728 = t48 * t727;
        const auto t729 = t1 * t153;
        const auto t730 = t151 * t533;
        const auto t731 = t38 * t552;
        const auto t732 = -t143 * t552 + t731;
        const auto t733 = -t125 * t127 * t145 * t472 * t545
            - 3 * t125 * t145 * t507 * t542 * t55 - t127 * t155 * t542 * t55
            + t155 * t168 - t155 * t472 * t545
            - t55 * t7
                * (2 * t1 * t153 * t49 * t55 * t7 - t149 * t259 * t518
                   - t150 * t726 + t151 * t49 * t525)
            + t564 * t609 + t568 + t675 + t699 * t7;
        const auto t734 = t187 - t38 - 1;
        const auto t735 = t141 + t151;
        const auto t736 = t510
            * (t570 * t692 + t570 * t698 - t576 * t609
               + t7 * (t141 * t734 + t151 * t566 - t38 * t735 + t97));
        const auto t737 = t510 * (t583 * t692 + t583 * t698 + t590 * t609);
        const auto t738 = t211 * t51;
        const auto t739 = t510
            * (t125 * t127 * t145 * t472 * t594 + t155 * t472 * t594
               - t3 * (-t1 * t738 + t10 * t5 * t593 - t151 * t600 - t157 * t735)
               - t602 * t609);
        const auto t740 = t606 + t98;
        const auto t741 = t166 * t316;
        const auto t742 = t192 * t49;
        const auto t743 = 2 * t170;
        const auto t744 = t224 * t7;
        const auto t745 = -t29 * t744;
        const auto t746 = t172 * t48;
        const auto t747 = t55 * t746;
        const auto t748 = 2 * t521;
        const auto t749 = t245 * t748;
        const auto t750 = -t1 * t10 * t209 * t51 * t55 * t7 + t209 * t747 + t372
            + t745 + t749;
        const auto t751 = t154 - t259;
        const auto t752 = t55 * t8;
        const auto t753 = t172 * t71 + t569;
        const auto t754 = t170 * t230;
        const auto t755 = -t1 * t753 + t3 * (-t153 * t172 + t754) + t751 * t752;
        const auto t756 = t228 * t755;
        const auto t757 = 2 * t49;
        const auto t758 = t231 * t49 * t743;
        const auto t759 = -t113 * t172 + t571 + t585;
        const auto t760 = t269 * t755;
        const auto t761 = t272 * t759;
        const auto t762 = t228 * t759;
        const auto t763 = t281 * t762;
        const auto t764 = t229 * t759;
        const auto t765 = -t169 - t173 - t228 * t265 * t759 + t236 * t756
            + t236 * t761 + t266 * t760 + t29 * t755
            - t3 * t55 * (-t171 * t193 + t172 * t193 + t754 * t757 - t758)
            + t3 * t763 + t644 * t764;
        const auto t766 = t1 * t189;
        const auto t767 = t521 * t766;
        const auto t768 = t168 * t292;
        const auto t769 = t297 * t55;
        const auto t770 = t219 * t290;
        const auto t771 = 2 * t305;
        const auto t772 = t154 * t771 + t219 * t296 + t51 * t770;
        const auto t773 = t314 * t748;
        const auto t774 = t168 * t75;
        const auto t775 = t312 * t774;
        const auto t776 = -t1 * t10 * t307 * t7 + t313 * t746 + t773 - t775;
        const auto t777 = t317 * t66;
        const auto t778 = -t777;
        const auto t779 = t168 * t74;
        const auto t780 = t291 * t779 + t778;
        const auto t781 = t1 * t763 + t1 * (t166 * t748 + t172 * t770 + t780)
            + t138 * t755 - t228 * t338 * t759 + t326 * t756 + t326 * t761
            + t340 * t760 + t659 * t764 + t753;
        const auto t782 = p2z + t17 + t349 * t7;
        const auto t783 = t168 * t3;
        const auto t784 = t366 * t7;
        const auto t785 = -t782;
        const auto t786 = 2 * t153;
        const auto t787 = t170 * t51;
        const auto t788 = t172 * t394 + t352 * t787 + t785 * t786;
        const auto t789 = t168 * t755 - t279 + t660 + t7 * t763;
        const auto t790 = -t228 * t393 * t759 + t381 * t756 + t381 * t761
            + t395 * t760 + t676 * t764 + t789;
        const auto t791 = t170 * t65;
        const auto t792 = t521 * t612;
        const auto t793 = -t792;
        const auto t794 = t168 * t678;
        const auto t795 = t172 * t411 + t793 + t794;
        const auto t796 = t141 * t687;
        const auto t797 = -t177 * t687 + t796;
        const auto t798 = t468 * t7;
        const auto t799 = t127 * t184;
        const auto t800 = t180 * t472;
        const auto t801 = t518 * t74;
        const auto t802 = t180 * t504;
        const auto t803 = t159 * t799;
        const auto t804 = -t10 * t180 * t3 - t125 * t127 * t184 * t472 * t691
            - 3 * t125 * t184 * t507 * t55 * t696 + t174
            - t3 * t55
                * (-t171 * t424 + t172 * t424 + t230 * t801 - t231 * t801)
            + t3 * t803 + t433 * t799 + t691 * t800 + t696 * t802;
        const auto t805 = t149 * t454;
        const auto t806 = t3 * t452;
        const auto t807 = -t767;
        const auto t808 = t168 * t447;
        const auto t809 = t172 * t452 + t807 + t808;
        const auto t810 = t38 * t465;
        const auto t811 = -t177 * t465 + t810;
        const auto t812 = t177 * t3;
        const auto t813 = t603 * t799;
        const auto t814 = -t1 * t803 + t138 * t180
            - t138
                * (2 * t1 * t10 * t133 * t3 * t7 - t157 * t3 * t487
                   - t487 * t812 - t519 * t88)
            + t179 + t184 * t503 * t701 - t471 * t800 + t471 * t813
            + t502 * t799 - t503 * t802;
        const auto t815 = 2 * t752 - 1;
        const auto t816 = t219 * t65;
        const auto t817 = t250 * t8 + t7;
        const auto t818 = 2 * t715;
        const auto t819 = t219 * t525;
        const auto t820 = t533 * t783;
        const auto t821 = -t815;
        const auto t822 = -t817;
        const auto t823 = t51 * t533;
        const auto t824 = -t564;
        const auto t825 = t122 * t533 + t153 * t544 + t525 * t719;
        const auto t826 = t268 * t825;
        const auto t827 = t123 * t541 + t215 * t537 + t323 * t537;
        const auto t828 = t55 * t827;
        const auto t829 = t537 * t7;
        const auto t830 = 2 * t123;
        const auto t831 = t229 * t826;
        const auto t832 = t221 * t48;
        const auto t833 = t157 + t172;
        const auto t834 = t510
            * (-t1 * (t10 * t734 * t8 - t172 * t567 - t38 * t833 - t7 * t832)
               + t125 * t127 * t184 * t472 * t570 - t570 * t800 - t576 * t799);
        const auto t835 = t510
            * (t3 * (-t141 * t833 + t157 * t581 + t172 * t579 + t606)
               - t583 * t800 + t583 * t813 + t590 * t799);
        const auto t836 = t510
            * (-t1 * (t219 * (t600 * t7 + t746) - t783 * (t171 + t349))
               + t125 * t127 * t184 * t472 * t594 - t594 * t800 - t602 * t799);
        const auto t837 = t10 * t166;
        const auto t838 = t166 * t320;
        const auto t839 = -t838;
        const auto t840 = t202 * t74;
        const auto t841 = t138 * t840;
        const auto t842 = t837 + t839 + t841;
        const auto t843 = t110 * t138;
        const auto t844 = t113 * t168;
        const auto t845 = t110 * t286;
        const auto t846 = t845 * t96;
        const auto t847 = 2 * t113;
        const auto t848 = t343 * t847;
        const auto t849 = t209 * t51;
        const auto t850 = t374 * t849;
        const auto t851 = 2 * t248;
        const auto t852 = -t64;
        const auto t853 = -t344;
        const auto t854 = t792 + t853;
        const auto t855 = t168 * t65;
        const auto t856 = t234 * t779 + t855;
        const auto t857 = t10 * t361;
        const auto t858 = t362 - t857;
        const auto t859 = 2 * t55;
        const auto t860 = -t10 * t125;
        const auto t861 = -2 * t49 * t57 - 4 * t50 * t55 * t62 + t62;
        const auto t862 = -t861;
        const auto t863 = t113 * t305;
        const auto t864 = 4 * t107;
        const auto t865 = t221 * t864;
        const auto t866 = 2 * t220 + t223 + t865;
        const auto t867 = std::pow(t49, 3);
        const auto t868 = t204 * t867;
        const auto t869 = 4 * t62;
        const auto t870 = p0x - 3 * p1x + t403;
        const auto t871 = t117 * t48;
        const auto t872 = t204 * t862;
        const auto t873 = std::pow(t3, 3);
        const auto t874 = -t515 * t57 + 4 * t83 * t873 * t95 - 3 * t85 - t86;
        const auto t875 = t10 * t113;
        const auto t876 = 4 * t95;
        const auto t877 = -t208 * t515 + t21 * t873 * t876 - 3 * t41 + t870;
        const auto t878 = t193 * t68 + t202 * t71 + t234 * t65;
        const auto t879 = t162 / std::pow(t134, 2);
        const auto t880 = t204 * t879;
        const auto t881 = t159 * t472;
        const auto t882 = t878 * t881;
        const auto t883 = t110 * t305;
        const auto t884 = t113 * t250;
        const auto t885 = t117 * t224 + t209 * t883 + t209 * t884;
        const auto t886 = t127 * t885;
        const auto t887 = t159 * t886;
        const auto t888 = std::pow(t192, 2) * t204;
        const auto t889 = t189 * t51;
        const auto t890 = t342 * t48;
        const auto t891 = t204 * std::pow(t209, 2);
        const auto t892 = 2 * t110;
        const auto t893 = t305 * t892;
        const auto t894 = t250 * t847;
        const auto t895 = 2 * t117;
        const auto t896 = t472 * t859;
        const auto t897 = t48 * t71;
        const auto t898 = t113 * t651 - t117 * t291 - t312 * t652 + t312 * t897;
        const auto t899 = t486 + t90;
        const auto t900 = t489 * t899;
        const auto t901 = -t304 + t464;
        const auto t902 = t485 * t901;
        const auto t903 = 2 * t79;
        const auto t904 = 2 * t10 * t5 * t83 - t82 - t903;
        const auto t905 = t10 * t45 * t904;
        const auto t906 = t138 * t304;
        const auto t907 = t22 + 1;
        const auto t908 = 2 * t21 * t5 * t95 - t906 - t907;
        const auto t909 = t87 * t908;
        const auto t910 = t110 * t55;
        const auto t911 = t291 * t910;
        const auto t912 = t262 * t313;
        const auto t913 = t113 * t769;
        const auto t914 = t307 * t65;
        const auto t915 = t911 - t912 - t913 + t914;
        const auto t916 = -t149 * t898 - t3 * t915 - t474 + t475
            - t7 * (-t900 + t902 + t905 - t909);
        const auto t917 = t234 * t305;
        const auto t918 = t49 * t59;
        const auto t919 = t51 * t57;
        const auto t920 = 4 * t49;
        const auto t921 = t67 * t920 + t918 + t919;
        const auto t922 = t304 * t49;
        const auto t923 = t208 * t51;
        const auto t924 = t109 * t920 + t922 + t923;
        const auto t925 = t262 * t859;
        const auto t926 = 2 * t51 * t59;
        const auto t927 = 8 * t62;
        const auto t928 = -t191 + t86;
        const auto t929 = t219 * t926 + t294 * t57 + t578 * t927 + t928;
        const auto t930 = 8 * t107;
        const auto t931 = t2 - t206 + t207 + t42;
        const auto t932 = t208 * t294 + t578 * t930 + t771 * t922 + t931;
        const auto t933 = 2 * t199;
        const auto t934 = t221 * t51;
        const auto t935 = -t289 + t90;
        const auto t936 = t305 * t933 + t59 * t616 + t927 * t934 + t935;
        const auto t937 = t0 + t24 + t303 - t311;
        const auto t938 = t304 * t616 + t621 * t923 + t930 * t934 + t937;
        const auto t939 = t219 * t312;
        const auto t940 = t117 * t219;
        const auto t941 = t110 * t307 + t312 * t884 + t312 * t940;
        const auto t942 = t472 * t55;
        const auto t943 = t209 * t349;
        const auto t944 = t224 * t49;
        const auto t945 = t192 * t349;
        const auto t946 = t291 * t65 + t297 * t68 + t651 * t71;
        const auto t947 = t162 * t95;
        const auto t948 = t700 * t885;
        const auto t949 = t127 * t941;
        const auto t950 = t159 * t3;
        const auto t951 = t878 * t880;
        const auto t952 = t74 * t881;
        const auto t953 = t128 * t472 * t55;
        const auto t954 = t885 * t953;
        const auto t955 = t878 * t953;
        const auto t956 = t510
            * (t1 * t10 * t264 - t1 * t887
               + t125 * t127 * t55
                   * (t110 * t932 + t117 * t938 + t307 * t849 + t312 * t943
                      + t312 * t944 + t894 * t924)
               + t125 * t472 * t55
                   * (t193 * t769 + t202 * t770 + t290 * t945
                      + t48 * t707 * t921 + t68 * t929 + t71 * t936)
               + t127 * t264 * t941 - t149 * t882
               - t149
                   * (t113 * t936 + t202 * t312 * t48 * t55
                      + t209 * t290 * t48 * t49 * t55 - t224 * t291
                      - t250 * t895 * t921 - t312 * t620
                      + 2 * t48 * t55 * t71 * t924 - t65 * t938)
               - t156 * t947 + t249 - t252 - t254 + t255
               + t264 * t472 * t55 * t946 - t29 * t916
               + t55 * t7
                   * (t121 * t929 - t124 * t932 + t153 * t938 - t193 * t939
                      + t201 * t325 + t209 * t290 * t49 * t51 * t55
                      - t214 * t936 - t225 * t296)
               - t74
                   * (2 * t110 * t48 * t55 * t921 - t113 * t929
                      + t192 * t307 * t48 + t209 * t290 * t48 * t51 * t55
                      - t312 * t917 - t624 * t769 + t65 * t932 - t924 * t925)
               - t878 * t916 * t942 - t886 * t916 - t915 - t941 * t948
               - t941 * t955 - t946 * t951 - t946 * t952 - t946 * t954
               - t949 * t950);
        const auto t957 = t193 * t248;
        const auto t958 = t209 * t398;
        const auto t959 = t110 * t253;
        const auto t960 = t224 * t68;
        const auto t961 = t353 * t65 + t394 * t68 + t665 * t71;
        const auto t962 = t113 * t369 + t366 * t883 + t366 * t940;
        const auto t963 = t127 * t962;
        const auto t964 = t219 * t366;
        const auto t965 = t49 * t61;
        const auto t966 = t48 * t57;
        const auto t967 = t64 * t920 + t965 + t966;
        const auto t968 = t365 * t49;
        const auto t969 = t208 * t48;
        const auto t970 = t112 * t920 + t968 + t969;
        const auto t971 = t859 * t970;
        const auto t972 = t132 - t345;
        const auto t973 = t250 * t933 + t61 * t616 + t832 * t927 + t972;
        const auto t974 = t32 + t364 + t6;
        const auto t975 = -t363 + t974;
        const auto t976 = t365 * t616 + t621 * t969 + t832 * t930 + t975;
        const auto t977 = t305 * t366;
        const auto t978 = t305 * t967;
        const auto t979 = t209 * t305;
        const auto t980 = 2 * t48 * t61;
        const auto t981 = t211 * t49;
        const auto t982 = t219 * t980 + t350 * t57 + t927 * t981 + t928;
        const auto t983 = t208 * t350 + t358 * t968 + t930 * t981 + t931;
        const auto t984 = t192 * t293;
        const auto t985 = t219 * t346;
        const auto t986 = t209 * t293;
        const auto t987 = t510
            * (t10 * t264 * t7 + t10 * t3 * t392
               + t125 * t127 * t55
                   * (t113 * t983 + t117 * t976 + t366 * t944 + t366 * t986
                      + t369 * t624 + t893 * t970)
               + t125 * t472 * t55
                   * (t189 * t978 + t202 * t985 + t234 * t354 + t346 * t984
                      + t65 * t982 + t71 * t973)
               + t127 * t264 * t962 + t127 * t392 * t885
               - t149
                   * (t113 * t973 - t117 * t982 + t202 * t369
                      + t209 * t346 * t48 * t49 * t55 - t224 * t353
                      - t366 * t620 - t65 * t976 + t71 * t983)
               - t170 * t882 - t185 * t947 + t264 * t472 * t55 * t961 + t382
               - t384 - t385 + t386 + t392 * t472 * t55 * t878
               + t55 * t7
                   * (2 * t121 * t51 * t55 * t967 + t153 * t976 - t193 * t964
                      + t209 * t346 * t49 * t51 * t55 - t214 * t973
                      + t225 * t346 * t51 - t259 * t971 - t261 * t373)
               - t7 * t887
               - t74
                   * (t110 * t982 - t193 * t369 + t234 * t977 + t353 * t979
                      - t357 * t624 + t383 * t971 - t68 * t983 - t847 * t978)
               - t948 * t962 - t950 * t963 - t951 * t961 - t952 * t961
               - t954 * t961 - t955 * t962 - t957 + t958 + t959 - t960);
        const auto t988 = 8 * t409;
        const auto t989 = t3 * t988;
        const auto t990 = t200 + 1;
        const auto t991 = 2 * t407 + t80 * t859 + t989 + t990;
        const auto t992 = t51 * t991;
        const auto t993 = t49 * t930;
        const auto t994 = t455 * t993;
        const auto t995 = t535 + 1;
        const auto t996 = t208 * t301 + t428 * t55 + t994 + t995;
        const auto t997 = t48 * t991;
        const auto t998 =
            t100 * t187 - t107 * t3 * t55 + t208 * t75 + t209 + t3 * t865;
        const auto t999 = t14 + t403 + t56;
        const auto t1000 = t187 * t404 + t191 + t219 * t80 + t221 * t3 * t869
            - t3 * t55 * t62 + t999;
        const auto t1001 = t193 * t55;
        const auto t1002 = t419 * t51;
        const auto t1003 = t225 * t51;
        const auto t1004 = t411 * t51;
        const auto t1005 = -t1000;
        const auto t1006 = -t998;
        const auto t1007 = t29 * t434;
        const auto t1008 = t228 * t236;
        const auto t1009 = t268 * t434;
        const auto t1010 = t266 * t55;
        const auto t1011 = t281 * t3;
        const auto t1012 = t229 * t236;
        const auto t1013 = t236 * t272;
        const auto t1014 = std::pow(t267, -2);
        const auto t1015 = t1014 * t271;
        const auto t1016 = t1015 * t435;
        const auto t1017 = t281 * t74;
        const auto t1018 = t1017 * t228;
        const auto t1019 = t266 * t268;
        const auto t1020 = t204 * t229;
        const auto t1021 = t1019 * t1020;
        const auto t1022 = -t281;
        const auto t1023 = t1022 + t271 * t96;
        const auto t1024 = t285
            * (t10 * t265 * t3 - t1007 - t1008 * t1011 - t1008 * t434
               - t1009 * t1010 - t1010 * t1016 - t1011 * t436 - t1012 * t436
               - t1013 * t438 - t1017 * t1019 - t1018 * t437 - t1021 * t437
               - t1023 + t228 * t265 * t437 * t55 + t265 * t268 * t435
               - t276
                   * (t1005 * t656 - t122 * t997 - t153 * t992 - t201 * t411
                      + t405 * t945 + t405 * t984)
               - t441
                   * (t1006 * t440 - t215 * t996 - t217 * t996 - t225 * t429
                      + t419 * t943 + t419 * t986)
               + t55
                   * (-t1
                          * (t1000 * t847 - t117 * t997 + t209 * t416
                             - t224 * t422 + t253 * t695 - t342 * t998
                             - t429 * t631 + t897 * t996)
                      - t263
                      - t3
                          * (t110 * t48 * t991 - t113 * t992 - t262 * t996
                             + t51 * t65 * t996)
                      - t426
                      + t7
                          * (-t1001 * t429 - t1002 * t261 + t1003 * t405
                             + t1004 * t209 + t1005 * t655 - t1006 * t786
                             + t257 * t991 - t259 * t996)));
        const auto t1025 = t204 * t499;
        const auto t1026 = t1 * t208;
        const auto t1027 = t1 * t221;
        const auto t1028 = t12 - t476;
        const auto t1029 = t102 * t616 + t1026 * t621 + t1027 * t930 + t1028;
        const auto t1030 = t444 * t49;
        const auto t1031 = t1 * t51 * t988 + t1030 * t771 + t192 + t668 * t919;
        const auto t1032 = t11 + t443 + t58;
        const auto t1033 = t1032 - t442;
        const auto t1034 = t1027 * t927 + t1033 + t149 * t933 + t444 * t616;
        const auto t1035 = t51 * t727;
        const auto t1036 = t1035 * t993 + t209 + t219 * t498 + t668 * t923;
        const auto t1037 = 4 * t1;
        const auto t1038 = t1 * t57 + t1030 + t1037 * t70;
        const auto t1039 = t1038 * t859;
        const auto t1040 = t102 * t49;
        const auto t1041 = t1026 + t1037 * t116 + t1040;
        const auto t1042 = t1041 * t859;
        const auto t1043 = t452 * t48;
        const auto t1044 = t209 * t219;
        const auto t1045 = -t1029;
        const auto t1046 = -t1034;
        const auto t1047 = 2 * t218;
        const auto t1048 = t233 * t250;
        const auto t1049 = t219 * t445;
        const auto t1050 = t268 * t718;
        const auto t1051 = t1008 * t281;
        const auto t1052 = t1015 * t720;
        const auto t1053 = t1019 * t281;
        const auto t1054 = t271 * t95;
        const auto t1055 = t1054 * t156;
        const auto t1056 = t1011 * t721 + t1018 * t722 + t1055 + t29 * t718
            - t495 + t497 + t500 - t501;
        const auto t1057 = t285
            * (t1 * t10 * t265 - t1 * t1051
               + t1 * t55
                   * (-t1039 * t331 + t1042 * t232 - t1044 * t446 + t1045 * t122
                      - t1046 * t123 - t225 * t446 + t261 * t481 + t477 * t620)
               - t1008 * t718 - t1010 * t1050 - t1010 * t1052 - t1013 * t723
               - t1021 * t722 - t1053 * t149 - t1056 + t228 * t265 * t55 * t722
               - t236 * t725 - t256 + t265 * t268 * t720
               - t276
                   * (-t1031 * t153 - t1038 * t1048 + t1046 * t124
                      - t1049 * t201 + t192 * t445 * t53 * t55
                      + t192 * t452 * t51)
               + t3 * t55
                   * (-t1031 * t123 + t1036 * t122 + t1039 * t335 - t1042 * t231
                      + t1043 * t209 - t446 * t979 + t477 * t917 - t499 * t631)
               - t441
                   * (-t1036 * t214 - t1041 * t1047 + t1045 * t121
                      + t209 * t477 * t53 * t55 + t209 * t499 * t51 * t55
                      - t225 * t478)
               - t7
                   * (-t1025 * t202 + t1029 * t530 + t1031 * t248 - t1034 * t910
                      - t1036 * t715 + t193 * t204 * t478 - t204 * t454 * t849
                      + t224 * t452));
        const auto t1058 = t204 * t541;
        const auto t1059 = t208 * t7;
        const auto t1060 = t221 * t7;
        const auto t1061 = t18 - t536;
        const auto t1062 = t104 * t616 + t1059 * t621 + t1060 * t930 + t1061;
        const auto t1063 = -t1062;
        const auto t1064 = t17 + t512 + t60;
        const auto t1065 = t1064 - t524;
        const auto t1066 = t1060 * t927 + t1065 + t170 * t933 + t513 * t616;
        const auto t1067 = -t1066;
        const auto t1068 = t49 * t513;
        const auto t1069 = t1068 * t358 + t192 + t48 * t7 * t988 + t743 * t966;
        const auto t1070 = t204 * t48;
        const auto t1071 = t1070 * t7;
        const auto t1072 = t1071 * t993 + t209 + t219 * t540 + t743 * t969;
        const auto t1073 = t305 * t537;
        const auto t1074 = 4 * t7;
        const auto t1075 = t305 * (t1068 + t1074 * t70 + t57 * t7);
        const auto t1076 = t104 * t49;
        const auto t1077 = t1059 + t1074 * t116 + t1076;
        const auto t1078 = t1077 * t859;
        const auto t1079 = t219 * t537;
        const auto t1080 = t51 * t537;
        const auto t1081 = 2 * t216;
        const auto t1082 = t268 * t824;
        const auto t1083 = t1015 * t825;
        const auto t1084 = t1054 * t185;
        const auto t1085 =
            t1011 * t826 + t1018 * t827 + t1084 + t29 * t824 + t562;
        const auto t1086 = t285
            * (t1
                   * (t1058 * t201 + t1063 * t717 - t1067 * t337 - t1069 * t213
                      + t1072 * t563 + t204 * t234 * t538 - t204 * t624 * t726
                      - t225 * t533)
               + t10 * t265 * t7 - t1008 * t824 - t1010 * t1082 - t1010 * t1083
               - t1013 * t828 - t1021 * t827 - t1051 * t7 - t1053 * t170 - t1085
               - t170
                   * (-t1044 * t526 + t1062 * t68 - t1066 * t110 + t1075 * t895
                      - t1078 * t197 + t1079 * t193 - t1080 * t253
                      + t224 * t526)
               + t228 * t265 * t55 * t827 - t236 * t831 + t265 * t268 * t825
               - t276
                   * (t1067 * t124 - t1069 * t122 - t1075 * t786
                      + t192 * t48 * t533 + t192 * t52 * t525 * t55
                      - t201 * t819)
               - t441
                   * (t1063 * t121 - t1072 * t123 - t1077 * t1081
                      + t209 * t48 * t541 * t55 + t209 * t52 * t537 * t55
                      - t225 * t538)
               - t74
                   * (-t1001 * t541 + t1069 * t110 - t1072 * t68 + t1073 * t234
                      - t1075 * t847 + t1078 * t383 + t209 * t823 - t544 * t624)
               + t957 - t958 - t959 + t960);
        const auto t1087 = t3 + t55 * t867;
        const auto t1088 = t55 * t847;
        const auto t1089 = t209 * t250 * t566 + t745;
        const auto t1090 = t7 * t74;
        const auto t1091 = t624 * t74;
        const auto t1092 = t314 * t515 - t314 - t370 * t515 + t370;
        const auto t1093 = t567 * t849;
        const auto t1094 = t138 * t68;
        const auto t1095 = t766 * t96;
        const auto t1096 = -t1095;
        const auto t1097 = -t576;
        const auto t1098 = t1097 * t29;
        const auto t1099 = t265 * t268;
        const auto t1100 = t268 * t570;
        const auto t1101 = t1011 * t1100;
        const auto t1102 = t29 * t370;
        const auto t1103 = t29 * t314;
        const auto t1104 = -t1103;
        const auto t1105 = t1102 + t1104;
        const auto t1106 = t1008 * t1097 - 3 * t1014 * t226 * t266 * t55 * t570
            + t1097 * t644 + t1098 + t1099 * t570 - t1101 + t1105
            - t226 * t228 * t236 * t268 * t570;
        const auto t1107 = t117 * t315;
        const auto t1108 = t1107 * t491;
        const auto t1109 = -t846;
        const auto t1110 = -t113 * t515 + t113 + 2 * t571;
        const auto t1111 = t305 * t335;
        const auto t1112 = t123 * t580;
        const auto t1113 =
            t150 * (t330 - t331) + t3 * (t1111 + t1112) + t589 * t7;
        const auto t1114 = t1113 * t29;
        const auto t1115 = t1008 * t1113;
        const auto t1116 = t1099 * t583;
        const auto t1117 = t294 * t330;
        const auto t1118 = -t1117 + 2 * t214 * t48 * t49 * t51 * t55
            - t293 * t624 - t580 * t624;
        const auto t1119 = t55 * t579;
        const auto t1120 = -2 * t3 * t5 * t68 * t95;
        const auto t1121 = t1119 * t193 + t1120 + t234 * t625 + t778;
        const auto t1122 = t1113 * t644;
        const auto t1123 = 2 * t323;
        const auto t1124 = -t243 * t748 + t318;
        const auto t1125 = t3 * (t1124 + t592 * t979 + t626);
        const auto t1126 = -2 * t4 * t7 * t71 * t95;
        const auto t1127 = t10 * t72;
        const auto t1128 = t146 * t748;
        const auto t1129 = -t1128;
        const auto t1130 = t55 * t592;
        const auto t1131 = t1129 + t1130 * t234 + t168 * t840 + t635;
        const auto t1132 = -t602;
        const auto t1133 = t1132 * t29;
        const auto t1134 = t1008 * t1132;
        const auto t1135 = t1099 * t594;
        const auto t1136 = -t109;
        const auto t1137 = t214 * t616;
        const auto t1138 = t1003 + t219 * t849 + 2 * t574;
        const auto t1139 = t1132 * t644;
        const auto t1140 = t268 * t594;
        const auto t1141 = t1010 * t1015;
        const auto t1142 = t29 * t71;
        const auto t1143 = t138 * t75;
        const auto t1144 = t1142 + t1143 * t290 + t680;
        const auto t1145 = -t10 * t243 + t628;
        const auto t1146 = t10 * t69;
        const auto t1147 = t627 * t69;
        const auto t1148 = -t1147;
        const auto t1149 = t297 * t625;
        const auto t1150 = t1146 + t1148 + t1149;
        const auto t1151 = t146 * t459;
        const auto t1152 = t459 * t72;
        const auto t1153 = t149 * t7;
        const auto t1154 = t1143 * t312;
        const auto t1155 = -t1108;
        const auto t1156 = t117 * t29;
        const auto t1157 = 2 * t910;
        const auto t1158 = t491 * t7;
        const auto t1159 = t1158 * t847;
        const auto t1160 = -t1159;
        const auto t1161 = t312 * t48;
        const auto t1162 = t1160 + t1161 * t625 + t844;
        const auto t1163 = -t70;
        const auto t1164 = t122 * t7;
        const auto t1165 = 2 * t168;
        const auto t1166 = 2 * t51 * t59 + 4 * t52 * t55 * t62 - t62;
        const auto t1167 = t205 * t864;
        const auto t1168 = t1167 + t223 + 2 * t306;
        const auto t1169 = t3 * t908;
        const auto t1170 = t21 * t491;
        const auto t1171 = std::pow(t1, 3);
        const auto t1172 = -4 * t1171 * t83 * t95 + t459 * t59 + 3 * t89 + t90;
        const auto t1173 =
            p0y - 3 * p1y + t1171 * t21 * t876 - 3 * t23 - t304 * t459 + t443;
        const auto t1174 = t110 * t48;
        const auto t1175 = t1166 * t204;
        const auto t1176 = t291 * t55;
        const auto t1177 = -t1172;
        const auto t1178 = t881 * t946;
        const auto t1179 = t204 * std::pow(t290, 2);
        const auto t1180 = t49 * t611;
        const auto t1181 = t204 * std::pow(t312, 2);
        const auto t1182 = t219 * t895;
        const auto t1183 = t10 * t110;
        const auto t1184 = t896 * t946;
        const auto t1185 = 2 * t916;
        const auto t1186 = t132 + t511;
        const auto t1187 = -t365 + t551;
        const auto t1188 = 2 * t81;
        const auto t1189 = t881 * t961;
        const auto t1190 = t1 * t159;
        const auto t1191 = t159 * t7;
        const auto t1192 = t51 * t61;
        const auto t1193 = t48 * t59;
        const auto t1194 = 4 * t64;
        const auto t1195 = t219 * (t1192 + t1193 + t1194 * t51);
        const auto t1196 = t365 * t51;
        const auto t1197 = t304 * t48;
        const auto t1198 = 4 * t109;
        const auto t1199 = t1196 + t1197 + t1198 * t48;
        const auto t1200 = t305 * t980 + t350 * t59 + t738 * t927 + t935;
        const auto t1201 = t1196 * t358 + t304 * t350 + t738 * t930 + t937;
        const auto t1202 = t1 * t61;
        const auto t1203 = t59 * t7;
        const auto t1204 = t1 * t365;
        const auto t1205 = t304 * t7;
        const auto t1206 = 8 * t7;
        const auto t1207 = t205 * t48;
        const auto t1208 = t1207 * t927 + t250 * t926 + t294 * t61 + t972;
        const auto t1209 = t1197 * t771 + t1207 * t930 + t294 * t365 + t975;
        const auto t1210 = t187 * t290;
        const auto t1211 = t187 * t312;
        const auto t1212 = t510
            * (t1 * t10 * t392
               + t1 * t55
                   * (-t1161 * t985 + t117 * t1200 - t1195 * t847
                      + t1199 * t219 * t342 - t1201 * t71 + t291 * t964
                      + t353 * t939 - t369 * t651)
               + t10 * t1187 * t133 * t3
               + t10 * t45 * (2 * t10 * t8 * t83 - t1188 - t79 - t80)
               - t1178 * t170 - t1186 * t29 * t35 - t1189 * t149 - t1190 * t963
               - t1191 * t949
               + t125 * t127 * t55
                   * (t110 * t1209 + t113 * t1201 + t1161 * t369 + t1182 * t1199
                      + t1211 * t366 + t307 * t373)
               + t125 * t472 * t55
                   * (t1195 * t611 + t1200 * t65 + t1208 * t68 + t1210 * t346
                      + t291 * t354 + t297 * t357)
               + t127 * t392 * t941 - t168 * t916
               - t168
                   * (t1 * t10 * t1186 * t3 * t901 + t10 * t1187 * t3 * t904
                      + 2 * t10 * t3 * t91 * (-t1074 * t23 + t1204 + t1205)
                      - t1169 * t1186 - t1187 * t203 * t899
                      + t45
                          * (-8 * t1158 * t83 + t1186 + t168 * t903
                             + t459 * t61)
                      - 2 * t489 * (-t1074 * t89 + t1202 + t1203)
                      - t87
                          * (-t1170 * t1206 + 2 * t1205 * t138 + t365 * t459
                             + t551 + t974))
               + t3 * t55
                   * (-t110 * t1200 + t113 * t1208 + t1201 * t68 - t1209 * t65
                      - t291 * t977 + t297 * t369 - t307 * t353
                      + t312 * t346 * t48 * t51 * t55)
               + t392 * t472 * t55 * t946 - t608 * t947 - t700 * t941 * t962
               - t87 * (-t168 * t365 + 2 * t21 * t8 * t95 - t907)
               - t880 * t946 * t961 + t900 - t902 - t905 + t909
               - t916 * t942 * t961 - t916 * t963 - t941 * t953 * t961
               - t946 * t953 * t962);
        const auto t1213 = t204 * t49;
        const auto t1214 = t204 * t429;
        const auto t1215 = t3 * t304;
        const auto t1216 = t205 * t3;
        const auto t1217 = t15 - t418;
        const auto t1218 = t100 * t294 + t1215 * t771 + t1216 * t930 + t1217;
        const auto t1219 = -t1218;
        const auto t1220 = -t402 + t999;
        const auto t1221 = t1216 * t927 + t1220 + t294 * t404 + t74 * t926;
        const auto t1222 = -t1221;
        const auto t1223 = t404 * t51;
        const auto t1224 = t1223 * t621 + t290 + t301 * t918 + t51 * t989;
        const auto t1225 = t214 * t55;
        const auto t1226 = t301 * t922 + t305 * t428 + t312 + t456 * t993;
        const auto t1227 = t305 * t419;
        const auto t1228 = t305 * t422;
        const auto t1229 = 4 * t3;
        const auto t1230 = t1223 + t1229 * t67 + t3 * t59;
        const auto t1231 = t1230 * t250;
        const auto t1232 = t100 * t51;
        const auto t1233 = t1198 * t3 + t1215 + t1232;
        const auto t1234 = t219 * t419;
        const auto t1235 = t250 * t611;
        const auto t1236 = t228 * t326;
        const auto t1237 = t312 * t349;
        const auto t1238 = t325 * t51;
        const auto t1239 = t290 * t349;
        const auto t1240 = t340 * t55;
        const auto t1241 = t229 * t326;
        const auto t1242 = t272 * t326;
        const auto t1243 = t268 * t340;
        const auto t1244 = t1020 * t1243;
        const auto t1245 = t1 * t281;
        const auto t1246 = t149 * t281;
        const auto t1247 = t1246 * t228;
        const auto t1248 = t1245 * t436 + t1247 * t437 + t138 * t434 + t431;
        const auto t1249 = t285
            * (t10 * t3 * t338 - t1009 * t1240 - t1011 * t1236 - t1016 * t1240
               - t1017 * t1243 - t1055 - t1236 * t434 - t1241 * t436
               - t1242 * t438 - t1244 * t437 - t1248
               - t149
                   * (t113 * t1224 - t1176 * t429 - t1226 * t65 - t1231 * t895
                      + t1233 * t1235 + t1234 * t291 + t312 * t416
                      - t422 * t939)
               + t228 * t338 * t437 * t55 + t268 * t338 * t435
               - t276
                   * (-t1048 * t1230 + t1222 * t153 - t1224 * t124
                      + t1239 * t405 - t296 * t406 + t411 * t651)
               - t441
                   * (-t1047 * t1233 - t121 * t1226 + t1219 * t214
                      + t1237 * t419 - t1238 * t419 + t429 * t939)
               + t7
                   * (t1002 * t204 * t651 - t1213 * t312 * t424 + t1214 * t296
                      + t1219 * t563 - t1222 * t213 - t1224 * t1225
                      + t1226 * t432 - t325 * t411)
               - t74
                   * (-t113 * t1221 + t1218 * t65 + t1227 * t291 - t1228 * t312
                      + t1231 * t892 - t1233 * t925 + t307 * t422 - t695 * t769)
               + t911 - t912 - t913 + t914);
        const auto t1250 = t250 * t477;
        const auto t1251 = t325 * t48;
        const auto t1252 = 8 * t451;
        const auto t1253 = t1 * t1252;
        const auto t1254 = t1253 + 2 * t450 + t55 * t903 + t990;
        const auto t1255 = t1035 * t930 + t304 * t668 + t498 * t55 + t995;
        const auto t1256 = t1 * t205 * t869 - t1 * t55 * t62 + t1032 + t289
            + t293 * t444 + t305 * t79;
        const auto t1257 = -t1256;
        const auto t1258 =
            -t1 * t107 * t55 + t1 * t1167 + t102 * t293 + t150 * t304 + t312;
        const auto t1259 = -t1258;
        const auto t1260 = t138 * t718;
        const auto t1261 = t122 * t48;
        const auto t1262 = t1022 + t271 * t491;
        const auto t1263 = t285
            * (t1 * t10 * t338 - t1050 * t1240 - t1052 * t1240 - t1236 * t1245
               - t1236 * t718 - t1242 * t723 - t1243 * t1246 - t1244 * t722
               - t1245 * t721 - t1247 * t722 - t1260 - t1262
               + t228 * t338 * t55 * t722 + t268 * t338 * t720
               - t276
                   * (t1210 * t445 + t1239 * t445 - t1254 * t1261 - t1254 * t339
                      + t1257 * t786 - t296 * t452)
               - t326 * t725
               - t441
                   * (t1211 * t477 + t1237 * t477 - t1255 * t217 - t1255 * t323
                      + t1259 * t655 - t325 * t499)
               + t55
                   * (t1
                          * (t1254 * t330 - t1254 * t331 + t1255 * t232
                             - t1255 * t332)
                      + t3
                          * (t1043 * t312 - t1176 * t499 - t1250 * t296
                             + t1251 * t445 + t1254 * t335 - t1255 * t231
                             + t1257 * t830 - t1259 * t233)
                      + t484
                      - t7
                          * (-t1254 * t390 + t1255 * t557 + t1256 * t895
                             - t1258 * t611 + t219 * t297 * t477 - t307 * t454
                             + t312 * t712 - t499 * t770)
                      + t898));
        const auto t1264 = t205 * t7;
        const auto t1265 = t104 * t294 + t1061 + t1205 * t771 + t1264 * t930;
        const auto t1266 = t51 * t513;
        const auto t1267 =
            t1193 * t743 + t1206 * t451 * t48 + t1266 * t358 + t290;
        const auto t1268 = t1065 + t1264 * t927 + t170 * t926 + t294 * t513;
        const auto t1269 = t113 * t55;
        const auto t1270 = t1071 * t930;
        const auto t1271 = t1197 * t743 + t1270 * t51 + t305 * t540 + t312;
        const auto t1272 = t1074 * t67 + t1203 + t1266;
        const auto t1273 = t1272 * t219;
        const auto t1274 = t104 * t51;
        const auto t1275 = t1074 * t109 + t1205 + t1274;
        const auto t1276 = t219 * t541;
        const auto t1277 = t49 * t533;
        const auto t1278 = 2 * t324;
        const auto t1279 = t281 * t7;
        const auto t1280 = t170 * t281;
        const auto t1281 = t1054 * t608;
        const auto t1282 = t113 * t819 - t117 * t533 + t1245 * t826
            + t1247 * t827 + t1281 + t138 * t824 - t537 * t816 + t541 * t715;
        const auto t1283 = t285
            * (t1 * t55
                   * (t1079 * t291 - t1161 * t819 - t121 * t1267 + t124 * t1271
                      + t1272 * t330 * t859 - t1275 * t219 * t233 - t1276 * t290
                      + t1277 * t312)
               + t10 * t338 * t7 - t1082 * t1240 - t1083 * t1240 - t1236 * t1279
               - t1236 * t824 - t1242 * t828 - t1243 * t1280 - t1244 * t827
               - t1282
               - t170
                   * (-t1080 * t770 + t117 * t1268 - t1265 * t71 - t1273 * t892
                      + 2 * t1275 * t49 * t55 * t68 + t297 * t49 * t537 * t55
                      - t307 * t726 + t312 * t49 * t51 * t525 * t55)
               + t228 * t338 * t55 * t827 + t268 * t338 * t825
               - t276
                   * (t1210 * t525 - t122 * t1267 - t1268 * t153 - t1273 * t656
                      + t291 * t533 - t296 * t544)
               - t3
                   * (-t1058 * t297 + t1080 * t204 * t291 - t1161 * t204 * t526
                      + t1265 * t430 + t1267 * t910 - t1268 * t1269
                      - t1271 * t530 + t307 * t533)
               - t326 * t831 - t334
               - t441
                   * (t1211 * t537 - t123 * t1271 - t1238 * t537 - t1265 * t214
                      - t1275 * t1278 + t250 * t312 * t541));
        const auto t1284 = 2 * t221;
        const auto t1285 = t314 * t320;
        const auto t1286 = -t671;
        const auto t1287 =
            t1 * (-t1285 - t1286 + t312 * t48 * t55 * t566 - t775);
        const auto t1288 = t10 * t165;
        const auto t1289 = t297 * t374 + t566 * t770 + t780 + t839;
        const auto t1290 = t1097 * t138;
        const auto t1291 = t1097 * t1236;
        const auto t1292 = t268 * t338;
        const auto t1293 = t1292 * t570;
        const auto t1294 = -t112;
        const auto t1295 = t123 * t294;
        const auto t1296 = 2 * t1111 + t1161 * t305 + t1251;
        const auto t1297 = t1097 * t659;
        const auto t1298 = t1015 * t1240;
        const auto t1299 = t1 + std::pow(t51, 3) * t55;
        const auto t1300 = t309 + t579 * t939;
        const auto t1301 = t1161 * t580;
        const auto t1302 = -t330;
        const auto t1303 = t1158 * t342;
        const auto t1304 = -t1303;
        const auto t1305 = t1113 * t1236;
        const auto t1306 = t1292 * t583;
        const auto t1307 = t1113 * t659;
        const auto t1308 = t268 * t583;
        const auto t1309 = t1113 * t138;
        const auto t1310 = t138 * t361;
        const auto t1311 = t1245 * t1308;
        const auto t1312 = t1104 - t1309 + t1310 + t1311;
        const auto t1313 = t117 * t592;
        const auto t1314 = t49 * t600;
        const auto t1315 = 2 * t123 * t48 * t49 * t51 * t55 - t1237 * t49
            - t1314 * t312 - t257 * t350;
        const auto t1316 = t312 * t657;
        const auto t1317 = -t117 * t459 + t117 + 2 * t244;
        const auto t1318 = t1132 * t138;
        const auto t1319 = t521 * t845;
        const auto t1320 = t1132 * t659;
        const auto t1321 = t1132 * t1236;
        const auto t1322 = -2 * t1 * t65 * t8 * t95;
        const auto t1323 = t1130 * t291 + t1322 + t290 * t774 + t378;
        const auto t1324 = t1292 * t594;
        const auto t1325 = -t67;
        const auto t1326 = t233 * t48;
        const auto t1327 = t1094 + t346 * t634 + t807;
        const auto t1328 = t10 * t314;
        const auto t1329 = -t1328 + t773;
        const auto t1330 = t10 * t146;
        const auto t1331 = t168 * t353 * t74;
        const auto t1332 = t1129 + t1330 + t1331;
        const auto t1333 = t366 * t634;
        const auto t1334 = -t1319;
        const auto t1335 = t1107 * t521;
        const auto t1336 = -t1335;
        const auto t1337 = t1156 + t1336 + t366 * t774;
        const auto t1338 = -2 * t48 * t61 - 4 * t53 * t55 * t62 + t62;
        const auto t1339 = -t1338;
        const auto t1340 = t211 * t864;
        const auto t1341 = t1340 + t223 + 2 * t368;
        const auto t1342 = std::pow(t48, 3);
        const auto t1343 = t1342 * t204;
        const auto t1344 = p0z - 3 * p1z + t512;
        const auto t1345 = t113 * t51;
        const auto t1346 = t1339 * t204;
        const auto t1347 = std::pow(t7, 3);
        const auto t1348 =
            -3 * t131 - t132 + 4 * t1347 * t83 * t95 - t519 * t61;
        const auto t1349 = t1344 + t1347 * t21 * t876 - 3 * t31 - t365 * t519;
        const auto t1350 = t204 * std::pow(t346, 2);
        const auto t1351 = t204 * std::pow(t366, 2);
        const auto t1352 = t3 * t365;
        const auto t1353 = t211 * t3;
        const auto t1354 = t100 * t350 + t1217 + t1352 * t358 + t1353 * t930;
        const auto t1355 = t1220 + t1353 * t927 + t350 * t404 + t74 * t980;
        const auto t1356 = t404 * t48;
        const auto t1357 = t1356 * t621 + t301 * t965 + t346 + t48 * t989;
        const auto t1358 = t250 * t428 + t301 * t968 + t366 + t48 * t994;
        const auto t1359 = t1194 * t3 + t1356 + t3 * t61;
        const auto t1360 = t1359 * t305;
        const auto t1361 = t100 * t48;
        const auto t1362 = t112 * t1229 + t1352 + t1361;
        const auto t1363 = t228 * t381;
        const auto t1364 = t257 * t859;
        const auto t1365 = t293 * t366;
        const auto t1366 = t293 * t346;
        const auto t1367 = t395 * t55;
        const auto t1368 = t229 * t381;
        const auto t1369 = t272 * t381;
        const auto t1370 = t268 * t395;
        const auto t1371 = t1020 * t1370;
        const auto t1372 = t1280 * t228;
        const auto t1373 = -t110 * t411 + t117 * t406 + t1279 * t436
            + t1372 * t437 + t168 * t434 - t398 * t419 + t429 * t530;
        const auto t1374 = t285
            * (-t1
                   * (-t1213 * t366 * t422 - t1214 * t353 + t1269 * t1357
                      + t1354 * t715 - t1355 * t248 - t1358 * t430
                      + t204 * t665 * t695 + t369 * t411)
               + t10 * t3 * t393 - t1009 * t1367 - t1011 * t1363 - t1016 * t1367
               - t1017 * t1370 - t1084 - t1363 * t434 - t1368 * t436
               - t1369 * t438 - t1371 * t437 - t1373 + t228 * t393 * t437 * t55
               + t268 * t393 * t435
               - t276
                   * (-t122 * t1355 - t124 * t1357 - t1360 * t786 + t1366 * t405
                      - t352 * t427 + t411 * t665)
               - t387
               - t441
                   * (-t1081 * t1362 - t121 * t1358 - t123 * t1354
                      + t1365 * t419 - t380 * t695 + t429 * t964)
               + t55 * t7
                   * (t1004 * t366 + t1234 * t394 - t1357 * t214 + t1358 * t153
                      + t1359 * t1364 - t1362 * t259 * t859 - t357 * t429
                      - t424 * t964)
               - t74
                   * (t110 * t1355 - t1354 * t68 - t1360 * t847
                      + 2 * t1362 * t51 * t55 * t65 + t353 * t419 * t51 * t55
                      - t357 * t695 + t366 * t405 * t48 * t51 * t55
                      - t369 * t424));
        const auto t1375 = t1 * t211;
        const auto t1376 = -t102 * t350 - t1028 - t1204 * t358 - t1375 * t930;
        const auto t1377 = -t1033 - t1375 * t927 - t149 * t980 - t350 * t444;
        const auto t1378 = t444 * t48;
        const auto t1379 = t1192 * t668 + t1253 * t48 + t1378 * t771 + t346;
        const auto t1380 =
            t1196 * t668 + t250 * t498 + t366 + t51 * t728 * t930;
        const auto t1381 = t219 * t477;
        const auto t1382 = t1037 * t64 + t1202 + t1378;
        const auto t1383 = t1382 * t219;
        const auto t1384 = t102 * t48;
        const auto t1385 = t1037 * t112 + t1204 + t1384;
        const auto t1386 = t1385 * t219;
        const auto t1387 = t219 * t352;
        const auto t1388 = t187 * t366;
        const auto t1389 = t187 * t346;
        const auto t1390 = t1279 * t721 + t1372 * t722 + t168 * t718 + t716;
        const auto t1391 = t285
            * (t1 * t10 * t393
               + t1 * t55
                   * (t121 * t1377 + 2 * t123 * t1382 * t49 * t55 - t124 * t1376
                      - t1386 * t233 - t1387 * t477
                      + t366 * t445 * t48 * t49 * t55 + t380 * t445 * t49
                      - t481 * t985)
               - t1050 * t1367 - t1052 * t1367 + t113 * t346 * t49 * t55
               - t117 * t354 - t1245 * t1363 - t1246 * t1370 - t1281
               - t1363 * t718 - t1369 * t723 - t1371 * t722 - t1390
               - t170
                   * (-t1049 * t373 + t117 * t1379 - t1380 * t71 + t1381 * t394
                      - t1383 * t892 + t1386 * t189 + t366 * t712 - t499 * t985)
               + t228 * t393 * t55 * t722 + t268 * t393 * t720
               - t276
                   * (t122 * t1377 - t1379 * t153 - t1383 * t656 + t1389 * t445
                      - t352 * t496 + t394 * t452)
               + t3
                   * (t1025 * t352 - t1225 * t1377 + t1376 * t432 - t1379 * t337
                      + t1380 * t717 - t204 * t373 * t446 + t204 * t394 * t481
                      - t380 * t452)
               - t366 * t816 + t369 * t71 - t381 * t725
               - t441
                   * (t123 * t1376 - t1278 * t1385 - t1380 * t214 + t1388 * t477
                      - t380 * t481 + t499 * t977));
        const auto t1392 = t1270 + t365 * t743 + t540 * t55 + t995;
        const auto t1393 = t1188 * t55 + t1206 * t532 + 2 * t531 + t990;
        const auto t1394 = t1393 * t51;
        const auto t1395 =
            t104 * t349 - t107 * t55 * t7 + t1340 * t7 + t171 * t365 + t366;
        const auto t1396 = t1064 + t211 * t7 * t869 + t250 * t81 + t345
            + t349 * t513 - t55 * t62 * t7;
        const auto t1397 = t380 * t49;
        const auto t1398 = -t1396;
        const auto t1399 = -t1395;
        const auto t1400 = t168 * t824;
        const auto t1401 = t1022 + t271 * t521;
        const auto t1402 = t285
            * (t10 * t393 * t7 - t1082 * t1367 - t1083 * t1367 - t1279 * t1363
               - t1279 * t826 - t1280 * t1370 - t1363 * t824 - t1369 * t828
               - t1371 * t827 - t1372 * t827 - t1400 - t1401
               + t228 * t393 * t55 * t827 + t268 * t393 * t825
               - t276
                   * (t1366 * t525 + t1389 * t525 - t1393 * t339 - t1394 * t153
                      + t1398 * t233 - t352 * t533)
               - t381 * t831
               - t441
                   * (t1365 * t537 + t1388 * t537 - t1392 * t215 - t1392 * t323
                      + t1399 * t830 - t380 * t541)
               + t55
                   * (t1
                          * (-t1276 * t346 + t1277 * t366 - t1387 * t537
                             - t1392 * t332 + t1393 * t330 + t1397 * t525
                             + t1398 * t440 - t1399 * t656)
                      - t3
                          * (-t113 * t1394 + t1392 * t383 - t1395 * t189
                             + t1396 * t892 + t305 * t353 * t537 - t357 * t541
                             + t366 * t823 - t369 * t526)
                      - t391 + t555 + t556 - t558 - t559
                      - t7
                          * (t117 * t1393 * t51 - t1392 * t197
                             + t1392 * t49 * t68 - t1393 * t390)));
        const auto t1403 = 2 * t121 * t48 * t49 * t51 * t55 - t187 * t373
            - t335 * t616 - t373 * t567;
        const auto t1404 = -t110 * t519 + t110 + 2 * t596;
        const auto t1405 = t1097 * t168;
        const auto t1406 = t1097 * t676;
        const auto t1407 = t1097 * t1363;
        const auto t1408 = t1126 + t374 * t394 + t566 * t985 + t633;
        const auto t1409 = t268 * t393;
        const auto t1410 = t1409 * t570;
        const auto t1411 = t361 * t627;
        const auto t1412 = t7 * (-t1411 + t375 + t579 * t964);
        const auto t1413 = t10 * t66;
        const auto t1414 = t1148 + t353 * t625 + t357 * t579 + t379;
        const auto t1415 = t1113 * t168;
        const auto t1416 = t1113 * t1363;
        const auto t1417 = t1409 * t583;
        const auto t1418 = -t116;
        const auto t1419 = t121 * t350;
        const auto t1420 = t1397 + t48 * t964 + 2 * t599;
        const auto t1421 = t1113 * t676;
        const auto t1422 = t1015 * t1367;
        const auto t1423 = t1342 * t55 + t7;
        const auto t1424 = t592 * t977 + t670;
        const auto t1425 = t243 * t519;
        const auto t1426 = t319 * t519;
        const auto t1427 = t1314 * t366;
        const auto t1428 = t1132 * t168;
        const auto t1429 = t1140 * t1279;
        const auto t1430 = -t1102;
        const auto t1431 = t1310 + t1430;
        const auto t1432 = -3 * t1014 * t226 * t395 * t55 * t594 + t1132 * t1363
            + t1132 * t676 + t1409 * t594 + t1428 - t1429 + t1431
            - t226 * t228 * t268 * t381 * t594;
        const auto t1433 = t168 * t3 * t91;
        const auto t1434 = t29 * t679;
        const auto t1435 = t320 * t72;
        const auto t1436 = -t1127 + t1434 + t1435;
        const auto t1437 = -t28;
        const auto t1438 = -t36;
        const auto t1439 = t3 * t796;
        const auto t1440 = t28 * t515;
        const auto t1441 = t36 * t515;
        const auto t1442 = -t1094;
        const auto t1443 = t1095 + t1442;
        const auto t1444 = t44 + t45 * t515 + t685;
        const auto t1445 = -t855;
        const auto t1446 = t1445 + t344;
        const auto t1447 = t219 * t424;
        const auto t1448 = 4 * t108;
        const auto t1449 = -t100 * t315 + t107 - t1448 * t4;
        const auto t1450 = -t1449;
        const auto t1451 = t204 * t869;
        const auto t1452 = t1451 * t4;
        const auto t1453 = -t55 * t62 - 1;
        const auto t1454 = t1452 + t1453 + t301 * t404;
        const auto t1455 = t1213 * t4 * t864 + t1418 + t419 + t428 * t74;
        const auto t1456 =
            -3 * p2x + p3x + t1163 + t1452 * t49 + t195 * t404 + t207 + t402;
        const auto t1457 = -t1454;
        const auto t1458 = -t1455;
        const auto t1459 = -t1456;
        const auto t1460 = t204 * std::pow(t405, 2);
        const auto t1461 = t305 * t786;
        const auto t1462 = 2 * t563;
        const auto t1463 = std::pow(t419, 2);
        const auto t1464 = t228 * t434;
        const auto t1465 = t437 * t859;
        const auto t1466 = t204 * t272;
        const auto t1467 = t228 * t281;
        const auto t1468 = t204 * t993;
        const auto t1469 = t1040 * t301 + t1468 * t156 + t149 * t428 + t477;
        const auto t1470 = t204 * t51;
        const auto t1471 = t1470 * t930;
        const auto t1472 = t1232 * t668 + t1471 * t156 + t419 + t498 * t74;
        const auto t1473 = t3 * t444;
        const auto t1474 = t1223 * t668 + t1252 * t156 + t1473 * t771 + t405;
        const auto t1475 = t1 * t404;
        const auto t1476 = t1030 * t301 + t1475 * t621 + t156 * t988 + t445;
        const auto t1477 = t1 * t100 + t102 * t3 + t1037 * t417;
        const auto t1478 = t1037 * t401 + t1473 + t1475;
        const auto t1479 = t1478 * t250;
        const auto t1480 = t250 * t499;
        const auto t1481 = t228 * t438;
        const auto t1482 = t349 * t419;
        const auto t1483 = t219 * t429;
        const auto t1484 = t1466 * t437;
        const auto t1485 = t436 * t441;
        const auto t1486 = t285
            * (t1016 * t720 + t1056 + t1248 + t1464 * t723 + t1481 * t718
               + t1484 * t722 + t1485 * t722
               - t149
                   * (t113 * t1476 + t1234 * t446 + t1235 * t1477 - t1381 * t422
                      - t1469 * t65 - t1479 * t895 + t416 * t477 - t429 * t496)
               + t170
                   * (t1002 * t1049 + t121 * t1474 - t124 * t1472 - t1447 * t477
                      + t1469 * t153 - t1476 * t214 + t411 * t499 - t429 * t452)
               - t276
                   * (-t1048 * t1478 - t124 * t1476 - t1474 * t153
                      + t349 * t405 * t445 + t411 * t454 + t424 * t452)
               + t434 * t721 + t436 * t718 + t438 * t725
               - t441
                   * (-t1047 * t1477 - t121 * t1469 + t1227 * t499
                      - t1472 * t214 + t1482 * t477 + t1483 * t477)
               - t74
                   * (-t1043 * t419 - t113 * t1474 + t1227 * t446 - t1228 * t477
                      + t1472 * t65 - t1477 * t925 + t1479 * t892
                      + t1480 * t405));
        const auto t1487 = t100 * t7 + t104 * t3 + t1074 * t417;
        const auto t1488 = t3 * t513;
        const auto t1489 = t404 * t7;
        const auto t1490 = t1074 * t401 + t1488 + t1489;
        const auto t1491 = t1076 * t301 + t1468 * t185 + t170 * t428 + t537;
        const auto t1492 = t1068 * t301 + t1489 * t621 + t185 * t988 + t525;
        const auto t1493 = t1070 * t930;
        const auto t1494 = t1361 * t743 + t1493 * t185 + t419 + t540 * t74;
        const auto t1495 = 8 * t532;
        const auto t1496 = t1356 * t743 + t1488 * t358 + t1495 * t185 + t405;
        const auto t1497 = t293 * t537;
        const auto t1498 = t250 * t419;
        const auto t1499 = t285
            * (t1016 * t825 + t1085 + t1373 + t1464 * t828 + t1481 * t824
               + t1484 * t827 + t1485 * t827
               - t149
                   * (-t1079 * t422 + t113 * t1492 - t117 * t1496 - t1491 * t65
                      + t1494 * t71 + t411 * t541 - t429 * t533 + t695 * t819)
               + t170
                   * (t1004 * t537 + t1234 * t526 + t1364 * t1490 - t1447 * t537
                      - t1487 * t259 * t859 + t1491 * t153 - t1492 * t214
                      - t429 * t544)
               - t276
                   * (-t122 * t1496 - t124 * t1492 - t1461 * t1490
                      + t293 * t405 * t525 + t411 * t726 + t422 * t533)
               + t434 * t826 + t436 * t824 + t438 * t831
               - t441
                   * (-t1081 * t1487 - t121 * t1491 - t123 * t1494
                      + t1483 * t537 + t1497 * t419 + t1498 * t541)
               - t74
                   * (t110 * t1496 + 2 * t1487 * t51 * t55 * t65
                      - t1490 * t305 * t847 - t1494 * t68
                      + t405 * t48 * t51 * t537 * t55 - t406 * t541
                      + t419 * t51 * t533 - t544 * t695));
        const auto t1500 = t1002 * t567 + t1483 * t51 + t310 * t655;
        const auto t1501 = t433 * t472;
        const auto t1502 = t472 * t691;
        const auto t1503 = t695 * t74;
        const auto t1504 = t504 * t696;
        const auto t1505 = t691 * t879;
        const auto t1506 = t472 * t950;
        const auto t1507 = t696 * t953;
        const auto t1508 = -t372;
        const auto t1509 = t137 * t459;
        const auto t1510 = t1504 * t590;
        const auto t1511 = t1502 * t590;
        const auto t1512 = t1505 * t583;
        const auto t1513 = t777 + t838;
        const auto t1514 = t165 * t627;
        const auto t1515 = t1514 + t406 * t579 + t413 + t422 * t625;
        const auto t1516 = t1507 * t583;
        const auto t1517 = t1124 - t419 * t51 * t55 * t592 + t625 * t695;
        const auto t1518 = t1128 + t405 * t634 + t427 * t592 + t632;
        const auto t1519 = t149 * t429;
        const auto t1520 = t1002 * t74;
        const auto t1521 = -t596 + t597;
        const auto t1522 = t1501 * t594 + t1502 * t602 + t1504 * t602
            - t1505 * t594 + t1506 * t594 - t1507 * t594 - t29 * t602;
        const auto t1523 = -t1142;
        const auto t1524 = t1143 * t445 + t1523 + t613;
        const auto t1525 = t305 * t446;
        const auto t1526 = t26 + t27 * t459 + t462;
        const auto t1527 = -t1146 + t1147;
        const auto t1528 = -t137;
        const auto t1529 = t1 * t466;
        const auto t1530 = t36 * t459;
        const auto t1531 = t1303 + t1445;
        const auto t1532 = -t102 * t286 + t107 - t1448 * t5;
        const auto t1533 = t1451 * t5;
        const auto t1534 = t1453 + t1533 + t444 * t668;
        const auto t1535 = -t1534;
        const auto t1536 = t1136 + t1470 * t5 * t864 + t149 * t498 + t477;
        const auto t1537 = -t1536;
        const auto t1538 =
            -3 * p2y + p3y + t1325 + t1533 * t51 + t303 + t442 + t444 * t448;
        const auto t1539 = -t1538;
        const auto t1540 = t204 * std::pow(t445, 2);
        const auto t1541 = t219 * t656;
        const auto t1542 = std::pow(t477, 2);
        const auto t1543 = t228 * t859;
        const auto t1544 = t1274 * t668 + t1471 * t608 + t170 * t498 + t537;
        const auto t1545 = t1384 * t743 + t149 * t540 + t1493 * t608 + t477;
        const auto t1546 = t1 * t513;
        const auto t1547 = t1378 * t743 + t1495 * t608 + t1546 * t358 + t445;
        const auto t1548 = t444 * t7;
        const auto t1549 = t1252 * t608 + t1266 * t668 + t1548 * t771 + t525;
        const auto t1550 = t1 * t104 + t1 * t1074 * t108 + t102 * t7;
        const auto t1551 = t1550 * t219;
        const auto t1552 = t1 * t1074 * t63 + t1546 + t1548;
        const auto t1553 = t187 * t477;
        const auto t1554 = t285
            * (t1052 * t825 + t1282 + t1390 + t1466 * t722 * t827
               + t149
                   * (-t121 * t1547 + 2 * t123 * t1552 * t49 * t55
                      + t124 * t1545 - t1276 * t445 - t1551 * t233
                      + t445 * t48 * t49 * t537 * t55 + t477 * t49 * t533
                      - t481 * t819)
               - t170
                   * (-t1049 * t1080 + t117 * t1549 + t1381 * t526 - t1544 * t71
                      + t1551 * t189 - t1552 * t219 * t892 + t452 * t538
                      - t499 * t819)
               + t228 * t718 * t828 + t228 * t723 * t824
               - t276
                   * (-t122 * t1547 - t153 * t1549 - t1541 * t1552
                      + t187 * t445 * t525 + t446 * t533 + t452 * t526)
               + t441 * t721 * t827 + t441 * t722 * t826
               - t441
                   * (t1073 * t499 - t123 * t1545 + t1250 * t541 - t1278 * t1550
                      - t1544 * t214 + t1553 * t537)
               + t718 * t826 + t721 * t824
               + t74
                   * (t122 * t1544 - t123 * t1549 - t1525 * t537 - t153 * t1545
                      + t1547 * t214 + t452 * t541 + t481 * t544
                      - t499 * t533));
        const auto t1555 = t149 * t481;
        const auto t1556 = t1049 * t566 + t138 * t806 + t1513 + t446 * t779;
        const auto t1557 = t472 * t502;
        const auto t1558 = t471 * t879;
        const auto t1559 = t503 * t953;
        const auto t1560 = -t1 * t10 * t125 * t472 * t570
            + t1 * (t1285 + t477 * t48 * t55 * t566 - t477 * t774 - t671)
            - t127 * t503 * t55 * t576 + t138 * t576 + t1557 * t570
            + t1558 * t570 + t1559 * t570 - t471 * t472 * t576 + t573;
        const auto t1561 = t149 * t478;
        const auto t1562 = t503 * t504;
        const auto t1563 = t471 * t472;
        const auto t1564 = -t318;
        const auto t1565 = t499 * t74;
        const auto t1566 = t28 * t519;
        const auto t1567 = t138 * t602;
        const auto t1568 = t1558 * t594;
        const auto t1569 = t66 * t748;
        const auto t1570 = t1569 + t377 + t445 * t774 + t496 * t592 + t708;
        const auto t1571 = t1559 * t594;
        const auto t1572 = t1557 * t594;
        const auto t1573 = t1523 + t525 * t774;
        const auto t1574 = t34 + t35 * t519 + t549;
        const auto t1575 = t1442 + t525 * t634 + t767;
        const auto t1576 = t168 * t534;
        const auto t1577 = -t1413 + t1569 + t1576;
        const auto t1578 = t7 * t731;
        const auto t1579 = t137 * t519;
        const auto t1580 = -t104 * t518 + t107 - t1448 * t8;
        const auto t1581 = -t1580;
        const auto t1582 = t1451 * t8;
        const auto t1583 = t1453 + t1582 + t513 * t743;
        const auto t1584 = t1070 * t8 * t864 + t1294 + t170 * t540 + t537;
        const auto t1585 =
            -3 * p2z + p3z + t1582 * t48 + t364 + t513 * t528 + t524 + t852;
        const auto t1586 = -t1583;
        const auto t1587 = -t1584;
        const auto t1588 = -t1585;
        const auto t1589 = t204 * std::pow(t525, 2);
        const auto t1590 = std::pow(t537, 2);
        const auto t1591 = t149 * t541;
        const auto t1592 = t1435 + t374 * t526 + t566 * t819 + t632 + t820;
        const auto t1593 = t472 * t564;
        const auto t1594 = t472 * t545;
        const auto t1595 = t504 * t542;
        const auto t1596 = t545 * t879;
        const auto t1597 = t1191 * t472;
        const auto t1598 = t542 * t953;
        const auto t1599 = t139 + t1593 * t570 + t1594 * t576 + t1595 * t576
            - t1596 * t570 + t1597 * t570 - t1598 * t570 - t168 * t576
            - t27 * t39
            + t7
                * (2 * t110 * t4 * t7 * t95 - t1143 * t537
                   + t51 * t537 * t55 * t566 - t671);
        const auto t1600 = t1143 * t525 + t1147 + t377 + t544 * t579;
        const auto t1601 = -t10 * t125 * t472 * t583 * t7 - t10 * t590 * t7
            + t1594 * t590 + t1595 * t590 + t1596 * t583 + t1598 * t583
            - t472 * t564 * t583 + t589
            + t7 * (t1079 * t579 - t1080 * t374 + t1411 + t1508);
        const auto t1602 = t1276 * t48 + t440 * t746 + t538 * t600;
        const auto t1603 = t117 * t250;
        const auto t1604 = t153 * t305;
        const auto t1605 = t122 * t250;
        const auto t1606 = t229 * t269;
        const auto t1607 = t55 * t566;
        const auto t1608 = 3 * t603;
        const auto t1609 = t130 * t509 / std::pow(t134, 3.0 / 2.0);
        const auto t1610 = t1608 * t570;
        const auto t1611 = t1609
            * (t156 * t159 * (t158 + t187 + t293) - t1610 * t583 - t570 * t590
               + t576 * t583);
        const auto t1612 = t349 - 2;
        const auto t1613 = t1609
            * (t159 * t185 * (t141 + t1612 + t187) - t1610 * t594 + t570 * t602
               + t576 * t594);
        const auto t1614 = -t585 + t586;
        const auto t1615 = -t1310;
        const auto t1616 = t1286 - t1519 * t168;
        const auto t1617 = t151 * t51;
        const auto t1618 = t1609
            * (t159 * t608 * (t1612 + t293 + t38) - t1608 * t583 * t594
               + t583 * t602 - t590 * t594);
        const auto t1619 = t1313 - t246;
        const auto t1620 = t117 * t51;
        hess[0] = t136
            * (t125 * (std::pow(t76, 2) + t99) - t129 * std::pow(t47, 2)
               + 2 * t47 * t94);
        hess[1] = t164;
        hess[2] = t186;
        hess[3] = t285
            * (t1 * t55 * (t196 * t232 + t235)
               - t170 * (t190 + t194 - t198 + t202 * t203)
               - t229
                   * (-t196 * t216 - t196 * t218 + t205 * t210 + t210 * t211
                      + 2 * t212 * t213 - t225 * t76)
               - t284 + t3 * t55 * (t196 * t230 - t196 * t231));
        hess[4] = t285
            * (t226 * t228 * (-t243 * t302 - t322) - t341
               - t4
                   * (-t10 * t292 + t10 * t297 * t55 * t7 - t288
                      + t295 * t55 * t65)
               + t55 * t7
                   * (t154 * t301 * t51 + t296 * t76 + t298 * t75 + t3 * t300));
        hess[5] = t285
            * (-t1 * (t3 * t351 * t55 * t71 - t344 - t348 - t354 * t76)
               + t226 * t228 * (-t245 * t360 - t376) - t396
               + t4 * t55 * (t356 + t359));
        hess[6] = t285
            * (t149
                   * (2 * t122 * t415 - t232 * t414 - t3 * t416
                      + t405 * t48 * t76)
               + t226 * t274 * t436 + t241 * t436 + t242 * t438 + t273 * t438
               + t274 * t434 + t283
               - t441
                   * (t215 * t414 + t217 * t414 + t293 * t439 + t349 * t439
                      + t415 * t440 + t429 * t76)
               - t7 * (-t397 * t398 + t399 * t400 + t406 * t76 + t413)
               + t74 * (t153 * t414 * t48 - t230 * t414));
        hess[7] = t510
            * (-t4 * (-t10 * t447 + t288 + t430 * t449 + t453)
               - t469 * (t463 + t467) - t508
               + t7
                   * (t154 * t286 * t455 - t449 * t457 * t55 + t454 * t456
                      - t458));
        hess[8] = t510
            * (t1 * t10 * t3 * t87
               - t1
                   * (t133 * t518 * t96 + t3 * t490 * t520 + t343 * t514
                      - t40 * t522)
               + t10 * t7 * t94 + t125 * t127 * t47 * t472 * t545
               + 3 * t125 * t47 * t507 * t542 * t55
               + t4 * (-t10 * t527 + t10 * t534 + t523 * t66 + t529 * t530)
               + t40 * t91 - t469 * (t550 + t554) - t517 - t543 - t546 - t547
               - t565);
        hess[9] = t577;
        hess[10] = t591;
        hess[11] = t605;
        hess[12] = t164;
        hess[13] = t136
            * (t125 * (std::pow(t151, 2) + t607) - t129 * std::pow(t145, 2)
               - 2 * t145 * t155);
        hess[14] = t610;
        hess[15] = t285
            * (t226 * t228 * (-t318 - t319 * t623 - t630)
               + t5 * t55 * (t619 + t622) - t646
               - t7 * (t1 * t55 * t617 * t68 - t151 * t253 - t613 - t615));
        hess[16] = t285
            * (-t149
                   * (t1 * t10 * t290 * t3 * t48 + t449 * t48 * t71
                      - t449 * t652 - t650 * t651)
               - t229
                   * (-t151 * t325 + t211 * t653 - t218 * t449 + t221 * t653
                      - t324 * t449 + t55 * t654 * t655)
               + t55 * t7 * (t154 * t449 + t658) - t663
               - t74 * (t297 * t650 + t342 * t647 + t648 - t649));
        hess[17] = t285
            * (t226 * t228 * (-t314 * t360 - t674)
               + t3 * t55
                   * (t1 * t356 + t150 * t667 + t151 * t352 + t230 * t48 * t668)
               - t5
                   * (t10 * t3 * t353 * t55 - t10 * t666 + t351 * t55 * t71
                      - t664)
               - t677);
        hess[18] = t510
            * (t5 * (-t10 * t678 + t10 * t679 + t196 * t430 + t316 * t72)
               - t690 * (t686 + t689) + t7 * (-t149 * t196 * t68 + t683)
               + t703);
        hess[19] = t285
            * (-t149
                   * (t1 * t10 * t3 * t445 * t48 - t454 * t650
                      + t48 * t705 * t71 - t652 * t705)
               + t170
                   * (2 * t124 * t711 + t151 * t445 * t49 - t154 * t710 - t713)
               - t3 * (-t494 * t705 + t709)
               - t441
                   * (t151 * t499 + t187 * t724 + t217 * t710 + t323 * t710
                      + t349 * t724 + t655 * t711)
               + t636 * t721 + t637 * t723 + t639 * t725 + t641 * t723
               + t642 * t718 + t662);
        hess[20] = t510
            * (t3
                   * (t204 * t230 * t286 * t7 + t526 * t728 - t529 * t55 * t729
                      - t730)
               - t5 * (-t10 * t170 * t726 + t29 * t533 + t529 * t715 + t664)
               - t690 * (t550 + t732) - t733);
        hess[21] = t736;
        hess[22] = t737;
        hess[23] = t739;
        hess[24] = t186;
        hess[25] = t610;
        hess[26] = t136
            * (t125 * (std::pow(t172, 2) + t740) - t129 * std::pow(t184, 2)
               + 2 * t180 * t184);
        hess[27] = t285
            * (t1 * t55
                   * (t171 * t742 + t172 * t201 + t232 * t49 * t743 + t619 * t7)
               + t226 * t228 * (-t361 * t623 - t750) - t765
               - t8
                   * (t1 * t10 * t202 * t55 - t10 * t614 + t55 * t617 * t68
                      - t741));
        hess[28] = t285
            * (t226 * t228 * (-t302 * t370 - t671 - t776)
               - t3 * (-t172 * t769 + t295 * t55 * t65 * t7 - t767 - t768)
               + t55 * t8 * (t300 + t772) - t781);
        hess[29] = t285
            * (-t149 * (t172 * t665 + t353 * t783 - t529 * t652 + t611 * t782)
               - t229
                   * (-t172 * t380 + t205 * t784 - t216 * t529 + t221 * t784
                      - t324 * t529 + 2 * t337 * t785)
               + t3 * t55 * (t230 * t529 + t788)
               + t55 * t7 * (t154 * t529 - t259 * t529) - t790);
        hess[30] = t510
            * (-t1 * (-t196 * t791 + t795) - t798 * (t686 + t797)
               - t8 * (-t10 * t681 + t196 * t530 + t412 + t741) - t804);
        hess[31] = t510
            * (t3 * (-t449 * t791 + t809) - t798 * (t463 + t811)
               + t8 * (-t10 * t805 + t10 * t806 + t165 * t287 + t449 * t715)
               + t814);
        hess[32] = t285
            * (-t1 * (t172 * t819 - t815 * t816 + t817 * t818 + t820)
               + t170 * (t124 * t51 * t821 - t154 * t821)
               - t441
                   * (t172 * t541 + t187 * t829 + t215 * t821 + t293 * t829
                      + t323 * t821 + t822 * t830)
               + t74
                   * (2 * t153 * t822 + t172 * t51 * t525 - t230 * t821
                      - t7 * t823)
               + t755 * t826 + t756 * t828 + t759 * t831 + t761 * t828
               + t762 * t824 + t789);
        hess[33] = t834;
        hess[34] = t835;
        hess[35] = t836;
        hess[36] = t285
            * (t1 * t55 * (t195 * t232 + t232 + t235)
               + t226 * t228
                   * (t10 * t209 * t3 * t48 * t55 * t7 - t188 * t851
                      - t224 * t76 + t843 + t844 - t846 - t848 + t850)
               - t284 + t3 * t55 * (-t195 * t231 + t237 + t327)
               - t7 * (t190 * t55 + t194 * t55 + t842));
        hess[37] = t285
            * (t226 * t228 * (t1 * t10 * t117 - t321 - t630)
               + t5 * t55 * (t132 - t187 * t233 + t622 + t852)
               + t55 * t7
                   * (2 * t1 * t153 * t50 * t55 - t150 * t742 - t151 * t201
                      - 2 * t259 * t657 - t729)
               - t646);
        hess[38] = t285
            * (-t1 * (t172 * t253 + t854 + t856)
               - t157
                   * (t1 * t202 * t55 + 2 * t10 * t4 * t68 - 2 * t239 - t614
                      - t68)
               + t226 * t228 * (-t750 - t858) - t765);
        hess[39] = t510
            * (2 * t10 * t264 * t3 - t110 * t234 * t859
               + 2 * t113 * t192 * t51 * t55
               + t125 * t127
                   * (-t10 * t877 * t895 + std::pow(t224, 2) + t52 * t891
                      + t53 * t891 + t866 * t893 + t866 * t894)
               + t125 * t472
                   * (-t10 * t611 * t874 + std::pow(t202, 2) * t204 + t52 * t888
                      + t53 * t888 + t872 * t889 + t872 * t890)
               + 2 * t127 * t264 * t885 - t128 * t878 * t885 * t896 - t162 * t96
               - t209 * t383 * t859 + 2 * t209 * t48 * t55 * t68
               + 2 * t264 * t472 * t55 * t878
               - t286
                   * (t10 * t65 * t877 + t202 * t204 * t209 * t48 - t224 * t631
                      + t48 * t55 * t71 * t866 - t871 * t872 - t874 * t875)
               - t301 * t882
               - t301
                   * (t110 * t48 * t55 * t862 - t262 * t866 + t51 * t65 * t866
                      - t862 * t863)
               - t315 * t887
               + 2 * t55 * t7
                   * (-t124 * t51 * t866
                      - t153 * (3 * t116 - t208 * t616 - t864 * t868 + t870)
                      + t192 * t225 * t51
                      + t214
                          * (3 * t49 * t55 * t62 - t57 * t616 - t86
                             - t868 * t869)
                      - t258 * t861 - t261 * t849)
               - t700 * std::pow(t885, 2) - t860 - std::pow(t878, 2) * t880);
        hess[40] = t956;
        hess[41] = t987;
        hess[42] = t1024;
        hess[43] = t1057;
        hess[44] = t1086;
        hess[45] = t285
            * (-t1 * (t1087 * t1088 + t1089 + t858) - t1106
               + t226 * t268
                   * (-t1087 * t818 - t1094 - t1096 - t253 * t566 - t615 - t853
                      - t856)
               - t29 * (-t1 * t1091 + t1090 * t849 + t1092)
               + t55 * t7
                   * (-t1087 * t655 - t1093 + 2 * t121 * t50 * t51 * t55
                      + t225 * t49 * t51 - t257));
        hess[46] = t285
            * (t10 * t226 * t268 * t3 * t583
               + 3 * t1014 * t226 * t266 * t55 * t583 - t1114 - t1115 - t1116
               + t1118 * t3 * t55 - t1122 - t141 * (t1091 + t1110 - t744)
               + t226 * t228 * t236 * t268 * t583
               + t226 * t268 * (-t1121 - t842) - t587
               - t7 * (-t1108 - t1109 + t224 * t579 - t843 - t850));
        hess[47] = t285
            * (t1
                   * (-t1123 * t211 + 2 * t123 * t204 * t48 * t50 - t209 * t981
                      - t218 + t225 * t600)
               + t1011 * t1140 + t1012 * t1140 - t1125 - t1133 - t1134 - t1135
               - t1139 + t1141 * t594 + t171 * (t1136 - t1137 + t1138 + t25)
               + t275 * (-t1126 - t1127 - t1131) + t598);
        hess[48] = t285
            * (t226 * t228 * (-t1145 - t322) - t341
               - t38
                   * (2 * t10 * t5 * t65 - 2 * t148 - t292 + t297 * t55 * t7
                      - t65)
               - t7 * (t1095 + t1144 + t76 * t769));
        hess[49] = t285
            * (-t138
                   * (t1 * t290 * t3 * t48 * t55 - t1151 + t1152 - t1153 * t651
                      + t146 - t72)
               + t226 * t228
                   * (t1154 + t1155 + t1156 - t1157 * t647 + t1162
                      - t151 * t307)
               - t3 * (t1150 + t55 * t648 + t647 * t707)
               + t55 * t7 * (t154 * t448 + t154 + t658) - t663);
        hess[50] = t285
            * (t226 * t228 * (t10 * t110 * t7 - t673 - t776)
               + t3 * t55
                   * (-t1164 + 2 * t122 * t52 * t55 * t7 - t171 * t298
                      - t172 * t296 - 2 * t231 * t787)
               + t55 * t8 * (t1163 - t124 * t294 + t772 + t86) - t781);
        hess[51] = t956;
        hess[52] = t510
            * (2 * t1 * t55
                   * (-t113 * t1166 * t219 + t1166 * t117 * t48 * t55
                      + t1168 * t49 * t65 - t1168 * t897)
               + 2 * t10 * t133 * t3 * t901 + 2 * t10 * t45 * t7 * t899
               - t1165 * t87 * t901
               - t1165
                   * (t10 * t3 * t901 * t904 - t1169 * t899 + t1172 * t45
                      + t1173 * t87 + t3 * t91 * (-4 * t1170 + 2 * t906 + t907)
                      - t489 * (-4 * t141 * t83 + 3 * t79 + t82))
               - t1178 * t668 - t1184 * t128 * t941 - t1184 * t916
               - t1185 * t138 - t1185 * t949
               + t125 * t127
                   * (t1168 * t1182 + t1168 * t894 - 2 * t1173 * t1183
                      + t1181 * t50 + t1181 * t53 + std::pow(t307, 2))
               + t125 * t472
                   * (-t10 * t1177 * t189 + t1175 * t1180 + t1175 * t890
                      + t1179 * t50 + t1179 * t53 + t204 * std::pow(t297, 2))
               - t159 * t286 * t949 - t162 * t491 - 2 * t29 * t35 * t899
               + 2 * t3
                   * (t10 * t1173 * t65 + t1168 * t48 * t55 * t68
                      - t1174 * t1175 - t1176 * t307 - t1177 * t875
                      + t204 * t297 * t312 * t48)
               - t700 * std::pow(t941, 2) - t860 - t880 * std::pow(t946, 2));
        hess[53] = t1212;
        hess[54] = t1249;
        hess[55] = t1263;
        hess[56] = t1283;
        hess[57] = t285
            * (t1100 * t1241 + t1100 * t1245 - t1287 - t1290 - t1291 - t1293
               - t1297 + t1298 * t570 + t275 * (-t1120 - t1288 - t1289) + t573
               + t7
                   * (2 * t121 * t204 * t49 * t52 - t1284 * t215 - t312 * t934
                      - t324 + t325 * t567)
               + t75 * (t1294 - t1295 + t1296 + t33));
        hess[58] = t285
            * (t1241 * t1308 + t1298 * t583 - t1305 - t1306 - t1307 + t1312
               + t149 * (t1117 + t1302 - t294 * t331 + t331)
               + t275
                   * (-t1144 - t1299 * t400 - t1304 - t579 * t769 - t768 - t855)
               - t7 * (t1145 + t1299 * t851 + t1300)
               + t74
                   * (-t1299 * t830 - t1301 + 2 * t214 * t48 * t52 * t55
                      + t325 * t48 * t51 - t335));
        hess[59] = t285
            * (t1 * t10 * t226 * t268 * t594 + t1 * t1315 * t55
               + 3 * t1014 * t226 * t340 * t55 * t594 + t1313 - t1318 - t1320
               - t1321 - t1324 - t157 * (t1316 + t1317 - t308)
               + t226 * t228 * t268 * t326 * t594
               + t226 * t268 * (-t1150 - t1323) - t246
               - t3 * (-t1162 - t1319 + t307 * t592));
        hess[60] = t285
            * (t1 * t55
                   * (2 * t124 * t3 * t53 * t55 - t1326 * t75 - t352 * t76
                      - t457 - t667 * t75)
               + t226 * t228 * (t10 * t113 * t3 - t376 - t749) - t396
               + t4 * t55 * (t1325 - t153 * t350 + t359 + t90));
        hess[61] = t285
            * (-t141
                   * (2 * t10 * t71 * t8 + t3 * t353 * t55 - 2 * t569 - t666
                      - t71)
               + t226 * t228 * (-t1329 - t674)
               - t3 * (t1303 + t1327 + t151 * t354) - t677);
        hess[62] = t285
            * (-t1 * (t1332 + t172 * t985 + t782 * t818)
               + t226 * t228
                   * (-t1088 * t782 + t1333 + t1334 + t1337 - t172 * t369
                      + t843)
               + t3 * t55 * (t230 * t528 + t230 + t788)
               + t55 * t7 * (-t259 * t528 + t751 + t758) - t790);
        hess[63] = t987;
        hess[64] = t1212;
        hess[65] = t510
            * (2 * t1 * t55
                   * (t121
                          * (-t132 - t1343 * t869 - t350 * t61
                             + 3 * t48 * t55 * t62)
                      - t122 * t1341 * t49
                      - t124 * (3 * t112 - t1343 * t864 + t1344 - t350 * t365)
                      - t1338 * t388 + t346 * t380 * t49 - t352 * t964)
               + 2 * t10 * t392 * t7 + 2 * t110 * t346 * t49 * t55
               - t1189 * t743
               + t125 * t127
                   * (t1182 * t1341 + t1341 * t893 - 2 * t1349 * t875
                      + t1351 * t50 + t1351 * t52 + std::pow(t369, 2))
               + t125 * t472
                   * (-t10 * t1348 * t342 + t1180 * t1346 + t1346 * t889
                      + t1350 * t50 + t1350 * t52 + t204 * std::pow(t353, 2))
               + 2 * t127 * t392 * t962 - t128 * t896 * t961 * t962
               - t159 * t518 * t963 - t162 * t521 - t189 * t964
               - t315
                   * (t10 * t1349 * t68 - t1183 * t1348
                      + t1341 * t51 * t55 * t65 - t1345 * t1346
                      + t204 * t353 * t366 * t51 - t357 * t369)
               - t357 * t895 + 2 * t366 * t51 * t55 * t71
               + 2 * t392 * t472 * t55 * t961 - t700 * std::pow(t962, 2)
               - t743
                   * (t117 * t1339 * t51 * t55 - t1339 * t333 - t1341 * t197
                      + t1341 * t49 * t68)
               - t860 - t880 * std::pow(t961, 2));
        hess[66] = t1374;
        hess[67] = t1391;
        hess[68] = t1402;
        hess[69] = t285
            * (-t1 * (-t1337 + t369 * t566 - t848)
               + t10 * t226 * t268 * t570 * t7
               + 3 * t1014 * t226 * t395 * t55 * t570 + t110 * t566
               + t1403 * t55 * t7 - t1405 - t1406 - t1407 - t1410
               + t226 * t228 * t268 * t381 * t570
               + t226 * t268 * (-t1332 - t1408)
               - t38 * (t1404 + t170 * t373 - t669) - t638);
        hess[70] = t285
            * (t1279 * t1308 + t1308 * t1368 - t1412 - t1415 - t1416 - t1417
               - t1421 + t1422 * t583 + t150 * (t1418 - t1419 + t1420 + t43)
               + t275 * (-t1322 - t1413 - t1414)
               + t3
                   * (-t1207 * t366 + 2 * t204 * t214 * t51 * t53
                      - 2 * t205 * t217 - t216 + t380 * t580)
               + t589);
        hess[71] = t285
            * (t1 * t55
                   * (2 * t123 * t49 * t53 * t55 - t1423 * t440 - t1427 - t330
                      + t380 * t48 * t49)
               - t1432
               - t168
                   * (-t1090 * t373 + t1425 - t1426 - t243 + t319 + t657 * t784)
               + t226 * t268
                   * (-t1142 - t1327 - t1423 * t707 - t348 - t354 * t592 - t793)
               - t3 * (t1157 * t1423 + t1329 + t1424));
        hess[72] = t510
            * (t1 * (t1436 + t399 * t707 + t427 * t76) + t10 * t3 * t94
               + t10 * t3
                   * (t1 * t678 + t515 * t66 - t515 * t69 - t66 - t681 * t7
                      + t69)
               + t125 * t127 * t47 * t472 * t691
               + 3 * t125 * t47 * t507 * t55 * t696 - t138 * t175 + t1433
               - t161 * t433
               - t168
                   * (t156 * (-t29 * t404 - t492 * t96 - t493) + t315 * t92
                      + t515 * t88 - t516 * t693 - t88)
               - t3 * t506
               - t468
                   * (t1437 + t1438 + t1439 + t1440 + t1441 + t3 * t688
                      - t315 * t46 - t40 * t684)
               - t473 * t691 - t505 * t696);
        hess[73] = t510
            * (t141 * (-t342 * t38 + t65 - t678 + t679 + 2 * t73)
               - t690 * (t1444 + t689) + t7 * (t1443 + t683) + t703);
        hess[74] = t510
            * (-t1 * (t1446 + t795)
               + t55 * t8
                   * (-t1447 + 2 * t153 * t3 * t49 * t55 - t259 * t301
                      + t411 * t51 - t68)
               - t798 * (t1444 + t797) - t804);
        hess[75] = t1024;
        hess[76] = t1249;
        hess[77] = t1374;
        hess[78] = t285
            * (2 * t1007 + t1015 * std::pow(t435, 2) + t1023 + t1464 * t1465
               + t1465 * t229 * t436 + t1466 * std::pow(t437, 2)
               + t1467 * t301 * t437
               - t275
                   * (t1048 * t1457 + t1457 * t1461 + t1459 * t1462
                      + t1460 * t52 + t1460 * t53 + std::pow(t411, 2))
               + t281 * t315 * t436 + 2 * t434 * t436
               - t441
                   * (t1047 * t1449 + t1081 * t1449 + t1458 * t440
                      + t1463 * t293 + t1463 * t349 + std::pow(t429, 2) * t55)
               + t859
                   * (-t1
                          * (t113 * t1456 - t117 * t1454 * t48 + t1450 * t251
                             - t1455 * t65 + t416 * t419 - t427 * t429)
                      - t3
                          * (t110 * t1454 * t48 - t1345 * t1454 - t1450 * t494
                             + t1450 * t51 * t55 * t65)
                      + t420 - t421 + t423 - t425
                      + t7
                          * (t124 * t1449 * t51 * t55 - t1457 * t257
                             - t1458 * t153 + t1459 * t214 - t406 * t429
                             + t411 * t419 * t51)));
        hess[79] = t1486;
        hess[80] = t1499;
        hess[81] = t510
            * (-t1 * (t1088 * t310 + t1498 * t566 - t362 - t429 * t779 + t857)
               + t1105 + t1501 * t570 + t1502 * t576 + t1504 * t576
               - t1505 * t570 + t1506 * t570 - t1507 * t570
               + t170 * (2 * t121 * t3 * t49 * t51 * t55 - t1500 + t257)
               - t29 * t576
               - t29 * (-t1 * t1503 - t1092 + t3 * t419 * t51 * t55 * t7)
               + t603
                   * (t1443 + t1446 + t310 * t818 + t411 * t566 + t682 + t794));
        hess[82] = t510
            * (t141 * (t1110 - t1503 + t170 * t429) - t142 * t35 + t1501 * t583
               + t1506 * t583 - t1510 - t1511 - t1512 - t1516
               - t168 * (t142 * t684 - t1439 - t1440 + t1509 + t28) + t182
               + t29 * t590
               + t3 * (t1498 * t579 + t1508 + t245 * t627 - t419 * t634)
               + t603 * (t1513 + t1515 - t837));
        hess[83] = t510
            * (t149
                   * (2 * t123 * t3 * t48 * t49 * t55 + t123 * t48 - t1419 * t3
                      - t1482 * t49 - t429 * t600)
               + t1517 * t3 + t1521 + t1522
               - t157 * (-t110 * t515 + t110 + t1519 - t1520 + 2 * t638)
               + t603 * (t1436 + t1518));
        hess[84] = t510
            * (t4 * t55
                   * (2 * t1 * t122 * t51 * t55 - t1525 - t231 * t668
                      + t452 * t48 - t65)
               - t469 * (t1526 + t467) - t508 - t7 * (t1096 + t1524 + t458));
        hess[85] = t510
            * (-t1 * t699 - t138 * t155
               - t138
                   * (t1 * t3 * t445 * t48 * t55 + t1151 - t1152 - t147
                      - t7 * t805)
               + t170 * (t151 * t454 + t154 * t704 - t154 + t656 * t711 - t713)
               - t3 * (t1527 + t709)
               - t468
                   * (t1 * t810 - t143 * t460 + t1438 - t144 * t286 + t1509
                      + t1528 + t1529 + t1530)
               + t471 * t692 + t471 * t698 + t502 * t609 + t503 * t697
               + t503 * t702 + t661);
        hess[86] = t510
            * (t157 * (-t459 * t71 + 2 * t568 + t71 - t805 + t806)
               + t3 * (t1531 + t809) - t798 * (t1526 + t811) + t814);
        hess[87] = t1057;
        hess[88] = t1263;
        hess[89] = t1391;
        hess[90] = t1486;
        hess[91] = t285
            * (t1015 * std::pow(t720, 2) + 2 * t1260 + t1262
               + t1466 * std::pow(t722, 2) + t1467 * t668 * t722
               + t1543 * t718 * t722
               - t275
                   * (t1048 * t1535 + t1535 * t1541 + 2 * t1539 * t432
                      + t1540 * t50 + t1540 * t53 + std::pow(t452, 2))
               + t281 * t286 * t721
               - t441
                   * (t1047 * t1532 + t1278 * t1532 + t1537 * t655
                      + t1542 * t187 + t1542 * t349 + std::pow(t499, 2) * t55)
               + 2 * t718 * t721 + t722 * t725 * t859
               + t859
                   * (t1
                          * (t121 * t1535 * t48 + t122 * t1532 * t49 * t55
                             - t1532 * t232 * t55 - t1535 * t330)
                      + t3
                          * (-t122 * t1537 + t123 * t1539 - t1480 * t445
                             + t153 * t1532 * t48 * t55 - t1535 * t335
                             + t452 * t477 * t48)
                      - t479 - t480 + t482 + t483
                      - t7
                          * (-t1049 * t499 + t117 * t1538 - t1532 * t714
                             - t1534 * t390 - t1536 * t71 + t452 * t478)));
        hess[92] = t1554;
        hess[93] = t510
            * (t125 * t472 * (-t1288 + t1514 + t1556) - t1560
               - t38 * (-t113 * t459 + t113 - t1555 + t170 * t499 + 2 * t585)
               + t55 * t7
                   * (-t1 * t1137 + 2 * t1 * t121 * t49 * t51 * t55 + t121 * t49
                      - t1553 * t51 - t499 * t567));
        hess[94] = t510
            * (t1 * t10 * t125 * t472 * t583 + t1 * t10 * t3 * t35
               + t1 * t10 * t590
               + t1 * t10
                   * (-t1 * t481 * t74 + t1561 * t7 - t245 * t459 + t361 * t459
                      + t584)
               + t125 * t472
                   * (t1524 + t1531 + t189 * t629 + t452 * t579 + t808)
               - t1557 * t583 - t1558 * t583 - t1559 * t583 - t1562 * t590
               - t1563 * t590
               - t168
                   * (t142 * t286 * t45 + t142 * t3 * t465 - t3 * t461
                      - t459 * t488 + t488)
               + t3
                   * (t10 * t370 + t1250 * t579 - t499 * t625 + t629 * t847
                      - t672)
               - t45 * t650);
        hess[95] = t510
            * (t1 * (t1381 * t592 + t1564 + t319 * t748 - t481 * t779)
               + t1190 * t472 * t594 + t1562 * t602 + t1563 * t602 - t1567
               - t1568 + t157 * (t1317 - t1561 + t1565) - t1571 - t1572
               - t176 * t45 - t29 * (-t1529 - t1530 + t1566 + t176 * t460 + t36)
               + t37 + t603 * (t1527 + t1570));
        hess[96] = t510
            * (t1 * (t1573 + t533 * t76 + t854) + t168 * t94
               + t38 * (-t157 * t189 + 2 * t169 - t527 + t534 + t68)
               - t469 * (t1574 + t554) + t47 * t542 * t701 - t517 - t543
               + t545 * t604 - t546 - t547 - t565 + t93);
        hess[97] = t510
            * (-t3 * (t1304 + t1575 + t730)
               + t5 * t55
                   * (-t1164 * t621 + 2 * t124 * t48 * t55 * t7 - t48 * t819
                      + t49 * t533 - t71)
               - t690 * (t1574 + t732) - t733);
        hess[98] = t510
            * (t1 * t10 * t7 * t87 + t10 * t180 * t7
               + t10 * t7
                   * (2 * t10 * t3 * t68 * t8 - t1153 * t726 - t166 * t519
                      - t167 + t3 * t51 * t525 * t55 * t7)
               + t125 * t127 * t184 * t472 * t545
               + 3 * t125 * t184 * t507 * t542 * t55
               - t138
                   * (t175 * t519 - t175 + t178 * t518 + t185 * t522
                      - t514 * t812)
               - t1433 + t3 * (t1577 + t172 * t544 + t400 * t817)
               - t468
                   * (t1437 + t1528 + t1566 + t1578 + t1579 - t177 * t548
                      - t183 * t518 + t553 * t7)
               - t542 * t802 - t545 * t800 - t564 * t799 - t7 * t803);
        hess[99] = t1086;
        hess[100] = t1283;
        hess[101] = t1402;
        hess[102] = t1499;
        hess[103] = t1554;
        hess[104] = t285
            * (t1015 * std::pow(t825, 2) + 2 * t1400 + t1401
               + t1466 * std::pow(t827, 2) + t1467 * t743 * t827
               + t1543 * t824 * t827
               - t275
                   * (t1461 * t1586 + t1541 * t1586 + t1588 * t233 * t55
                      + t1589 * t50 + t1589 * t52 + std::pow(t533, 2))
               + t281 * t518 * t826
               - t441
                   * (t1081 * t1580 + t1278 * t1580 + t1587 * t830
                      + t1590 * t187 + t1590 * t293 + std::pow(t541, 2) * t55)
               + 2 * t824 * t826 + t827 * t831 * t859
               + t859
                   * (t1
                          * (t121 * t1588 + t122 * t1580 * t49 * t55
                             - t124 * t1587 - t1276 * t525 - t1586 * t330
                             + t49 * t533 * t537)
                      - t3
                          * (t110 * t1585 - t1345 * t1583 + t1581 * t561
                             - t1584 * t68 + t537 * t823 - t541 * t544)
                      - t560
                      - t7
                          * (t117 * t1583 * t51 - t1581 * t398
                             + t1581 * t49 * t55 * t68 - t1583 * t390)));
        hess[105] = t510
            * (-t138 * (t137 + t1441 - t1578 - t1579 + t39 * t548) + t1599
               + t38 * (-t1080 * t170 + t1404 + t1591)
               + t603 * (t1128 - t1330 + t1592));
        hess[106] = t510
            * (t125 * t472 * (t1577 + t1600)
               - t141
                   * (-t117 * t519 + t117 - t170 * t538 + 2 * t246 + t541 * t74)
               - t1601
               + t3 * t55
                   * (-t1295 * t7 - t1497 * t48
                      + 2 * t214 * t48 * t51 * t55 * t7 + t214 * t51
                      - t541 * t580));
        hess[107] = t510
            * (t1431 + t149 * (2 * t123 * t48 * t49 * t55 * t7 - t1302 - t1602)
               + t1593 * t594 + t1594 * t602 + t1595 * t602 - t1596 * t594
               + t1597 * t594 - t1598 * t594 - t168 * t602
               - t168 * (-t1080 * t1090 + t1153 * t538 - t1425 + t1426 + t595)
               - t3
                   * (t1073 * t592 + t1157 * t746 + t1328 - t1591 * t168 - t773)
               + t603 * (t1573 + t1575 + t533 * t592 + t707 * t746 + t792));
        hess[108] = t577;
        hess[109] = t736;
        hess[110] = t834;
        hess[111] = t285
            * (-t1 * (t1089 - t1603 * t617 + t572 * t621) - t1106
               - t276
                   * (t153 * t51 * t618 + t201 * t567 + 2 * t339 * t567
                      + t48 * t619 + t49 * t945 + t49 * t984)
               + t55 * t7 * (t1003 * t49 - t1093 - t257 * t618 + t575 * t757)
               - t74
                   * (t1174 * t617 - t1345 * t617 - t203 * t624 + t783 * t849));
        hess[112] = t285
            * (t1 * t10 * t226 * t268 * t570
               + 3 * t1014 * t226 * t340 * t55 * t570 - t1287 - t1290 - t1291
               - t1293 - t1297 + t226 * t228 * t268 * t326 * t570
               + t226 * t268 * (-t1289 - t295 * t714)
               + t3 * t49 * t55 * (t123 * t299 + t1296) + t571 - t572
               - t7 * (t1154 + t295 * t940 - t307 * t566 + t846));
        hess[113] = t285
            * (t1 * (t1284 * t217 + t324 * t355 + t366 * t832 - t380 * t567)
               + t110 * t566 + t1100 * t1279 + t1100 * t1368 + t1403 * t170
               - t1405 - t1406 - t1407 - t1410 + t1422 * t570
               + t275 * (-t1331 - t1408 - t351 * t816)
               - t3 * (-t29 * t669 + t318 + t333 * t351 + t373 * t779) - t638);
        hess[114] = t285
            * (-t1009 * t570 - t1016 * t570 + t1097 * t228 * t437 * t55
               + t1097 * t268 * t435 + t1098 - t1101 - t1103 - t1430
               - t149 * (-t196 * t871 + t310 * t847 - t429 * t783 + t566 * t695)
               - t1606 * t437 * t570
               - t275
                   * (-t1462 * t310 - t1604 * t196 - t1605 * t196
                      + t204 * t405 * t49 * t52 + t204 * t405 * t49 * t53
                      - t411 * t567)
               + t55 * t7 * (t121 * t196 * t51 - t1500)
               - t74
                   * (t1002 * t783 + t1174 * t196 - t1345 * t196
                      - t203 * t695));
        hess[115] = t510
            * (t125 * t472 * (t1556 + t449 * t714) - t1560
               + t3 * t49 * t55
                   * (-t123 * t449 - t1480 + t305 * t481 + t335 * t668)
               - t7 * (t1109 + t1143 * t477 - t1607 * t499 + t449 * t940));
        hess[116] = t510
            * (t1 * (-t1607 * t541 + t529 * t940 + t537 * t774 - t848) + t1599
               + t3
                   * (t1 * t10 * t3 * t541 * t55 - t1080 * t779 - t1564
                      - t333 * t529)
               + t603 * (t1592 + t529 * t816));
        hess[117] = t1609
            * (t125 * (std::pow(t566, 2) + t99) - t1608 * std::pow(t570, 2)
               + 2 * t570 * t576);
        hess[118] = t1611;
        hess[119] = t1613;
        hess[120] = t591;
        hess[121] = t737;
        hess[122] = t835;
        hess[123] = t285
            * (-t1 * (t1091 * t138 - t138 * t744 + t617 * t863 + t671)
               + t1011 * t1308 + t1012 * t1308 - t1114 - t1115 - t1116
               + t1118 * t74 - t1122 + t1141 * t583 + t1614
               + t275 * (-t1121 - t398 * t617 - t841)
               + t7 * (t1123 * t205 + t209 * t578 + t216 * t618 - t225 * t580));
        hess[124] = t285
            * (t1 * t55 * (t121 * t299 * t48 - t299 * t330)
               + 3 * t1014 * t226 * t340 * t55 * t583 - t1103 - t1305 - t1306
               - t1307 - t1309 + t1311 - t1615
               + t226 * t228 * t268 * t326 * t583
               - t276
                   * (t1210 * t51 + t1239 * t51 + t1261 * t299 + t296 * t580
                      + t300 * t49 + t51 * t580 * t786)
               + t3 * t55
                   * (2 * t1112 * t51 + t1251 * t51 - t1301 - t299 * t335)
               - t7 * (t1300 - t295 * t333 + t588 * t771));
        hess[125] = t285
            * (t1 * t51 * t55 * (t121 * t355 + t1420)
               + t10 * t226 * t268 * t583 * t7
               + 3 * t1014 * t226 * t395 * t55 * t583 - t1412 - t1415 - t1416
               - t1417 - t1421 + t226 * t228 * t268 * t381 * t583
               + t226 * t268 * (-t1414 - t360 * t383) + t244
               - t3 * (t1159 + t1333 + t351 * t883 - t369 * t579) - t588);
        hess[126] = t510
            * (-t1 * (t138 * t1503 + t1616 + t196 * t863)
               + t10 * t125 * t3 * t472 * t583 + t10 * t3 * t590
               + t125 * t472 * (t1515 + t198 * t55 + t777) - t1510 - t1511
               - t1512 - t1516 - t1614
               + t3 * t55
                   * (-t1295 * t3 + 2 * t214 * t3 * t48 * t51 * t55
                      - t293 * t695 - t580 * t695)
               + t433 * t472 * t583
               - t7
                   * (-t1155 - t138 * t1520 - t196 * t883 + t429 * t55 * t579));
        hess[127] = t285
            * (t1 * t55 * (t330 * t449 - t331 * t449) - t1050 * t583
               - t1052 * t583 + t1113 * t228 * t55 * t722 + t1113 * t268 * t720
               - t1312 - t1606 * t583 * t722
               - t170 * (t1617 * t895 - t203 * t499 - t390 * t449 + t478 * t579)
               - t275
                   * (-t1605 * t449 + t204 * t445 * t50 * t51
                      + t204 * t445 * t51 * t53 - t449 * t719 - t452 * t580
                      - t629 * t786)
               + t3 * t55
                   * (-t1617 * t830 + t214 * t449 * t48 - t305 * t48 * t499
                      - t481 * t580));
        hess[128] = t510
            * (t1 * t51 * t55
                   * (t1079 * t48 - t121 * t529 - t1276 + t330 * t743)
               + t125 * t472 * (t1576 + t1600 + t529 * t561) - t1601
               - t3 * (-t1119 * t541 + t1160 + t529 * t883 + t537 * t634));
        hess[129] = t1611;
        hess[130] = t1609
            * (t125 * (std::pow(t579, 2) + t607) - t1608 * std::pow(t583, 2)
               - 2 * t583 * t590);
        hess[131] = t1618;
        hess[132] = t605;
        hess[133] = t739;
        hess[134] = t836;
        hess[135] = t285
            * (-t1 * (t1091 * t168 + t1335 - t224 * t592 + t617 * t884)
               + t10 * t226 * t268 * t3 * t594
               + 3 * t1014 * t226 * t266 * t55 * t594 - t1125 - t1133 - t1134
               - t1135 - t1139 - t1521 + t226 * t228 * t236 * t268 * t594
               + t226 * t268 * (-t1131 - t251 * t617)
               + t48 * t55 * t7 * (t1138 + t214 * t618));
        hess[136] = t285
            * (t1140 * t1241 + t1140 * t1245 + t1298 * t594 + t1315 * t149
               - t1318 - t1320 - t1321 - t1324 + t1619
               + t275 * (-t1149 - t1323 - t262 * t302)
               + t3
                   * (2 * t211 * t215 + t218 * t299 + t312 * t738 - t325 * t600)
               - t7 * (t1316 * t168 + t1603 * t295 - t168 * t308 + t372));
        hess[137] = t285
            * (t1 * t55 * (t1397 * t48 - t1427 - t330 * t355 + 2 * t48 * t601)
               - t1432
               - t170
                   * (t1620 * t351 - t351 * t390 + t366 * t49 * t650
                      - t373 * t783)
               - t276
                   * (t1326 * t600 + t1366 * t48 + t1389 * t48 + t339 * t355
                      + t352 * t600 + t356 * t51)
               - t3 * (t1424 - t351 * t863 + t358 * t597));
        hess[138] = t510
            * (t1 * (-t1336 - t1503 * t168 - t196 * t884 + t429 * t55 * t592)
               - t140 + t1517 * t3 + t1522 + t176 * t27
               + t603 * (t1434 + t1518 + t196 * t251)
               + t7 * (t110 * t196 * t250 + t1520 * t168 + t1616));
        hess[139] = t510
            * (t1 * t10 * t125 * t472 * t594
               + t1 * t55
                   * (2 * t1 * t123 * t48 * t49 * t55 - t1 * t1419 - t349 * t478
                      - t478 * t600)
               + t125 * t472 * (t1570 + t55 * t649) + t127 * t503 * t55 * t602
               - t1567 - t1568 - t1571 - t1572 - t1619
               - t3 * (-t1334 - t1555 * t168 - t449 * t884 + t499 * t55 * t592)
               + t471 * t472 * t602
               - t7 * (t1508 + t1561 * t168 - t1565 * t168 + t1603 * t449));
        hess[140] = t285
            * (t1 * t55 * (t123 * t49 * t529 - t1602) - t1082 * t594
               - t1083 * t594 - t1102 + t1132 * t228 * t55 * t827
               + t1132 * t268 * t825 + t1428 - t1429 - t1606 * t594 * t827
               - t1615
               - t170
                   * (-t1080 * t783 + t1620 * t529 - t390 * t529 + t538 * t650)
               - t275
                   * (-t1604 * t529 + t204 * t48 * t50 * t525
                      + t204 * t48 * t52 * t525 - t233 * t747 - t529 * t719
                      - t533 * t600)
               - t74
                   * (t1080 * t592 - t1345 * t529 - t541 * t650 + t746 * t892));
        hess[141] = t1613;
        hess[142] = t1618;
        hess[143] = t1609
            * (t125 * (std::pow(t592, 2) + t740) - t1608 * std::pow(t594, 2)
               + 2 * t594 * t602);
    }


    void similarity_gradient(
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
        double c0x,
        double c0y,
        double c0z,
        double c1x,
        double c1y,
        double c1z,
        double grad[12])
    {
        const auto t0 = c0x - c1y;
        const auto t1 = c0y - c1x + t0;
        const auto t2 = p0y - p1y;
        const auto t3 = -p3z;
        const auto t4 = p0z + t3;
        const auto t5 = -p3y;
        const auto t6 = p0y + t5;
        const auto t7 = p0z - p1z;
        const auto t8 = -p2z;
        const auto t9 = p0z + t8;
        const auto t10 = -p2y;
        const auto t11 = p0y + t10;
        const auto t12 = -p3x;
        const auto t13 = p0x + t12;
        const auto t14 = p0x - p1x;
        const auto t15 = -p2x;
        const auto t16 = p0x + t15;
        const auto t17 = -c0x * t14 - c0y * t16 + c0z * (-t11 * t7 + t2 * t9)
            + c1x * t13 + c1y * t14;
        const auto t18 = c1z * (t2 * t4 - t6 * t7) + t17;
        const auto t19 = c0z * (p1y + t10) + c1z * (p1y + t5);
        const auto t20 = -c0x * t7 - c0y * t9 + c0z * (t11 * t14 - t16 * t2)
            + c1x * t4 + c1y * t7;
        const auto t21 = c1z * (-t13 * t2 + t14 * t6) + t20;
        const auto t22 = c0z * (p1z + t8) + c1z * (p1z + t3);
        const auto t23 = -t14;
        const auto t24 = -t7;
        const auto t25 = c0x * t2 + c0y * t11 + c0z * (t16 * t24 - t23 * t9)
            - c1x * t6 - c1y * t2 + c1z * (-t13 * t7 + t14 * t4);
        const auto t26 = c0z * (p1x + t15) + c1z * (p1x + t12);
        const auto t27 = c0z * t11 + c1z * t6;
        const auto t28 = c0z * t9 + c1z * t4;
        const auto t29 = c0z * t16 + c1z * t13;
        const auto t30 = c0z * t21;
        const auto t31 = -t2;
        const auto t32 = -t6;
        const auto t33 = c1z * (-t24 * t32 - t31 * t4) + t17;
        const auto t34 = c1z * (t13 * t31 + t23 * t32) + t20;
        const auto t35 = -t25;
        grad[0] = -2 * t1 * t18 + 2 * t19 * t21 + 2 * t22 * t25;
        grad[1] = 2 * t1 * t25 + 2 * t18 * t22 - 2 * t21 * t26;
        grad[2] = -2 * t1 * t21 - 2 * t18 * t19 - 2 * t25 * t26;
        grad[3] = 2 * t0 * t18 - 2 * t21 * t27 - 2 * t25 * t28;
        grad[4] = -2 * t0 * t25 - 2 * t18 * t28 + 2 * t21 * t29;
        grad[5] = 2 * t0 * t21 + 2 * t18 * t27 + 2 * t25 * t29;
        grad[6] = 2 * c0y * t18 + 2 * c0z * t25 * t7 + 2 * t2 * t30;
        grad[7] = -2 * c0y * t25 + 2 * c0z * t18 * t7 - 2 * t14 * t30;
        grad[8] = 2 * c0y * t21 - 2 * c0z * t14 * t25 - 2 * c0z * t18 * t2;
        grad[9] = -2 * c1x * t33 + 2 * c1z * t2 * t34 - 2 * c1z * t35 * t7;
        grad[10] = -2 * c1x * t35 - 2 * c1z * t14 * t34 + 2 * c1z * t33 * t7;
        grad[11] = -2 * c1x * t34 + 2 * c1z * t14 * t35 - 2 * c1z * t2 * t33;
    }

    // hess is (144×1) flattened in column-major order
    void similarity_hessian(
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
        double c0x,
        double c0y,
        double c0z,
        double c1x,
        double c1y,
        double c1z,
        double hess[144])
    {
        const auto t0 = -p2y;
        const auto t1 = -p3y;
        const auto t2 = c0z * (p1y + t0) + c1z * (p1y + t1);
        const auto t3 = std::pow(t2, 2);
        const auto t4 = -p2z;
        const auto t5 = -p3z;
        const auto t6 = c0z * (p1z + t4) + c1z * (p1z + t5);
        const auto t7 = c0x - c1y;
        const auto t8 = c0y - c1x + t7;
        const auto t9 = std::pow(t8, 2);
        const auto t10 = std::pow(t6, 2) + t9;
        const auto t11 = -p2x;
        const auto t12 = -p3x;
        const auto t13 = c0z * (p1x + t11) + c1z * (p1x + t12);
        const auto t14 = 2 * t13;
        const auto t15 = -t14 * t2;
        const auto t16 = -t14 * t6;
        const auto t17 = p0y + t0;
        const auto t18 = p0y + t1;
        const auto t19 = c0z * t17 + c1z * t18;
        const auto t20 = t19 * t2;
        const auto t21 = t7 * t8;
        const auto t22 = p0z + t4;
        const auto t23 = p0z + t5;
        const auto t24 = c0z * t22 + c1z * t23;
        const auto t25 = t21 + t24 * t6;
        const auto t26 = -2 * t20 - 2 * t25;
        const auto t27 = t6 * t7;
        const auto t28 = t24 * t8;
        const auto t29 = p0x + t11;
        const auto t30 = p0x + t12;
        const auto t31 = c0z * t29 + c1z * t30;
        const auto t32 = c0z + c1z;
        const auto t33 = p0x - p1x;
        const auto t34 = p0y - p1y;
        const auto t35 = p0z - p1z;
        const auto t36 = -c0x * t35 - c0y * t22 + c0z * (t17 * t33 - t29 * t34)
            + c1x * t23 + c1y * t35;
        const auto t37 = c1z * (t18 * t33 - t30 * t34) + t36;
        const auto t38 = t32 * t37;
        const auto t39 = 2 * t2 * t31 - 2 * t27 + 2 * t28 + 2 * t38;
        const auto t40 = -t33;
        const auto t41 = -t35;
        const auto t42 = c0x * t34 + c0y * t17 + c0z * (-t22 * t40 + t29 * t41)
            - c1x * t18 - c1y * t34 + c1z * (t23 * t33 - t30 * t35);
        const auto t43 = -t19 * t8 + t2 * t7 + t32 * t42;
        const auto t44 = 2 * t31 * t6 + 2 * t43;
        const auto t45 = c0z * t2;
        const auto t46 = t34 * t45;
        const auto t47 = -c0y * t8;
        const auto t48 = c0z * t35;
        const auto t49 = t47 + t48 * t6;
        const auto t50 = 2 * t46 + 2 * t49;
        const auto t51 = -t18;
        const auto t52 = -t34;
        const auto t53 = c1z * (t30 * t52 + t40 * t51) + t36;
        const auto t54 = c0y * t6 + t48 * t8;
        const auto t55 = -2 * c0z * t53 - 2 * t33 * t45 - 2 * t54;
        const auto t56 = -c0z * t42;
        const auto t57 = t33 * t6;
        const auto t58 = t34 * t8;
        const auto t59 = c0y * t2 + c0z * t58;
        const auto t60 = -2 * c0z * t57 + 2 * t56 + 2 * t59;
        const auto t61 = c1z * t2;
        const auto t62 = t34 * t61;
        const auto t63 = c1x * t8;
        const auto t64 = c1z * t35;
        const auto t65 = t6 * t64 + t63;
        const auto t66 = 2 * t62 + 2 * t65;
        const auto t67 = t64 * t8;
        const auto t68 = c1x * t6;
        const auto t69 = c1z * t53;
        const auto t70 = -2 * t33 * t61 - 2 * t67 + 2 * t68 - 2 * t69;
        const auto t71 = c1x * t2;
        const auto t72 = c1z * t58;
        const auto t73 = -t42;
        const auto t74 = c1z * t73;
        const auto t75 = -t74;
        const auto t76 = -2 * c1z * t57 - 2 * t71 + 2 * t72 - 2 * t75;
        const auto t77 = std::pow(t13, 2);
        const auto t78 = -2 * t2 * t6;
        const auto t79 = 2 * t13 * t19 + 2 * t27 - 2 * t28 - 2 * t38;
        const auto t80 = t13 * t31;
        const auto t81 = -2 * t25 - 2 * t80;
        const auto t82 = t13 * t7;
        const auto t83 = t31 * t8;
        const auto t84 = -c0x * t33 - c0y * t29 + c0z * (-t17 * t35 + t22 * t34)
            + c1x * t30 + c1y * t33;
        const auto t85 = c1z * (-t18 * t35 + t23 * t34) + t84;
        const auto t86 = t32 * t85;
        const auto t87 = 2 * t19 * t6 - 2 * t82 + 2 * t83 + 2 * t86;
        const auto t88 = c0z * t37;
        const auto t89 = c0z * t13;
        const auto t90 = -2 * t34 * t89 + 2 * t54 + 2 * t88;
        const auto t91 = t33 * t89;
        const auto t92 = 2 * t49 + 2 * t91;
        const auto t93 = c1z * (-t23 * t52 - t41 * t51) + t84;
        const auto t94 = t34 * t6;
        const auto t95 = t33 * t8;
        const auto t96 = c0y * t13 + c0z * t95;
        const auto t97 = -2 * c0z * t93 - 2 * c0z * t94 - 2 * t96;
        const auto t98 = c1z * t13;
        const auto t99 = -t69;
        const auto t100 = -2 * t34 * t98 + 2 * t67 - 2 * t68 - 2 * t99;
        const auto t101 = t33 * t98;
        const auto t102 = 2 * t101 + 2 * t65;
        const auto t103 = c1z * t95;
        const auto t104 = c1x * t13;
        const auto t105 = c1z * t93;
        const auto t106 = -2 * c1z * t94 - 2 * t103 + 2 * t104 - 2 * t105;
        const auto t107 = 2 * t13 * t24 - 2 * t43;
        const auto t108 = 2 * t2 * t24 + 2 * t82 - 2 * t83 - 2 * t86;
        const auto t109 = -2 * t20 - 2 * t21 - 2 * t80;
        const auto t110 = -2 * c0z * t73 - 2 * t13 * t48 - 2 * t59;
        const auto t111 = c0z * t85;
        const auto t112 = 2 * t111 - 2 * t35 * t45 + 2 * t96;
        const auto t113 = 2 * t46 + 2 * t47 + 2 * t91;
        const auto t114 = -2 * t13 * t64 + 2 * t71 - 2 * t72 - 2 * t74;
        const auto t115 = -t105;
        const auto t116 = 2 * t103 - 2 * t104 - 2 * t115 - 2 * t35 * t61;
        const auto t117 = 2 * t101 + 2 * t62 + 2 * t63;
        const auto t118 = std::pow(t19, 2);
        const auto t119 = std::pow(t7, 2);
        const auto t120 = t119 + std::pow(t24, 2);
        const auto t121 = 2 * t31;
        const auto t122 = -t121 * t19;
        const auto t123 = -t121 * t24;
        const auto t124 = c0z * t19;
        const auto t125 = t124 * t34;
        const auto t126 = -c0y * t7;
        const auto t127 = t126 + t24 * t48;
        const auto t128 = -2 * t125 - 2 * t127;
        const auto t129 = c0y * t24 + t48 * t7 + t88;
        const auto t130 = 2 * t124 * t33 + 2 * t129;
        const auto t131 = t34 * t7;
        const auto t132 = c0y * t19 + c0z * t131 + t56;
        const auto t133 = 2 * c0z * t24 * t33 - 2 * t132;
        const auto t134 = c1z * t19;
        const auto t135 = t134 * t34;
        const auto t136 = c1x * t7;
        const auto t137 = t136 + t24 * t64;
        const auto t138 = -2 * t135 - 2 * t137;
        const auto t139 = t64 * t7;
        const auto t140 = c1x * t24;
        const auto t141 = 2 * t134 * t33 + 2 * t139 - 2 * t140 + 2 * t69;
        const auto t142 = c1z * t131;
        const auto t143 = c1x * t19;
        const auto t144 = 2 * c1z * t24 * t33 - 2 * t142 + 2 * t143 + 2 * t75;
        const auto t145 = std::pow(t31, 2);
        const auto t146 = -2 * t19 * t24;
        const auto t147 = 2 * c0z * t31 * t34 - 2 * t129;
        const auto t148 = c0z * t31 * t33;
        const auto t149 = -2 * t127 - 2 * t148;
        const auto t150 = t24 * t34;
        const auto t151 = t33 * t7;
        const auto t152 = c0y * t31 + c0z * t151 + t111;
        const auto t153 = 2 * c0z * t150 + 2 * t152;
        const auto t154 = c1z * t31;
        const auto t155 = -2 * t139 + 2 * t140 + 2 * t154 * t34 + 2 * t99;
        const auto t156 = t154 * t33;
        const auto t157 = -2 * t137 - 2 * t156;
        const auto t158 = c1z * t151;
        const auto t159 = c1x * t31;
        const auto t160 = 2 * c1z * t150 + 2 * t105 + 2 * t158 - 2 * t159;
        const auto t161 = 2 * t132 + 2 * t31 * t48;
        const auto t162 = 2 * c0z * t19 * t35 - 2 * t152;
        const auto t163 = -2 * t125 - 2 * t126 - 2 * t148;
        const auto t164 = 2 * t142 - 2 * t143 + 2 * t31 * t64 + 2 * t74;
        const auto t165 = 2 * t115 - 2 * t158 + 2 * t159 + 2 * t19 * t64;
        const auto t166 = -2 * t135 - 2 * t136 - 2 * t156;
        const auto t167 = std::pow(c0z, 2);
        const auto t168 = std::pow(t34, 2);
        const auto t169 = t167 * t168;
        const auto t170 = std::pow(c0y, 2);
        const auto t171 = std::pow(t35, 2);
        const auto t172 = t167 * t171 + t170;
        const auto t173 = t33 * t34;
        const auto t174 = 2 * t167;
        const auto t175 = -t173 * t174;
        const auto t176 = t174 * t35;
        const auto t177 = -t176 * t33;
        const auto t178 = c0z * c1z;
        const auto t179 = t168 * t178;
        const auto t180 = -c0y * c1x;
        const auto t181 = t171 * t178 + t180;
        const auto t182 = 2 * t179 + 2 * t181;
        const auto t183 = t173 * t178;
        const auto t184 = c0y * t64 + c1x * t48;
        const auto t185 = -2 * t183 + 2 * t184;
        const auto t186 = c1z * t48;
        const auto t187 = t186 * t33;
        const auto t188 = c0y * c1z;
        const auto t189 = c0z * c1x;
        const auto t190 = t188 * t34 + t189 * t34;
        const auto t191 = -2 * t187 - 2 * t190;
        const auto t192 = std::pow(t33, 2);
        const auto t193 = t167 * t192;
        const auto t194 = -t176 * t34;
        const auto t195 = -2 * t183 - 2 * t184;
        const auto t196 = t178 * t192;
        const auto t197 = 2 * t181 + 2 * t196;
        const auto t198 = t186 * t34;
        const auto t199 = t188 * t33 + t189 * t33;
        const auto t200 = -2 * t198 + 2 * t199;
        const auto t201 = -2 * t187 + 2 * t190;
        const auto t202 = -2 * t198 - 2 * t199;
        const auto t203 = 2 * t179 + 2 * t180 + 2 * t196;
        const auto t204 = std::pow(c1z, 2);
        const auto t205 = t168 * t204;
        const auto t206 = std::pow(c1x, 2);
        const auto t207 = t171 * t204 + t206;
        const auto t208 = 2 * t204;
        const auto t209 = -t173 * t208;
        const auto t210 = t208 * t35;
        const auto t211 = -t210 * t33;
        const auto t212 = t192 * t204;
        const auto t213 = -t210 * t34;
        hess[0] = 2 * t10 + 2 * t3;
        hess[1] = t15;
        hess[2] = t16;
        hess[3] = t26;
        hess[4] = t39;
        hess[5] = t44;
        hess[6] = t50;
        hess[7] = t55;
        hess[8] = t60;
        hess[9] = t66;
        hess[10] = t70;
        hess[11] = t76;
        hess[12] = t15;
        hess[13] = 2 * t10 + 2 * t77;
        hess[14] = t78;
        hess[15] = t79;
        hess[16] = t81;
        hess[17] = t87;
        hess[18] = t90;
        hess[19] = t92;
        hess[20] = t97;
        hess[21] = t100;
        hess[22] = t102;
        hess[23] = t106;
        hess[24] = t16;
        hess[25] = t78;
        hess[26] = 2 * t3 + 2 * t77 + 2 * t9;
        hess[27] = t107;
        hess[28] = t108;
        hess[29] = t109;
        hess[30] = t110;
        hess[31] = t112;
        hess[32] = t113;
        hess[33] = t114;
        hess[34] = t116;
        hess[35] = t117;
        hess[36] = t26;
        hess[37] = t79;
        hess[38] = t107;
        hess[39] = 2 * t118 + 2 * t120;
        hess[40] = t122;
        hess[41] = t123;
        hess[42] = t128;
        hess[43] = t130;
        hess[44] = t133;
        hess[45] = t138;
        hess[46] = t141;
        hess[47] = t144;
        hess[48] = t39;
        hess[49] = t81;
        hess[50] = t108;
        hess[51] = t122;
        hess[52] = 2 * t120 + 2 * t145;
        hess[53] = t146;
        hess[54] = t147;
        hess[55] = t149;
        hess[56] = t153;
        hess[57] = t155;
        hess[58] = t157;
        hess[59] = t160;
        hess[60] = t44;
        hess[61] = t87;
        hess[62] = t109;
        hess[63] = t123;
        hess[64] = t146;
        hess[65] = 2 * t118 + 2 * t119 + 2 * t145;
        hess[66] = t161;
        hess[67] = t162;
        hess[68] = t163;
        hess[69] = t164;
        hess[70] = t165;
        hess[71] = t166;
        hess[72] = t50;
        hess[73] = t90;
        hess[74] = t110;
        hess[75] = t128;
        hess[76] = t147;
        hess[77] = t161;
        hess[78] = 2 * t169 + 2 * t172;
        hess[79] = t175;
        hess[80] = t177;
        hess[81] = t182;
        hess[82] = t185;
        hess[83] = t191;
        hess[84] = t55;
        hess[85] = t92;
        hess[86] = t112;
        hess[87] = t130;
        hess[88] = t149;
        hess[89] = t162;
        hess[90] = t175;
        hess[91] = 2 * t172 + 2 * t193;
        hess[92] = t194;
        hess[93] = t195;
        hess[94] = t197;
        hess[95] = t200;
        hess[96] = t60;
        hess[97] = t97;
        hess[98] = t113;
        hess[99] = t133;
        hess[100] = t153;
        hess[101] = t163;
        hess[102] = t177;
        hess[103] = t194;
        hess[104] = 2 * t169 + 2 * t170 + 2 * t193;
        hess[105] = t201;
        hess[106] = t202;
        hess[107] = t203;
        hess[108] = t66;
        hess[109] = t100;
        hess[110] = t114;
        hess[111] = t138;
        hess[112] = t155;
        hess[113] = t164;
        hess[114] = t182;
        hess[115] = t195;
        hess[116] = t201;
        hess[117] = 2 * t205 + 2 * t207;
        hess[118] = t209;
        hess[119] = t211;
        hess[120] = t70;
        hess[121] = t102;
        hess[122] = t116;
        hess[123] = t141;
        hess[124] = t157;
        hess[125] = t165;
        hess[126] = t185;
        hess[127] = t197;
        hess[128] = t202;
        hess[129] = t209;
        hess[130] = 2 * t207 + 2 * t212;
        hess[131] = t213;
        hess[132] = t76;
        hess[133] = t106;
        hess[134] = t117;
        hess[135] = t144;
        hess[136] = t160;
        hess[137] = t166;
        hess[138] = t191;
        hess[139] = t200;
        hess[140] = t203;
        hess[141] = t211;
        hess[142] = t213;
        hess[143] = 2 * t205 + 2 * t206 + 2 * t212;
    }

}
