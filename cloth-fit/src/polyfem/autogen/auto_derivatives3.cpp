#include "auto_derivatives.hpp"
#include <cmath>

namespace polyfem::autogen {

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
        double grad[12])
    {
        const auto t0 = -p2x;
        const auto t1 = p1x + t0;
        const auto t2 = -t1;
        const auto t3 = std::pow(t2, 2);
        const auto t4 = -p2y;
        const auto t5 = p1y + t4;
        const auto t6 = -t5;
        const auto t7 = std::pow(t6, 2);
        const auto t8 = -p2z;
        const auto t9 = p1z + t8;
        const auto t10 = -t9;
        const auto t11 = std::pow(t10, 2);
        const auto t12 = t11 + t3 + t7;
        const auto t13 = 1.0 / t12;
        const auto t14 = t13 * t2;
        const auto t15 = p2x - p3x;
        const auto t16 = p2y - p3y;
        const auto t17 = p2z - p3z;
        const auto t18 = -t10 * t17 - t15 * t2 - t16 * t6;
        const auto t19 = t13 * t18;
        const auto t20 = t19 * t2;
        const auto t21 = t15 + t20;
        const auto t22 = std::pow(t1, 2);
        const auto t23 = std::pow(t5, 2);
        const auto t24 = std::pow(t9, 2);
        const auto t25 = 1.0 / (t22 + t23 + t24);
        const auto t26 = p0x - p1x;
        const auto t27 = p0y - p1y;
        const auto t28 = p0z - p1z;
        const auto t29 = t25 * (t1 * t26 + t27 * t5 + t28 * t9);
        const auto t30 = -p0x + p1x;
        const auto t31 = -t1 * t29 - t30;
        const auto t32 = -p0y + p1y;
        const auto t33 = -t29 * t5 - t32;
        const auto t34 = -p0z + p1z;
        const auto t35 = -t29 * t9 - t34;
        const auto t36 = std::pow(t31, 2) + std::pow(t33, 2) + std::pow(t35, 2);
        const auto t37 = 1.0 / t36;
        const auto t38 = t1 * t25;
        const auto t39 = t38 * t5;
        const auto t40 = t38 * t9;
        const auto t41 = t25 * (t1 * t15 + t16 * t5 + t17 * t9);
        const auto t42 = -p3x - t0 - t1 * t41;
        const auto t43 = -p3y - t4 - t41 * t5;
        const auto t44 = -p3z - t41 * t9 - t8;
        const auto t45 = t31 * t42 + t33 * t43 + t35 * t44;
        const auto t46 = t19 * t6;
        const auto t47 = t16 + t46;
        const auto t48 = t10 * t19;
        const auto t49 = t17 + t48;
        const auto t50 = t39 * t47 + t40 * t49;
        const auto t51 = std::pow(t42, 2) + std::pow(t43, 2) + std::pow(t44, 2);
        const auto t52 = 1 / (std::sqrt(t36) * std::sqrt(t51));
        const auto t53 = t13 * t6;
        const auto t54 = t25 * t5 * t9;
        const auto t55 = t21 * t39 + t49 * t54;
        const auto t56 = t10 * t13;
        const auto t57 = t21 * t40 + t47 * t54;
        const auto t58 = t15 + 2 * t20;
        const auto t59 = -t2 * t26;
        const auto t60 = -t27 * t6;
        const auto t61 = -t10 * t28;
        const auto t62 = t60 + t61;
        const auto t63 = t59 + t62;
        const auto t64 = t13 * t63;
        const auto t65 = t6 * t64;
        const auto t66 = t27 + t65;
        const auto t67 = t53 * t66;
        const auto t68 = t10 * t64;
        const auto t69 = t28 + t68;
        const auto t70 = t56 * t69;
        const auto t71 = t2 * t64;
        const auto t72 = 2 * t71;
        const auto t73 = p0x - 2 * p1x + p2x;
        const auto t74 = t72 + t73;
        const auto t75 = t53 * t74;
        const auto t76 = t56 * t74;
        const auto t77 = t26 + t71;
        const auto t78 = -2 * t13 * t18 * t3 - t15 * t2 + t18;
        const auto t79 = std::pow(t12, -2);
        const auto t80 = t63 * t79;
        const auto t81 = -t64 - 1;
        const auto t82 = t14 * t73 + 2 * t3 * t80 + t81;
        const auto t83 = -t47;
        const auto t84 = t6 * t83;
        const auto t85 = -t49;
        const auto t86 = t10 * t85;
        const auto t87 = -t21;
        const auto t88 = t21 * t77 + t47 * t66 + t49 * t69;
        const auto t89 = std::pow(t21, 2) + std::pow(t47, 2) + std::pow(t49, 2);
        const auto t90 = t88 / t89;
        const auto t91 = t13 * t90;
        const auto t92 = -t66;
        const auto t93 = -t69;
        const auto t94 = -t77;
        const auto t95 = std::pow(t66, 2) + std::pow(t69, 2) + std::pow(t77, 2);
        const auto t96 = t88 / t95;
        const auto t97 = 1 / (std::sqrt(t89) * std::sqrt(t95));
        const auto t98 = t16 + 2 * t46;
        const auto t99 = t14 * t77;
        const auto t100 = 2 * t65;
        const auto t101 = p0y - 2 * p1y + p2y;
        const auto t102 = t100 + t101;
        const auto t103 = t102 * t14;
        const auto t104 = t102 * t56;
        const auto t105 = -2 * t13 * t18 * t7 - t16 * t6 + t18;
        const auto t106 = t101 * t53 + 2 * t7 * t80 + t81;
        const auto t107 = t2 * t87;
        const auto t108 = t17 + 2 * t48;
        const auto t109 = 2 * t68;
        const auto t110 = p0z - 2 * p1z + p2z;
        const auto t111 = t109 + t110;
        const auto t112 = t111 * t14;
        const auto t113 = t111 * t53;
        const auto t114 = -t10 * t17 - 2 * t11 * t13 * t18 + t18;
        const auto t115 = 2 * t11 * t80 + t110 * t56 + t81;
        const auto t116 = 2 * t1;
        const auto t117 = t116 * t64 + t30;
        const auto t118 = t47 * t53;
        const auto t119 = t49 * t56;
        const auto t120 = p1x - 2 * p2x + p3x;
        const auto t121 = t116 * t19 + t120;
        const auto t122 = t1 * t72 + 2 * t59 + t62;
        const auto t123 = t18 * t79;
        const auto t124 = t19 + 1;
        const auto t125 = t116 * t123 * t2 + t120 * t14 + t124;
        const auto t126 = t6 * t92;
        const auto t127 = t10 * t93;
        const auto t128 = t13 * t96;
        const auto t129 = t53 * t83;
        const auto t130 = t56 * t85;
        const auto t131 = 2 * t5;
        const auto t132 = t131 * t64 + t32;
        const auto t133 = t14 * t21;
        const auto t134 = p1y - 2 * p2y + p3y;
        const auto t135 = t131 * t19 + t134;
        const auto t136 = t100 * t5 + t59 + 2 * t60 + t61;
        const auto t137 = t123 * t131 * t6 + t124 + t134 * t53;
        const auto t138 = t2 * t94;
        const auto t139 = t14 * t87;
        const auto t140 = 2 * t9;
        const auto t141 = t140 * t64 + t34;
        const auto t142 = p1z - 2 * p2z + p3z;
        const auto t143 = t140 * t19 + t142;
        const auto t144 = t109 * t9 + t59 + t60 + 2 * t61;
        const auto t145 = t10 * t123 * t140 + t124 + t142 * t56;
        const auto t146 = t13 * t3 - 1;
        const auto t147 = t45 / t51;
        const auto t148 = t13 * t7 - 1;
        const auto t149 = t11 * t13 - 1;
        grad[0] = t52
            * (t21 * (t1 * t14 + 1)
               + t37 * t45 * (-t31 * (-t22 * t25 + 1) + t33 * t39 + t35 * t40)
               - t50);
        grad[1] = t52
            * (t37 * t45 * (t31 * t39 - t33 * (-t23 * t25 + 1) + t35 * t54)
               + t47 * (t5 * t53 + 1) - t55);
        grad[2] = t52
            * (t37 * t45 * (t31 * t40 + t33 * t54 - t35 * (-t24 * t25 + 1))
               + t49 * (t56 * t9 + 1) - t57);
        grad[3] = t97
            * (-t13 * t77 * t78 + t21 * t82 + t47 * t75 + t49 * t76 + t58 * t67
               + t58 * t70 - t91 * (-t58 * t84 - t58 * t86 + t78 * t87)
               - t96 * (-t75 * t92 - t76 * t93 - t82 * t94));
        grad[4] = t97
            * (t103 * t21 + t104 * t49 - t105 * t13 * t66 + t106 * t47
               + t70 * t98 - t91 * (t105 * t83 - t107 * t98 - t86 * t98)
               - t96 * (-t103 * t94 - t104 * t93 - t106 * t92) + t98 * t99);
        grad[5] = t97
            * (t108 * t67 + t108 * t99 + t112 * t21 + t113 * t47
               - t114 * t13 * t69 + t115 * t49
               - t91 * (-t107 * t108 - t108 * t84 + t114 * t85)
               - t96 * (-t112 * t94 - t113 * t92 - t115 * t93));
        grad[6] = t97
            * (t117 * t118 + t117 * t119 + t121 * t67 + t121 * t70
               + t122 * t13 * t21 + t125 * t77
               + t128 * (t117 * t126 + t117 * t127 + t122 * t94)
               + t90 * (t121 * t129 + t121 * t130 + t125 * t87));
        grad[7] = t97
            * (t119 * t132 + t128 * (t127 * t132 + t132 * t138 + t136 * t92)
               + t13 * t136 * t47 + t132 * t133 + t135 * t70 + t135 * t99
               + t137 * t66 + t90 * (t130 * t135 + t135 * t139 + t137 * t83));
        grad[8] = t97
            * (t118 * t141 + t128 * (t126 * t141 + t138 * t141 + t144 * t93)
               + t13 * t144 * t49 + t133 * t141 + t143 * t67 + t143 * t99
               + t145 * t69 + t90 * (t129 * t143 + t139 * t143 + t145 * t85));
        grad[9] = t52
            * (t146 * t77 - t147 * (t146 * t21 + t50) + t39 * t66 + t40 * t69);
        grad[10] = t52
            * (-t147 * (t148 * t47 + t55) + t148 * t66 + t39 * t77 + t54 * t69);
        grad[11] = t52
            * (-t147 * (t149 * t49 + t57) + t149 * t69 + t40 * t77 + t54 * t66);
    }

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
        double hess[144])
    {
        const auto t0 = -p2y;
        const auto t1 = p1y + t0;
        const auto t2 = -t1;
        const auto t3 = std::pow(t2, 2);
        const auto t4 = -p2x;
        const auto t5 = p1x + t4;
        const auto t6 = std::pow(t5, 2);
        const auto t7 = -t5;
        const auto t8 = std::pow(t7, 2);
        const auto t9 = -p2z;
        const auto t10 = p1z + t9;
        const auto t11 = -t10;
        const auto t12 = std::pow(t11, 2);
        const auto t13 = t12 + t3 + t8;
        const auto t14 = std::pow(t13, -2);
        const auto t15 = t14 * t6;
        const auto t16 = 1.0 / t13;
        const auto t17 = t16 * t7;
        const auto t18 = t17 * t5;
        const auto t19 = t18 + 1;
        const auto t20 = -p1x;
        const auto t21 = p0x + t20;
        const auto t22 = -t21;
        const auto t23 = t22 * t7;
        const auto t24 = -p1y;
        const auto t25 = p0y + t24;
        const auto t26 = -t25;
        const auto t27 = t2 * t26;
        const auto t28 = -p1z;
        const auto t29 = p0z + t28;
        const auto t30 = -t29;
        const auto t31 = t11 * t30;
        const auto t32 = t27 + t31;
        const auto t33 = t23 + t32;
        const auto t34 = t16 * t33;
        const auto t35 = t34 * t7;
        const auto t36 = t21 + t35;
        const auto t37 = -p3x;
        const auto t38 = p2x + t37;
        const auto t39 = -p3y;
        const auto t40 = p2y + t39;
        const auto t41 = -p3z;
        const auto t42 = p2z + t41;
        const auto t43 = -t11 * t42 - t2 * t40 - t38 * t7;
        const auto t44 = t16 * t43;
        const auto t45 = t44 * t7;
        const auto t46 = t38 + t45;
        const auto t47 = t2 * t34;
        const auto t48 = t25 + t47;
        const auto t49 = t2 * t44;
        const auto t50 = t40 + t49;
        const auto t51 = t11 * t34;
        const auto t52 = t29 + t51;
        const auto t53 = t11 * t44;
        const auto t54 = t42 + t53;
        const auto t55 = t36 * t46 + t48 * t50 + t52 * t54;
        const auto t56 = t19 * t36;
        const auto t57 = std::pow(t1, 2);
        const auto t58 = std::pow(t10, 2);
        const auto t59 = t57 + t58 + t6;
        const auto t60 = 1.0 / t59;
        const auto t61 = t5 * t60;
        const auto t62 = t48 * t61;
        const auto t63 = t52 * t61;
        const auto t64 = t1 * t62 + t10 * t63;
        const auto t65 = -t56 + t64;
        const auto t66 = -t50;
        const auto t67 = t16 * t2;
        const auto t68 = t66 * t67;
        const auto t69 = -t54;
        const auto t70 = t11 * t16;
        const auto t71 = t69 * t70;
        const auto t72 = -t46;
        const auto t73 = t19 * t72 + t5 * t68 + t5 * t71;
        const auto t74 = std::pow(t36, 2) + std::pow(t48, 2) + std::pow(t52, 2);
        const auto t75 = 1.0 / t74;
        const auto t76 = std::pow(t46, 2) + std::pow(t50, 2) + std::pow(t54, 2);
        const auto t77 = std::pow(t76, -1.0 / 2.0);
        const auto t78 = t77 / std::pow(t74, 3.0 / 2.0);
        const auto t79 = t1 * t61;
        const auto t80 = t21 * t5;
        const auto t81 = t1 * t25;
        const auto t82 = t10 * t29;
        const auto t83 = t81 + t82;
        const auto t84 = t80 + t83;
        const auto t85 = t60 * t84;
        const auto t86 = t5 * t85;
        const auto t87 = -p0x;
        const auto t88 = p1x + t87;
        const auto t89 = t86 + t88;
        const auto t90 = -t89;
        const auto t91 = t38 * t5;
        const auto t92 = t1 * t40;
        const auto t93 = t10 * t42;
        const auto t94 = t92 + t93;
        const auto t95 = t91 + t94;
        const auto t96 = t60 * t95;
        const auto t97 = t5 * t96;
        const auto t98 = p3x + t4;
        const auto t99 = -t97 - t98;
        const auto t100 = t1 * t85;
        const auto t101 = -p0y;
        const auto t102 = p1y + t101;
        const auto t103 = t100 + t102;
        const auto t104 = -t103;
        const auto t105 = t1 * t96;
        const auto t106 = p3y + t0;
        const auto t107 = -t105 - t106;
        const auto t108 = t10 * t85;
        const auto t109 = -p0z;
        const auto t110 = p1z + t109;
        const auto t111 = t108 + t110;
        const auto t112 = -t111;
        const auto t113 = t10 * t96;
        const auto t114 = p3z + t9;
        const auto t115 = -t113 - t114;
        const auto t116 = t104 * t107 + t112 * t115 + t90 * t99;
        const auto t117 = t6 * t60;
        const auto t118 = t57 * t60;
        const auto t119 = t58 * t60;
        const auto t120 = t119 - 2;
        const auto t121 = t116 * (-t117 - t118 - t120);
        const auto t122 = t19 * t46;
        const auto t123 = t50 * t61;
        const auto t124 = t54 * t61;
        const auto t125 = t1 * t123 + t10 * t124;
        const auto t126 = -t122 + t125;
        const auto t127 = t5 * t90;
        const auto t128 = t1 * t60;
        const auto t129 = t127 * t128;
        const auto t130 = t10 * t112;
        const auto t131 = t128 * t130;
        const auto t132 = t118 - 1;
        const auto t133 = -t132;
        const auto t134 = t104 * t133;
        const auto t135 = t129 + t131 - t134;
        const auto t136 = t1 * t67;
        const auto t137 = t136 + 1;
        const auto t138 = t137 * t50;
        const auto t139 = t46 * t61;
        const auto t140 = t128 * t54;
        const auto t141 = t1 * t139 + t10 * t140;
        const auto t142 = -t138 + t141;
        const auto t143 = t1 * t104;
        const auto t144 = t143 * t61;
        const auto t145 = t130 * t61;
        const auto t146 = t117 - 1;
        const auto t147 = -t146;
        const auto t148 = t147 * t90;
        const auto t149 = t144 + t145 - t148;
        const auto t150 =
            std::pow(t104, 2) + std::pow(t112, 2) + std::pow(t90, 2);
        const auto t151 = 1.0 / t150;
        const auto t152 = t149 * t151;
        const auto t153 = 3 * t116;
        const auto t154 = t152 * t153;
        const auto t155 =
            std::pow(t107, 2) + std::pow(t115, 2) + std::pow(t99, 2);
        const auto t156 = std::pow(t155, -1.0 / 2.0);
        const auto t157 = t156 / std::pow(t150, 3.0 / 2.0);
        const auto t158 =
            t157 * (t121 * t79 - t126 * t135 + t135 * t154 - t142 * t149);
        const auto t159 = t10 * t61;
        const auto t160 = t10 * t70;
        const auto t161 = t160 + 1;
        const auto t162 = t161 * t54;
        const auto t163 = t128 * t50;
        const auto t164 = t10 * t139 + t10 * t163;
        const auto t165 = -t162 + t164;
        const auto t166 = t10 * t60;
        const auto t167 = t127 * t166;
        const auto t168 = t143 * t166;
        const auto t169 = t119 - 1;
        const auto t170 = -t169;
        const auto t171 = t112 * t170;
        const auto t172 = t167 + t168 - t171;
        const auto t173 =
            t157 * (t121 * t159 - t126 * t172 - t149 * t165 + t154 * t172);
        const auto t174 = 2 * t97;
        const auto t175 = t174 + t98;
        const auto t176 = t175 * t5;
        const auto t177 = t5 * t99;
        const auto t178 = 2 * t146;
        const auto t179 = 2 * t117;
        const auto t180 = t179 - 1;
        const auto t181 = t1 * t107;
        const auto t182 = t10 * t115;
        const auto t183 = 2 * t91;
        const auto t184 = -t183 + 2 * t6 * t60 * t95 - t94;
        const auto t185 = 2 * t18;
        const auto t186 = t185 + 1;
        const auto t187 = t48 * t67;
        const auto t188 = t52 * t70;
        const auto t189 = 2 * p1x;
        const auto t190 = p0x + p2x - t189;
        const auto t191 = t17 * t190;
        const auto t192 = t14 * t8;
        const auto t193 = 2 * t33;
        const auto t194 = -t34 - 1;
        const auto t195 = t191 + t192 * t193 + t194;
        const auto t196 = t16 * t8;
        const auto t197 = p2x + t196 * t5 + t20;
        const auto t198 = 2 * t36;
        const auto t199 = t16 * t198;
        const auto t200 = 2 * t35;
        const auto t201 = t190 + t200;
        const auto t202 = t201 * t61;
        const auto t203 = t136 * t202;
        const auto t204 = t160 * t202;
        const auto t205 = t19 * t195 + t197 * t199 - t203 - t204;
        const auto t206 = t116 * t151;
        const auto t207 = t187 * t201 + t188 * t201 + t195 * t36;
        const auto t208 = t126 * t151;
        const auto t209 = t175 * t60;
        const auto t210 = 2 * t86;
        const auto t211 = -t190 + t210;
        const auto t212 = t211 * t60;
        const auto t213 = t190 * t61;
        const auto t214 = std::pow(t59, -2);
        const auto t215 = t85 + 1;
        const auto t216 = -t213 + 2 * t214 * t6 * t84 - t215;
        const auto t217 = t130 * t209 + t143 * t209 + t181 * t212 + t182 * t212
            + t184 * t60 * t90 + t216 * t99;
        const auto t218 = 2 * t45;
        const auto t219 = t218 + t38;
        const auto t220 = t2 * t50;
        const auto t221 = t11 * t54;
        const auto t222 = -2 * t16 * t43 * t8 - t38 * t7 + t43;
        const auto t223 = -t222;
        const auto t224 = t219 * t220 + t219 * t221 + t223 * t46;
        const auto t225 = 1.0 / t155;
        const auto t226 = t126 * t225;
        const auto t227 = t16 * t226;
        const auto t228 = t153 / std::pow(t150, 2);
        const auto t229 = t149 * t228;
        const auto t230 = t116 * t225;
        const auto t231 = t16 * t230;
        const auto t232 = t152 * t231;
        const auto t233 =
            t152 * t217 + t207 * t208 - t207 * t229 + t224 * t227 - t224 * t232;
        const auto t234 = std::pow(t150, -1.0 / 2.0);
        const auto t235 = t156 * t234;
        const auto t236 = t16 * t5;
        const auto t237 = t16 * t3;
        const auto t238 = 2 * t237;
        const auto t239 = t238 - 1;
        const auto t240 = t239 * t48;
        const auto t241 = 2 * t47;
        const auto t242 = 2 * p1y;
        const auto t243 = p0y + p2y - t242;
        const auto t244 = t241 + t243;
        const auto t245 = t19 * t7;
        const auto t246 = t16 * t245;
        const auto t247 = t243 * t67;
        const auto t248 = t14 * t3;
        const auto t249 = t193 * t248 + t194 + t247;
        const auto t250 = t249 * t79;
        const auto t251 = t160 * t61;
        const auto t252 = t244 * t251;
        const auto t253 = t214 * t6;
        const auto t254 = t198 * t253;
        const auto t255 = t1 * t254;
        const auto t256 = 2 * t52;
        const auto t257 = t10 * t256;
        const auto t258 = t1 * t214;
        const auto t259 = t257 * t258 * t5;
        const auto t260 = t255 + t259;
        const auto t261 = -t250 - t252 + t260;
        const auto t262 = t244 * t246 + t261;
        const auto t263 = t236 * t50;
        const auto t264 = 2 * t1;
        const auto t265 = t253 * t46;
        const auto t266 = t264 * t265;
        const auto t267 = 2 * t5;
        const auto t268 = t10 * t258;
        const auto t269 = t267 * t268;
        const auto t270 = t269 * t54;
        const auto t271 = t266 + t270;
        const auto t272 = t17 * t36;
        const auto t273 = t188 * t244 + t244 * t272 + t249 * t48;
        const auto t274 = 2 * t105;
        const auto t275 = t106 + t274;
        const auto t276 = t275 * t60;
        const auto t277 = 2 * t100;
        const auto t278 = -t243 + t277;
        const auto t279 = t61 * t99;
        const auto t280 = t182 * t60;
        const auto t281 = 2 * t92;
        const auto t282 = t91 + t93;
        const auto t283 = -t281 - t282 + 2 * t57 * t60 * t95;
        const auto t284 = t128 * t243;
        const auto t285 = 2 * t214 * t57 * t84 - t215 - t284;
        const auto t286 = t104 * t283 * t60 + t107 * t285 + t127 * t276
            + t130 * t276 + t278 * t279 + t278 * t280;
        const auto t287 = 2 * t49;
        const auto t288 = t287 + t40;
        const auto t289 = t46 * t7;
        const auto t290 = -2 * t16 * t3 * t43 - t2 * t40 + t43;
        const auto t291 = -t290;
        const auto t292 = t221 * t288 + t288 * t289 + t291 * t50;
        const auto t293 = t16 * t291;
        const auto t294 = t293 * t79;
        const auto t295 = t251 * t288;
        const auto t296 = t152 * t286 + t208 * t273 + t227 * t292 - t229 * t273
            - t232 * t292 + t246 * t288 - t294 - t295;
        const auto t297 = t12 * t16;
        const auto t298 = 2 * t297;
        const auto t299 = t298 - 1;
        const auto t300 = t236 * t299;
        const auto t301 = 2 * t51;
        const auto t302 = 2 * p1z;
        const auto t303 = p0z + p2z - t302;
        const auto t304 = t301 + t303;
        const auto t305 = t303 * t70;
        const auto t306 = t12 * t14;
        const auto t307 = t193 * t306 + t194 + t305;
        const auto t308 = t159 * t307;
        const auto t309 = t136 * t61;
        const auto t310 = t304 * t309;
        const auto t311 = t10 * t254;
        const auto t312 = 2 * t48;
        const auto t313 = t1 * t312;
        const auto t314 = t10 * t214 * t313 * t5;
        const auto t315 = t311 + t314;
        const auto t316 = -t308 - t310 + t315;
        const auto t317 = t246 * t304 + t316;
        const auto t318 = 2 * t10;
        const auto t319 = t265 * t318;
        const auto t320 = t269 * t50;
        const auto t321 = t319 + t320;
        const auto t322 = t187 * t304 + t272 * t304 + t307 * t52;
        const auto t323 = 2 * t113;
        const auto t324 = t114 + t323;
        const auto t325 = t324 * t60;
        const auto t326 = 2 * t108;
        const auto t327 = -t303 + t326;
        const auto t328 = t181 * t60;
        const auto t329 = 2 * t93;
        const auto t330 = t91 + t92;
        const auto t331 = -t329 - t330 + 2 * t58 * t60 * t95;
        const auto t332 = t166 * t303;
        const auto t333 = 2 * t214 * t58 * t84 - t215 - t332;
        const auto t334 = t112 * t331 * t60 + t115 * t333 + t127 * t325
            + t143 * t325 + t279 * t327 + t327 * t328;
        const auto t335 = 2 * t53;
        const auto t336 = t335 + t42;
        const auto t337 = -t11 * t42 - 2 * t12 * t16 * t43 + t43;
        const auto t338 = -t337;
        const auto t339 = t220 * t336 + t289 * t336 + t338 * t54;
        const auto t340 = t16 * t338;
        const auto t341 = t159 * t340;
        const auto t342 = t309 * t336;
        const auto t343 = t152 * t334 + t208 * t322 + t227 * t339 - t229 * t322
            - t232 * t339 + t246 * t336 - t341 - t342;
        const auto t344 = 2 * p2x;
        const auto t345 = p1x + p3x - t344;
        const auto t346 = t345 * t61;
        const auto t347 = 2 * t95;
        const auto t348 = -t60 * t95 - 1;
        const auto t349 = -t253 * t347 - t346 - t348;
        const auto t350 = t214 * t57;
        const auto t351 = t174 + t345;
        const auto t352 = t351 * t5;
        const auto t353 = t214 * t58;
        const auto t354 = -t180;
        const auto t355 = 2 * t16;
        const auto t356 = t355 * t6 - 1;
        const auto t357 = t2 * t48;
        const auto t358 = t11 * t52;
        const auto t359 = t34 * t5;
        const auto t360 = 2 * t359;
        const auto t361 = t360 + t88;
        const auto t362 = t17 * t6 + t5;
        const auto t363 = 2 * t23;
        const auto t364 = t267 * t35 + t32 + t363;
        const auto t365 = t16 * t206;
        const auto t366 = -t210 - t88;
        const auto t367 = t351 * t60;
        const auto t368 = -t179 * t84 + 2 * t80 + t83;
        const auto t369 = -t130 * t367 - t143 * t367 + t280 * t366 + t328 * t366
            + t349 * t90 + t368 * t60 * t99;
        const auto t370 = t44 * t5;
        const auto t371 = 2 * t370;
        const auto t372 = t345 + t371;
        const auto t373 = t50 * t67;
        const auto t374 = t54 * t70;
        const auto t375 = t17 * t345;
        const auto t376 = t14 * t43;
        const auto t377 = t376 * t7;
        const auto t378 = t44 + 1;
        const auto t379 = t267 * t377 + t375 + t378;
        const auto t380 = t372 * t373 + t372 * t374 + t379 * t46;
        const auto t381 = t357 * t361 + t358 * t361 + t36 * t364;
        const auto t382 = t16 * t208;
        const auto t383 = t152 * t230;
        const auto t384 = t16 * t229;
        const auto t385 =
            t152 * t369 + t226 * t380 - t380 * t383 + t381 * t382 - t381 * t384;
        const auto t386 = -t102 - t277;
        const auto t387 = 2 * p2y;
        const auto t388 = p1y + p3y - t387;
        const auto t389 = t274 + t388;
        const auto t390 = t389 * t60;
        const auto t391 = 2 * t118;
        const auto t392 = t80 + t82;
        const auto t393 = -t391 * t84 + t392 + 2 * t81;
        const auto t394 = t128 * t388;
        const auto t395 = -t347 * t350 - t348 - t394;
        const auto t396 = t104 * t395 + t107 * t393 * t60 - t127 * t390
            - t130 * t390 + t279 * t386 + t280 * t386;
        const auto t397 = 2 * t44;
        const auto t398 = t1 * t397;
        const auto t399 = t388 + t398;
        const auto t400 = t17 * t46;
        const auto t401 = t388 * t67;
        const auto t402 = t2 * t376;
        const auto t403 = t264 * t402 + t378 + t401;
        const auto t404 = t374 * t399 + t399 * t400 + t403 * t50;
        const auto t405 = 2 * t34;
        const auto t406 = t1 * t405;
        const auto t407 = t102 + t406;
        const auto t408 = t36 * t7;
        const auto t409 = 2 * t27;
        const auto t410 = t23 + t264 * t47 + t31 + t409;
        const auto t411 = t358 * t407 + t407 * t408 + t410 * t48;
        const auto t412 = 2 * t136;
        const auto t413 = t412 + 1;
        const auto t414 = t403 * t79;
        const auto t415 = t391 - 1;
        const auto t416 = -t415;
        const auto t417 = t128 * t393 + 2 * t129 + 2 * t131;
        const auto t418 = -t104 * t416 + t417;
        const auto t419 = t119 * t386;
        const auto t420 = -t147 * t386 + t419;
        const auto t421 = t206 * t61;
        const auto t422 = t383 * t404;
        const auto t423 = t384 * t411;
        const auto t424 = t251 * t399;
        const auto t425 = -t270;
        const auto t426 = -t266 + t425;
        const auto t427 = -t110 - t326;
        const auto t428 = 2 * p2z;
        const auto t429 = p1z + p3z - t428;
        const auto t430 = t323 + t429;
        const auto t431 = t430 * t60;
        const auto t432 = 2 * t119;
        const auto t433 = t80 + t81;
        const auto t434 = -t432 * t84 + t433 + 2 * t82;
        const auto t435 = t166 * t429;
        const auto t436 = -t347 * t353 - t348 - t435;
        const auto t437 = t112 * t436 + t115 * t434 * t60 - t127 * t431
            - t143 * t431 + t279 * t427 + t328 * t427;
        const auto t438 = t10 * t397;
        const auto t439 = t429 + t438;
        const auto t440 = t429 * t70;
        const auto t441 = t11 * t376;
        const auto t442 = t318 * t441 + t378 + t440;
        const auto t443 = t373 * t439 + t400 * t439 + t442 * t54;
        const auto t444 = t10 * t405;
        const auto t445 = t110 + t444;
        const auto t446 = 2 * t31;
        const auto t447 = t23 + t27 + t318 * t51 + t446;
        const auto t448 = t357 * t445 + t408 * t445 + t447 * t52;
        const auto t449 = 2 * t160;
        const auto t450 = t449 + 1;
        const auto t451 = t450 * t54;
        const auto t452 = t159 * t442;
        const auto t453 = t432 - 1;
        const auto t454 = -t453;
        const auto t455 = t166 * t434 + 2 * t167 + 2 * t168;
        const auto t456 = -t112 * t454 + t455;
        const auto t457 = t118 * t427;
        const auto t458 = -t147 * t427 + t457;
        const auto t459 = t383 * t443;
        const auto t460 = t384 * t448;
        const auto t461 = t309 * t439;
        const auto t462 = -t320;
        const auto t463 = -t319 + t462;
        const auto t464 = t253 * t57;
        const auto t465 = t253 * t58;
        const auto t466 = t196 - 1;
        const auto t467 = t46 * t466;
        const auto t468 = t125 + t467;
        const auto t469 = t36 * t466 + t64;
        const auto t470 = t235
            * (t126 * t225 * t468 + t149 * t151 * t469 + t19 * t466
               - t383 * t468 - t464 - t465);
        const auto t471 = t237 - 1;
        const auto t472 = -t471;
        const auto t473 = 1.0 / t76;
        const auto t474 = t471 * t50;
        const auto t475 = t141 + t474;
        const auto t476 = -t36;
        const auto t477 = t17 * t476;
        const auto t478 = -t52;
        const auto t479 = t11 * t478;
        const auto t480 = -t48;
        const auto t481 = t2 * t477 - t472 * t480 + t479 * t67;
        const auto t482 = t65 * t75;
        const auto t483 = t473 * t55;
        const auto t484 = t482 * t483;
        const auto t485 = std::pow(t74, -1.0 / 2.0);
        const auto t486 = t485 * t77;
        const auto t487 = t486
            * (t473 * t475 * t73 - t475 * t484 - t481 * t482
               - t67 * (-t245 - t297 * t5 + t472 * t5));
        const auto t488 = t297 - 1;
        const auto t489 = -t488;
        const auto t490 = t488 * t54;
        const auto t491 = t164 + t490;
        const auto t492 = t480 * t67;
        const auto t493 = t11 * t477 + t11 * t492 - t478 * t489;
        const auto t494 = t486
            * (t473 * t491 * t73 - t482 * t493 - t484 * t491
               - t70 * (-t237 * t5 - t245 + t489 * t5));
        const auto t495 = t14 * t57;
        const auto t496 = t137 * t48;
        const auto t497 = t36 * t61;
        const auto t498 = t128 * t52;
        const auto t499 = t1 * t497 + t10 * t498;
        const auto t500 = -t496 + t499;
        const auto t501 = t17 * t72;
        const auto t502 = t1 * t501 + t1 * t71 + t137 * t66;
        const auto t503 = t10 * t128;
        const auto t504 = t135 * t206;
        const auto t505 =
            t157 * (t121 * t503 - t135 * t165 - t142 * t172 + 3 * t172 * t504);
        const auto t506 = t1 * t16;
        const auto t507 = 2 * t196;
        const auto t508 = t507 - 1;
        const auto t509 = t36 * t508;
        const auto t510 = t312 * t350;
        const auto t511 = t5 * t510;
        const auto t512 = t259 + t511;
        const auto t513 = t137 * t2;
        const auto t514 = t16 * t513;
        const auto t515 = t128 * t160;
        const auto t516 = -t195 * t79 - t201 * t515;
        const auto t517 = t201 * t514 + t516;
        const auto t518 = t350 * t50;
        const auto t519 = t267 * t518;
        const auto t520 = t270 + t519;
        const auto t521 = t142 * t151;
        const auto t522 = t135 * t151;
        const auto t523 = t142 * t225;
        const auto t524 = t16 * t523;
        const auto t525 = t135 * t228;
        const auto t526 = t16 * t223;
        const auto t527 = t526 * t79;
        const auto t528 = t219 * t515;
        const auto t529 = t225 * t365;
        const auto t530 = t135 * t529;
        const auto t531 = t207 * t521 - t207 * t525 + t217 * t522 + t219 * t514
            + t224 * t524 - t224 * t530 - t527 - t528;
        const auto t532 = t1 * t275;
        const auto t533 = 2 * t132;
        const auto t534 = t128 * t18;
        const auto t535 = t244 * t534;
        const auto t536 = t244 * t515;
        const auto t537 = p2y + t1 * t237 + t24;
        const auto t538 = t273 * t525;
        const auto t539 = t292 * t530;
        const auto t540 = t299 * t506;
        const auto t541 = t307 * t503;
        const auto t542 = t304 * t534;
        const auto t543 = t10 * t510;
        const auto t544 = t198 * t5;
        const auto t545 = t268 * t544;
        const auto t546 = t543 + t545;
        const auto t547 = -t541 - t542 + t546;
        const auto t548 = t304 * t514 + t547;
        const auto t549 = t318 * t518;
        const auto t550 = t267 * t46;
        const auto t551 = t268 * t550;
        const auto t552 = t549 + t551;
        const auto t553 = t340 * t503;
        const auto t554 = t336 * t534;
        const auto t555 = t322 * t521 - t322 * t525 + t334 * t522 + t336 * t514
            + t339 * t524 - t339 * t530 - t553 - t554;
        const auto t556 = 2 * t144 + 2 * t145 + t368 * t61;
        const auto t557 = -t354 * t90 + t556;
        const auto t558 = t119 * t366;
        const auto t559 = -t133 * t366 + t558;
        const auto t560 = t225 * t380;
        const auto t561 = t16 * t525;
        const auto t562 = t372 * t515 + t379 * t79;
        const auto t563 = -t135 * t151 * t369 - t137 * t16 * t2 * t372
            - t142 * t151 * t16 * t381 - t142 * t225 * t380 + t381 * t561
            + t504 * t560 + t562;
        const auto t564 = t1 * t389;
        const auto t565 = t355 * t57 - 1;
        const auto t566 = t1 + t57 * t67;
        const auto t567 = 2 * t60;
        const auto t568 = t225 * t404;
        const auto t569 = t16 * t411 * t521 + t396 * t522 + t404 * t523
            - t411 * t561 - t504 * t568;
        const auto t570 = t117 * t427;
        const auto t571 = -t133 * t427 + t570;
        const auto t572 = t225 * t443;
        const auto t573 = t439 * t534 + t442 * t503 + t552;
        const auto t574 = -t135 * t151 * t437 - t137 * t16 * t2 * t439
            - t142 * t151 * t16 * t448 - t142 * t225 * t443 + t448 * t561
            + t504 * t572 + t573;
        const auto t575 = -t466;
        const auto t576 = t2 * t480;
        const auto t577 = t17 * t479 + t17 * t576 - t476 * t575;
        const auto t578 = t500 * t75;
        const auto t579 = t483 * t578;
        const auto t580 = t486
            * (-t17 * (-t1 * t297 + t1 * t575 - t513) + t468 * t473 * t502
               - t468 * t579 - t577 * t578);
        const auto t581 = t350 * t58;
        const auto t582 = t471 * t48 + t499;
        const auto t583 = t225 * t475;
        const auto t584 = t235
            * (t135 * t151 * t582 + t137 * t471 + t142 * t225 * t475 - t464
               - t504 * t583 - t581);
        const auto t585 = t486
            * (t473 * t491 * t502 - t491 * t579 - t493 * t578
               - t70 * (-t1 * t196 + t1 * t489 - t513));
        const auto t586 = t14 * t58;
        const auto t587 = t161 * t52;
        const auto t588 = t128 * t48;
        const auto t589 = t10 * t497 + t10 * t588;
        const auto t590 = -t587 + t589;
        const auto t591 = t10 * t501 + t10 * t68 + t161 * t69;
        const auto t592 = t10 * t16;
        const auto t593 = t256 * t353;
        const auto t594 = t5 * t593;
        const auto t595 = t314 + t594;
        const auto t596 = t11 * t161;
        const auto t597 = t16 * t596;
        const auto t598 = t136 * t166;
        const auto t599 = -t159 * t195 - t201 * t598;
        const auto t600 = t201 * t597 + t599;
        const auto t601 = t353 * t54;
        const auto t602 = t267 * t601;
        const auto t603 = t320 + t602;
        const auto t604 = t151 * t165;
        const auto t605 = t151 * t172;
        const auto t606 = t165 * t225;
        const auto t607 = t16 * t606;
        const auto t608 = t172 * t228;
        const auto t609 = t159 * t526;
        const auto t610 = t219 * t598;
        const auto t611 = t172 * t529;
        const auto t612 = t207 * t604 - t207 * t608 + t217 * t605 + t219 * t597
            + t224 * t607 - t224 * t611 - t609 - t610;
        const auto t613 = t1 * t593;
        const auto t614 = t545 + t613;
        const auto t615 = t249 * t503;
        const auto t616 = t166 * t18;
        const auto t617 = t244 * t616;
        const auto t618 = -t615 - t617;
        const auto t619 = t244 * t597 + t618;
        const auto t620 = t264 * t601;
        const auto t621 = t551 + t620;
        const auto t622 = t293 * t503;
        const auto t623 = t288 * t616;
        const auto t624 = t273 * t604 - t273 * t608 + t286 * t605 + t288 * t597
            + t292 * t607 - t292 * t611 - t622 - t623;
        const auto t625 = t10 * t324;
        const auto t626 = 2 * t169;
        const auto t627 = t304 * t616;
        const auto t628 = t304 * t598;
        const auto t629 = p2z + t10 * t297 + t28;
        const auto t630 = t322 * t608;
        const auto t631 = t339 * t611;
        const auto t632 = t118 * t366;
        const auto t633 = -t170 * t366 + t632;
        const auto t634 = t172 * t206;
        const auto t635 = t16 * t608;
        const auto t636 = t159 * t379 + t372 * t598;
        const auto t637 = -t11 * t16 * t161 * t372 - t151 * t16 * t165 * t381
            - t151 * t172 * t369 - t165 * t225 * t380 + t381 * t635
            + t560 * t634 + t636;
        const auto t638 = t117 * t386;
        const auto t639 = -t170 * t386 + t638;
        const auto t640 = t399 * t616 + t403 * t503;
        const auto t641 = -t11 * t16 * t161 * t399 - t151 * t16 * t165 * t411
            - t151 * t172 * t396 - t165 * t225 * t404 + t411 * t635
            + t568 * t634 + t640;
        const auto t642 = t10 * t430;
        const auto t643 = t355 * t58 - 1;
        const auto t644 = t2 * t503;
        const auto t645 = t10 + t58 * t70;
        const auto t646 = t16 * t448 * t604 + t437 * t605 + t443 * t606
            - t448 * t635 - t572 * t634;
        const auto t647 = t590 * t75;
        const auto t648 = t483 * t647;
        const auto t649 = t486
            * (-t17 * (-t10 * t237 + t10 * t575 - t596) + t468 * t473 * t591
               - t468 * t648 - t577 * t647);
        const auto t650 = t486
            * (t473 * t475 * t591 - t475 * t648 - t481 * t647
               - t67 * (-t10 * t196 + t10 * t472 - t596));
        const auto t651 = t488 * t52 + t589;
        const auto t652 = t225 * t491;
        const auto t653 = t235
            * (t151 * t172 * t651 + t161 * t488 + t165 * t225 * t491 - t465
               - t581 - t634 * t652);
        const auto t654 = -t588;
        const auto t655 = t166 * t52;
        const auto t656 = -t655;
        const auto t657 = t253 * t313;
        const auto t658 = t253 * t257;
        const auto t659 = t654 + t656 + t657 + t658;
        const auto t660 = t355 * t46;
        const auto t661 = t219 * t309;
        const auto t662 = t219 * t251;
        const auto t663 = -t163;
        const auto t664 = t166 * t54;
        const auto t665 = -t664;
        const auto t666 = t264 * t50;
        const auto t667 = t253 * t666;
        const auto t668 = t318 * t54;
        const auto t669 = t253 * t668;
        const auto t670 = t663 + t665 + t667 + t669;
        const auto t671 = t260 + t511;
        const auto t672 = -t128 * t36 + t671;
        const auto t673 = t128 * t46;
        const auto t674 = t271 + t519;
        const auto t675 = -t673 + t674;
        const auto t676 = t315 + t594;
        const auto t677 = -t166 * t36 + t676;
        const auto t678 = t166 * t46;
        const auto t679 = t321 + t602;
        const auto t680 = -t678 + t679;
        const auto t681 = std::pow(t155, -2);
        const auto t682 = t153 * t681;
        const auto t683 = t14 * t682;
        const auto t684 = t14 * std::pow(t219, 2);
        const auto t685 = 2 * t14;
        const auto t686 = t685 * (4 * t16 * t43 * t8 + 2 * t38 * t7 - t43);
        const auto t687 = 4 * t95;
        const auto t688 = t214 * std::pow(t5, 3);
        const auto t689 = -t687 * t688 + 3 * t97;
        const auto t690 = t179 * t38 + t689 + t98;
        const auto t691 = t14 * std::pow(t201, 2);
        const auto t692 = 4 * t33;
        const auto t693 = t192 * t692;
        const auto t694 = 2 * t191 + t194 + t693;
        const auto t695 = t312 * t67;
        const auto t696 = t256 * t70;
        const auto t697 = 4 * t84;
        const auto t698 = t688 * t697;
        const auto t699 = p0x - 3 * p1x - 2 * t190 * t6 * t60 + t344
            - 3 * t5 * t60 * t84 + t698;
        const auto t700 = t217 * t225;
        const auto t701 = t224 * t355;
        const auto t702 = t151 * t207;
        const auto t703 = t206 * t207;
        const auto t704 = t175 * t211;
        const auto t705 = t60 * (-t117 * t687 + 3 * t91 + t94);
        const auto t706 = 2 * t213 + t215 - t253 * t697;
        const auto t707 = std::pow(t76, -2);
        const auto t708 = t288 * t7;
        const auto t709 = t11 * t69;
        const auto t710 = -t288 * t709 + t290 * t66 - t708 * t72;
        const auto t711 = t2 * t219;
        const auto t712 = -t219 * t709 + t222 * t72 - t66 * t711;
        const auto t713 = std::pow(t74, -2);
        const auto t714 = t478 * t70;
        const auto t715 = -t249;
        const auto t716 = -t244 * t477 - t244 * t714 + t480 * t715;
        const auto t717 = -t195;
        const auto t718 = -t201 * t492 - t201 * t714 + t476 * t717;
        const auto t719 = t187 * t219 + t188 * t219 + t195 * t46 + t201 * t373
            + t201 * t374 + t36 * t526;
        const auto t720 = t473 * t719;
        const auto t721 = t16 * t720;
        const auto t722 = t188 * t288 + t244 * t374 + t244 * t400 + t249 * t50
            + t272 * t288 + t293 * t48;
        const auto t723 = t473 * t722;
        const auto t724 = t16 * t712;
        const auto t725 = t719 * t75;
        const auto t726 = t718 * t75;
        const auto t727 = t40 * t7;
        const auto t728 = t2 * t38;
        const auto t729 = 4 * t2;
        const auto t730 = t45 * t729 + t727 + t728;
        const auto t731 = 2 * t69;
        const auto t732 = t70 * t731;
        const auto t733 = t17 * t222;
        const auto t734 = t219 * t67;
        const auto t735 = t38 * t7;
        const auto t736 = 2 * t735;
        const auto t737 = 8 * t43;
        const auto t738 = t192 * t2;
        const auto t739 = t106 - t287;
        const auto t740 = t40 * t507 + t67 * t736 + t737 * t738 + t739;
        const auto t741 = t2 * t40;
        const auto t742 = 2 * t741;
        const auto t743 = t248 * t7;
        const auto t744 = -t218 + t98;
        const auto t745 = t17 * t742 + t238 * t38 + t737 * t743 + t744;
        const auto t746 = t16 * t483;
        const auto t747 = t243 * t7;
        const auto t748 = t190 * t2;
        const auto t749 = t35 * t729 + t747 + t748;
        const auto t750 = 2 * t478;
        const auto t751 = t70 * t750;
        const auto t752 = t7 * t717;
        const auto t753 = t2 * t201;
        const auto t754 = 2 * t17;
        const auto t755 = 8 * t33;
        const auto t756 = t0 + t101 - t241 + t242;
        const auto t757 = t243 * t507 + t738 * t755 + t748 * t754 + t756;
        const auto t758 = 2 * t67;
        const auto t759 = t189 - t200 + t4 + t87;
        const auto t760 = t190 * t238 + t743 * t755 + t747 * t758 + t759;
        const auto t761 = t55 * t75;
        const auto t762 = t16 * t761;
        const auto t763 = t244 * t297;
        const auto t764 = t288 * t297;
        const auto t765 = 2 * t374;
        const auto t766 = t17 * t223;
        const auto t767 = t486
            * (3 * t14 * t55 * t707 * t710 * t712
               + t16 * t473 * t55 * t710 * t718 * t75
               + t16 * t473 * t55 * t712 * t716 * t75
               + t16
                   * (t195 * t708 + t201 * t291 * t67 + t201 * t764
                      + t219 * t763 + t244 * t766 + t249 * t711 + t36 * t740
                      + t46 * t757 + t48 * t745 + t50 * t760 + t696 * t730
                      + t749 * t765)
               + 3 * t55 * t713 * t716 * t718 - t710 * t721 - t716 * t725
               - t722 * t726 - t723 * t724
               - t746
                   * (t12 * t16 * t219 * t288 - t288 * t733 - t290 * t734
                      - t66 * t745 - t72 * t740 - t730 * t732)
               - t762
                   * (t12 * t16 * t201 * t244 - t244 * t752 - t476 * t757
                      - t480 * t760 - t715 * t753 - t749 * t751));
        const auto t768 = t336 * t7;
        const auto t769 = t2 * t66;
        const auto t770 = -t336 * t769 + t337 * t69 - t72 * t768;
        const auto t771 = -t307;
        const auto t772 = -t304 * t477 - t304 * t492 + t478 * t771;
        const auto t773 = t187 * t336 + t272 * t336 + t304 * t373 + t304 * t400
            + t307 * t54 + t340 * t52;
        const auto t774 = t473 * t773;
        const auto t775 = t42 * t7;
        const auto t776 = t11 * t38;
        const auto t777 = 4 * t11;
        const auto t778 = t45 * t777 + t775 + t776;
        const auto t779 = 2 * t66;
        const auto t780 = t67 * t779;
        const auto t781 = t219 * t70;
        const auto t782 = t11 * t192;
        const auto t783 = t114 - t335;
        const auto t784 = t42 * t507 + t70 * t736 + t737 * t782 + t783;
        const auto t785 = t11 * t42;
        const auto t786 = 2 * t785;
        const auto t787 = t306 * t7;
        const auto t788 = t17 * t786 + t298 * t38 + t737 * t787 + t744;
        const auto t789 = t303 * t7;
        const auto t790 = t11 * t190;
        const auto t791 = t35 * t777 + t789 + t790;
        const auto t792 = 2 * t480;
        const auto t793 = t67 * t792;
        const auto t794 = t11 * t201;
        const auto t795 = t109 - t301 + t302 + t9;
        const auto t796 = t303 * t507 + t754 * t790 + t755 * t782 + t795;
        const auto t797 = 2 * t70;
        const auto t798 = t190 * t298 + t755 * t787 + t759 + t789 * t797;
        const auto t799 = t237 * t304;
        const auto t800 = t237 * t336;
        const auto t801 = 2 * t373;
        const auto t802 = t11 * t219;
        const auto t803 = t486
            * (3 * t14 * t55 * t707 * t712 * t770
               + t16 * t473 * t55 * t712 * t75 * t772
               + t16 * t473 * t55 * t718 * t75 * t770
               + t16
                   * (t195 * t768 + t201 * t338 * t70 + t201 * t800
                      + t219 * t799 + t304 * t766 + t307 * t802 + t36 * t784
                      + t46 * t796 + t52 * t788 + t54 * t798 + t695 * t778
                      + t791 * t801)
               + 3 * t55 * t713 * t718 * t772 - t721 * t770 - t724 * t774
               - t725 * t772 - t726 * t773
               - t746
                   * (t16 * t219 * t3 * t336 - t336 * t733 - t337 * t781
                      - t69 * t788 - t72 * t784 - t778 * t780)
               - t762
                   * (t16 * t201 * t3 * t304 - t304 * t752 - t476 * t796
                      - t478 * t798 - t771 * t794 - t791 * t793));
        const auto t804 = t219 * t361;
        const auto t805 = t201 * t372;
        const auto t806 = 8 * t377;
        const auto t807 = t5 * t806;
        const auto t808 = t397 + 1;
        const auto t809 = t16 * t183 + 2 * t375 + t807 + t808;
        const auto t810 = t355 * t5;
        const auto t811 = t5 * t755;
        const auto t812 = t14 * t7;
        const auto t813 = t811 * t812;
        const auto t814 = t405 + 1;
        const auto t815 = t16 * t363 + t190 * t810 + t813 + t814;
        const auto t816 = t18 * t190 + t196 * t22 + t201 - t359 + t5 * t693;
        const auto t817 = 4 * t43;
        const auto t818 = t20 + t344 + t37;
        const auto t819 =
            t17 * t91 + t192 * t5 * t817 + t196 * t345 + t218 - t370 + t818;
        const auto t820 = t361 * t479 + t361 * t576 + t364 * t476;
        const auto t821 = 3 * t55;
        const auto t822 = t713 * t821;
        const auto t823 = t16 * t718 * t822;
        const auto t824 = t372 * t68 + t372 * t71 + t379 * t72;
        const auto t825 = t707 * t821;
        const auto t826 = t824 * t825;
        const auto t827 = t473 * t761;
        const auto t828 = t14 * t712 * t827;
        const auto t829 = t16 * t364;
        const auto t830 = t187 * t372 + t188 * t372 + t36 * t379 + t361 * t373
            + t361 * t374 + t46 * t829;
        const auto t831 = t473 * t830;
        const auto t832 = t473 * t824;
        const auto t833 = t55 * t832;
        const auto t834 = t201 * t361;
        const auto t835 = 2 * t476;
        const auto t836 = t219 * t372;
        const auto t837 = 2 * t72;
        const auto t838 = t486
            * (t14 * t223 * t364 + t16 * t725 * t820 + t187 * t809 + t188 * t809
               + t195 * t379 + t199 * t819 + t248 * t804 + t248 * t805
               + t306 * t804 + t306 * t805 + t373 * t815 + t374 * t815
               + t660 * t816 + t720 * t824 - t724 * t826 - t724 * t831
               - t726 * t830 - t726 * t833
               - t746
                   * (-t222 * t379 + t237 * t836 + t297 * t836 - t709 * t809
                      - t769 * t809 - t819 * t837)
               - t762
                   * (t237 * t834 + t297 * t834 - t364 * t717 - t479 * t815
                      - t576 * t815 - t816 * t835)
               - t820 * t823 - t820 * t828);
        const auto t839 = t476 * t7;
        const auto t840 = t407 * t479 + t407 * t839 + t410 * t480;
        const auto t841 = t399 * t501 + t399 * t71 + t403 * t66;
        const auto t842 = t825 * t841;
        const auto t843 = t16 * t410;
        const auto t844 = t188 * t399 + t272 * t399 + t374 * t407 + t400 * t407
            + t403 * t48 + t50 * t843;
        const auto t845 = t473 * t844;
        const auto t846 = t483 * t726;
        const auto t847 = t26 * t7;
        const auto t848 = t1 * t190;
        const auto t849 = 4 * t1;
        const auto t850 = t35 * t849 + t847 + t848;
        const auto t851 = t1 * t192;
        const auto t852 = t25 - t406;
        const auto t853 = t26 * t507 + t754 * t848 + t755 * t851 + t852;
        const auto t854 = t1 * t355;
        const auto t855 = t1 * t755;
        const auto t856 = t2 * t855;
        const auto t857 = t17 * t409 + t201 + t748 * t854 + t812 * t856;
        const auto t858 = t388 * t7;
        const auto t859 = t1 * t38 + t45 * t849 + t858;
        const auto t860 = t24 + t387 + t39;
        const auto t861 = -t398 + t860;
        const auto t862 = t388 * t507 + t735 * t854 + t737 * t851 + t861;
        const auto t863 = t1 * t2;
        const auto t864 = t219 + t728 * t854 + t758 * t858 + t806 * t863;
        const auto t865 = t297 * t407;
        const auto t866 = t297 * t399;
        const auto t867 = t195 * t7;
        const auto t868 = t486
            * (t16 * t719 * t75 * t840
               + t16
                   * (t201 * t866 + t219 * t865 + t36 * t862 + t399 * t867
                      + t403 * t753 + t407 * t766 + t410 * t734 + t46 * t853
                      + t48 * t864 + t50 * t857 + t696 * t859 + t765 * t850)
               + t473 * t719 * t841 - t724 * t842 - t724 * t845 - t726 * t844
               - t746
                   * (t12 * t16 * t219 * t399 + t2 * t219 * t403 - t399 * t733
                      - t66 * t864 - t72 * t862 - t732 * t859)
               - t762
                   * (t12 * t16 * t201 * t407 + t16 * t2 * t201 * t410
                      - t407 * t752 - t476 * t853 - t480 * t857 - t751 * t850)
               - t823 * t840 - t828 * t840 - t841 * t846);
        const auto t869 = t445 * t576 + t445 * t839 + t447 * t478;
        const auto t870 = t439 * t501 + t439 * t68 + t442 * t69;
        const auto t871 = t825 * t870;
        const auto t872 = t16 * t447;
        const auto t873 = t187 * t439 + t272 * t439 + t373 * t445 + t400 * t445
            + t442 * t52 + t54 * t872;
        const auto t874 = t473 * t873;
        const auto t875 = t30 * t7;
        const auto t876 = t10 * t190;
        const auto t877 = 4 * t10;
        const auto t878 = t35 * t877 + t875 + t876;
        const auto t879 = t10 * t192;
        const auto t880 = t29 - t444;
        const auto t881 = t30 * t507 + t754 * t876 + t755 * t879 + t880;
        const auto t882 = t10 * t355;
        const auto t883 = t10 * t755;
        const auto t884 = t11 * t883;
        const auto t885 = t17 * t446 + t201 + t790 * t882 + t812 * t884;
        const auto t886 = t429 * t7;
        const auto t887 = t10 * t38 + t45 * t877 + t886;
        const auto t888 = t28 + t41 + t428;
        const auto t889 = -t438 + t888;
        const auto t890 = t429 * t507 + t735 * t882 + t737 * t879 + t889;
        const auto t891 = t10 * t11;
        const auto t892 = t219 + t776 * t882 + t797 * t886 + t806 * t891;
        const auto t893 = t237 * t445;
        const auto t894 = t237 * t439;
        const auto t895 = t486
            * (t16 * t719 * t75 * t869
               + t16
                   * (t201 * t894 + t219 * t893 + t36 * t890 + t439 * t867
                      + t442 * t794 + t445 * t766 + t447 * t781 + t46 * t881
                      + t52 * t892 + t54 * t885 + t695 * t887 + t801 * t878)
               + t473 * t719 * t870 - t724 * t871 - t724 * t874 - t726 * t873
               - t746
                   * (t11 * t219 * t442 + t16 * t219 * t3 * t439 - t439 * t733
                      - t69 * t892 - t72 * t890 - t780 * t887)
               - t762
                   * (t11 * t16 * t201 * t447 + t16 * t201 * t3 * t445
                      - t445 * t752 - t476 * t881 - t478 * t885 - t793 * t878)
               - t823 * t869 - t828 * t869 - t846 * t870);
        const auto t896 = t223 * t466;
        const auto t897 = t16 * std::pow(t7, 3) + t5;
        const auto t898 = t163 + t664 - t667 - t669;
        const auto t899 = t16 * t225;
        const auto t900 = t224 * t899;
        const auto t901 = t225 * t468;
        const auto t902 = t16 * t682;
        const auto t903 = t224 * t468 * t902 - t468 * t700 - t469 * t702
            - t469 * t900 + t703 * t901;
        const auto t904 = -t519;
        const auto t905 = t471 * t734 + t527 + t528 + t904;
        const auto t906 = -t116 * t151 * t207 * t225 * t475
            - 3 * t116 * t16 * t224 * t475 * t681 - t16 * t2 * t201 * t471
            + t475 * t700 + t516 + t582 * t702 + t582 * t900;
        const auto t907 = -t602;
        const auto t908 = t488 * t781 + t609 + t610 + t907;
        const auto t909 = -t11 * t16 * t201 * t488
            - t116 * t151 * t207 * t225 * t491
            - 3 * t116 * t16 * t224 * t491 * t681 + t491 * t700 + t599
            + t651 * t702 + t651 * t900;
        const auto t910 = -t123;
        const auto t911 = -t62;
        const auto t912 = t511 + t911;
        const auto t913 = t350 * t544;
        const auto t914 = t257 * t350;
        const auto t915 = t497 + t535 + t536 + t655 - t913 - t914;
        const auto t916 = t350 * t550;
        const auto t917 = t350 * t668;
        const auto t918 = t139 + t664 - t916 - t917;
        const auto t919 = t288 * t515 + t288 * t534 + t918;
        const auto t920 = t546 + t613;
        const auto t921 = -t166 * t48 + t920;
        const auto t922 = t166 * t50;
        const auto t923 = t552 + t620;
        const auto t924 = -t922 + t923;
        const auto t925 = t14 * std::pow(t288, 2);
        const auto t926 = t685 * (4 * t16 * t3 * t43 + 2 * t2 * t40 - t43);
        const auto t927 = std::pow(t1, 3) * t214;
        const auto t928 = 3 * t105 - t687 * t927;
        const auto t929 = t106 + t391 * t40 + t928;
        const auto t930 = t14 * std::pow(t244, 2);
        const auto t931 = t248 * t692;
        const auto t932 = t194 + 2 * t247 + t931;
        const auto t933 = t17 * t198;
        const auto t934 = t697 * t927;
        const auto t935 = p0y - 3 * p1y - 3 * t1 * t60 * t84
            - 2 * t243 * t57 * t60 + t387 + t934;
        const auto t936 = t225 * t286;
        const auto t937 = t292 * t355;
        const auto t938 = t151 * t273;
        const auto t939 = t206 * t273;
        const auto t940 = t275 * t278;
        const auto t941 = t60 * (-t118 * t687 + t282 + 3 * t92);
        const auto t942 = t215 + 2 * t284 - t350 * t697;
        const auto t943 = t16 * t710;
        const auto t944 = t16 * t770;
        const auto t945 = t716 * t75;
        const auto t946 = t75 * t772;
        const auto t947 = t196 * t336;
        const auto t948 = t2 * t42;
        const auto t949 = t11 * t40;
        const auto t950 = t49 * t777 + t948 + t949;
        const auto t951 = t17 * t837;
        const auto t952 = t290 * t67;
        const auto t953 = t288 * t70;
        const auto t954 = t11 * t248;
        const auto t955 = t238 * t42 + t70 * t742 + t737 * t954 + t783;
        const auto t956 = t2 * t306;
        const auto t957 = t298 * t40 + t67 * t786 + t737 * t956 + t739;
        const auto t958 = t196 * t304;
        const auto t959 = t2 * t303;
        const auto t960 = t11 * t243;
        const auto t961 = t47 * t777 + t959 + t960;
        const auto t962 = t17 * t835;
        const auto t963 = t2 * t715;
        const auto t964 = t11 * t244;
        const auto t965 = t238 * t303 + t755 * t954 + t758 * t960 + t795;
        const auto t966 = t243 * t298 + t755 * t956 + t756 + t797 * t959;
        const auto t967 = 2 * t400;
        const auto t968 = t291 * t67;
        const auto t969 = t244 * t70;
        const auto t970 = t2 * t249;
        const auto t971 = t11 * t288;
        const auto t972 = t486
            * (3 * t14 * t55 * t707 * t710 * t770
               + t16 * t473 * t55 * t710 * t75 * t772
               + t16 * t473 * t55 * t716 * t75 * t770
               + t16
                   * (t244 * t947 + t288 * t958 + t304 * t968 + t307 * t971
                      + t336 * t970 + t338 * t969 + t48 * t955 + t50 * t965
                      + t52 * t957 + t54 * t966 + t933 * t950 + t961 * t967)
               + 3 * t55 * t713 * t716 * t772 - t722 * t946 - t723 * t944
               - t746
                   * (t288 * t947 - t336 * t952 - t337 * t953 - t66 * t955
                      - t69 * t957 - t950 * t951)
               - t762
                   * (t244 * t958 - t304 * t963 - t478 * t966 - t480 * t965
                      - t771 * t964 - t961 * t962)
               - t773 * t945 - t774 * t943);
        const auto t973 = t16 * t716 * t822;
        const auto t974 = t14 * t710 * t827;
        const auto t975 = t2 * t22;
        const auto t976 = t243 * t5;
        const auto t977 = 4 * t5;
        const auto t978 = t47 * t977 + t975 + t976;
        const auto t979 = t17 * t364;
        const auto t980 = t21 - t360;
        const auto t981 = t22 * t238 + t248 * t811 + t758 * t976 + t980;
        const auto t982 = t2 * t813 + t244 + t363 * t67 + t747 * t810;
        const auto t983 = t2 * t345;
        const auto t984 = t40 * t5 + t49 * t977 + t983;
        const auto t985 = t248 * t737;
        const auto t986 = -t371 + t818;
        const auto t987 = t238 * t345 + t5 * t985 + t741 * t810 + t986;
        const auto t988 = t2 * t807 + t288 + t727 * t810 + t754 * t983;
        const auto t989 = t379 * t7;
        const auto t990 = t486
            * (t16 * t722 * t75 * t820
               + t16
                   * (t244 * t989 + t288 * t979 + t36 * t988 + t361 * t764
                      + t361 * t968 + t372 * t763 + t372 * t970 + t46 * t982
                      + t48 * t987 + t50 * t981 + t696 * t984 + t765 * t978)
               + t473 * t722 * t824
               - t746
                   * (t372 * t764 - t372 * t952 + t379 * t708 - t66 * t987
                      - t72 * t988 - t732 * t984)
               - t762
                   * (t244 * t979 + t361 * t763 - t361 * t963 - t476 * t982
                      - t480 * t981 - t751 * t978)
               - t820 * t973 - t820 * t974 - t826 * t943 - t830 * t945
               - t831 * t943 - t833 * t945);
        const auto t991 = t288 * t407;
        const auto t992 = t244 * t399;
        const auto t993 = 8 * t402;
        const auto t994 = t1 * t993;
        const auto t995 = t16 * t281 + 2 * t401 + t808 + t994;
        const auto t996 = t27 * t355;
        const auto t997 = t14 * t856;
        const auto t998 = t243 * t854 + t814 + t996 + t997;
        const auto t999 = t1 * t34;
        const auto t1000 = t1 * t931 + t136 * t243 + t237 * t26 + t244 - t999;
        const auto t1001 = t355 * t50;
        const auto t1002 = t1 * t44;
        const auto t1003 =
            t1 * t248 * t817 - t1002 + t237 * t388 + t287 + t67 * t92 + t860;
        const auto t1004 = t355 * t48;
        const auto t1005 = t16 * t75;
        const auto t1006 = t483 * t945;
        const auto t1007 = t196 * t244;
        const auto t1008 = t196 * t288;
        const auto t1009 = t7 * t72;
        const auto t1010 = t486
            * (t1000 * t1001 + t1003 * t1004 + t1005 * t722 * t840
               - t1006 * t841 + t14 * t291 * t410 + t188 * t995 + t192 * t991
               + t192 * t992 + t249 * t403 + t272 * t995 + t306 * t991
               + t306 * t992 + t374 * t998 + t400 * t998 + t723 * t841
               - t746
                   * (-t1003 * t779 + t1008 * t399 - t1009 * t995 - t290 * t403
                      + t399 * t764 - t709 * t995)
               - t762
                   * (-t1000 * t792 + t1007 * t407 + t407 * t763 - t410 * t715
                      - t479 * t998 - t839 * t998)
               - t840 * t973 - t840 * t974 - t842 * t943 - t844 * t945
               - t845 * t943);
        const auto t1011 = t2 * t30;
        const auto t1012 = t10 * t243;
        const auto t1013 = t1011 + t1012 + t47 * t877;
        const auto t1014 = t1012 * t758 + t238 * t30 + t248 * t883 + t880;
        const auto t1015 = t14 * t884;
        const auto t1016 = t1015 * t2 + t244 + t446 * t67 + t882 * t960;
        const auto t1017 = t2 * t429;
        const auto t1018 = t10 * t40 + t1017 + t49 * t877;
        const auto t1019 = t10 * t985 + t238 * t429 + t741 * t882 + t889;
        const auto t1020 = t1017 * t797 + t288 + t882 * t949 + t891 * t993;
        const auto t1021 = t486
            * (-t1006 * t870 + t16 * t722 * t75 * t869
               + t16
                   * (t1007 * t439 + t1008 * t445 + t1013 * t967 + t1014 * t50
                      + t1016 * t54 + t1018 * t933 + t1019 * t48 + t1020 * t52
                      + t439 * t970 + t442 * t964 + t445 * t968 + t447 * t953)
               + t473 * t722 * t870
               - t746
                   * (t1008 * t439 - t1018 * t951 - t1019 * t66 - t1020 * t69
                      - t439 * t952 + t442 * t971)
               - t762
                   * (t1007 * t445 - t1013 * t962 - t1014 * t480 - t1016 * t478
                      - t445 * t963 + t447 * t969)
               - t869 * t973 - t869 * t974 - t871 * t943 - t873 * t945
               - t874 * t943);
        const auto t1022 = t469 * t938;
        const auto t1023 = t468 * t936;
        const auto t1024 = t17 * t466;
        const auto t1025 = t1024 * t288 + t294 + t295 + t426;
        const auto t1026 = t292 * t899;
        const auto t1027 = t1026 * t469;
        const auto t1028 = t291 * t471;
        const auto t1029 = t1 + t16 * std::pow(t2, 3);
        const auto t1030 = t292 * t902;
        const auto t1031 = -t1026 * t582 + t1030 * t475 - t475 * t936
            - t582 * t938 + t583 * t939;
        const auto t1032 = t651 * t938;
        const auto t1033 = t491 * t936;
        const auto t1034 = -t551;
        const auto t1035 = t1034 - t549;
        const auto t1036 = -t620;
        const auto t1037 = t1036 + t488 * t953 + t622 + t623;
        const auto t1038 = t1026 * t651;
        const auto t1039 = -t124;
        const auto t1040 = -t63;
        const auto t1041 = t1040 + t594;
        const auto t1042 = -t140;
        const auto t1043 = -t498;
        const auto t1044 = t1043 + t613;
        const auto t1045 = t353 * t544;
        const auto t1046 = t313 * t353;
        const auto t1047 = -t1045 - t1046 + t497 + t588 + t627 + t628;
        const auto t1048 = t353 * t550;
        const auto t1049 = t353 * t666;
        const auto t1050 = -t1048 - t1049 + t139 + t163;
        const auto t1051 = t1050 + t336 * t598 + t336 * t616;
        const auto t1052 = t14 * std::pow(t336, 2);
        const auto t1053 = t685 * (2 * t11 * t42 + 4 * t12 * t16 * t43 - t43);
        const auto t1054 = std::pow(t10, 3) * t214;
        const auto t1055 = -t1054 * t687 + 3 * t113;
        const auto t1056 = t1055 + t114 + t42 * t432;
        const auto t1057 = t14 * std::pow(t304, 2);
        const auto t1058 = t306 * t692;
        const auto t1059 = t1058 + t194 + 2 * t305;
        const auto t1060 = t1054 * t697;
        const auto t1061 = p0z - 3 * p1z - 3 * t10 * t60 * t84 + t1060
            - 2 * t303 * t58 * t60 + t428;
        const auto t1062 = t225 * t334;
        const auto t1063 = t339 * t355;
        const auto t1064 = t151 * t322;
        const auto t1065 = t206 * t322;
        const auto t1066 = t324 * t327;
        const auto t1067 = t60 * (-t119 * t687 + t330 + 3 * t93);
        const auto t1068 = t215 + 2 * t332 - t353 * t697;
        const auto t1069 = t820 * t822;
        const auto t1070 = t16 * t772;
        const auto t1071 = t14 * t770 * t827;
        const auto t1072 = t11 * t22;
        const auto t1073 = t303 * t5;
        const auto t1074 = t1072 + t1073 + t51 * t977;
        const auto t1075 = t11 * t771;
        const auto t1076 = t1073 * t797 + t22 * t298 + t306 * t811 + t980;
        const auto t1077 = t11 * t813 + t304 + t363 * t70 + t789 * t810;
        const auto t1078 = t11 * t345;
        const auto t1079 = t1078 + t42 * t5 + t53 * t977;
        const auto t1080 = t337 * t70;
        const auto t1081 = t306 * t737;
        const auto t1082 = t1081 * t5 + t298 * t345 + t785 * t810 + t986;
        const auto t1083 = t1078 * t754 + t11 * t807 + t336 + t775 * t810;
        const auto t1084 = t338 * t70;
        const auto t1085 = t11 * t307;
        const auto t1086 = t486
            * (-t1069 * t1070 - t1071 * t820 + t16 * t75 * t773 * t820
               + t16
                   * (t1074 * t801 + t1076 * t54 + t1077 * t46 + t1079 * t695
                      + t1082 * t52 + t1083 * t36 + t1084 * t361 + t1085 * t372
                      + t304 * t989 + t336 * t979 + t361 * t800 + t372 * t799)
               + t473 * t773 * t824
               - t746
                   * (-t1079 * t780 - t1080 * t372 - t1082 * t69 - t1083 * t72
                      + t372 * t800 + t379 * t768)
               - t762
                   * (-t1074 * t793 - t1075 * t361 - t1076 * t478 - t1077 * t476
                      + t304 * t979 + t361 * t799)
               - t826 * t944 - t830 * t946 - t831 * t944 - t833 * t946);
        const auto t1087 = t1070 * t822;
        const auto t1088 = t483 * t946;
        const auto t1089 = t11 * t26;
        const auto t1090 = t1 * t303;
        const auto t1091 = t1089 + t1090 + t51 * t849;
        const auto t1092 = t410 * t67;
        const auto t1093 = t1090 * t797 + t26 * t298 + t306 * t855 + t852;
        const auto t1094 = t11 * t997 + t304 + t409 * t70 + t854 * t959;
        const auto t1095 = t11 * t388;
        const auto t1096 = t1 * t42 + t1095 + t53 * t849;
        const auto t1097 = t2 * t403;
        const auto t1098 = t1 * t1081 + t298 * t388 + t785 * t854 + t861;
        const auto t1099 = t1095 * t758 + t11 * t994 + t336 + t854 * t948;
        const auto t1100 = t486
            * (-t1071 * t840 - t1087 * t840 - t1088 * t841
               + t16 * t75 * t773 * t840
               + t16
                   * (t1084 * t407 + t1085 * t399 + t1091 * t967 + t1092 * t336
                      + t1093 * t54 + t1094 * t50 + t1096 * t933 + t1097 * t304
                      + t1098 * t52 + t1099 * t48 + t399 * t958 + t407 * t947)
               + t473 * t773 * t841
               - t746
                   * (-t1080 * t399 - t1096 * t951 + t1097 * t336 - t1098 * t69
                      - t1099 * t66 + t399 * t947)
               - t762
                   * (-t1075 * t407 - t1091 * t962 + t1092 * t304 - t1093 * t478
                      - t1094 * t480 + t407 * t958)
               - t842 * t944 - t844 * t946 - t845 * t944);
        const auto t1101 = t336 * t445;
        const auto t1102 = t304 * t439;
        const auto t1103 = 8 * t10 * t441;
        const auto t1104 = t1103 + t16 * t329 + 2 * t440 + t808;
        const auto t1105 = t31 * t355;
        const auto t1106 = t1015 + t1105 + t303 * t882 + t814;
        const auto t1107 =
            t10 * t1058 - t10 * t34 + t160 * t303 + t297 * t30 + t304;
        const auto t1108 = t355 * t54;
        const auto t1109 = t10 * t306 * t817 - t10 * t44 + t297 * t429 + t335
            + t70 * t93 + t888;
        const auto t1110 = t355 * t52;
        const auto t1111 = t486
            * (t1005 * t773 * t869 - t1071 * t869 - t1087 * t869 - t1088 * t870
               + t1101 * t192 + t1101 * t248 + t1102 * t192 + t1102 * t248
               + t1104 * t187 + t1104 * t272 + t1106 * t373 + t1106 * t400
               + t1107 * t1108 + t1109 * t1110 + t14 * t338 * t447 + t307 * t442
               - t746
                   * (-t1009 * t1104 - t1104 * t769 - t1109 * t731 - t337 * t442
                      + t439 * t800 + t439 * t947)
               - t762
                   * (-t1106 * t576 - t1106 * t839 - t1107 * t750 + t445 * t799
                      + t445 * t958 - t447 * t771)
               + t774 * t870 - t871 * t944 - t873 * t946 - t874 * t944);
        const auto t1112 = t1064 * t469;
        const auto t1113 = t1062 * t468;
        const auto t1114 = t1024 * t336 + t341 + t342 + t463;
        const auto t1115 = t339 * t899;
        const auto t1116 = t1115 * t469;
        const auto t1117 = t1064 * t582;
        const auto t1118 = t1062 * t475;
        const auto t1119 = t471 * t67;
        const auto t1120 = t1035 + t1119 * t336 + t553 + t554;
        const auto t1121 = t1115 * t582;
        const auto t1122 = t338 * t488;
        const auto t1123 = t10 + std::pow(t11, 3) * t16;
        const auto t1124 = t339 * t902;
        const auto t1125 = -t1062 * t491 - t1064 * t651 + t1065 * t652
            - t1115 * t651 + t1124 * t491;
        const auto t1126 = -t143;
        const auto t1127 = -t130;
        const auto t1128 = t5 * t558 + t5 * t632;
        const auto t1129 = t206 * t60;
        const auto t1130 = t309 * t372;
        const auto t1131 = t251 * t372;
        const auto t1132 = t179 * t90 + t556 + t89;
        const auto t1133 = t14 * t228;
        const auto t1134 = std::pow(t361, 2);
        const auto t1135 = 4 * t16 * t33 * t6 + 2 * t22 * t5 - t33;
        const auto t1136 = t151 * t355;
        const auto t1137 = t14 * std::pow(t372, 2);
        const auto t1138 = t15 * t817;
        const auto t1139 = -t44 - 1;
        const auto t1140 = t1138 + t1139 + t345 * t810;
        const auto t1141 = -3 * p2x + p3x + t189;
        const auto t1142 = t206 * t355;
        const auto t1143 = -t117 * t697 + 3 * t80 + t83;
        const auto t1144 = -t253 * t687 - 2 * t346 - t348;
        const auto t1145 = t1069 * t14;
        const auto t1146 = t1005 * t830;
        const auto t1147 = t1005 * t820;
        const auto t1148 = t762 * t832;
        const auto t1149 = t473 * t762;
        const auto t1150 = t1149 * t820;
        const auto t1151 = t1 * t22 + t26 * t5 + t359 * t849;
        const auto t1152 = t1 * t813 + t363 * t506 + t407 + t810 * t847;
        const auto t1153 = t14 * t811;
        const auto t1154 = t1153 * t863 + t27 * t810 + t361 + t854 * t975;
        const auto t1155 = t388 * t5;
        const auto t1156 = t1 * t345;
        const auto t1157 = t1155 + t1156 + t370 * t849;
        const auto t1158 = t1 * t807 + t1156 * t754 + t399 + t810 * t858;
        const auto t1159 = t1155 * t758 + t372 + t5 * t994 + t854 * t983;
        const auto t1160 = t486
            * (t1145 * t840 + t1146 * t840 + t1147 * t844 + t1148 * t840
               + t1150 * t841
               + t16
                   * (t1092 * t372 + t1097 * t361 + t1151 * t765 + t1152 * t46
                      + t1154 * t50 + t1157 * t696 + t1158 * t36 + t1159 * t48
                      + t361 * t866 + t372 * t865 + t399 * t979 + t407 * t989)
               - t746
                   * (t1097 * t372 - t1157 * t732 - t1158 * t72 - t1159 * t66
                      + t372 * t866 + t399 * t989)
               - t762
                   * (t1092 * t361 - t1151 * t751 - t1152 * t476 - t1154 * t480
                      + t361 * t865 + t407 * t979)
               + t826 * t841 + t831 * t841 + t832 * t844);
        const auto t1161 = t10 * t22 + t30 * t5 + t359 * t877;
        const auto t1162 = t447 * t70;
        const auto t1163 = t10 * t813 + t363 * t592 + t445 + t810 * t875;
        const auto t1164 = t1072 * t882 + t1153 * t891 + t31 * t810 + t361;
        const auto t1165 = t429 * t5;
        const auto t1166 = t10 * t345;
        const auto t1167 = t1165 + t1166 + t370 * t877;
        const auto t1168 = t11 * t442;
        const auto t1169 = t10 * t807 + t1166 * t754 + t439 + t810 * t886;
        const auto t1170 = t1078 * t882 + t1103 * t5 + t1165 * t797 + t372;
        const auto t1171 = t486
            * (t1145 * t869 + t1146 * t869 + t1147 * t873 + t1148 * t869
               + t1150 * t870
               + t16
                   * (t1161 * t801 + t1162 * t372 + t1163 * t46 + t1164 * t54
                      + t1167 * t695 + t1168 * t361 + t1169 * t36 + t1170 * t52
                      + t361 * t894 + t372 * t893 + t439 * t979 + t445 * t989)
               - t746
                   * (-t1167 * t780 + t1168 * t372 - t1169 * t72 - t1170 * t69
                      + t372 * t894 + t439 * t989)
               - t762
                   * (-t1161 * t793 + t1162 * t361 - t1163 * t476 - t1164 * t478
                      + t361 * t893 + t445 * t979)
               + t826 * t870 + t831 * t870 + t832 * t873);
        const auto t1172 = t469 * t560;
        const auto t1173 = t225 * t369;
        const auto t1174 = t1173 * t468;
        const auto t1175 = t1130 + t1131 + t122 * t754 + t379 * t466;
        const auto t1176 = t151 * t16;
        const auto t1177 = t1176 * t381;
        const auto t1178 = t1177 * t469;
        const auto t1179 = t380 * t682;
        const auto t1180 = t365 * t381;
        const auto t1181 = t1119 * t372 + t562;
        const auto t1182 = t1119 * t361 - t1173 * t475 - t1177 * t582
            + t1179 * t475 + t1180 * t583 + t361 * t515 - t560 * t582
            + t79 * t829;
        const auto t1183 = t488 * t70;
        const auto t1184 = t1183 * t372 + t636;
        const auto t1185 = -t1173 * t491 - t1177 * t651 + t1179 * t491
            + t1180 * t652 + t1183 * t361 + t159 * t829 + t361 * t598
            - t560 * t651;
        const auto t1186 = t103 + t104 * t391 + t417;
        const auto t1187 = t271 + t414 + t424;
        const auto t1188 = t519 + t910;
        const auto t1189 = -t127;
        const auto t1190 = t1 * t419 + t1 * t638;
        const auto t1191 = t399 * t534;
        const auto t1192 = t399 * t515;
        const auto t1193 = std::pow(t407, 2);
        const auto t1194 = 2 * t1 * t26 + 4 * t16 * t33 * t57 - t33;
        const auto t1195 = t14 * std::pow(t399, 2);
        const auto t1196 = t495 * t817;
        const auto t1197 = t1139 + t1196 + t388 * t854;
        const auto t1198 = -3 * p2y + p3y + t242;
        const auto t1199 = -t118 * t697 + t392 + 3 * t81;
        const auto t1200 = -t348 - t350 * t687 - 2 * t394;
        const auto t1201 = t196 * t407;
        const auto t1202 = t1 * t30 + t10 * t26 + t877 * t999;
        const auto t1203 = t10 * t996 + t10 * t997 + t1011 * t854 + t445;
        const auto t1204 = t1089 * t882 + t14 * t855 * t891 + t31 * t854 + t407;
        const auto t1205 = t196 * t399;
        const auto t1206 = t1 * t429;
        const auto t1207 = t10 * t388;
        const auto t1208 = t1002 * t877 + t1206 + t1207;
        const auto t1209 = t10 * t994 + t1017 * t854 + t1207 * t758 + t439;
        const auto t1210 = t1 * t1103 + t1095 * t882 + t1206 * t797 + t399;
        const auto t1211 = t486
            * (t1005 * t840 * t873 + t1005 * t844 * t869 + t1149 * t840 * t870
               + t1149 * t841 * t869 + t14 * t822 * t840 * t869
               + t16
                   * (t1092 * t439 + t1097 * t445 + t1162 * t399 + t1168 * t407
                      + t1201 * t439 + t1202 * t967 + t1203 * t50 + t1204 * t54
                      + t1205 * t445 + t1208 * t933 + t1209 * t48 + t1210 * t52)
               - t746
                   * (t1097 * t439 + t1168 * t399 + t1205 * t439 - t1208 * t951
                      - t1209 * t66 - t1210 * t69)
               - t762
                   * (t1092 * t445 + t1162 * t407 + t1201 * t445 - t1202 * t962
                      - t1203 * t480 - t1204 * t478)
               + t841 * t874 + t842 * t870 + t845 * t870);
        const auto t1212 = t1024 * t399 + t1187;
        const auto t1213 = t225 * t396;
        const auto t1214 = t1176 * t411;
        const auto t1215 = t404 * t682;
        const auto t1216 = t365 * t411;
        const auto t1217 = t1024 * t407 - t1213 * t468 - t1214 * t469
            + t1215 * t468 + t1216 * t901 + t251 * t407 - t469 * t568
            + t79 * t843;
        const auto t1218 = -t497;
        const auto t1219 = t1214 * t582;
        const auto t1220 = t568 * t582;
        const auto t1221 = t1213 * t475;
        const auto t1222 = -t139;
        const auto t1223 = t1191 + t1192 + t138 * t758 + t403 * t471;
        const auto t1224 = t1183 * t399 + t640;
        const auto t1225 = t1183 * t407 - t1213 * t491 - t1214 * t651
            + t1215 * t491 + t1216 * t652 + t407 * t616 + t503 * t843
            - t568 * t651;
        const auto t1226 = t111 + t112 * t432 + t455;
        const auto t1227 = t321 + t452 + t461;
        const auto t1228 = t1039 + t602;
        const auto t1229 = t1042 + t620;
        const auto t1230 = t10 * t457 + t10 * t570;
        const auto t1231 = t439 * t616;
        const auto t1232 = t439 * t598;
        const auto t1233 = std::pow(t445, 2);
        const auto t1234 = 2 * t10 * t30 + 4 * t16 * t33 * t58 - t33;
        const auto t1235 = t14 * std::pow(t439, 2);
        const auto t1236 = t586 * t817;
        const auto t1237 = t1139 + t1236 + t429 * t882;
        const auto t1238 = -3 * p2z + p3z + t302;
        const auto t1239 = -t119 * t697 + t433 + 3 * t82;
        const auto t1240 = -t348 - t353 * t687 - 2 * t435;
        const auto t1241 = t1024 * t439 + t1227;
        const auto t1242 = t225 * t437;
        const auto t1243 = t1176 * t448;
        const auto t1244 = t443 * t682;
        const auto t1245 = t365 * t448;
        const auto t1246 = t1024 * t445 - t1242 * t468 - t1243 * t469
            + t1244 * t468 + t1245 * t901 + t159 * t872 + t309 * t445
            - t469 * t572;
        const auto t1247 = t1119 * t439 + t573;
        const auto t1248 = t1119 * t445 - t1242 * t475 - t1243 * t582
            + t1244 * t475 + t1245 * t583 + t445 * t534 + t503 * t872
            - t572 * t582;
        const auto t1249 = t1243 * t651;
        const auto t1250 = t572 * t651;
        const auto t1251 = t1242 * t491;
        const auto t1252 = t1231 + t1232 + t162 * t797 + t442 * t488;
        const auto t1253 = t211 * t5;
        const auto t1254 = t354 * t60;
        const auto t1255 = t17 * t50;
        const auto t1256 = t17 * t299;
        const auto t1257 = t413 * t48;
        const auto t1258 = t450 * t52;
        const auto t1259 = t192 * t3;
        const auto t1260 = t12 * t192;
        const auto t1261 = 3 * t483;
        const auto t1262 = t485 / std::pow(t76, 3.0 / 2.0);
        const auto t1263 = t234 / std::pow(t155, 3.0 / 2.0);
        const auto t1264 = t1263
            * (3 * t116 * t225 * t468 * t475 - t116 * t79 * (t120 + t196 + t237)
               - t468 * t582 - t469 * t475);
        const auto t1265 = t297 - 2;
        const auto t1266 = t1263
            * (-t116 * t159 * (t118 + t1265 + t196)
               + 3 * t116 * t225 * t468 * t491 - t468 * t651 - t469 * t491);
        const auto t1267 = t46 * t67;
        const auto t1268 = t1 * t278;
        const auto t1269 = t416 * t60;
        const auto t1270 = -t545;
        const auto t1271 = t299 * t67;
        const auto t1272 = t186 * t36;
        const auto t1273 = t12 * t248;
        const auto t1274 = t1263
            * (3 * t116 * t225 * t475 * t491
               - t116 * t503 * (t117 + t1265 + t237) - t475 * t651
               - t491 * t582);
        const auto t1275 = t46 * t70;
        const auto t1276 = t50 * t70;
        const auto t1277 = t10 * t327;
        const auto t1278 = t454 * t60;
        hess[0] = t78
            * (3 * t55 * std::pow(t65, 2) * t75
               - t55 * (t12 * t15 + t15 * t3 + std::pow(t19, 2))
               - 2 * t65 * t73);
        hess[1] = t158;
        hess[2] = t173;
        hess[3] = t235
            * (-t206 * (t186 * t187 + t186 * t188 + t205) + t233
               + t60
                   * (-t118 * t176 - t119 * t176 + t147 * t184 + t177 * t178
                      + t180 * t181 + t180 * t182));
        hess[4] =
            t235 * (-t206 * (t236 * t240 + t262) + t239 * t263 + t271 + t296);
        hess[5] =
            t235 * (-t206 * (t300 * t52 + t317) + t300 * t54 + t321 + t343);
        hess[6] = t235
            * (2 * t147 * t279 + t147 * t349 + t280 * t354 + t328 * t354
               + t350 * t352 + t352 * t353
               - t365
                   * (-t11 * t159 * t361 + t19 * t364 + t198 * t362
                      - t2 * t361 * t79 + t356 * t357 + t356 * t358)
               + t385);
        hess[7] = t235
            * (t152 * t396 + t226 * t404 + t246 * t399 + t263 * t413
               + t382 * t411 - t414 + t421 * (t418 + t420) - t422 - t423 - t424
               + t426);
        hess[8] = t235
            * (t152 * t437 + t226 * t443 + t236 * t451 + t246 * t439
               + t382 * t448 + t421 * (t456 + t458) - t452 - t459 - t460 - t461
               + t463);
        hess[9] = t470;
        hess[10] = t487;
        hess[11] = t494;
        hess[12] = t158;
        hess[13] = t78
            * (3 * std::pow(t500, 2) * t55 * t75 - 2 * t500 * t502
               - t55 * (t12 * t495 + std::pow(t137, 2) + t495 * t8));
        hess[14] = t505;
        hess[15] = t235
            * (-t206 * (t506 * t509 + t512 + t517) + t46 * t506 * t508 + t520
               + t531);
        hess[16] = t235
            * (-t206
                   * (t137 * t249 + t188 * t413 + t272 * t413
                      + t355 * t48 * t537 - t535 - t536)
               + t273 * t521 + t286 * t522 + t292 * t524 - t538 - t539
               + t60
                   * (-t117 * t532 - t119 * t532 + t133 * t283 + t177 * t415
                      + t181 * t533 + t182 * t415));
        hess[17] =
            t235 * (-t206 * (t52 * t540 + t548) + t54 * t540 + t552 + t555);
        hess[18] = t235
            * (t1 * t116 * t151 * t60 * (t557 + t559) + t1 * t16 * t186 * t46
               - t520 - t563);
        hess[19] = t235
            * (t133 * t181 * t567 + t133 * t395 + t253 * t564 + t279 * t416
               + t280 * t416 + t353 * t564
               - t365
                   * (-t11 * t407 * t503 + t137 * t410 + t312 * t566
                      + t358 * t565 - t407 * t7 * t79 + t408 * t565)
               + t569);
        hess[20] = t235
            * (t1 * t116 * t151 * t60 * (t456 + t571) + t1 * t16 * t450 * t54
               - t574);
        hess[21] = t580;
        hess[22] = t584;
        hess[23] = t585;
        hess[24] = t173;
        hess[25] = t505;
        hess[26] = t78
            * (3 * t55 * std::pow(t590, 2) * t75
               - t55 * (std::pow(t161, 2) + t3 * t586 + t586 * t8)
               - 2 * t590 * t591);
        hess[27] = t235
            * (-t206 * (t509 * t592 + t595 + t600) + t46 * t508 * t592 + t603
               + t612);
        hess[28] = t235
            * (-t206 * (t240 * t592 + t614 + t619) + t239 * t50 * t592 + t621
               + t624);
        hess[29] = t235
            * (-t206
                   * (t161 * t307 + t187 * t450 + t272 * t450
                      + t355 * t52 * t629 - t627 - t628)
               + t322 * t604 + t334 * t605 + t339 * t607
               + t60
                   * (-t117 * t625 - t118 * t625 + t170 * t331 + t177 * t453
                      + t181 * t453 + t182 * t626)
               - t630 - t631);
        hess[30] = t235
            * (t10 * t116 * t151 * t60 * (t557 + t633) + t10 * t16 * t186 * t46
               - t603 - t637);
        hess[31] = t235
            * (t10 * t116 * t151 * t60 * (t418 + t639) + t10 * t16 * t413 * t50
               - t621 - t641);
        hess[32] = t235
            * (t170 * t182 * t567 + t170 * t436 + t253 * t642 + t279 * t454
               + t328 * t454 + t350 * t642
               - t365
                   * (-t159 * t445 * t7 + t161 * t447 + t256 * t645
                      + t357 * t643 + t408 * t643 - t445 * t644)
               + t646);
        hess[33] = t649;
        hess[34] = t650;
        hess[35] = t653;
        hess[36] = t235
            * (t19 * t526 + t197 * t660 - t206 * (t205 + t659) + t233 - t661
               - t662 + t670);
        hess[37] = t235 * (-t206 * (t517 + t672) + t531 + t675);
        hess[38] = t235 * (-t206 * (t600 + t677) + t612 + t680);
        hess[39] = t235
            * (-t206
                   * (t12 * t691 + std::pow(t195, 2) - t198 * t60 * t699
                      + t3 * t691 + t694 * t695 + t694 * t696)
               + std::pow(t207, 2) * t228 - 2 * t217 * t702
               + std::pow(t224, 2) * t683 + t225 * t701 * t703
               - t230
                   * (t12 * t684 + t14 * std::pow(t223, 2) + t220 * t686
                      + t221 * t686 + t3 * t684 + t46 * t567 * t690)
               + t567
                   * (t118 * t704 + t119 * t704 + t130 * t705 + t143 * t705
                      + t181 * t706 + t182 * t706 + t184 * t216 + t690 * t90
                      - t699 * t99)
               - t700 * t701);
        hess[40] = t767;
        hess[41] = t803;
        hess[42] = t838;
        hess[43] = t868;
        hess[44] = t895;
        hess[45] = t235
            * (t195 * t466 + t199 * t897 + t203 + t204
               - t230 * (t16 * t896 + t660 * t897 + t661 + t662 + t898) + t588
               + t655 - t657 - t658 + t903);
        hess[46] = t235 * (-t230 * (t426 + t673 + t905) - t672 - t906);
        hess[47] = t235 * (-t230 * (t463 + t678 + t908) - t677 - t909);
        hess[48] = t235 * (-t206 * (t262 + t912) + t296 + t674 + t910);
        hess[49] = t235
            * (t135 * t151 * t286 + t137 * t16 * t291 + t142 * t151 * t273
               + t142 * t16 * t225 * t292 + 2 * t16 * t50 * t537
               - t206 * (t137 * t249 + 2 * t16 * t48 * t537 - t915) - t538
               - t539 - t919);
        hess[50] = t235 * (-t206 * (t619 + t921) + t624 + t924);
        hess[51] = t767;
        hess[52] = t235
            * (-t206
                   * (t12 * t930 + std::pow(t249, 2) - t312 * t60 * t935
                      + t696 * t932 + t8 * t930 + t932 * t933)
               + t225 * t937 * t939 + t228 * std::pow(t273, 2)
               - t230
                   * (t12 * t925 + t14 * std::pow(t291, 2) + t221 * t926
                      + t289 * t926 + t50 * t567 * t929 + t8 * t925)
               - 2 * t286 * t938 + std::pow(t292, 2) * t683
               + t567
                   * (t104 * t929 - t107 * t935 + t117 * t940 + t119 * t940
                      + t127 * t941 + t130 * t941 + t177 * t942 + t182 * t942
                      + t283 * t285)
               - t936 * t937);
        hess[53] = t972;
        hess[54] = t990;
        hess[55] = t1010;
        hess[56] = t1021;
        hess[57] = t235
            * (-t1022 - t1023 - t1027 + t116 * t151 * t225 * t273 * t468
               + 3 * t116 * t16 * t292 * t468 * t681 + t16 * t244 * t466 * t7
               - t230 * (t1025 + t123 + t904) - t261 - t912);
        hess[58] = t235
            * (t1004 * t1029 + t1031
               - t230 * (t1001 * t1029 + t1028 * t16 + t919) + t249 * t471
               + t915);
        hess[59] = t235
            * (-t1032 - t1033 - t1038 + t11 * t16 * t244 * t488
               + t116 * t151 * t225 * t273 * t491
               + 3 * t116 * t16 * t292 * t491 * t681
               - t230 * (t1035 + t1037 + t922) - t618 - t921);
        hess[60] = t235 * (t1039 - t206 * (t1041 + t317) + t343 + t679);
        hess[61] = t235 * (t1042 - t206 * (t1044 + t548) + t555 + t923);
        hess[62] = t235
            * (-t1051 + t151 * t165 * t322 + t151 * t172 * t334
               + t16 * t161 * t338 + t16 * t165 * t225 * t339
               + 2 * t16 * t54 * t629
               - t206 * (-t1047 + 2 * t16 * t52 * t629 + t161 * t307) - t630
               - t631);
        hess[63] = t803;
        hess[64] = t972;
        hess[65] = t235
            * (-t1062 * t1063 + t1063 * t1065 * t225 - 2 * t1064 * t334
               - t206
                   * (t1057 * t3 + t1057 * t8 + t1059 * t695 + t1059 * t933
                      - t1061 * t256 * t60 + std::pow(t307, 2))
               + t228 * std::pow(t322, 2)
               - t230
                   * (t1052 * t3 + t1052 * t8 + t1053 * t220 + t1053 * t289
                      + t1056 * t54 * t567 + t14 * std::pow(t338, 2))
               + std::pow(t339, 2) * t683
               + t567
                   * (t1056 * t112 - t1061 * t115 + t1066 * t117 + t1066 * t118
                      + t1067 * t127 + t1067 * t143 + t1068 * t177
                      + t1068 * t181 + t331 * t333));
        hess[66] = t1086;
        hess[67] = t1100;
        hess[68] = t1111;
        hess[69] = t235
            * (-t1041 - t1112 - t1113 - t1116 + t116 * t151 * t225 * t322 * t468
               + 3 * t116 * t16 * t339 * t468 * t681 + t16 * t304 * t466 * t7
               - t230 * (t1114 + t124 + t907) - t316);
        hess[70] = t235
            * (-t1044 - t1117 - t1118 - t1121 + t116 * t151 * t225 * t322 * t475
               + 3 * t116 * t16 * t339 * t475 * t681 + t16 * t2 * t304 * t471
               - t230 * (t1036 + t1120 + t140) - t547);
        hess[71] = t235
            * (t1047 + t1110 * t1123 + t1125
               - t230 * (t1051 + t1108 * t1123 + t1122 * t16) + t307 * t488);
        hess[72] = t235
            * (t1129
                   * (t1126 + t1127 + t1128 + t130 * t179 + t143 * t179
                      - t147 * t368 - t148 * t267)
               - t1130 - t1131 + t19 * t379 + t362 * t660 + t385 + t898);
        hess[73] =
            t235 * (t1 * t116 * t151 * t60 * (t1132 + t559) - t563 - t675);
        hess[74] =
            t235 * (t10 * t116 * t151 * t60 * (t1132 + t633) - t637 - t680);
        hess[75] = t838;
        hess[76] = t990;
        hess[77] = t1086;
        hess[78] = t235
            * (t1133 * std::pow(t381, 2) - t1136 * t369 * t381
               + t1142 * t381 * t560
               - t230
                   * (t1137 * t12 + t1137 * t3 + t1140 * t765 + t1140 * t801
                      + std::pow(t379, 2)
                      + t660 * (t1138 * t7 + t1141 + t185 * t345 + t371 - t45))
               - t365
                   * (t1134 * t237 + t1134 * t297 + t1135 * t695 + t1135 * t696
                      + t16 * std::pow(t364, 2)
                      + t198 * (t15 * t692 * t7 + t236 * t363 - t35 + t361))
               - 2 * t369 * t560 + std::pow(t380, 2) * t682
               + t567
                   * (t1143 * t280 + t1143 * t328 + t1144 * t130 + t1144 * t143
                      + t349 * t368 - t351 * t558 - t351 * t632
                      + t90 * (t1141 - t179 * t345 + t689)
                      + t99 * (t179 * t21 - t698 + 3 * t86 + t88)));
        hess[79] = t1160;
        hess[80] = t1171;
        hess[81] = t235
            * (-t1172 - t1174 - t1178 + t1179 * t468 + t1180 * t901
               - t230 * (t1175 + t670) + t251 * t361 + t309 * t361 + t466 * t829
               + t56 * t754 + t659);
        hess[82] = t235 * (t1182 - t230 * (t1181 + t675) + t672);
        hess[83] = t235 * (t1185 - t230 * (t1184 + t680) + t677);
        hess[84] = t235
            * (t116 * t151 * t5 * t60 * (t1186 + t420) - t1187 - t1188
               + t126 * t151 * t16 * t411 + t126 * t225 * t404
               + t149 * t151 * t396 + t16 * t19 * t399 * t7 - t422 - t423);
        hess[85] = t235
            * (t1001 * t566
               + t1129
                   * (t1127 + t1189 + t1190 + t127 * t391 + t130 * t391
                      - t133 * t393 - t134 * t264)
               - t1191 - t1192 + t137 * t403 + t569 + t918);
        hess[86] =
            t235 * (t10 * t116 * t151 * t60 * (t1186 + t639) - t641 - t924);
        hess[87] = t868;
        hess[88] = t1010;
        hess[89] = t1100;
        hess[90] = t1160;
        hess[91] = t235
            * (t1133 * std::pow(t411, 2) - t1136 * t396 * t411
               + t1142 * t411 * t568
               - t230
                   * (t1001 * (t1196 * t2 + t1198 + t388 * t412 + t398 - t49)
                      + t1195 * t12 + t1195 * t8 + t1197 * t765 + t1197 * t967
                      + std::pow(t403, 2))
               - t365
                   * (t1193 * t196 + t1193 * t297 + t1194 * t696 + t1194 * t933
                      + t16 * std::pow(t410, 2)
                      + t312 * (t1 * t996 + t2 * t495 * t692 + t407 - t47))
               - 2 * t396 * t568 + std::pow(t404, 2) * t682
               + t567
                   * (t104 * (t1198 - t388 * t391 + t928)
                      + t107 * (3 * t100 + t102 + t25 * t391 - t934)
                      + t1199 * t279 + t1199 * t280 + t1200 * t127
                      + t1200 * t130 - t389 * t419 - t389 * t638
                      + t393 * t395));
        hess[92] = t1211;
        hess[93] = t235 * (t1217 - t230 * (t1188 + t1212) + t671 + t911);
        hess[94] = t235
            * (t1215 * t475 + t1216 * t583 + t1218 - t1219 - t1220 - t1221
               - t230 * (t1222 + t1223 + t665 + t916 + t917) + t407 * t515
               + t407 * t534 + t471 * t843 + t496 * t758 + t656 + t913 + t914);
        hess[95] = t235 * (t1225 - t230 * (t1224 + t924) + t921);
        hess[96] = t235
            * (t116 * t151 * t5 * t60 * (t1226 + t458) - t1227 - t1228
               + t126 * t151 * t16 * t448 + t126 * t225 * t443
               + t149 * t151 * t437 + t16 * t19 * t439 * t7 - t459 - t460);
        hess[97] =
            t235 * (t1 * t116 * t151 * t60 * (t1226 + t571) - t1229 - t574);
        hess[98] = t235
            * (t1050 + t1108 * t645
               + t1129
                   * (t1126 + t1189 + t1230 + t127 * t432 + t143 * t432
                      - t170 * t434 - t171 * t318)
               - t1231 - t1232 + t161 * t442 + t646);
        hess[99] = t895;
        hess[100] = t1021;
        hess[101] = t1111;
        hess[102] = t1171;
        hess[103] = t1211;
        hess[104] = t235
            * (t1133 * std::pow(t448, 2) - t1136 * t437 * t448
               + t1142 * t448 * t572
               - t230
                   * (t1108 * (t11 * t1236 + t1238 + t429 * t449 + t438 - t53)
                      + t1235 * t3 + t1235 * t8 + t1237 * t801 + t1237 * t967
                      + std::pow(t442, 2))
               - t365
                   * (t1233 * t196 + t1233 * t237 + t1234 * t695 + t1234 * t933
                      + t16 * std::pow(t447, 2)
                      + t256 * (t10 * t1105 + t11 * t586 * t692 + t445 - t51))
               - 2 * t437 * t572 + std::pow(t443, 2) * t682
               + t567
                   * (t112 * (t1055 + t1238 - t429 * t432)
                      + t115 * (-t1060 + 3 * t108 + t110 + t29 * t432)
                      + t1239 * t279 + t1239 * t328 + t1240 * t127
                      + t1240 * t143 - t430 * t457 - t430 * t570
                      + t434 * t436));
        hess[105] = t235 * (t1040 + t1246 - t230 * (t1228 + t1241) + t676);
        hess[106] = t235 * (t1043 + t1248 - t230 * (t1229 + t1247) + t920);
        hess[107] = t235
            * (t1045 + t1046 + t1218 + t1244 * t491 + t1245 * t652 - t1249
               - t1250 - t1251 - t230 * (t1048 + t1049 + t1222 + t1252 + t663)
               + t445 * t598 + t445 * t616 + t488 * t872 + t587 * t797 + t654);
        hess[108] = t470;
        hess[109] = t580;
        hess[110] = t649;
        hess[111] = t235
            * (t1253 * t350 + t1253 * t353 + t1254 * t130 + t1254 * t143
               + t146 * t216 + 2 * t148 * t61
               - t231
                   * (t159 * t802 + t220 * t508 + t221 * t508 + 2 * t467 * t7
                      + t711 * t79 + t896)
               + t903);
        hess[112] = t235
            * (-t1022 - t1023 + t1024 * t244 - t1027 + t1030 * t468 + t17 * t240
               - t230 * (t1025 + t1255 * t239) + t250 + t252 - t255 - t259
               + t901 * t939);
        hess[113] = t235
            * (t1024 * t304 + t1065 * t901 - t1112 - t1113 - t1116
               + t1124 * t468 + t1256 * t52 - t230 * (t1114 + t1256 * t54)
               + t308 + t310 - t311 - t314);
        hess[114] = t235
            * (t116 * t151 * t16 * t225 * t381 * t468
               + 3 * t116 * t380 * t468 * t681 - t1172 - t1174 - t1178
               - t230 * (t1175 + t186 * t373 + t186 * t374)
               + t60
                   * (t1128 + t127 * t178 + t130 * t180 + t143 * t180
                      + t146 * t368));
        hess[115] =
            t235 * (t1217 + t1257 * t17 - t230 * (t1212 + t1255 * t413) + t260);
        hess[116] =
            t235 * (t1246 + t1258 * t17 - t230 * (t1241 + t17 * t451) + t315);
        hess[117] = t1262
            * (t1261 * std::pow(t468, 2) + 2 * t468 * t577
               - t55 * (t1259 + t1260 + std::pow(t575, 2)));
        hess[118] = t1264;
        hess[119] = t1266;
        hess[120] = t487;
        hess[121] = t584;
        hess[122] = t650;
        hess[123] = t235
            * (t16 * t2 * t36 * t508 - t230 * (t1267 * t508 + t425 + t905)
               - t512 - t906);
        hess[124] = t235
            * (t1031 + t1268 * t253 + t1268 * t353 + t1269 * t127 + t1269 * t130
               + 2 * t128 * t134 + t132 * t285
               - t231
                   * (t1028 + 2 * t2 * t474 + t221 * t239 + t239 * t289
                      + t503 * t971 + t708 * t79));
        hess[125] = t235
            * (t1065 * t583 - t1117 - t1118 + t1119 * t304 - t1121
               + t1124 * t475 + t1270 + t1271 * t52
               - t230 * (t1120 + t1271 * t54) + t541 + t542 - t543);
        hess[126] = t235
            * (t1182 + t1272 * t67 - t230 * (t1181 + t1267 * t186 + t520)
               + t512);
        hess[127] = t235
            * (t116 * t151 * t16 * t225 * t411 * t475
               + 3 * t116 * t404 * t475 * t681 - t1219 - t1220 - t1221
               - t230 * (t1223 + t374 * t413 + t400 * t413)
               + t60
                   * (t1190 + t127 * t415 + t130 * t415 + t132 * t393
                      + t143 * t533));
        hess[128] =
            t235 * (t1248 + t1258 * t67 - t230 * (t1247 + t451 * t67) + t546);
        hess[129] = t1264;
        hess[130] = t1262
            * (t1261 * std::pow(t475, 2) + 2 * t475 * t481
               - t55 * (t1259 + t1273 + std::pow(t472, 2)));
        hess[131] = t1274;
        hess[132] = t494;
        hess[133] = t585;
        hess[134] = t653;
        hess[135] = t235
            * (t11 * t16 * t36 * t508 - t230 * (t1275 * t508 + t462 + t908)
               - t595 - t909);
        hess[136] = t235
            * (t1030 * t491 - t1032 - t1033 - t1038 + t1270
               - t230 * (t1034 + t1037 + t1276 * t239) + t240 * t70
               + t488 * t969 - t613 + t615 + t617 + t652 * t939);
        hess[137] = t235
            * (t1125 + t127 * t1278 + t1277 * t253 + t1277 * t350 + t1278 * t143
               + 2 * t166 * t171 + t169 * t333
               - t231
                   * (2 * t11 * t490 + t1122 + t159 * t768 + t220 * t299
                      + t289 * t299 + t336 * t644));
        hess[138] = t235
            * (t1185 + t1272 * t70 - t230 * (t1184 + t1275 * t186 + t603)
               + t595);
        hess[139] = t235
            * (t1225 + t1257 * t70 - t230 * (t1224 + t1276 * t413 + t621)
               + t614);
        hess[140] = t235
            * (t116 * t151 * t16 * t225 * t448 * t491
               + 3 * t116 * t443 * t491 * t681 - t1249 - t1250 - t1251
               - t230 * (t1252 + t373 * t450 + t400 * t450)
               + t60
                   * (t1230 + t127 * t453 + t130 * t626 + t143 * t453
                      + t169 * t434));
        hess[141] = t1266;
        hess[142] = t1274;
        hess[143] = t1262
            * (t1261 * std::pow(t491, 2) + 2 * t491 * t493
               - t55 * (t1260 + t1273 + std::pow(t489, 2)));
    }

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
        double dA[12])
    {
        const auto t0 = t0_x - t1_x;
        const auto t1 = -t0;
        const auto t2 = -t2_z;
        const auto t3 = t0_z + t2;
        const auto t4 = -t3;
        const auto t5 = -t2_x;
        const auto t6 = t0_x + t5;
        const auto t7 = -t6;
        const auto t8 = t0_z - t1_z;
        const auto t9 = -t8;
        const auto t10 = t1 * t4 - t7 * t9;
        const auto t11 = -t2_y;
        const auto t12 = t0_y + t11;
        const auto t13 = t0_y - t1_y;
        const auto t14 = t0 * t12 - t13 * t6;
        const auto t15 = -t12 * t8 + t13 * t3;
        const auto t16 = std::pow(t14, 2) + std::pow(t15, 2);
        const auto t17 = std::pow(t10, 2) + t16;
        const auto t18 = std::pow(t17, -1.0 / 2.0);
        const auto t19 = t1_z + t2;
        const auto t20 = t11 + t1_y;
        const auto t21 = -t12;
        const auto t22 = -t13;
        const auto t23 = t1 * t21 - t22 * t7;
        const auto t24 = -t10;
        const auto t25 = (-t10 * t3_y + t14 * t3_z + t15 * t3_x) / t17;
        const auto t26 = t1_x + t5;
        const auto t27 = -t21 * t9 + t22 * t4;
        const auto t28 = t0 * t3 - t6 * t8;
        const auto t29 = std::pow(t16 + std::pow(t28, 2), -1.0 / 2.0);
        dA[0] =
            t18 * (t19 * t3_y - t20 * t3_z + t25 * (-t19 * t24 + t20 * t23));
        dA[1] =
            t18 * (-t19 * t3_x + t25 * (t19 * t27 - t23 * t26) + t26 * t3_z);
        dA[2] =
            t18 * (t20 * t3_x + t25 * (-t20 * t27 + t24 * t26) - t26 * t3_y);
        dA[3] = t18 * (t12 * t3_z + t25 * (t21 * t23 + t24 * t3) - t3 * t3_y);
        dA[4] = t18 * (t25 * (t23 * t6 + t27 * t4) + t3 * t3_x - t3_z * t6);
        dA[5] = t18 * (-t12 * t3_x + t25 * (t12 * t27 + t24 * t7) + t3_y * t6);
        dA[6] = t18 * (-t13 * t3_z + t25 * (t13 * t23 + t24 * t9) + t3_y * t8);
        dA[7] = t18 * (t0 * t3_z + t25 * (t1 * t23 + t27 * t8) - t3_x * t8);
        dA[8] = t18 * (-t0 * t3_y + t13 * t3_x + t25 * (t0 * t24 + t22 * t27));
        dA[9] = -t15 * t29;
        dA[10] = t28 * t29;
        dA[11] = -t14 * t29;
    }

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
        double dA[144])
    {
        const auto t0 = t0_x - t1_x;
        const auto t1 = -t0;
        const auto t2 = -t2_z;
        const auto t3 = t0_z + t2;
        const auto t4 = -t3;
        const auto t5 = t1 * t4;
        const auto t6 = -t2_x;
        const auto t7 = t0_x + t6;
        const auto t8 = -t7;
        const auto t9 = -t1_z;
        const auto t10 = t0_z + t9;
        const auto t11 = -t10;
        const auto t12 = t11 * t8;
        const auto t13 = -t12 + t5;
        const auto t14 = -t2_y;
        const auto t15 = t0_y + t14;
        const auto t16 = -t1_y;
        const auto t17 = t0_y + t16;
        const auto t18 = t17 * t7;
        const auto t19 = t0 * t15 - t18;
        const auto t20 = t10 * t15;
        const auto t21 = t17 * t3 - t20;
        const auto t22 = std::pow(t19, 2) + std::pow(t21, 2);
        const auto t23 = std::pow(t13, 2) + t22;
        const auto t24 = std::pow(t23, -3.0 / 2.0);
        const auto t25 = t1_z + t2;
        const auto t26 = t14 + t1_y;
        const auto t27 = -t25 * t3_y + t26 * t3_z;
        const auto t28 = -t15;
        const auto t29 = t1 * t28;
        const auto t30 = -t17;
        const auto t31 = t30 * t8;
        const auto t32 = -t31;
        const auto t33 = t29 + t32;
        const auto t34 = -t25;
        const auto t35 = -t13;
        const auto t36 = t26 * t33 + t34 * t35;
        const auto t37 = -t13 * t3_y + t19 * t3_z + t21 * t3_x;
        const auto t38 = 1.0 / t23;
        const auto t39 = t37 * t38;
        const auto t40 = 3 * t39;
        const auto t41 = t1_x + t6;
        const auto t42 = t25 * t3_x - t3_z * t41;
        const auto t43 = -t41;
        const auto t44 = t30 * t4;
        const auto t45 = t11 * t28;
        const auto t46 = -t45;
        const auto t47 = t44 + t46;
        const auto t48 = t25 * t47 + t33 * t43;
        const auto t49 = t37 * t41;
        const auto t50 = t36 * t38;
        const auto t51 = 3 * t37;
        const auto t52 = t50 * t51;
        const auto t53 = t24 * (-t26 * t49 + t27 * t48 + t36 * t42 - t48 * t52);
        const auto t54 = -t26 * t3_x + t3_y * t41;
        const auto t55 = -t26;
        const auto t56 = t35 * t41 + t47 * t55;
        const auto t57 = t24 * (-t25 * t49 + t27 * t56 + t36 * t54 - t52 * t56);
        const auto t58 = -t15 * t3_z + t3 * t3_y;
        const auto t59 = t28 * t33 + t3 * t35;
        const auto t60 = t24
            * (t27 * t59 + t36 * t58 + t37 * (t26 * t28 + t3 * t34)
               - t52 * t59);
        const auto t61 = std::pow(t23, -1.0 / 2.0);
        const auto t62 = -t3_z;
        const auto t63 = -t3 * t3_x + t3_z * t7;
        const auto t64 = t33 * t7 + t4 * t47;
        const auto t65 = t27 * t38;
        const auto t66 = t51 / std::pow(t23, 2);
        const auto t67 = t36 * t66;
        const auto t68 = t61
            * (t39 * (t26 * t7 + t33) + t50 * t63 + t62 + t64 * t65
               - t64 * t67);
        const auto t69 = t15 * t3_x - t3_y * t7;
        const auto t70 = t15 * t47 + t35 * t8;
        const auto t71 = t61
            * (t39 * (t13 + t34 * t8) + t3_y + t50 * t69 + t65 * t70
               - t67 * t70);
        const auto t72 = -t10 * t3_y + t17 * t3_z;
        const auto t73 = t11 * t35 + t17 * t33;
        const auto t74 = t24
            * (t27 * t73 + t36 * t72 + t37 * (t11 * t34 + t17 * t26)
               - t52 * t73);
        const auto t75 = -t0 * t3_z + t10 * t3_x;
        const auto t76 = t1 * t33 + t10 * t47;
        const auto t77 = -t29 + t31;
        const auto t78 = t61
            * (t39 * (t1 * t26 + t77) + t3_z + t50 * t75 + t65 * t76
               - t67 * t76);
        const auto t79 = -t3_y;
        const auto t80 = t0 * t3_y - t17 * t3_x;
        const auto t81 = t0 * t35 + t30 * t47;
        const auto t82 = t12 - t5;
        const auto t83 = t61
            * (t39 * (t0 * t34 + t82) + t50 * t80 + t65 * t81 - t67 * t81
               + t79);
        const auto t84 = t0 * t3;
        const auto t85 = -t10 * t7 + t84;
        const auto t86 = std::pow(t22 + std::pow(t85, 2), -3.0 / 2.0);
        const auto t87 = t21 * t86;
        const auto t88 = t87 * (t13 * t25 + t19 * t26);
        const auto t89 = t2_z + t9;
        const auto t90 = t61 * (-t13 * t50 - t89);
        const auto t91 = t61 * (t16 + t19 * t50 + t2_y);
        const auto t92 = t40 * t48;
        const auto t93 =
            t24 * (-t25 * t26 * t37 + t42 * t56 + t48 * t54 - t56 * t92);
        const auto t94 = t38 * t42;
        const auto t95 = t38 * t48;
        const auto t96 = t48 * t66;
        const auto t97 = t61
            * (t39 * (t28 * t43 + t77) + t3_z + t58 * t95 + t59 * t94
               - t59 * t96);
        const auto t98 = t24
            * (t37 * (t25 * t4 + t43 * t7) + t42 * t64 + t48 * t63 - t64 * t92);
        const auto t99 = -t3_x;
        const auto t100 = t61
            * (t39 * (t15 * t25 + t47) + t69 * t95 + t70 * t94 - t70 * t96
               + t99);
        const auto t101 = t61
            * (t39 * (t17 * t43 + t33) + t62 + t72 * t95 + t73 * t94
               - t73 * t96);
        const auto t102 = t24
            * (t37 * (t1 * t43 + t10 * t25) + t42 * t76 + t48 * t75
               - t76 * t92);
        const auto t103 = -t44 + t45;
        const auto t104 = t61
            * (t39 * (t103 + t25 * t30) + t3_x + t80 * t95 + t81 * t94
               - t81 * t96);
        const auto t105 = t61 * (t21 * t95 + t89);
        const auto t106 = t85 * t86;
        const auto t107 = -t106 * (-t19 * t41 + t21 * t25);
        const auto t108 = t61 * (t19 * t95 + t41);
        const auto t109 = t38 * t54;
        const auto t110 = t38 * t56;
        const auto t111 = t56 * t66;
        const auto t112 = t61
            * (t109 * t59 + t110 * t58 - t111 * t59 + t39 * (t3 * t41 + t82)
               + t79);
        const auto t113 = t61
            * (t109 * t64 + t110 * t63 - t111 * t64 + t39 * (t103 + t4 * t55)
               + t3_x);
        const auto t114 = t40 * t56;
        const auto t115 = t24
            * (-t114 * t70 + t37 * (t15 * t55 + t41 * t8) + t54 * t70
               + t56 * t69);
        const auto t116 = t61
            * (t109 * t73 + t110 * t72 - t111 * t73 + t39 * (t11 * t41 + t13)
               + t3_y);
        const auto t117 = t61
            * (t109 * t76 + t110 * t75 - t111 * t76 + t39 * (t10 * t55 + t47)
               + t99);
        const auto t118 = t24
            * (-t114 * t81 + t37 * (t0 * t41 + t30 * t55) + t54 * t81
               + t56 * t80);
        const auto t119 = t61 * (t110 * t21 + t26);
        const auto t120 = -t61 * (t110 * t13 + t41);
        const auto t121 = t19 * t86;
        const auto t122 = -t121 * (t13 * t41 + t21 * t26);
        const auto t123 = t37 * t7;
        const auto t124 = t40 * t59;
        const auto t125 =
            t24 * (-t123 * t15 - t124 * t64 + t58 * t64 + t59 * t63);
        const auto t126 =
            t24 * (-t123 * t3 - t124 * t70 + t58 * t70 + t59 * t69);
        const auto t127 = t24
            * (-t124 * t73 + t37 * (t11 * t3 + t17 * t28) + t58 * t73
               + t59 * t72);
        const auto t128 = t38 * t59;
        const auto t129 = t38 * t58;
        const auto t130 = t59 * t66;
        const auto t131 = t61
            * (t128 * t75 + t129 * t76 - t130 * t76 + t39 * (2 * t29 + t32)
               + t62);
        const auto t132 = t61
            * (t128 * t80 + t129 * t81 - t130 * t81 + t39 * (t13 + t84) + t3_y);
        const auto t133 = -t87 * (t13 * t3 + t15 * t19);
        const auto t134 = -t61 * (t128 * t13 + t3);
        const auto t135 = t61 * (t128 * t19 + t15);
        const auto t136 = t40 * t64;
        const auto t137 =
            t24 * (-t136 * t70 - t15 * t3 * t37 + t63 * t70 + t64 * t69);
        const auto t138 = t38 * t63;
        const auto t139 = t38 * t64;
        const auto t140 = t64 * t66;
        const auto t141 = t61
            * (t138 * t73 + t139 * t72 - t140 * t73 + t39 * (t18 + t77) + t3_z);
        const auto t142 = t24
            * (-t136 * t76 + t37 * (t1 * t7 + t10 * t4) + t63 * t76
               + t64 * t75);
        const auto t143 = t61
            * (t138 * t81 + t139 * t80 - t140 * t81 + t39 * (2 * t44 + t46)
               + t99);
        const auto t144 = t61 * (t139 * t21 + t3);
        const auto t145 = -t106 * (t19 * t7 - t21 * t3);
        const auto t146 = -t0_x + t2_x;
        const auto t147 = t61 * (t139 * t19 + t146);
        const auto t148 = t38 * t69;
        const auto t149 = t38 * t70;
        const auto t150 = t66 * t70;
        const auto t151 = t61
            * (t148 * t73 + t149 * t72 - t150 * t73 + t39 * (2 * t11 * t8 - t5)
               + t79);
        const auto t152 = t61
            * (t148 * t76 + t149 * t75 - t150 * t76 + t39 * (t103 + t20)
               + t3_x);
        const auto t153 = t40 * t81;
        const auto t154 = t24
            * (-t153 * t70 + t37 * (t0 * t8 + t15 * t30) + t69 * t81
               + t70 * t80);
        const auto t155 = -t0_y;
        const auto t156 = t61 * (t149 * t21 + t155 + t2_y);
        const auto t157 = t61 * (-t13 * t149 - t146);
        const auto t158 = t121 * (t13 * t7 + t15 * t21);
        const auto t159 = t0 * t37;
        const auto t160 =
            t24 * (-t159 * t17 - t40 * t73 * t76 + t72 * t76 + t73 * t75);
        const auto t161 =
            t24 * (-t10 * t159 - t153 * t73 + t72 * t81 + t73 * t80);
        const auto t162 = t87 * (t10 * t13 + t17 * t19);
        const auto t163 = t38 * t73;
        const auto t164 = -t0_z + t1_z;
        const auto t165 = t61 * (-t13 * t163 - t164);
        const auto t166 = t61 * (t155 + t163 * t19 + t1_y);
        const auto t167 =
            t24 * (-t10 * t17 * t37 - t153 * t76 + t75 * t81 + t76 * t80);
        const auto t168 = t38 * t76;
        const auto t169 = t61 * (t164 + t168 * t21);
        const auto t170 = -t106 * (-t0 * t19 + t10 * t21);
        const auto t171 = t61 * (t0 + t168 * t19);
        const auto t172 = t38 * t81;
        const auto t173 = t61 * (t17 + t172 * t21);
        const auto t174 = -t61 * (t0 + t13 * t172);
        const auto t175 = -t121 * (t0 * t13 + t17 * t21);
        dA[0] = t24
            * (2 * t27 * t36 - std::pow(t36, 2) * t40
               + t37 * (std::pow(t26, 2) + std::pow(t34, 2)));
        dA[1] = t53;
        dA[2] = t57;
        dA[3] = t60;
        dA[4] = t68;
        dA[5] = t71;
        dA[6] = t74;
        dA[7] = t78;
        dA[8] = t83;
        dA[9] = t88;
        dA[10] = t90;
        dA[11] = t91;
        dA[12] = t53;
        dA[13] = t24
            * (t37 * (std::pow(t25, 2) + std::pow(t43, 2))
               - t40 * std::pow(t48, 2) + 2 * t42 * t48);
        dA[14] = t93;
        dA[15] = t97;
        dA[16] = t98;
        dA[17] = t100;
        dA[18] = t101;
        dA[19] = t102;
        dA[20] = t104;
        dA[21] = t105;
        dA[22] = t107;
        dA[23] = t108;
        dA[24] = t57;
        dA[25] = t93;
        dA[26] = t24
            * (t37 * (std::pow(t41, 2) + std::pow(t55, 2))
               - t40 * std::pow(t56, 2) + 2 * t54 * t56);
        dA[27] = t112;
        dA[28] = t113;
        dA[29] = t115;
        dA[30] = t116;
        dA[31] = t117;
        dA[32] = t118;
        dA[33] = t119;
        dA[34] = t120;
        dA[35] = t122;
        dA[36] = t60;
        dA[37] = t97;
        dA[38] = t112;
        dA[39] = t24
            * (t37 * (std::pow(t28, 2) + std::pow(t3, 2))
               - t40 * std::pow(t59, 2) + 2 * t58 * t59);
        dA[40] = t125;
        dA[41] = t126;
        dA[42] = t127;
        dA[43] = t131;
        dA[44] = t132;
        dA[45] = t133;
        dA[46] = t134;
        dA[47] = t135;
        dA[48] = t68;
        dA[49] = t98;
        dA[50] = t113;
        dA[51] = t125;
        dA[52] = t24
            * (t37 * (std::pow(t4, 2) + std::pow(t7, 2))
               - t40 * std::pow(t64, 2) + 2 * t63 * t64);
        dA[53] = t137;
        dA[54] = t141;
        dA[55] = t142;
        dA[56] = t143;
        dA[57] = t144;
        dA[58] = t145;
        dA[59] = t147;
        dA[60] = t71;
        dA[61] = t100;
        dA[62] = t115;
        dA[63] = t126;
        dA[64] = t137;
        dA[65] = t24
            * (t37 * (std::pow(t15, 2) + std::pow(t8, 2))
               - t40 * std::pow(t70, 2) + 2 * t69 * t70);
        dA[66] = t151;
        dA[67] = t152;
        dA[68] = t154;
        dA[69] = t156;
        dA[70] = t157;
        dA[71] = t158;
        dA[72] = t74;
        dA[73] = t101;
        dA[74] = t116;
        dA[75] = t127;
        dA[76] = t141;
        dA[77] = t151;
        dA[78] = t24
            * (t37 * (std::pow(t11, 2) + std::pow(t17, 2))
               - t40 * std::pow(t73, 2) + 2 * t72 * t73);
        dA[79] = t160;
        dA[80] = t161;
        dA[81] = t162;
        dA[82] = t165;
        dA[83] = t166;
        dA[84] = t78;
        dA[85] = t102;
        dA[86] = t117;
        dA[87] = t131;
        dA[88] = t142;
        dA[89] = t152;
        dA[90] = t160;
        dA[91] = t24
            * (t37 * (std::pow(t1, 2) + std::pow(t10, 2))
               - t40 * std::pow(t76, 2) + 2 * t75 * t76);
        dA[92] = t167;
        dA[93] = t169;
        dA[94] = t170;
        dA[95] = t171;
        dA[96] = t83;
        dA[97] = t104;
        dA[98] = t118;
        dA[99] = t132;
        dA[100] = t143;
        dA[101] = t154;
        dA[102] = t161;
        dA[103] = t167;
        dA[104] = t24
            * (t37 * (std::pow(t0, 2) + std::pow(t30, 2))
               - t40 * std::pow(t81, 2) + 2 * t80 * t81);
        dA[105] = t173;
        dA[106] = t174;
        dA[107] = t175;
        dA[108] = t88;
        dA[109] = t105;
        dA[110] = t119;
        dA[111] = t133;
        dA[112] = t144;
        dA[113] = t156;
        dA[114] = t162;
        dA[115] = t169;
        dA[116] = t173;
        dA[117] = 0;
        dA[118] = 0;
        dA[119] = 0;
        dA[120] = t90;
        dA[121] = t107;
        dA[122] = t120;
        dA[123] = t134;
        dA[124] = t145;
        dA[125] = t157;
        dA[126] = t165;
        dA[127] = t170;
        dA[128] = t174;
        dA[129] = 0;
        dA[130] = 0;
        dA[131] = 0;
        dA[132] = t91;
        dA[133] = t108;
        dA[134] = t122;
        dA[135] = t135;
        dA[136] = t147;
        dA[137] = t158;
        dA[138] = t166;
        dA[139] = t171;
        dA[140] = t175;
        dA[141] = 0;
        dA[142] = 0;
        dA[143] = 0;
    }

}
