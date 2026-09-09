def add_cloth_fit_arguments(parser):


    cloth_fit_group = parser.add_argument_group('Cloth-fit Reshaping Options')

    cloth_fit_group.add_argument('--enable_cloth_fit_reshaping', action='store_true',
                                help="Enable cloth-fit reshaping instead of SMPL-based reshaping")

    cloth_fit_group.add_argument('--cloth_fit_original_mesh_basename', type=str,
                                default='nerf_original.obj',
                                help="Basename for original full-resolution mesh (default: nerf_original.obj)")

    cloth_fit_group.add_argument('--cloth_fit_simplified_mesh_basename', type=str,
                                default='nerf_simp_cleaned.obj',
                                help="Basename for original simplified mesh (default: nerf_simp_cleaned.obj)")

    cloth_fit_group.add_argument('--cloth_fit_deformed_mesh_path', type=str,
                                help="Full path to deformed simplified mesh (cloth-fit result)")

    cloth_fit_group.add_argument(
        '--cloth_fit_deformed_mesh_already_restored',
        action='store_true',
        help=(
            "If set, treat --cloth_fit_deformed_mesh_path vertices as already in the source/world space "
            "(i.e., no additional restore-from-normalized using SMPL skeleton center/scale). "
            "Use this when cloth-fit runs in height-aware mode that restores translation and avoids global scaling."
        ),
    )


def validate_cloth_fit_arguments(args):

    if args.enable_cloth_fit_reshaping and not args.cloth_fit_deformed_mesh_path:
        raise ValueError("--cloth_fit_deformed_mesh_path is required when --enable_cloth_fit_reshaping is enabled")

    return True
