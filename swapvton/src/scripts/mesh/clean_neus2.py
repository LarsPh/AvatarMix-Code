import trimesh
import os
import argparse

def clean_mesh(input_filepath, output_dir, n_fill):

    try:

        mesh = trimesh.load(input_filepath)
        print(f"Loaded mesh: {input_filepath}")

        initial_vertices = len(mesh.vertices)
        initial_faces = len(mesh.faces)
        print(f"Initial vertices: {initial_vertices}, Initial faces: {initial_faces}")


        mesh.remove_unreferenced_vertices()
        print(f"Removed unreferenced vertices. Vertices left: {len(mesh.vertices)}")


        mesh.remove_degenerate_faces()
        print(f"Removed degenerate faces. Faces left: {len(mesh.faces)}")


        mesh.remove_duplicate_faces()
        print(f"Removed duplicate faces. Faces left: {len(mesh.faces)}")


        components = mesh.split(only_watertight=False)
        if components:

            largest_component = max(components, key=lambda m: len(m.faces))
            mesh = largest_component
            print(f"Kept the largest connected component. Vertices left: {len(mesh.vertices)}, Faces left: {len(mesh.faces)}")
        else:
            print("No connected components found after cleaning.")

            if len(mesh.faces) == 0:
                print(f"Mesh has no faces after cleaning, skipping save for {input_filepath}")
                return


        mesh.process()
        print(f"Post-split cleaning. Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")


        filename_base = os.path.basename(input_filepath)
        frame_str = filename_base.replace('frame_', '').replace('.obj', '')

        padded_frame_str = frame_str.zfill(n_fill)
        output_filename = f"{padded_frame_str}.obj"
        output_filepath = os.path.join(output_dir, output_filename)


        os.makedirs(output_dir, exist_ok=True)


        mesh.export(output_filepath)
        print(f"Saved cleaned mesh to: {output_filepath}")

    except Exception as e:
        print(f"Error processing file {input_filepath}: {e}")


parser = argparse.ArgumentParser(description='Clean NEUS2 meshes for a single subject.')
parser.add_argument(
    '--subject_root_dir',
    type=str,
    required=True,
    help='Root directory for a single subject where mesh/neus2_raw and mesh/trimesh_cleaned are located (e.g., .../dataset_neus2/0000 or .../dataset_neus2/subject_name).'
)
parser.add_argument(
    '--mesh_frame_idx',
    type=int,
    default=-1,
    help='Specific frame index of the mesh to process (e.g., 0 for frame_0.obj or frame_0000.obj). Default is -1, which processes all .obj files in neus2_raw.'
)
parser.add_argument(
    '--dataset_type',
    type=str,
    choices=['thuman2', 'avatarrex', 'other'],
    default='other',
    help='Type of the dataset. This determines N_FILL for output filenames (thuman2: 4, avatarrex: 8, other: 8).'
)
args = parser.parse_args()

if args.dataset_type == 'thuman2' or args.dataset_type == 'actorshq':
    n_fill = 4
elif args.dataset_type == 'avatarrex':
    n_fill = 8
else:
    n_fill = 8

input_directory = os.path.join(args.subject_root_dir, 'mesh/neus2_raw')
output_directory = os.path.join(args.subject_root_dir, 'mesh/trimesh_cleaned')


if not os.path.exists(args.subject_root_dir):
    print(f"Subject root directory not found: {args.subject_root_dir}")
elif not os.path.exists(input_directory):
    print(f"Input directory 'mesh/neus2_raw' not found in: {args.subject_root_dir}")
else:
    os.makedirs(output_directory, exist_ok=True)

    obj_files_all = [f for f in os.listdir(input_directory) if f.endswith('.obj')]

    obj_files_to_process = []
    if args.mesh_frame_idx != -1:
        for f_name in obj_files_all:
            try:

                frame_num_str = f_name.replace('frame_', '').replace('.obj', '')
                if int(frame_num_str) == args.mesh_frame_idx:
                    obj_files_to_process.append(f_name)
            except ValueError:
                print(f"Could not parse frame number from {f_name} for filtering. Skipping this file.")
        if not obj_files_to_process:
             print(f"No mesh found for frame_idx {args.mesh_frame_idx} in {input_directory}")
    else:
        obj_files_to_process = obj_files_all

    if not obj_files_to_process:
        print(f"No .obj files found to process in {input_directory} (after filtering, if any).")
    else:
        print(f"Found {len(obj_files_to_process)} .obj files to process in {input_directory}.")
        for filename in sorted(obj_files_to_process):
            input_filepath = os.path.join(input_directory, filename)
            clean_mesh(input_filepath, output_directory, n_fill)

print("Mesh cleaning process finished.")
