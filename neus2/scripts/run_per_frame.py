import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default='.')
parser.add_argument('--output_dir', type=str, default='neus2_exp')
parser.add_argument('--config', type=str, default='base.json')
parser.add_argument('--frame_start', type=int, default=0)
parser.add_argument('--frame_end', type=int, default=-1)
parser.add_argument('--n_steps', type=int, default=-1)
parser.add_argument('--dynamic_test', action='store_true')
parser.add_argument('--dynamic_save_mesh', action='store_true')
parser.add_argument('--dynamic_save_mesh_only', action='store_true')
parser.add_argument('--save_every_n_steps', type=int, default=10000)
parser.add_argument('--white_bkgd', action='store_true')
args = parser.parse_args()

frames = sorted(glob.glob(os.path.join(args.base_dir, '*.json')))
config = args.config

for scene in frames[args.frame_start:args.frame_end]:
    num = os.path.basename(scene).split('.')[0]
    name = f"{args.output_dir}/{num}"
    
    cmd = f"python scripts/run_dynamic.py \
        --scene {scene} --mode nerf --name {name} --network {config} \
        --save_snapshot_per_frame --save_every_n_steps {args.save_every_n_steps}"
    
    if args.dynamic_test:
        cmd += " --dynamic_test"
    if args.dynamic_save_mesh:
        cmd += " --dynamic_save_mesh"
    if args.dynamic_save_mesh_only:
        cmd += " --dynamic_save_mesh_only"
    if args.white_bkgd:
        cmd += " --white_bkgd"
    os.system(cmd)