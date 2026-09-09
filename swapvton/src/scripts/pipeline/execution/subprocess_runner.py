import subprocess
import os
from loguru import logger
from ..execution.debug_support import get_debug_port
from ..execution.env_utils import build_python_command


def run_command(command, cwd=None, shell=False, debug_port=None):

    original_command = command.copy()
    logger.info(f"Original command: {' '.join(original_command)}")


    if debug_port:
        logger.info(f"Debug mode enabled on port {debug_port}")


        is_conda_command = len(command) >= 3 and command[0] == 'conda' and command[1] == 'run'

        if is_conda_command:
            logger.warning("CONDA DEBUGGING: External project debugging not supported")
            logger.warning("Skipping debugpy injection for conda command")
            logger.warning("TODO: Implement cross-project debugging for external conda environments")
            logger.warning("For now, debug the external project directly in its own environment")

        elif len(command) >= 3 and command[0] == 'uv' and command[1] == 'run' and command[2] == 'python':

            logger.info(f"UV DEBUGGING: Using direct venv Python (auto-attach enabled)")
            venv_python = os.path.join(cwd or '.', '.venv', 'Scripts', 'python.exe')
            if not os.path.exists(venv_python):

                venv_python = os.path.join(cwd or '.', '.venv', 'bin', 'python')

            if os.path.exists(venv_python):

                debugpy_cmd = [venv_python, '-m', 'debugpy', '--listen', f'localhost:{debug_port}', '--wait-for-client']
                command = debugpy_cmd + command[3:]
                logger.info(f"Modified command (using venv python): {' '.join(command)}")
                logger.info(f"✅ VSCode should auto-attach (subProcess=true)")
            else:
                logger.warning(f"venv python not found at {venv_python}, falling back to uv method")
                debugpy_args = ['-m', 'debugpy', '--listen', f'localhost:{debug_port}', '--wait-for-client']
                command = command[:3] + debugpy_args + command[3:]
                logger.info(f"Modified command: {' '.join(command)}")

        elif 'python' in command:

            python_index = command.index('python')
            debugpy_cmd = ['python', '-m', 'debugpy', '--listen', f'localhost:{debug_port}', '--wait-for-client']
            command = command[:python_index] + debugpy_cmd + command[python_index+1:]
            logger.info(f"Modified command: {' '.join(command)}")
        else:
            logger.warning(f"Could not inject debugpy into command: {command}")

    if cwd:
        logger.info(f"Working directory: {cwd}")

    logger.info(f"Executing: {' '.join(command)}")


    process = subprocess.Popen(command, cwd=cwd, shell=shell)


    return_code = process.wait()


    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def run_neus2_training_and_export(config, subject_name, base_dir, output_dir, stage_config, is_swapped=False, debug_subprocess=False):

    neus2_project = config['paths']['neus2_project']
    conda_env = config['conda_envs']['neus2']
    use_white_background = stage_config.get('use_white_background', False)

    logger.info(f"\n--- Training Subject {subject_name} ---")


    python_prefix = build_python_command(neus2_project, conda_env)
    train_cmd = python_prefix + [
        'scripts/run_per_frame.py',
        '--base_dir', base_dir,
        '--output_dir', output_dir,
        '--config', stage_config['config'],
        '--frame_start', '0', '--frame_end', '1',
        '--save_every_n_steps', str(stage_config['save_every_n_steps'])
    ]

    if use_white_background:
        train_cmd.append('--white_bkgd')

    logger.info(f"Running NeuS2 training: {' '.join(train_cmd)}")

    try:
        debug_port = get_debug_port('neus2_train_swapped' if is_swapped else 'neus2_train') if debug_subprocess else None
        run_command(train_cmd, cwd=neus2_project, debug_port=debug_port)

        logger.info(f"\n--- Exporting Mesh for Subject {subject_name} ---")


        export_cmd = python_prefix + [
            'scripts/run_per_frame.py',
            '--base_dir', base_dir,
            '--output_dir', output_dir,
            '--config', stage_config['config'],
            '--frame_start', '0', '--frame_end', '1',
            '--save_every_n_steps', str(stage_config['save_every_n_steps']),
            '--dynamic_test', '--dynamic_save_mesh'
        ]

        if use_white_background:
            export_cmd.append('--white_bkgd')

        run_command(export_cmd, cwd=neus2_project, debug_port=debug_port)

        logger.info(f"Successfully trained and exported mesh for subject: {subject_name}")

    except Exception as e:
        logger.error(f"Error training NeuS2 for {subject_name}: {e}")
        raise
