import os
from loguru import logger


def build_python_command(project_path: str, conda_env_name: str) -> list:

    venv_path = os.path.join(project_path, '.venv')
    project_name = os.path.basename(project_path)

    if os.path.exists(venv_path) or os.path.isfile(os.path.join(project_path, 'pyproject.toml')):
        logger.info(f"✓ Using UV environment for {project_name} (detected .venv/)")
        return ['uv', 'run', 'python']
    else:
        logger.info(f"✓ Using conda environment '{conda_env_name}' for {project_name} (.venv not found)")
        return ['conda', 'run', '-n', conda_env_name, '--no-capture-output', 'python']
