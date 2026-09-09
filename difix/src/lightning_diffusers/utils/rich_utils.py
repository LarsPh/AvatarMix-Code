from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict

from lightning_diffusers.utils import pylogger

log = pylogger.get_pylogger(__name__)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:


    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []


    for field in cfg:
        queue.append((field, cfg[field], tree))


    while queue:
        field, val, parent = queue.pop(0)

        if isinstance(val, DictConfig):

            branch = parent.add(f"[bold]{field}[/bold]", style=style, guide_style=style)
            for nested_field in val:
                queue.append((nested_field, val[nested_field], branch))
        else:

            parent.add(f"{field}: {val}", style=style, guide_style=style)


    rich.print(tree)


    if save_to_file:
        hydra_cfg = HydraConfig.get()
        with open(Path(hydra_cfg.runtime.output_dir, "config_tree.log"), "w") as fp:
            rich.print(tree, file=fp)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:


    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = input("Input a list of comma separated tags (dev): ")
        tags = tags.strip() or "dev"
        tags = [tag.strip() for tag in tags.split(",") if tag != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        hydra_cfg = HydraConfig.get()
        with open(Path(hydra_cfg.runtime.output_dir, "tags.log"), "w") as fp:
            rich.print(cfg.get("tags"), file=fp)
