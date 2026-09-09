#!/usr/bin/env python3


import argparse
import os
import sys
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from loguru import logger
from omegaconf import OmegaConf


template_src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(template_src_path))

from predict import predict


def main():

    parser = argparse.ArgumentParser(description="Batch refine images using DiFix3D Lightning")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with images")
    parser.add_argument("--output_dir", type=str, help="Output directory for refined images")
    parser.add_argument("--output_suffix", type=str, help="Output suffix for refined images (alternative to output_dir)")
    parser.add_argument("--model_id", type=str, default="nvidia/difix", help="HuggingFace model ID")
    parser.add_argument("--prompt", type=str, default="remove degradation", help="Refinement prompt")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for processing")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing output files")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")

    args = parser.parse_args()


    if args.output_dir and args.output_suffix:
        logger.error("Cannot specify both --output_dir and --output_suffix")
        sys.exit(1)
    elif args.output_suffix:


        input_path = Path(args.input_dir)


        input_parts = input_path.parts
        base_refined_parts = []

        for part in input_parts:
            if ('head_swapped_back_renders' in part or 'head_swapped_renders' in part) and not part.endswith('_refined'):

                base_refined_parts.append(f"{part}_refined")
                break
            else:
                base_refined_parts.append(part)


        if base_refined_parts and any('head_swapped' in part for part in base_refined_parts):
            args.output_dir = str(Path(*base_refined_parts))
            logger.info(f"Using base refined directory: {args.output_dir}")
            logger.info(f"Subject structure will be preserved from input: {args.input_dir}")
        else:

            args.output_dir = str(input_path.parent / f"{input_path.name}{args.output_suffix}")
            logger.warning(f"Could not detect SwapVTON directory pattern, using suffix fallback: {args.output_dir}")
    elif not args.output_dir:
        logger.error("Must specify either --output_dir or --output_suffix")
        sys.exit(1)


    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Created output directory: {args.output_dir}")


    GlobalHydra.instance().clear()
    logger.debug("Cleared existing Hydra global state")


    config_dir = str(Path(__file__).parent.parent / "configs")
    logger.debug(f"Using config directory: {config_dir}")


    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):

        cfg = compose(
            config_name="experiment/difix3d_batch_refine.yaml",
            overrides=[
                f"+data.input_dir={args.input_dir}",
                f"+data.output_dir={args.output_dir}",
                f"data.batch_size={args.batch_size}",
                f"data.skip_existing={args.skip_existing}",
                f"data._target_=lightning_diffusers.data.difix3d_datamodule.DiFix3DDataModule",
                f"model._target_=lightning_diffusers.models.difix3d_module.DiFix3DModule",
                f"model.prompt={args.prompt}",
                f"trainer.accelerator={args.device}",
                "trainer.devices=1",
            ]
        )

        logger.info("=== DiFix3D Batch Refinement Configuration ===")
        logger.info(f"Input directory: {args.input_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Model: {args.model_id}")
        logger.info(f"Prompt: '{args.prompt}'")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Device: {args.device}")
        logger.info(f"Skip existing: {args.skip_existing}")
        logger.info("=" * 47)

        try:
            logger.info("Starting batch refinement process...")
            logger.info(f"Input directory: {args.input_dir}")
            logger.info(f"Output directory: {args.output_dir}")
            logger.info(f"Batch size: {args.batch_size}")
            logger.info(f"Skip existing: {args.skip_existing}")


            input_path = Path(args.input_dir)
            if input_path.exists():
                logger.info(f"Input directory exists and will be processed by datamodule")
                logger.debug(f"Datamodule will handle image discovery and filtering")
            else:
                logger.warning(f"Input directory does not exist: {args.input_dir}")


            metric_dict, object_dict = predict(cfg)

            processed_batches = metric_dict.get('predictions', 0)
            logger.success(f"Successfully processed {processed_batches} batches")
            logger.info(f"Refined images saved to: {args.output_dir}")

        except Exception as e:
            logger.error(f"Error during batch refinement: {e}")
            logger.exception("Full error traceback:")
            sys.exit(1)


if __name__ == "__main__":
    main()
