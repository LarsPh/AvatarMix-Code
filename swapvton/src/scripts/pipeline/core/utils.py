from loguru import logger
import os

def process_subject_head_masks(data_root, data_type, color_tolerance=5, background_color=(0, 0, 0), skip_existing=False, dataset_type="thuman2"):


    from head_mask_extraction.utils.grid_parser import (
        parse_segmentation_grid,
        detect_image_resolution,
        get_camera_names_from_dataset,
        validate_camera_count
    )
    from head_mask_extraction.utils.mask_creator import create_head_mask, validate_head_mask
    from head_mask_extraction.utils.rgb_processor import create_masked_rgb_batch

    logger.debug(f"Processing head masks for: {data_root} (type: {data_type}, dataset: {dataset_type})")


    seg_path = os.path.join(data_root, "Semantic", "process", "parser", "parser-f0000.png")
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Segmentation grid not found: {seg_path}")


    logger.debug("Detecting image resolution from dataset...")
    image_resolution = detect_image_resolution(data_root, dataset_type)


    logger.debug("Discovering camera names from dataset...")
    camera_names = get_camera_names_from_dataset(data_root, dataset_type)


    if skip_existing and len(camera_names) > 0:
        first_camera_mask = os.path.join(data_root, camera_names[0], "mask", "head", "0000.png")
        if os.path.exists(first_camera_mask):
            logger.info(f"Head masks already exist, skipping: {data_root}")
            return


    logger.debug("Parsing segmentation grid...")
    camera_images = parse_segmentation_grid(seg_path, image_resolution, camera_names)
    validate_camera_count(len(camera_images), camera_names)


    masks = []
    valid_camera_names = []

    for camera_image, camera_name in zip(camera_images, camera_names):
        try:

            head_mask = create_head_mask(camera_image, tolerance=color_tolerance)


            if not validate_head_mask(head_mask, min_pixels=50):
                logger.warning(f"Invalid head mask for camera {camera_name}, skipping")
                continue


            camera_dir = os.path.join(data_root, camera_name)
            mask_dir = os.path.join(camera_dir, "mask", "head")
            os.makedirs(mask_dir, exist_ok=True)


            mask_path = os.path.join(mask_dir, "0000.png")
            from PIL import Image
            mask_image = Image.fromarray(head_mask)
            mask_image.save(mask_path)

            masks.append(head_mask)
            valid_camera_names.append(camera_name)

        except Exception as e:
            logger.error(f"Failed to create head mask for camera {camera_name}: {e}")
            continue

    logger.info(f"Created {len(masks)} head masks for {data_type} data")


    if data_type == 'gt' and len(masks) > 0:
        try:
            logger.debug("Applying masks to RGB images...")
            create_masked_rgb_batch(
                data_root=data_root,
                masks=masks,
                camera_names=valid_camera_names,
                background_color=background_color
            )
            logger.info("RGB masking complete for GT data")

        except Exception as e:
            logger.error(f"Failed to process RGB images for GT data: {e}")
