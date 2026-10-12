import cv2
import numpy as np
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

DATASET_ROOT = Path("yolo_keypoints")

# Only augment the training set
SPLIT = "train"

# Your YOLO class ID for thermo
THERMO_CLASS_ID = 1

# Rotations to create
ROTATIONS = {
    "rot90cw": -1,     # 90 degrees clockwise
    "rot90ccw": 1,     # 90 degrees counter-clockwise
}


# ============================================================
# ROTATE A YOLO LABEL
# ============================================================

def rotate_label_line(line, direction):
    """
    Rotate one YOLOv8 keypoint annotation by 90 degrees.

    Expected format:

        class
        bbox_x
        bbox_y
        bbox_width
        bbox_height
        kp1_x kp1_y kp1_visibility
        kp2_x kp2_y kp2_visibility
        kp3_x kp3_y kp3_visibility
        kp4_x kp4_y kp4_visibility

    Total = 17 values.

    IMPORTANT:
    Keypoint ordering is preserved.

    Keypoint 0 remains keypoint 0.
    Keypoint 1 remains keypoint 1.
    Keypoint 2 remains keypoint 2.
    Keypoint 3 remains keypoint 3.
    """

    values = list(map(float, line.split()))

    if len(values) != 17:
        raise ValueError(
            f"Expected 17 values, got {len(values)}"
        )

    # --------------------------------------------------------
    # Class
    # --------------------------------------------------------

    class_id = int(values[0])

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    x_center = values[1]
    y_center = values[2]
    box_width = values[3]
    box_height = values[4]

    if direction == -1:

        # 90 degrees clockwise
        #
        # (x, y) -> (1-y, x)

        new_bbox_x = 1.0 - y_center
        new_bbox_y = x_center

    else:

        # 90 degrees counter-clockwise
        #
        # (x, y) -> (y, 1-x)

        new_bbox_x = y_center
        new_bbox_y = 1.0 - x_center

    # Width and height swap after 90 degree rotation
    new_bbox_width = box_height
    new_bbox_height = box_width

    # --------------------------------------------------------
    # Keypoints
    # --------------------------------------------------------

    keypoints = np.array(
        values[5:],
        dtype=np.float32
    ).reshape(4, 3)

    new_keypoints = []

    for x, y, visibility in keypoints:

        if direction == -1:

            # 90 degrees clockwise
            new_x = 1.0 - y
            new_y = x

        else:

            # 90 degrees counter-clockwise
            new_x = y
            new_y = 1.0 - x

        # IMPORTANT:
        #
        # Do NOT reorder the keypoints.
        #
        # The transformed point is appended in exactly
        # the same position as the original point.

        new_keypoints.extend([
            new_x,
            new_y,
            visibility
        ])

    # --------------------------------------------------------
    # Construct new label
    # --------------------------------------------------------

    new_values = [
        class_id,
        new_bbox_x,
        new_bbox_y,
        new_bbox_width,
        new_bbox_height,
        *new_keypoints
    ]

    return " ".join(
        str(round(value, 6))
        for value in new_values
    )


# ============================================================
# AUGMENT DATASET
# ============================================================

def augment_split(split):

    images_dir = DATASET_ROOT / split / "images"
    labels_dir = DATASET_ROOT / split / "labels"

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not images_dir.exists():

        print(
            f"ERROR: Images directory does not exist:\n"
            f"{images_dir}"
        )

        return

    if not labels_dir.exists():

        print(
            f"ERROR: Labels directory does not exist:\n"
            f"{labels_dir}"
        )

        return

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    image_paths = [
        path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in image_extensions
    ]

    print("=" * 60)
    print("THERMO ROTATION AUGMENTATION")
    print("=" * 60)

    print(f"Dataset: {DATASET_ROOT}")
    print(f"Split:   {split}")
    print(f"Images:  {len(image_paths)}")
    print()

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    thermo_images = 0
    augmented_images = 0
    skipped_images = 0

    # --------------------------------------------------------
    # Process every image
    # --------------------------------------------------------

    for image_path in image_paths:

        label_path = (
            labels_dir
            / f"{image_path.stem}.txt"
        )

        # ----------------------------------------------------
        # Need a label file
        # ----------------------------------------------------

        if not label_path.exists():

            continue

        # ----------------------------------------------------
        # Read labels
        # ----------------------------------------------------

        with open(label_path, "r") as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        # ----------------------------------------------------
        # Check for thermo
        # ----------------------------------------------------

        has_thermo = False

        for line in lines:

            values = line.split()

            if len(values) == 0:
                continue

            class_id = int(float(values[0]))

            if class_id == THERMO_CLASS_ID:

                has_thermo = True
                break

        # ----------------------------------------------------
        # Skip non-thermo images
        # ----------------------------------------------------

        if not has_thermo:

            continue

        thermo_images += 1

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"WARNING: Could not read:\n"
                f"  {image_path}"
            )

            skipped_images += 1

            continue

        # ----------------------------------------------------
        # Generate both rotations
        # ----------------------------------------------------

        for suffix, direction in ROTATIONS.items():

            # ================================================
            # Rotate image
            # ================================================

            if direction == -1:

                # 90 degrees clockwise
                rotated_image = cv2.rotate(
                    image,
                    cv2.ROTATE_90_CLOCKWISE
                )

            else:

                # 90 degrees counter-clockwise
                rotated_image = cv2.rotate(
                    image,
                    cv2.ROTATE_90_COUNTERCLOCKWISE
                )

            # ================================================
            # Create filenames
            # ================================================

            new_stem = (
                f"{image_path.stem}_{suffix}"
            )

            new_image_path = (
                images_dir
                / f"{new_stem}{image_path.suffix}"
            )

            new_label_path = (
                labels_dir
                / f"{new_stem}.txt"
            )

            # ================================================
            # Avoid accidentally regenerating augmentations
            # ================================================

            if (
                new_image_path.exists()
                or new_label_path.exists()
            ):

                print(
                    f"Skipping existing:\n"
                    f"  {new_stem}"
                )

                continue

            # ================================================
            # Save rotated image
            # ================================================

            success = cv2.imwrite(
                str(new_image_path),
                rotated_image
            )

            if not success:

                print(
                    f"WARNING: Failed to save:\n"
                    f"  {new_image_path}"
                )

                continue

            # ================================================
            # Rotate annotations
            # ================================================

            new_lines = []

            for line in lines:

                try:

                    new_line = rotate_label_line(
                        line,
                        direction
                    )

                    new_lines.append(
                        new_line
                    )

                except Exception as e:

                    print(
                        f"WARNING: Could not transform "
                        f"label:\n"
                        f"  {label_path}\n"
                        f"  {e}"
                    )

            # ================================================
            # Save new annotations
            # ================================================

            with open(new_label_path, "w") as f:

                f.write(
                    "\n".join(new_lines)
                )

            augmented_images += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if thermo_images % 100 == 0:

            print(
                f"Processed {thermo_images} "
                f"thermo-positive images..."
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("AUGMENTATION COMPLETE")
    print("=" * 60)

    print(
        f"Thermo-positive images: {thermo_images}"
    )

    print(
        f"New rotated images:     {augmented_images}"
    )

    print(
        f"Skipped images:         {skipped_images}"
    )

    print(
        f"Expected new images:    {thermo_images * 2}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    augment_split(SPLIT)

    print()
    print("Done.")