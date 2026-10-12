import json
import shutil
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

COCO_ROOT = Path("COCO_Rebalanced")
OUTPUT_ROOT = Path("final_T1_dataset")


# Input COCO folder -> output YOLO folder
SPLITS = {
    "train": "train",
    "valid": "valid",
    "test": "test",
}


# COCO category ID -> YOLO class ID
CATEGORY_MAP = {
    1: 0,   # bpm
    2: 1,   # thermo
}


CLASS_NAMES = {
    0: "bpm",
    1: "thermo",
}


# ============================================================
# CORNER FUNCTIONS
# ============================================================

def order_corners(points):
    """
    Order four corners consistently:

        0 -------- 1
        |          |
        |          |
        3 -------- 2

    0 = top-left
    1 = top-right
    2 = bottom-right
    3 = bottom-left
    """

    points = np.array(points, dtype=np.float32)

    # Sort by y coordinate
    sorted_by_y = points[np.argsort(points[:, 1])]

    top = sorted_by_y[:2]
    bottom = sorted_by_y[2:]

    # Sort each pair by x coordinate
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]

    return np.array([
        top[0],       # top-left
        top[1],       # top-right
        bottom[1],    # bottom-right
        bottom[0],    # bottom-left
    ])


def get_four_corners(segmentation):
    """
    Convert a COCO segmentation polygon into exactly
    four corner points.
    """

    points = np.array(
        segmentation,
        dtype=np.float32
    ).reshape(-1, 2)

    # --------------------------------------------------------
    # Remove repeated final point
    # --------------------------------------------------------

    if (
        len(points) > 1
        and np.allclose(points[0], points[-1])
    ):
        points = points[:-1]

    # --------------------------------------------------------
    # Remove consecutive duplicate points
    # --------------------------------------------------------

    cleaned = []

    for point in points:

        if (
            len(cleaned) == 0
            or not np.allclose(point, cleaned[-1])
        ):
            cleaned.append(point)

    points = np.array(
        cleaned,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # If there are already exactly four points,
    # use them directly
    # --------------------------------------------------------

    if len(points) == 4:
        return order_corners(points)

    # --------------------------------------------------------
    # Try to reduce polygon to four points
    # --------------------------------------------------------

    contour = points.reshape(-1, 1, 2)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    # Try several approximation tolerances
    for epsilon_fraction in [
        0.005,
        0.01,
        0.015,
        0.02,
        0.03,
        0.05,
        0.08,
    ]:

        approx = cv2.approxPolyDP(
            contour,
            epsilon_fraction * perimeter,
            True
        )

        if len(approx) == 4:

            corners = approx.reshape(
                4,
                2
            )

            return order_corners(corners)

    # --------------------------------------------------------
    # Fallback: minimum-area rectangle
    # --------------------------------------------------------

    rect = cv2.minAreaRect(points)

    corners = cv2.boxPoints(rect)

    return order_corners(corners)


# ============================================================
# BOUNDING BOX
# ============================================================

def normalise_bbox(bbox, image_width, image_height):
    """
    Convert COCO bounding box:

        [x, y, width, height]

    to YOLO:

        [x_center, y_center, width, height]

    and normalise to 0-1.
    """

    x, y, width, height = bbox

    x_center = x + width / 2
    y_center = y + height / 2

    return [
        x_center / image_width,
        y_center / image_height,
        width / image_width,
        height / image_height,
    ]


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(input_split, output_split):

    input_dir = COCO_ROOT / input_split
    json_path = input_dir / "_annotations.coco.json"

    output_images = (
        OUTPUT_ROOT
        / output_split
        / "images"
    )

    output_labels = (
        OUTPUT_ROOT
        / output_split
        / "labels"
    )

    output_images.mkdir(
        parents=True,
        exist_ok=True
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 60)
    print(f"Processing: {input_split}")
    print(f"Input:      {input_dir}")
    print(f"Output:     {OUTPUT_ROOT / output_split}")
    print("=" * 60)

    # --------------------------------------------------------
    # Load COCO JSON
    # --------------------------------------------------------

    with open(json_path, "r") as f:
        coco = json.load(f)

    # --------------------------------------------------------
    # Create image lookup
    # --------------------------------------------------------

    images = {
        image["id"]: image
        for image in coco["images"]
    }

    # --------------------------------------------------------
    # Group annotations by image
    # --------------------------------------------------------

    annotations_by_image = {}

    for annotation in coco["annotations"]:

        image_id = annotation["image_id"]

        annotations_by_image.setdefault(
            image_id,
            []
        ).append(annotation)

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    converted_images = 0
    converted_annotations = 0
    skipped_annotations = 0

    for image_id, image_info in images.items():

        filename = image_info["file_name"]

        image_width = image_info["width"]
        image_height = image_info["height"]

        # ----------------------------------------------------
        # Find source image
        # ----------------------------------------------------

        source_image = input_dir / filename

        if not source_image.exists():

            # Try just the filename in case COCO contains
            # a relative path
            source_image = (
                input_dir
                / Path(filename).name
            )

        if not source_image.exists():

            print(
                f"WARNING: Image not found: {filename}"
            )

            continue

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        destination_image = (
            output_images
            / Path(filename).name
        )

        shutil.copy2(
            source_image,
            destination_image
        )

        converted_images += 1

        # ----------------------------------------------------
        # Generate YOLO labels
        # ----------------------------------------------------

        label_lines = []

        annotations = annotations_by_image.get(
            image_id,
            []
        )

        for annotation in annotations:

            category_id = annotation["category_id"]

            # Ignore categories that aren't in our map
            if category_id not in CATEGORY_MAP:

                skipped_annotations += 1

                continue

            class_id = CATEGORY_MAP[
                category_id
            ]

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            bbox = normalise_bbox(
                annotation["bbox"],
                image_width,
                image_height
            )

            # ------------------------------------------------
            # Segmentation
            # ------------------------------------------------

            segmentation = annotation.get(
                "segmentation"
            )

            if not segmentation:

                skipped_annotations += 1

                continue

            # COCO polygon format:
            #
            # [
            #     [x1, y1, x2, y2, ...]
            # ]
            #
            points = segmentation[0]

            try:

                corners = get_four_corners(
                    points
                )

            except Exception as e:

                print(
                    f"WARNING: Could not convert "
                    f"annotation {annotation['id']}: {e}"
                )

                skipped_annotations += 1

                continue

            # ------------------------------------------------
            # Normalise keypoints
            # ------------------------------------------------

            keypoints = []

            for x, y in corners:

                keypoints.extend([
                    x / image_width,
                    y / image_height,
                    2
                ])

            # ------------------------------------------------
            # Construct YOLO pose label
            # ------------------------------------------------

            values = [
                class_id,
                *bbox,
                *keypoints
            ]

            label_line = " ".join(
                str(round(value, 6))
                for value in values
            )

            label_lines.append(
                label_line
            )

            converted_annotations += 1

        # ----------------------------------------------------
        # Write label file
        #
        # Filename is based on image filename:
        #
        # image123.jpg -> image123.txt
        # ----------------------------------------------------

        label_filename = (
            Path(filename).stem
            + ".txt"
        )

        label_path = (
            output_labels
            / label_filename
        )

        with open(label_path, "w") as f:

            f.write(
                "\n".join(label_lines)
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print(f"Images converted:       {converted_images}")
    print(f"Annotations converted:  {converted_annotations}")
    print(f"Annotations skipped:    {skipped_annotations}")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONVERT TRAIN / VALIDATION / TEST
# ============================================================

for input_split, output_split in SPLITS.items():

    process_split(
        input_split,
        output_split
    )


# ============================================================
# CREATE data.yaml
# ============================================================

yaml_path = OUTPUT_ROOT / "data.yaml"

yaml_text = f"""path: {OUTPUT_ROOT.resolve()}

train: train/images
val: validation/images
test: test/images

kpt_shape: [4, 3]

names:
  0: bpm
  1: thermo

flip_idx: [1, 0, 3, 2]
"""

with open(yaml_path, "w") as f:
    f.write(yaml_text)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 60)
print("CONVERSION COMPLETE")
print("=" * 60)

print(f"Output directory:")
print(OUTPUT_ROOT.resolve())

print()
print("Created:")
print("  train/images")
print("  train/labels")
print("  validation/images")
print("  validation/labels")
print("  test/images")
print("  test/labels")
print("  data.yaml")