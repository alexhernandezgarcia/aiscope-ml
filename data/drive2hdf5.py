from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import zipfile
from pathlib import Path
from tqdm import tqdm

import h5py

import numpy as np
import pickle
from PIL import Image
from PIL.ImageOps import pad as pil_pad

from comet_ml import Experiment

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny


def main(argv=None):

    # Initialize New Comet Experiment
    exp = Experiment(project_name="aiscope-ml", display_summary_level=0)

    # Log arguments
    exp.log_parameters(vars(args))

    # Extract ZIP
    if args.data_zip:
        print(f"Extracting {args.data_zip}...")
        with zipfile.ZipFile(args.data_zip, "r") as f_zip:
            f_zip.extractall(args.data_dir)

    # Read data
    print(f"Reading images...")
    data_paths = []
    for ext in ["jpg", "jpeg"]:
        data_paths.extend(list(Path(args.data_dir).glob(r"**/image_*.{}".format(ext))))
    n_images = len(data_paths)
    print(f"Found {n_images} images in {args.data_dir}.")

    # Discard images without mask
    print(f"Discarding images without a corresponding mask...")
    del_list = []
    for path in tqdm(data_paths):
        mask_path = Path(str(path).replace("image_", "mask_").replace("jpg", "png"))
        if not mask_path.exists():
            print(f"{mask_path} not found\n{path} will not be included in the data set")
            del_list.append(path)
    for path in del_list:
        data_paths.remove(path)
    n_images = len(data_paths)
    print(f"Number of images after removal of missing masks: {n_images}")

    # Get image sizes
    if args.height is None and args.width is None:
        print(f"Height and width will be determine by the median...")
        heights = []
        widths = []
        for f in tqdm(data_paths):
            img_pil = Image.open(f).convert("RGBA")
            heights.append(img_pil.height)
            widths.append(img_pil.width)
        height = np.median(heights)
        width = np.median(widths)
    else:
        if args.height is not None:
            height = args.height
        else:
            height = args.width
        if args.width is not None:
            width = args.width
        else:
            width = args.height
    n_channels = 4
    print(f"Images shape: ({height} x {width} {n_channels}")

    # Open HDF5 file
    with h5py.File(args.output_hdf5, "w") as hdf5_file:

        # Create the Groups to store the Datasets
        grp_tr = hdf5_file.create_group("train")

        # Create the datasets that will contain the image data, mask and metadata
        images_tr = grp_tr.create_dataset(
            "images", shape=(n_images, height, width, n_channels), dtype=np.uint8
        )
        masks_tr = grp_tr.create_dataset(
            "masks", shape=(n_images, height, width, n_channels), dtype=np.uint8
        )
        paths_tr = grp_tr.create_dataset(
            "paths", shape=(n_images, 1), dtype=h5py.special_dtype(vlen=str)
        )

        # Permute the indices in order to shuffle the images in the HDF5 file
        if args.shuffle:
            indices_tr = np.random.permutation(n_images)
            print(f"Indices will be shuffled...")
        else:
            indices_tr = range(n_images)

        # Fill data
        print(f"Iterating over images...")
        for idx, path in enumerate(tqdm(data_paths)):
            img_pil_rgba = Image.open(path).convert("RGBA")
            mask_path = Path(str(path).replace("image_", "mask_").replace("jpg", "png"))
            mask_pil_rgba = Image.open(mask_path).convert("RGBA")
            img_cropped, mask_cropped, _ = crop_circle(
                img_pil_rgba,
                mask_pil_rgba,
                draw_circles=False,
                scale=args.scale,
                max_circles=1,
            )
            img_resized = img_cropped.resize((width, height))
            mask_resized = mask_cropped.resize((width, height))

            images_tr[idx, :, :, :] = img_resized
            masks_tr[idx, :, :, :] = mask_resized
            paths_tr[:, 0] = str(data_paths[idx])

            # Store images
            img_resized.save(Path(args.output_dir) / "image_{}.png".format(idx))
            mask_resized.save(Path(args.output_dir) / "mask{}.png".format(idx))
            f_filename = open(Path(args.output_dir) / "filename_{}.png".format(idx),
            "w")
            f_filename.write(str(path))
            f_filename.close()

            if args.max_n_images and idx > args.max_n_images:
                break


def crop_circle(
    img_orig, mask_orig, scale=10, margin=0.05, max_circles=3, draw_circles=False
):
    """
    Arguments
    ---------

    img_orig : PIL (RGB)
        The original image

    scale : int
        The factor to downscale the image before applying the Hough transform

    margin : float
        Percentage of margin around the circle before cropping

    max_cirlces : int
        Maximum number of circles detected (for debugging and viz purposes)

    draw_circles : bool
        For debugging and viz purposes

    See: https://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html
    """
    # Resize image
    (width, height) = (img_orig.width // scale, img_orig.height // scale)
    img_lowres = img_orig.resize((width, height))

    # Convert to grayscale
    img_gray = np.asarray(img_lowres.convert("L"))

    # Detect edges
    img_edges = canny(img_gray, sigma=1, low_threshold=10, high_threshold=50)

    # Detect circle
    radius_max = np.min(img_lowres.size)
    radius_min = np.min(img_lowres.size) / 4
    hough_radii = np.arange(radius_min, radius_max, 1)
    hough_res = hough_circle(img_edges, hough_radii)
    accums, cxs, cys, radii = hough_circle_peaks(
        hough_res, hough_radii, total_num_peaks=max_circles, normalize=True
    )

    # Draw circle
    if draw_circles:
        for cx, cy, radius in zip(cxs, cys, radii):
            cx = scale * cx
            cy = scale * cy
            radius = scale * radius
            x_min = cx - radius
            x_max = x_min + 2 * radius
            y_min = cy - radius
            y_max = y_min + 2 * radius
            draw = ImageDraw.Draw(img_orig)
            draw.arc(
                xy=[(x_min, y_min), (x_max, y_max)],
                start=0,
                end=360,
                fill=(255, 20, 20),
                width=20,
            )

    # Crop
    cx = scale * cxs[0]
    cy = scale * cys[0]
    radius = scale * radii[0]
    radius_margin = radius * (1.0 + margin)
    x_min = int(np.max([cx - radius_margin, 0]))
    x_max = int(np.min([cx + radius_margin, img_orig.width]))
    y_min = int(np.max([cy - radius_margin, 0]))
    y_max = int(np.min([cy + radius_margin, img_orig.height]))
    img_cropped = img_orig.crop((x_min, y_min, x_max, y_max))
    mask_cropped = mask_orig.crop((x_min, y_min, x_max, y_max))

    # Make image square
    cx = cx - x_min
    cy = cy - y_min
    size = np.max(img_cropped.size)
    img_cropped = pil_pad(
        img_cropped, size=(size, size), color=(0, 0, 0), centering=(cx, cy)
    )
    mask_cropped = pil_pad(
        mask_cropped, size=(size, size), color=(0, 0, 0), centering=(cx, cy)
    )

    return img_cropped, mask_cropped, img_edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="drive", help="Input directory")
    parser.add_argument("--data_zip", type=str, default=None, help="Input directory")
    parser.add_argument(
        "--output_hdf5", type=str, default=None, help="Output HDF5 file"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--height", type=int, default=None, help="Height of output images"
    )
    parser.add_argument(
        "--width", type=int, default=None, help="Width of output images"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        dest="shuffle",
        help="True to re-shuffle the train, test and validation partitions",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=20,
        help="The factor to downscale the image before applying the Hough transform",
    )
    parser.add_argument(
        "--max_n_images",
        type=int,
        default=None,
        help="Maximum number of images to process, for debugging",
    )
    args, unparsed = parser.parse_known_args()
    print("Args:\n" + "\n".join([f"    {k:20}: {v}" for k, v in vars(args).items()]))
    main()
