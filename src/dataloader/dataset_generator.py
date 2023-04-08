import numpy as np
import cv2
import pandas as pd
import os
from PIL import Image
import warnings
from utils.utils import convert_segmentation_mask_to_rgb
from tifffile.tifffile import TiffFile
import imagecodecs
from aicsimageio import AICSImage

warnings.simplefilter(action="ignore", category=FutureWarning)


class Dataset_Generator:
    """
    Create a tissue segmentation and tumor detection dataset of 300x300 patches
    from csv tile specifications, save to output folder
    """

    def __init__(self, data_dir: str, output_path: str, debug: bool = False) -> None:
        self.output_path = output_path
        self.data_path = data_dir
        self.debug = debug

    def generate_ds(self, set: str, sgm_flag: bool) -> None:
        """Generate ds for specified set and mode (segmentation vs tum det)"""

        print(
            f"Generate {set} dataset for {'segmentation' if sgm_flag else 'tumor detection'}"
        )

        # Init
        self.set = set
        self.sgm_flag = sgm_flag
        self.overlap = 0.5 if self.sgm_flag else 0.0
        self.min_relevant_pixels = 0.01 if self.sgm_flag else 0.5
        self.csv_path = (
            f"metadata_csvs/{set}_sgm_tile_specs.csv"
            if self.sgm_flag
            else f"metadata_csvs/{set}_tum_det_tile_specs.csv"
        )
        self.tile_df_name = f"{set}_sgm.csv" if sgm_flag else f"{set}_tum_det.csv"

        # read in csv
        df = pd.read_csv(self.csv_path)

        # Set output folder
        self.output_path = f"{self.output_path}/{set}"

        # Tile all images specified in set_df
        if self.sgm_flag:
            tile_df = self.tile_sgm(df)
        else:
            tile_df = self.tile_tum_det(df)

        # Save tile specs to csv
        tile_df.to_csv(
            f"{os.path.dirname(self.csv_path)}/{self.tile_df_name}", index=False
        )

    def tile_sgm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tile segmentation images and masks, save to output folder"""

        img_paths = df["image_path"].unique().tolist()
        mask_paths = df["mask_path"].unique().tolist()

        f = 1 / df["downscale_factor"].unique()[0]
        tile_size = df["tile_size"].unique()[0]

        for img_path, mask_path in zip(img_paths, mask_paths):
            # Load
            img = cv2.imread(f"{self.data_path}/{img_path}")
            mask = np.array(Image.open(f"{self.data_path}/{mask_path}"))

            # Downscale
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
            if self.debug:
                thumbnail = cv2.addWeighted(
                    img, 0.7, convert_segmentation_mask_to_rgb(mask, bgr=True), 0.3, 0
                )

            # Tile
            print("Tiling image", img_path)
            img_df = df[df["image_path"] == img_path]
            tile_idx = 0
            for idx, row in img_df.iterrows():
                x, y = row["x"], row["y"]
                tile = img[y : y + tile_size, x : x + tile_size]
                tile_mask = mask[y : y + tile_size, x : x + tile_size]

                # create output dir and save tile
                img_o_dir = f"{self.output_path}/{os.path.splitext(img_path)[0]}"
                mask_o_dir = f"{self.output_path}/{os.path.splitext(mask_path)[0]}"
                os.makedirs(img_o_dir, exist_ok=True)
                os.makedirs(mask_o_dir, exist_ok=True)
                tile_path = f"{img_o_dir}/{tile_idx}.npy"
                tile_mask_path = f"{mask_o_dir}/{tile_idx}.npy"
                np.save(tile_path, tile)
                np.save(tile_mask_path, tile_mask)

                df.at[idx, "tile_path"] = tile_path
                df.at[idx, "tile_mask_path"] = tile_mask_path
                df[:idx].to_csv(
                    f"{os.path.dirname(self.csv_path)}/{self.tile_df_name}", index=False
                )
                if self.debug:
                    thumbnail = self.mark_tile_in_thumbnail(
                        thumbnail, tile_idx, x, y, tile_size
                    )
                tile_idx += 1

            if self.debug:
                img_name = os.path.basename(tile_path.split("/")[-2])
                tbn_fp = f"{'/'.join(tile_path.split('/')[:-3])}/{img_name}_tiles_{tile_idx}.png"
                print("Create thumbnail", tbn_fp)
                if not os.path.exists(tbn_fp):
                    tbn_f = 1 / 3 if self.sgm_flag else 1 / 8
                    thumbnail = cv2.resize(
                        thumbnail,
                        None,
                        fx=tbn_f,
                        fy=tbn_f,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    cv2.imwrite(tbn_fp, thumbnail)

        return df

    def tile_tum_det(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tile tumor detection ome.tiff files, save to output folder"""

        ome_tiff_paths = df["ome_tiff_path"].unique().tolist()
        tile_size = df["tile_size"].unique()[0]

        for ome_tiff_path in ome_tiff_paths:
            # Load
            tiff = self.load_ome_tiff(f"{self.data_path}/{ome_tiff_path}")
            f = df["image_resize_factor"].unique()[0]

            # Downscale
            i = cv2.INTER_LINEAR if f > 1 else cv2.INTER_AREA
            tiff = cv2.resize(tiff, None, fx=f, fy=f, interpolation=i)
            if self.debug:
                thumbnail = tiff.copy()

            # Tile
            print("Tiling ome tiff", ome_tiff_path)
            tiff_df = df[df["ome_tiff_path"] == ome_tiff_path]
            tile_idx = 0
            for idx, row in tiff_df.iterrows():
                x, y = row["x"], row["y"]
                tile = tiff[y : y + tile_size, x : x + tile_size]
                tile_o_dir = f"{self.output_path}/{os.path.splitext(ome_tiff_path)[0].split('.')[0]}"
                os.makedirs(tile_o_dir, exist_ok=True)
                tile_path = f"{tile_o_dir}/{tile_idx}.npy"
                np.save(tile_path, tile)
                df.at[idx, "tile_path"] = tile_path

                df[:idx].to_csv(
                    f"{os.path.dirname(self.csv_path)}/{self.tile_df_name}", index=False
                )
                if self.debug:
                    thumbnail = self.mark_tile_in_thumbnail(
                        thumbnail, tile_idx, x, y, tile_size
                    )
                tile_idx += 1

            if self.debug:
                tiff_name = os.path.basename(tile_path.split("/")[-2])
                tbn_fp = f"{'/'.join(tile_path.split('/')[:-2])}/{tiff_name}_tiles_{tile_idx}.png"
                if not os.path.exists(tbn_fp):
                    tbn_f = 1 / 3 if self.sgm_flag else 1 / 8
                    thumbnail = cv2.resize(
                        thumbnail,
                        None,
                        fx=tbn_f,
                        fy=tbn_f,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    cv2.imwrite(tbn_fp, thumbnail)
        return df

    def load_ome_tiff(self, path: str) -> np.ndarray:
        """Load ome tiff file"""

        try:
            with TiffFile(path) as tif:
                print(f"Reading image {path}")
                img = tif.series[0].levels[1].asarray()
            if "Anonymized" in path:
                img = np.moveaxis(img, source=0, destination=-1)
        except imagecodecs._jpeg2k.Jpeg2kError:
            print("Jpeg2kError using AICSImage instead of tifffile")
            img = AICSImage(path)
            img = img.get_image_data("YXC", S=0, T=0)
            img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        return img

    def mark_tile_in_thumbnail(
        self, thumbnail: np.ndarray, tile_idx: int, xmin: int, ymin: int, tile_size: int
    ) -> np.ndarray:
        """Mark extracted tiles in thumbnail for debugging"""

        cv2.rectangle(
            thumbnail,
            (xmin, ymin),
            (xmin + tile_size - 1, ymin + tile_size - 1),
            (0, 0, 0),
            thickness=3 if self.sgm_flag else 8,
        )
        if self.sgm_flag:
            text_size = cv2.getTextSize(str(tile_idx), cv2.FONT_HERSHEY_SIMPLEX, 2, 8)[
                0
            ]
            cv2.putText(
                thumbnail,
                str(tile_idx),
                (
                    xmin + int(tile_size / 2) - int(text_size[0] / 2),
                    ymin + int(tile_size / 2) + int(text_size[1] / 2),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                color=(0, 0, 0),
                fontScale=2,
                thickness=8,
            )
        return thumbnail
