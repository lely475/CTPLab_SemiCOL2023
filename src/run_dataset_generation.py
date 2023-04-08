import argparse
from dataloader.dataset_generator import Dataset_Generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile segmentation images")
    parser.add_argument(
        "--o", help="Defines the output directory.", type=str, required=True
    )
    parser.add_argument(
        "--data", help="Defines the data directory.", type=str, required=True
    )
    args = parser.parse_args()

    ds_gen = Dataset_Generator(data_dir=args.data, output_path=args.o)
    ds_gen.generate_ds(set="train", sgm_flag=True)
    ds_gen.generate_ds(set="train", sgm_flag=False)
    ds_gen.generate_ds(set="val", sgm_flag=True)
    ds_gen.generate_ds(set="val", sgm_flag=False)
