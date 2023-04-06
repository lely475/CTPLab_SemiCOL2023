# Multi-task learning for tissue segmentation and tumor detection in colorectal cancer histology slides
The semi-supervised learning for CRC detection ([SemiCOL](https://www.semicol.org/)) challenge 2023 provides partially annotated data to encourage the development of automated solutions for tissue segmentation and tumor detection. We propose a U-Net based multi-task model combined with channel-wise and image-statistics-based color augmentations, as well as test-time augmentation (see Figure 1), as a candidate solution to the SemiCOL challenge. Our approach achieved a multi-task Dice score of .8655 (Arm 1) and .8515 (Arm 2) for tissue segmentation and AUROC of .9725  (Arm 1) and 0.9750 (Arm 2) for tumor detection on the challenge validation set.

![inference_pipeline3](https://user-images.githubusercontent.com/62755943/230338160-5ae2bdb8-640a-4fe3-9ee9-eff9fdf2bde0.png)
Figure 1: Proposed approach to tissue segmentation and tumor detection: We train a multi-task U-Net-based model for segmentation and classification, but only use the segmentation branch during inference. The tumor detection score is computed based on the amount of predicted class pixels.

## Setup
```
pip install -r requirements.txt
```

## Generate tiles for training
Add location of the data folder (parent folder of `01_MANUAL` and `02_WEAK`) and desired output folder, where the generated tiles will be stored. Approximate required storage for the tiles is 175 GB.
```
run_data_generation.py --o path/to/output/dir --data path/to/data/dir
```

## Train model
Train model on tiled challenge training set, tensorboard outputs and ckpts are logged to the output folder. 
```
run_train_multi_task_learner.py --o path/to/output/dir
```

## Validate model on internal validation set
To validate the model on the internal validation set, provide an output folder to which the results will be logged.
```
run_validation.py --o path/to/output/dir
```
