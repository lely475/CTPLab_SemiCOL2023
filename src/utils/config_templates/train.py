from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from typing import Dict, Any, List, Optional


@dataclass
class PathConfig(DataClassJsonMixin):
    """Config template for training paths"""

    train_sgm_csv: str
    val_sgm_csv: str
    train_tum_det_csv: str
    val_tum_det_csv: str
    output_path: str

    def __str__(self) -> str:
        return (
            f"\t\tTrain segmentation csv path: {self.train_sgm_csv}\n"
            f"\t\tValidation segmentation csv path: {self.val_sgm_csv}\n"
            f"\t\tTrain tumor detection csv path: {self.train_tum_det_csv}\n"
            f"\t\tValidation tumor detection csv path: {self.val_tum_det_csv}\n"
            f"\t\tOutput path: {self.output_path}\n"
        )


@dataclass
class ExperimentConfig(DataClassJsonMixin):
    """Config template for experiment params"""

    seed: int
    num_workers: int
    epochs: int
    batch_size: int
    sgm_class_weights: List[float]
    full_set: bool
    sgm_ratio: float = 0.5
    patch_size: int = 300

    def __str__(self) -> str:
        return (
            f"\t\tRandom seed: {self.seed}\n"
            f"\t\tNumber of workers: {self.num_workers}\n"
            f"\t\tNumber of epochs: {self.epochs}\n"
            f"\t\tBatch size: {self.batch_size}\n"
            f"\t\tPatch size (before crop): {self.patch_size}\n"
            f"\t\tRatio of sgm samples in batch: {self.sgm_ratio}\n"
            f"\t\tUse combined train+val set: {self.full_set}\n"
            f"\t\tSegmentation dice class weighting: {self.sgm_class_weights}\n"
        )


@dataclass
class LossConfig(DataClassJsonMixin):
    """Config template for loss params"""

    sgm_weight: float

    def __str__(self) -> str:
        return f"\t\tLoss weight sgm, tum det: {self.sgm_weight}, {1-self.sgm_weight}\n"


@dataclass
class OptimizerConfig(DataClassJsonMixin):
    """Config template for optimizer params"""

    name: str
    lr: float
    scheduler: Optional[str]
    params: Dict[str, Any]
    scheduler_params: Dict[str, Any]

    def __str__(self) -> str:
        return (
            f"\t\tName: {self.name}\n"
            f"\t\tLearning rate factor: {self.lr}\n"
            f"\t\tLearning rate schedule: {self.scheduler}\n"
            f"\t\tOptimizer parameters: {self.params}\n"
            f"\t\tScheduler parameters: {self.scheduler_params}\n"
        )


@dataclass
class DataAugmentationConfig(DataClassJsonMixin):
    """Config template for data augmentation params"""

    aug_probability: float
    zoom_factor: float
    bright_contrast_variation: float
    img_stat_ref_images: str

    def __str__(self) -> str:
        return (
            f"\t\tProbability for each augmentation: {self.aug_probability}\n"
            f"\t\tZoom Factor: +/- {self.zoom_factor}\n"
            f"\t\tBrightness+contrast variation: +/- {self.bright_contrast_variation}\n"
            f"\t\tReference imgs for image statistics augmentation: {self.img_stat_ref_images}\n"
        )


@dataclass
class ModelArchitectureConfig(DataClassJsonMixin):
    """Config template for model architecture"""

    num_classes: int
    start_filter: int

    def __str__(self) -> str:
        return (
            f"\t\tNumber of segmentation classes: {self.num_classes}\n"
            f"\t\tNumber of initial filter: {self.start_filter}\n"
        )


@dataclass
class ModelConfig(DataClassJsonMixin):
    """Config template for model params"""

    arch: ModelArchitectureConfig
    input_size: int

    def __str__(self) -> str:
        return (
            f"\t\tModel architecture: \n{self.arch}\n"
            f"\t\tInput size: {self.input_size}\n"
        )


@dataclass
class TrainConfig(DataClassJsonMixin):
    "Config template for training Multi Class Learner"

    paths: PathConfig
    experiment: ExperimentConfig
    optimizer: OptimizerConfig
    loss: LossConfig
    data_augmentation: DataAugmentationConfig
    model: ModelConfig

    def __str__(self) -> str:
        return (
            f"\n\tMulti task training configuration:\n"
            f"\t---------------------------------------\n"
            f"\tExperiment configuration\n"
            f"{self.experiment}\n"
            f"\t-----------------\n"
            f"\tPaths\n"
            f"{self.paths}\n"
            f"\t-----------------\n"
            f"\tModel parameters\n"
            f"{self.model}\n"
            f"\t-----------------\n"  #
            f"\tOptimizer parameters\n"
            f"{self.optimizer}\n"
            f"\t-----------------\n"
            f"\tLoss parameters\n"
            f"{self.loss}\n"
            f"\t-----------------\n"
            f"\tData Augmentation parameters\n"
            f"{self.data_augmentation}\n"
        )
