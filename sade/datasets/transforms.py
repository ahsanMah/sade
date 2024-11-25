from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

# from monai.transforms import *
from monai.transforms.compose import Compose
from monai.transforms.croppad.array import CenterSpatialCrop
from monai.transforms.croppad.dictionary import (
    CenterSpatialCropd,
    DivisiblePadd,
    SpatialCropd,
)
from monai.transforms.intensity.dictionary import (
    RandGibbsNoised,
    RandHistogramShiftd,
    RandKSpaceSpikeNoised,
    RandRicianNoised,
    RandScaleIntensityd,
    RandStdShiftIntensityd,
    ScaleIntensityRangePercentilesd,
)
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.smooth_field.dictionary import RandSmoothFieldAdjustIntensityd
from monai.transforms.spatial.array import RandDeformGrid, Resample
from monai.transforms.spatial.dictionary import (
    RandAffined,
    Spacingd,
)
from monai.transforms.transform import Randomizable, Transform
from monai.transforms.utility.dictionary import (
    EnsureChannelFirstd,
    RandLambdad,
    SqueezeDimd,
    TorchVisiond,
)
from monai.transforms.utils import create_control_grid, create_grid
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    ensure_tuple,
    fall_back_tuple,
)


class TumorGrowthGrid3D(RandDeformGrid):
    """
    Adapted from FastMRI
    """

    def __init__(
        self,
        spacing: Union[Sequence[float], float],
        magnitude_range: Tuple[float, float],
        device: Optional[torch.device] = None,
    ) -> None:
        max_tumor_size = magnitude_range[0]
        magnitude_range = magnitude_range[1:]
        super(TumorGrowthGrid3D, self).__init__(spacing, magnitude_range, device)

        self.max_tumor_size = max_tumor_size

    def randomize(self) -> None:
        h1, w1, d1 = self.R.randint(-self.max_tumor_size, self.max_tumor_size, size=3)
        h2, w2, d1 = self.R.randint(-self.max_tumor_size, self.max_tumor_size, size=3)
        self.center1 = np.array([h1, w1, d1, 1], dtype=np.float32).reshape(4, 1, 1, 1)
        self.center2 = np.array([h2, w2, d1, 1], dtype=np.float32).reshape(4, 1, 1, 1)
        self.rand_mag = self.R.uniform(self.magnitude[0], self.magnitude[1])

    def __call__(self, spatial_size: Sequence[int]) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            spatial_size: spatial size of the grid.
        """
        self.spacing = fall_back_tuple(self.spacing, (1.0,) * len(spatial_size))
        control_grid = create_control_grid(spatial_size, self.spacing)
        self.randomize()
        dist1 = np.sqrt(np.sum(np.square(control_grid - self.center1)[:3], axis=0) + 1e-6)
        dist2 = np.sqrt(np.sum(np.square(control_grid - self.center2)[:3], axis=0) + 1e-6)
        deform_mag = self.rand_mag / (dist1 + dist2)
        center = (self.center1 + self.center2) / 2.0
        deform = (control_grid - center)[:3] * deform_mag
        control_grid[: len(spatial_size)] -= deform
        # print(control_grid.shape, deform.shape)
        # print(self.center1, self.center2)

        control_grid = torch.as_tensor(
            np.ascontiguousarray(control_grid), device=self.device
        )
        return control_grid


class RandTumor(Randomizable, Transform):
    def __init__(
        self,
        spacing: Union[Tuple[float, float], float],
        max_tumor_size: float,
        magnitude_range: Tuple[float, float],
        prob: float = 0.1,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: Union[GridSampleMode, str] = GridSampleMode.BILINEAR,
        padding_mode: Union[GridSamplePadMode, str] = GridSamplePadMode.REFLECTION,
        device: Optional[torch.device] = None,
    ) -> None:
        self.deform_grid = TumorGrowthGrid3D(
            spacing=spacing,
            magnitude_range=(max_tumor_size, *magnitude_range),
            device=device,
        )
        self.resampler = Resample(device=device)

        self.spatial_size = spatial_size
        self.mode: GridSampleMode = GridSampleMode(mode)
        self.padding_mode: GridSamplePadMode = GridSamplePadMode(padding_mode)
        self.prob = prob
        self.do_transform = False

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ):
        self.deform_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self) -> None:
        self.do_transform = self.R.rand() < self.prob
        self.deform_grid.randomize()

    def __call__(
        self,
        img: Union[np.ndarray, torch.Tensor],
        spatial_size: Optional[Union[Tuple[int, int], int]] = None,
        mode: Optional[Union[GridSampleMode, str]] = None,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        sp_size = fall_back_tuple(spatial_size or self.spatial_size, img.shape[1:])
        self.randomize()
        if self.do_transform:
            grid = self.deform_grid(spatial_size=sp_size)
            grid = torch.nn.functional.interpolate(
                input=grid.unsqueeze(0),
                scale_factor=list(ensure_tuple(self.deform_grid.spacing)),
                mode=InterpolateMode.TRILINEAR.value,
                align_corners=False,
            )
            grid = CenterSpatialCrop(roi_size=sp_size)(grid[0])
        else:
            grid = create_grid(spatial_size=sp_size)
        return self.resampler(
            img,
            grid,
            mode=mode or self.mode,
            padding_mode=padding_mode or self.padding_mode,
        )


def get_train_transform(config):

    if config.data.spatial_dims == 3:
        spacing = [config.data.spacing_pix_dim] * 3
        return Compose(
            [
                LoadImaged("image", image_only=True),
                SqueezeDimd("image", dim=3),
                EnsureChannelFirstd("image"),
                SpatialCropd("image", roi_start=[11, 9, 0], roi_end=[172, 205, 152]),
                Spacingd("image", pixdim=spacing),
                DivisiblePadd("image", k=16),
                RandStdShiftIntensityd("image", (-0.05, 0.05)),
                RandScaleIntensityd("image", (-0.05, 0.05)),
                RandHistogramShiftd("image", num_control_points=[3, 5]),
                RandAffined(
                    "image",
                    prob=0.1,
                    rotate_range=[0.05, 0.05, 0.05],
                    translate_range=5,
                ),
                RandKSpaceSpikeNoised("image", prob=0.1),
                RandRicianNoised("image", prob=0.1, std=0.01, sample_std=True),
                RandGibbsNoised("image", prob=0.1, alpha=(0.0, 0.1)),
                ScaleIntensityRangePercentilesd(
                    "image",
                    lower=0.01,
                    upper=99.9,
                    b_min=-1.0,
                    b_max=1.0,
                    clip=True,
                    channel_wise=True,
                ),
            ]
        )
    else:
        crop_size = config.data.crop_size
        img_size = config.data.image_size
        assert (
            img_size[0] <= crop_size[0] and img_size[1] <= crop_size[1]
        ), f" Crop size = {crop_size} should upper bound image size = {img_size} as image is first (center) cropped, then resized."

        return Compose(
            [
                LoadImaged(
                    "image", image_only=True, converter=lambda image: image.convert("L")
                ),
                EnsureChannelFirstd("image"),
                CenterSpatialCropd("image", roi_size=crop_size),
                TorchVisiond("image", "Resize", size=img_size),
                RandStdShiftIntensityd("image", (-0.03, 0.03)),
                RandScaleIntensityd("image", (-0.03, 0.03)),
                RandHistogramShiftd("image", num_control_points=[3, 5]),
                # Adds Speckle-like Noise as it is multiplicative
                RandSmoothFieldAdjustIntensityd(
                    "image",
                    spatial_size=img_size,
                    rand_size=(img_size[0] // 2, img_size[1] // 2),
                    prob=0.1,
                    gamma=(0.8, 1.0),
                ),
                RandAffined(
                    "image",
                    prob=0.1,
                    rotate_range=0.05,
                    translate_range=5,
                ),
                ScaleIntensityRangePercentilesd(
                    "image",
                    lower=0.01,
                    upper=99.9,
                    b_min=-1.0,
                    b_max=1.0,
                    clip=True,
                    channel_wise=True,
                ),
            ]
        )


def get_val_transform(config):
    if config.data.spatial_dims == 3:
        spacing = [config.data.spacing_pix_dim] * 3
        return Compose(
            [
                LoadImaged("image", image_only=True),
                SqueezeDimd("image", dim=3),
                EnsureChannelFirstd("image"),
                SpatialCropd("image", roi_start=[11, 9, 0], roi_end=[172, 205, 152]),
                Spacingd("image", pixdim=spacing),
                DivisiblePadd("image", k=16),
                ScaleIntensityRangePercentilesd(
                    "image",
                    lower=0.01,
                    upper=99.9,
                    b_min=-1.0,
                    b_max=1.0,
                    clip=True,
                    channel_wise=True,
                ),
            ]
        )
    else:
        crop_size = config.data.crop_size
        img_size = config.data.image_size
        assert (
            img_size[0] <= crop_size[0] and img_size[1] <= crop_size[1]
        ), f" Crop size = {crop_size} should upper bound image size = {img_size} as image is first (center) cropped, then resized."

        return Compose(
            [
                LoadImaged(
                    "image", image_only=True, converter=lambda image: image.convert("L")
                ),
                EnsureChannelFirstd("image"),
                CenterSpatialCropd("image", roi_size=crop_size),
                TorchVisiond("image", "Resize", size=img_size),
                ScaleIntensityRangePercentilesd(
                    "image",
                    lower=0.01,
                    upper=99.9,
                    b_min=-1.0,
                    b_max=1.0,
                    clip=True,
                    channel_wise=True,
                ),
            ]
        )


def get_tumor_transform(config):
    spacing = [config.data.spacing_pix_dim] * 3
    deformer = RandTumor(
        spacing=1.0,
        max_tumor_size=5.0 / config.data.spacing_pix_dim,
        magnitude_range=(
            5.0 / config.data.spacing_pix_dim,
            15.0 / config.data.spacing_pix_dim,
        ),
        prob=1.0,
        spatial_size=config.data.image_size,
        padding_mode="zeros",
    )

    deformer.set_random_state(seed=0)

    return Compose(
        [
            LoadImaged("image", image_only=True),
            SqueezeDimd("image", dim=3),
            EnsureChannelFirstd("image"),
            SpatialCropd("image", roi_start=[11, 9, 0], roi_end=[172, 205, 152]),
            Spacingd("image", pixdim=spacing),
            DivisiblePadd("image", k=8),
            RandLambdad("image", deformer),
        ]
    )


def get_lesion_transform(config):
    spacing = [config.data.spacing_pix_dim] * 3

    return Compose(
        [
            LoadImaged(["image", "label"], image_only=True),
            SqueezeDimd(["image"], dim=3),
            EnsureChannelFirstd(["image"]),
            EnsureChannelFirstd(["label"], channel_dim="no_channel"),
            SpatialCropd(
                ["image", "label"],
                roi_start=[11, 9, 0],
                roi_end=[172, 205, 152],
            ),
            Spacingd(["image", "label"], pixdim=spacing),
            DivisiblePadd(["image", "label"], k=16),
            ScaleIntensityRangePercentilesd(
                "image",
                lower=0.01,
                upper=99.9,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
                channel_wise=True,
            ),
        ]
    )
