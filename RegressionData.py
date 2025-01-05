from torchvision.datasets import ImageFolder
from terratorch.datamodules import GenericNonGeoClassificationDataModule
from terratorch.datasets.generic_scalar_label_dataset import GenericScalarLabelDataset
from terratorch.datasets.utils import HLSBands
import numpy as np
from typing import Any,Iterable
from pathlib import Path
import albumentations as A
import os
from torchgeo.datamodules import NonGeoDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from terratorch.io.file import load_from_file_or_attribute
from terratorch.datamodules.generic_scalar_label_data_module import wrap_in_compose_is_list, Normalize

# class ModifiedImageFolder(ImageFolder):
#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (sample, target, regression_ground_truth) where target is class_index of the target class.
#         """
#         path, target = self.samples[index]
#         regression_ground_truth = os.path.split(path)[-1].split('.')[1].split('_')[-1] #ugly but should work
#         sample = self.loader(path)
#         if self.transform is not None:
#             sample = self.transform(sample)
#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return sample, target, regression_ground_truth


def itemgetter(index,samples,loader,transform,target_transform):
        """
        Workaround to load in the regression ground truth from filenames. 
        This is largely the __getitem__ function from torchvision.datasets.ImageFolder save for the regression_ground_truth
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = samples[index]
        regression_ground_truth = float(os.path.split(path)[-1].split('_')[-1].strip('.tif')) #ugly but should work if we input regression truth as _.... on the end before .tif
        sample = loader(path)
        if transform is not None:
            sample = transform(sample)
        if target_transform is not None:
            target = target_transform(target)

        return sample, target, regression_ground_truth


class GenericClassificationRegressionDataset(GenericScalarLabelDataset):
    def __init__(
        self,
        data_root: Path,
        num_classes: int,
        split: Path | None = None,
        ignore_split_file_extensions: bool = True,  # noqa: FBT001, FBT002
        allow_substring_split_file: bool = True,  # noqa: FBT001, FBT002
        rgb_indices: list[str] | None = None,
        dataset_bands: list[HLSBands | int] | None = None,
        output_bands: list[HLSBands | int] | None = None,
        class_names: list[str] | None = None,
        constant_scale: float = 1,
        transform: A.Compose | None = None,
        no_data_replace: float = 0,
        expand_temporal_dimension: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """
        Based heavily on TerraTorch's GenericNonGeoClassificationDataset. See the source code there for more information.
        """
        super().__init__(
            data_root,
            split=split,
            ignore_split_file_extensions=ignore_split_file_extensions,
            allow_substring_split_file=allow_substring_split_file,
            rgb_indices=rgb_indices,
            dataset_bands=dataset_bands,
            output_bands=output_bands,
            constant_scale=constant_scale,
            transform=transform,
            no_data_replace=no_data_replace,
            expand_temporal_dimension=expand_temporal_dimension,
        )
        self.num_classes = num_classes
        self.class_names = class_names

    def __getitem__(self, index: int) -> dict[str, Any]:
        """ 
        Altered version of the __getitem__ function of GenericScalarLabelDataset.
        """
        ##Ignore these two lines, kept only for reference
        #not sure if this will work since generic_scalar_label_dataset inherits from ImageFolder, but ay if it calls it calls
        # image, label,regression_ground_truth = ModifiedImageFolder.__getitem__(self, index) ##changed to ModifiedImageFolder

        ## Should work, not the best way to do it but it works 
        image, label,regression_ground_truth = itemgetter(index,self.samples,self.loader,self.transform,self.target_transform) 
        ## Commented out since idk where the rearrange is defined
        # if self.expand_temporal_dimension:
        #     image = rearrange(image, "h w (channels time) -> time h w channels", channels=len(self.output_bands))
        if self.filter_indices:
            image = image[..., self.filter_indices]

        image = image.astype(np.float32) * self.constant_scale

        if self.transforms:
            image = self.transforms(image=image)["image"]  # albumentations returns dict

        output = {
            "image": image,
            "label": label,
            'regr_label': regression_ground_truth, #ADDED for regression
            "filename": self.image_files[index]
        }

        return output
    

    def plot(self, sample, suptitle = None):
        pass
 

class GenericNonGeoClassificationRegressionDataModule(NonGeoDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        train_data_root: Path,
        val_data_root: Path,
        test_data_root: Path,
        means: list[float] | str,
        stds: list[float] | str,
        num_classes: int,
        predict_data_root: Path | None = None,
        train_split: Path | None = None,
        val_split: Path | None = None,
        test_split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        dataset_bands: list[HLSBands | int] | None = None,
        predict_dataset_bands: list[HLSBands | int] | None = None,
        output_bands: list[HLSBands | int] | None = None,
        constant_scale: float = 1,
        rgb_indices: list[int] | None = None,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        expand_temporal_dimension: bool = False,
        no_data_replace: float = 0,
        drop_last: bool = True,
        **kwargs: Any,
    ) -> None:
        """ 
        Heavily based on Terratorch's GenericNonGeoClassificationDataModule. See the source code there for more information.
        """
        super().__init__(GenericClassificationRegressionDataset, batch_size, num_workers, **kwargs) #giving the right dataset class 
        self.num_classes = num_classes
        self.train_root = train_data_root
        self.val_root = val_data_root
        self.test_root = test_data_root
        self.predict_root = predict_data_root
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.ignore_split_file_extensions = ignore_split_file_extensions
        self.allow_substring_split_file = allow_substring_split_file
        self.constant_scale = constant_scale
        self.no_data_replace = no_data_replace
        self.drop_last = drop_last

        self.dataset_bands = dataset_bands
        self.predict_dataset_bands = predict_dataset_bands if predict_dataset_bands else dataset_bands
        self.output_bands = output_bands
        self.rgb_indices = rgb_indices
        self.expand_temporal_dimension = expand_temporal_dimension

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)

        means = load_from_file_or_attribute(means)
        stds = load_from_file_or_attribute(stds)

        self.aug = Normalize(means, stds)

    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(
                self.train_root,
                self.num_classes,
                split=self.train_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.train_transform,
                no_data_replace=self.no_data_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(
                self.val_root,
                self.num_classes,
                split=self.val_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.val_transform,
                no_data_replace=self.no_data_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(
                self.test_root,
                self.num_classes,
                split=self.test_split,
                ignore_split_file_extensions=self.ignore_split_file_extensions,
                allow_substring_split_file=self.allow_substring_split_file,
                dataset_bands=self.dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
            )
        if stage in ["predict"] and self.predict_root:
            self.predict_dataset = self.dataset_class(
                self.predict_root,
                self.num_classes,
                dataset_bands=self.predict_dataset_bands,
                output_bands=self.output_bands,
                constant_scale=self.constant_scale,
                rgb_indices=self.rgb_indices,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                expand_temporal_dimension=self.expand_temporal_dimension,
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split == "train" and self.drop_last,
        )
