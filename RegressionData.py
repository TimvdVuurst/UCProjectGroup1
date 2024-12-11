from torchvision.datasets import ImageFolder
from terratorch.datamodules import GenericNonGeoClassificationDataModule
from terratorch.datasets import generic_scalar_label_dataset
import numpy as np
import pandas as pd
from typing import Any,Tuple,A,HLSBands,Path
import os

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
        regression_ground_truth = os.path.split(path)[-1].split('.')[1].split('_')[-1] #ugly but should work if we input regression truth as _.... on the end before .tif
        sample = loader(path)
        if transform is not None:
            sample = transform(sample)
        if target_transform is not None:
            target = target_transform(target)

        return sample, target, regression_ground_truth


class GenericClassificationRegressionDataset(generic_scalar_label_dataset):
 def __getitem__(self, index: int) -> dict[str, Any]:
        #not sure if this will work since generic_scalar_label_dataset inherits from ImageFolder, but ay if it calls it calls
        # image, label,regression_ground_truth = ModifiedImageFolder.__getitem__(self, index) ##changed to ModifiedImageFolder

        ## Should work, not the best way to do it but ay 
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
            'regr_label': regression_ground_truth, #ADDED
            "filename": self.image_files[index]
        }

        return output
 

 class GenericNonGeoClassificationRegressionDataModule(GenericNonGeoClassificationDataModule):
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
        **kwargs: Any) -> None:
            super().__init__( batch_size,
                            num_workers,
                            train_data_root,
                            val_data_root,
                            test_data_root,
                            means,
                            stds,
                            num_classes,
                            predict_data_root,
                            train_split,
                            val_split,
                            test_split,
                            ignore_split_file_extensions,
                            allow_substring_split_file,
                            dataset_bands,
                            predict_dataset_bands,
                            output_bands,
                            constant_scale,
                            rgb_indices,
                            train_transform,
                            val_transform, 
                            test_transform,
                            expand_temporal_dimension,
                            no_data_replace,
                            drop_last,
                            **kwargs)
            #this is really the only difference here, change it to the dataset defined above that is able to extract regression data from
            #the filenamebut since this variable is defined in a class that GenericNonGeoClassificationDataModule inherits from and this
            #felt like the easiest way to do it
            self.dataset_class = GenericClassificationRegressionDataset 