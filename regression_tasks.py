from typing import Any

import lightning
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import Callback
from segmentation_models_pytorch.losses import FocalLoss, JaccardLoss
from torch import Tensor, nn
from torchgeo.datasets.utils import unbind_samples
from torchgeo.trainers import BaseTask
from torchmetrics import ClasswiseWrapper, MetricCollection
# from torchmetrics.classification import MulticlassAccuracy, MulticlassFBetaScore, MulticlassJaccardIndex
from torchmetrics.regression import R2Score, SpearmanCorrCoef


from terratorch.models.model import AuxiliaryHead, Model, ModelOutput, get_factory
from terratorch.tasks.loss_handler import LossHandler
from terratorch.tasks.optimizer_factory import optimizer_factory


def to_class_prediction(y: ModelOutput) -> Tensor:
    y_hat = y.output
    # return y_hat.argmax(dim=1)
    print(y_hat.shape)
    return y_hat.reshape(y_hat.shape[0])


class RegressionTask(BaseTask):  # changed from ClassificationTask
    """Regression Task that accepts models from a range of sources.

    This class is heavily based on the ClassificationTask from Terratorch. See
    the documentation there for more details. Most code credit goes to that class,
    where changes are made these are noted explicitly.
    """

    def __init__(
        self,
        model_args: dict,
        model_factory: str,
        loss: str = "ce",
        aux_heads: list[AuxiliaryHead] | None = None,
        aux_loss: dict[str, float] | None = None,
        class_weights: list[float] | None = None,
        ignore_index: int | None = None,
        lr: float = 0.001,
        # the following are optional so CLI doesnt need to pass them
        optimizer: str | None = None,
        optimizer_hparams: dict | None = None,
        scheduler: str | None = None,
        scheduler_hparams: dict | None = None,
        #
        #
        freeze_backbone: bool = False,  # noqa: FBT001, FBT002
        freeze_decoder: bool = False,  # noqa: FBT002, FBT001
        class_names: list[str] | None = None,
    ) -> None:
        
        
        """Constructor

        Args:

            Defaults to None.
            model_args (Dict): Arguments passed to the model factory.
            model_factory (str): ModelFactory class to be used to instantiate the model.
            loss (str, optional): Loss to be used. Currently, supports 'ce', 'jaccard' or 'focal' loss.
                Defaults to "ce".
            aux_loss (dict[str, float] | None, optional): Auxiliary loss weights.
                Should be a dictionary where the key is the name given to the loss
                and the value is the weight to be applied to that loss.
                The name of the loss should match the key in the dictionary output by the model's forward
                method containing that output. Defaults to None.
            class_weights (Union[list[float], None], optional): List of class weights to be applied to the loss.
            class_weights (list[float] | None, optional): List of class weights to be applied to the loss.
                Defaults to None.
            ignore_index (int | None, optional): Label to ignore in the loss computation. Defaults to None.
            lr (float, optional): Learning rate to be used. Defaults to 0.001.
            optimizer (str | None, optional): Name of optimizer class from torch.optim to be used.
                If None, will use Adam. Defaults to None. Overriden by config / cli specification through LightningCLI.
            optimizer_hparams (dict | None): Parameters to be passed for instantiation of the optimizer.
                Overriden by config / cli specification through LightningCLI.
            scheduler (str, optional): Name of Torch scheduler class from torch.optim.lr_scheduler
                to be used (e.g. ReduceLROnPlateau). Defaults to None.
                Overriden by config / cli specification through LightningCLI.
            scheduler_hparams (dict | None): Parameters to be passed for instantiation of the scheduler.
                Overriden by config / cli specification through LightningCLI.
            freeze_backbone (bool, optional): Whether to freeze the backbone. Defaults to False.
            freeze_decoder (bool, optional): Whether to freeze the decoder and segmentation head. Defaults to False.
            class_names (list[str] | None, optional): List of class names passed to metrics for better naming.
                Defaults to numeric ordering.
        """
        self.aux_loss = aux_loss
        self.aux_heads = aux_heads
        self.model_factory = get_factory(model_factory)
        super().__init__()
        self.train_loss_handler = LossHandler(self.train_metrics.prefix)
        self.test_loss_handler = LossHandler(self.test_metrics.prefix)
        self.val_loss_handler = LossHandler(self.val_metrics.prefix)
        self.monitor = f"{self.val_metrics.prefix}loss"

    # overwrite early stopping
    def configure_callbacks(self) -> list[Callback]:
        return []

    def configure_models(self) -> None:
        self.model: Model = self.model_factory.build_model(
            "classification", aux_decoders=self.aux_heads, **self.hparams["model_args"] #if you change classification to regression it will try to build a pixelwise regression model
        )
        if self.hparams["freeze_backbone"]:
            self.model.freeze_encoder()
        if self.hparams["freeze_decoder"]:
            self.model.freeze_decoder()

    def configure_optimizers(
        self,
    ) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
        optimizer = self.hparams["optimizer"]
        if optimizer is None:
            optimizer = "Adam"
        return optimizer_factory(
            self.hparams["optimizer"],
            self.hparams["lr"],
            self.parameters(),
            self.hparams["optimizer_hparams"],
            self.hparams["scheduler"],
            self.monitor,
            self.hparams["scheduler_hparams"],
        )

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams["loss"]
        ignore_index = self.hparams["ignore_index"]

        # class_weights = (
        #             torch.Tensor(self.hparams["class_weights"]) if self.hparams["class_weights"] is not None else None
        #         )
        if loss == "mse":  # added to handle regression, for now no other loss is valid
            # ignore_value = -100 if ignore_index is None else ignore_index
            self.criterion = nn.MSELoss() 
        else:
            msg = f"Loss type '{loss}' is not valid."
            raise ValueError(msg)


    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["model_args"]["num_classes"]
        # ignore_index: int = self.hparams["ignore_index"]
        # class_names = self.hparams["class_names"]
        metrics = MetricCollection(
            {            
                "R2_Score": R2Score(
                    num_outputs=num_classes
                ),
                'Spearman Correlation': SpearmanCorrCoef(num_outputs=num_classes)
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Compute the train loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["regr_label"] #CHANGE TO REGRESSION GROUND TRUTH AS PER RegressionData.py (e.g. use the key regr_label)
        model_output: dict[str, Tensor] = self(x)
        print('TRAINING step reached! ')#, model_output)
        # print(type(y), y, y.shape)
        loss = self.train_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.train_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=x.shape[0])
        y_hat_hard = to_class_prediction(model_output)
        self.train_metrics.update(y_hat_hard, y)

        return loss["loss"]
   
    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute(), sync_dist=True)
        self.train_metrics.reset()
        return super().on_train_epoch_end()

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["regr_label"] #CHANGE TO REGRESSION GROUND TRUTH AS PER RegressionData.py (e.g. use the key regr_label)
        model_output: dict[str, Tensor] = self(x)
        print('VALIDATION step reached! ')#, model_output)
        # print(type(y), y,y.shape)
        loss = self.val_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.val_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=x.shape[0])
        y_hat_hard = to_class_prediction(model_output)
        self.val_metrics.update(y_hat_hard, y)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()
        return super().on_validation_epoch_end()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["regr_label"] #CHANGE TO REGRESSION GROUND TRUTH AS PER RegressionData.py (e.g. use the key regr_label)
        model_output: dict[str, Tensor] = self(x)
        loss = self.test_loss_handler.compute_loss(model_output, y, self.criterion, self.aux_loss)
        self.test_loss_handler.log_loss(self.log, loss_dict=loss, batch_size=x.shape[0])
        y_hat_hard = to_class_prediction(model_output)
        self.test_metrics.update(y_hat_hard, y)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.test_metrics.reset()
        return super().on_test_epoch_end()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch["image"]
        file_names = batch["filename"]

        y_hat = self(x).output
        y_hat = y_hat.argmax(dim=1)
        return y_hat, file_names