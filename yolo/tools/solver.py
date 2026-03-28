from math import ceil
from pathlib import Path

import torch
from lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision

from copy import deepcopy

from omegaconf import OmegaConf

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.drawer import draw_bboxes
from yolo.tools.loss_functions import DistillationLoss, create_loss_function
from yolo.utils.bounding_box_utils import create_converter, to_metrics_format
from yolo.utils.model_utils import PostProcess, create_optimizer, create_scheduler


class BaseModel(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.model = create_model(cfg.model, cfg=cfg, class_num=cfg.dataset.class_num, weight_path=cfg.weight)

    def forward(self, x):
        return self.model(x)


class ValidateModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        if self.cfg.task.task == "validation":
            self.validation_cfg = self.cfg.task
        else:
            self.validation_cfg = self.cfg.task.validation
        self.metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", backend="faster_coco_eval")
        self.metric.warn_on_many_detections = False
        self.val_loader = create_dataloader(self.validation_cfg.data, self.cfg.dataset, self.validation_cfg.task)
        self.ema = self.model

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.validation_cfg.nms)

    def val_dataloader(self):
        return self.val_loader

    def validation_step(self, batch, batch_idx):
        batch_size, images, targets, rev_tensor, img_paths = batch
        H, W = images.shape[2:]
        predicts = self.post_process(self.ema(images), image_size=[W, H])
        self.metric.update(
            [to_metrics_format(predict) for predict in predicts], [to_metrics_format(target) for target in targets]
        )
        return None

    def on_validation_epoch_end(self):
        epoch_metrics = self.metric.compute()
        del epoch_metrics["classes"]
        self.log_dict(epoch_metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log_dict(
            {"PyCOCO/AP @ .5:.95": epoch_metrics["map"], "PyCOCO/AP @ .5": epoch_metrics["map_50"]},
            sync_dist=True,
            rank_zero_only=True,
        )
        self.metric.reset()


class TrainModel(ValidateModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.train_loader = create_dataloader(self.cfg.task.data, self.cfg.dataset, self.cfg.task.task)

    def setup(self, stage):
        super().setup(stage)
        self.loss_fn = create_loss_function(self.cfg, self.vec2box)

    def train_dataloader(self):
        return self.train_loader

    def on_train_epoch_start(self):
        self.trainer.optimizers[0].next_epoch(
            ceil(len(self.train_loader) / self.trainer.world_size), self.current_epoch
        )
        self.vec2box.update(self.cfg.image_size)

    def training_step(self, batch, batch_idx):
        lr_dict = self.trainer.optimizers[0].next_batch()
        batch_size, images, targets, *_ = batch
        predicts = self(images)
        aux_predicts = self.vec2box(predicts["AUX"]) if "AUX" in predicts else None
        main_predicts = self.vec2box(predicts["Main"])
        loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)
        self.log_dict(
            loss_item,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log_dict(lr_dict, prog_bar=False, logger=True, on_epoch=False, rank_zero_only=True)
        return loss * batch_size

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.cfg.task.optimizer)
        scheduler = create_scheduler(optimizer, self.cfg.task.scheduler)
        return [optimizer], [scheduler]


class DistillTrainModel(TrainModel):
    """Training model with knowledge distillation from a frozen teacher.

    The teacher model runs in eval mode with no gradients. Its detection
    outputs are matched to the student's via :class:`DistillationLoss`
    (KL divergence on class logits + L2 on DFL distributions).

    Distillation config is read from ``cfg.task.distillation`` and must
    contain ``teacher_model`` (model config name, e.g. ``v9-c``) and
    ``teacher_weights`` (path to the teacher's ``.pt`` weight file).
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        distill_cfg = cfg.task.distillation

        # Build teacher model with the specified architecture
        teacher_model_cfg = deepcopy(cfg.model)
        teacher_model_yaml = Path(__file__).resolve().parent.parent / "config" / "model" / f"{distill_cfg.teacher_model}.yaml"
        teacher_model_override = OmegaConf.load(teacher_model_yaml)
        teacher_model_cfg = OmegaConf.merge(teacher_model_cfg, teacher_model_override)
        OmegaConf.set_struct(teacher_model_cfg, False)

        self.teacher = create_model(
            teacher_model_cfg, cfg=cfg,
            class_num=cfg.dataset.class_num,
            weight_path=Path(distill_cfg.teacher_weights),
            skip_heads=False,
        )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.distill_loss = DistillationLoss(
            temperature=getattr(distill_cfg, "temperature", 4.0),
            alpha_cls=getattr(distill_cfg, "alpha_cls", 1.0),
            alpha_dfl=getattr(distill_cfg, "alpha_dfl", 1.0),
        )
        self.distill_weight = getattr(distill_cfg, "weight", 1.0)

    def setup(self, stage):
        super().setup(stage)
        # Teacher uses same strides/image_size so we can share vec2box
        self.teacher = self.teacher.to(self.device)

    def training_step(self, batch, batch_idx):
        lr_dict = self.trainer.optimizers[0].next_batch()
        batch_size, images, targets, *_ = batch

        # Student forward
        predicts = self(images)
        aux_predicts = self.vec2box(predicts["AUX"]) if "AUX" in predicts else None
        main_predicts = self.vec2box(predicts["Main"])

        # Standard detection loss
        loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)

        # Teacher forward (no gradients)
        with torch.no_grad():
            teacher_out = self.teacher(images)
            teacher_predicts = self.vec2box(teacher_out["Main"])

        # Distillation loss
        distill_loss, distill_item = self.distill_loss(main_predicts, teacher_predicts)
        loss = loss + self.distill_weight * distill_loss
        loss_item.update(distill_item)

        self.log_dict(
            loss_item,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log_dict(lr_dict, prog_bar=False, logger=True, on_epoch=False, rank_zero_only=True)
        return loss * batch_size


class InferenceModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        # TODO: Add FastModel
        self.predict_loader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task)

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.cfg.task.nms)

    def predict_dataloader(self):
        return self.predict_loader

    def predict_step(self, batch, batch_idx):
        images, rev_tensor, origin_frame = batch
        predicts = self.post_process(self(images), rev_tensor=rev_tensor)
        img = draw_bboxes(origin_frame, predicts, idx2label=self.cfg.dataset.class_list)
        if getattr(self.predict_loader, "is_stream", None):
            fps = self._display_stream(img)
        else:
            fps = None
        if getattr(self.cfg.task, "save_predict", None):
            self._save_image(img, batch_idx)
        return img, fps

    def _save_image(self, img, batch_idx):
        save_image_path = Path(self.trainer.default_root_dir) / f"frame{batch_idx:03d}.png"
        img.save(save_image_path)
        print(f"💾 Saved visualize image at {save_image_path}")
