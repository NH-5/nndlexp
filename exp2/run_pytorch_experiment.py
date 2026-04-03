from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp2.pytorch_segmentation.evaluate import run_evaluation
from exp2.pytorch_segmentation.predict import run_prediction
from exp2.pytorch_segmentation.train import run_training
from exp2.pytorch_segmentation.utils import (
    build_run_stamp,
    resolve_experiment_dir,
    save_json,
    setup_logger,
)


EXPERIMENT_CONFIG = {
    "data_root": PROJECT_ROOT / "exp2" / "VOC2012",
    "output_root": PROJECT_ROOT / "exp2" / "outputs",
    "experiment_name": None,
    "device": "auto",
    "run_train": True,
    "run_evaluate": True,
    "run_predict": True,
}

TRAIN_CONFIG = {
    "train_split": "train",
    "val_split": "val",
    "epochs": 10,
    "batch_size": 4,
    "num_workers": 2,
    "crop_size": 513,
    "min_scale": 0.5,
    "max_scale": 2.0,
    "lr": 0.001,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "num_classes": 21,
    "ignore_label": 255,
    "seed": 42,
    "weights": "voc",
    "backbone_weights": "imagenet",
    "resume": None,
    "freeze_bn": False,
    "eval_every": 1,
    "eval_long_size": None,
    "log_interval": 20,
}

EVALUATE_CONFIG = {
    "split": "val",
    "checkpoint": None,
    "weights": "none",
    "backbone_weights": "imagenet",
    "num_classes": 21,
    "ignore_label": 255,
    "num_workers": 2,
    "long_size": None,
}

PREDICT_CONFIG = {
    "input": PROJECT_ROOT / "exp2" / "VOC2012" / "JPEGImages" / "2007_000032.jpg",
    "checkpoint": None,
    "weights": "none",
    "backbone_weights": "imagenet",
    "num_classes": 21,
    "long_size": 513,
}


def run_full_experiment() -> None:
    shared = {
        "data_root": EXPERIMENT_CONFIG["data_root"],
        "output_root": EXPERIMENT_CONFIG["output_root"],
        "experiment_name": EXPERIMENT_CONFIG["experiment_name"],
        "device": EXPERIMENT_CONFIG["device"],
    }

    if EXPERIMENT_CONFIG["run_train"]:
        train_args = Namespace(**shared, **TRAIN_CONFIG)
        run_training(train_args)

    experiment_name = EXPERIMENT_CONFIG["experiment_name"] or "latest"
    experiment_dir = resolve_experiment_dir(EXPERIMENT_CONFIG["output_root"], experiment_name)
    actual_experiment_name = experiment_dir.name

    pipeline_stamp = build_run_stamp()
    pipeline_log = experiment_dir / f"experiment_{pipeline_stamp}.log"
    logger = setup_logger("segmentation.experiment", pipeline_log)
    logger.info("Full experiment started")
    logger.info("Experiment directory: %s", experiment_dir)
    logger.info("Experiment config: %s", EXPERIMENT_CONFIG)
    logger.info("Train config: %s", TRAIN_CONFIG)
    logger.info("Evaluate config: %s", EVALUATE_CONFIG)
    logger.info("Predict config: %s", PREDICT_CONFIG)

    metrics = None
    if EXPERIMENT_CONFIG["run_evaluate"]:
        evaluate_args = Namespace(
            **{
                "data_root": EXPERIMENT_CONFIG["data_root"],
                "output_root": EXPERIMENT_CONFIG["output_root"],
                "experiment_name": actual_experiment_name,
                "device": EXPERIMENT_CONFIG["device"],
            },
            **EVALUATE_CONFIG,
        )
        metrics = run_evaluation(evaluate_args)
        logger.info(
            "Evaluation finished | loss=%.4f | pixel_accuracy=%.4f | mean_iou=%.4f",
            metrics["loss"],
            metrics["pixel_accuracy"],
            metrics["mean_iou"],
        )

    if EXPERIMENT_CONFIG["run_predict"]:
        predict_args = Namespace(
            **{
                "output_root": EXPERIMENT_CONFIG["output_root"],
                "experiment_name": actual_experiment_name,
                "device": EXPERIMENT_CONFIG["device"],
            },
            **PREDICT_CONFIG,
        )
        run_prediction(predict_args)
        logger.info("Prediction finished")

    summary = {
        "pipeline_stamp": pipeline_stamp,
        "experiment_dir": experiment_dir,
        "actual_experiment_name": actual_experiment_name,
        "experiment_config": EXPERIMENT_CONFIG,
        "train_config": TRAIN_CONFIG,
        "evaluate_config": EVALUATE_CONFIG,
        "predict_config": PREDICT_CONFIG,
        "evaluation_metrics": metrics,
        "stages": {
            "train": EXPERIMENT_CONFIG["run_train"],
            "evaluate": EXPERIMENT_CONFIG["run_evaluate"],
            "predict": EXPERIMENT_CONFIG["run_predict"],
        },
    }
    save_json(summary, experiment_dir / f"experiment_summary_{pipeline_stamp}.json")
    logger.info("Saved experiment summary: %s", experiment_dir / f"experiment_summary_{pipeline_stamp}.json")
    logger.info("Full experiment finished")


if __name__ == "__main__":
    run_full_experiment()
