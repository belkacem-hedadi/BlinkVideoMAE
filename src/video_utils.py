import io
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from datasets import Video, load_dataset
from PIL import Image
from transformers import TrainingArguments

try:
    import decord

    decord.bridge.set_bridge("torch")
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
DATASET_NAME_MAP = {
    "Bekhouche/HUST-LBEW": "HUST-LBEW",
    "Bekhouche/Epan-EyeBlink2": "Epan-EyeBlink2",
    "HUST-LBEW": "HUST-LBEW",
    "Epan-EyeBlink2": "Epan-EyeBlink2",
}
VALID_EYES = ("zuo", "you")
ID2LABEL = {0: "non_blink", 1: "blink"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# --- Seeding and dataset loading ---


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_blink_dataset(name):
    """Load a local or Hub video dataset without torchcodec (decode with PyAV)."""
    folder_name = DATASET_NAME_MAP.get(name, name.replace("Bekhouche/", ""))
    local_dir = DATASETS_DIR / folder_name
    source = str(local_dir) if local_dir.exists() else name
    dataset = load_dataset(source)
    first_split = next(iter(dataset))
    if "video" in dataset[first_split].column_names:
        dataset = dataset.cast_column("video", Video(decode=False))
    if "eye" in dataset[first_split].column_names:
        dataset = dataset.filter(lambda x: x["eye"] in VALID_EYES)
    return dataset


# --- Frame conversion and video decoding ---


def _to_uint8_hwc(frame):
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.permute(1, 2, 0)
        arr = arr.numpy()
    else:
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def _frames_from_thwc_or_tchw(video):
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu()
        if video.ndim == 4 and video.shape[1] in (1, 3):
            video = video.permute(0, 2, 3, 1)
        video = video.numpy()
    else:
        video = np.asarray(video)
        if video.ndim == 4 and video.shape[1] in (1, 3) and video.shape[-1] not in (1, 3):
            video = np.transpose(video, (0, 2, 3, 1))
    return [_to_uint8_hwc(frame) for frame in video]


def _decode_with_av(source):
    import av

    if isinstance(source, (bytes, bytearray, memoryview)):
        container = av.open(io.BytesIO(bytes(source)))
    else:
        container = av.open(str(source))
    frames = []
    try:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()
    return frames


def _decode_with_torchvision(path):
    from torchvision.io import read_video

    video, _, _ = read_video(str(path), pts_unit="sec", output_format="TCHW")
    return _frames_from_thwc_or_tchw(video)


def _decode_with_decord(path):
    vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
    batch = vr[:]
    if hasattr(batch, "asnumpy"):
        return _frames_from_thwc_or_tchw(batch.asnumpy())
    if hasattr(batch, "numpy"):
        return _frames_from_thwc_or_tchw(batch.numpy())
    return _frames_from_thwc_or_tchw(batch)


def _decode_path_or_bytes(source):
    errors = []
    is_bytes = isinstance(source, (bytes, bytearray, memoryview))
    tmp_path = None
    if is_bytes:
        fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        with os.fdopen(fd, "wb") as handle:
            handle.write(bytes(source))
        path = tmp_path
    else:
        path = str(source)

    try:
        try:
            return _decode_with_av(source if is_bytes else path)
        except Exception as exc:
            errors.append(f"av: {exc}")
        try:
            return _decode_with_torchvision(path)
        except Exception as exc:
            errors.append(f"torchvision: {exc}")
        if _HAS_DECORD:
            try:
                return _decode_with_decord(path)
            except Exception as exc:
                errors.append(f"decord: {exc}")
        raise RuntimeError("Could not decode video (" + "; ".join(errors) + ")")
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def load_video_frames(video_data):
    if video_data is None:
        return []

    if isinstance(video_data, dict):
        if video_data.get("bytes"):
            return _decode_path_or_bytes(video_data["bytes"])
        if video_data.get("path"):
            return _decode_path_or_bytes(video_data["path"])
        if "array" in video_data:
            return _frames_from_thwc_or_tchw(video_data["array"])
        if "frames" in video_data:
            video_data = video_data["frames"]
        else:
            return []

    type_name = type(video_data).__name__
    if "VideoDecoder" in type_name or "VideoReader" in type_name:
        try:
            frames = video_data[:]
            return _frames_from_thwc_or_tchw(frames)
        except Exception:
            frames_list = []
            for frame in video_data:
                if isinstance(frame, dict) and "data" in frame:
                    frames_list.append(_to_uint8_hwc(frame["data"]))
                else:
                    frames_list.append(_to_uint8_hwc(frame))
            return frames_list

    if isinstance(video_data, (str, os.PathLike)):
        return _decode_path_or_bytes(video_data)

    if isinstance(video_data, (bytes, bytearray, memoryview)):
        return _decode_path_or_bytes(video_data)

    if isinstance(video_data, (torch.Tensor, np.ndarray)) and getattr(video_data, "ndim", 0) == 4:
        return _frames_from_thwc_or_tchw(video_data)

    if isinstance(video_data, (list, tuple)):
        return [_to_uint8_hwc(frame) for frame in video_data]

    if hasattr(video_data, "__getitem__") and hasattr(video_data, "__len__"):
        try:
            return _frames_from_thwc_or_tchw(video_data[:])
        except Exception:
            return [_to_uint8_hwc(video_data[i]) for i in range(len(video_data))]

    return []


# --- Preprocessing and collation ---


def sample_or_pad_frames(frames_list, num_frames=16):
    if not frames_list:
        frames_list = [np.zeros((224, 224, 3), dtype=np.uint8)]
    if len(frames_list) > num_frames:
        indices = np.linspace(0, len(frames_list) - 1, num_frames, dtype=int)
        return [frames_list[i] for i in indices]
    if len(frames_list) < num_frames:
        frames_list = list(frames_list)
        while len(frames_list) < num_frames:
            frames_list.append(frames_list[-1])
    return frames_list


def augment_frames(frames_list, do_flip=True):
    apply_flip = do_flip and np.random.rand() > 0.5
    brightness = np.random.uniform(0.8, 1.2)
    contrast = np.random.uniform(0.8, 1.2)
    augmented = []
    for frame in frames_list:
        img = Image.fromarray(_to_uint8_hwc(frame))
        if apply_flip:
            img = TF.hflip(img)
        img = TF.adjust_contrast(TF.adjust_brightness(img, brightness), contrast)
        augmented.append(np.array(img))
    return augmented


def preprocess_video(example, is_train=False, num_frames=16, processor=None, do_flip=True):
    frames_list = sample_or_pad_frames(load_video_frames(example["video"]), num_frames)
    if is_train:
        frames_list = augment_frames(frames_list, do_flip=do_flip)
    inputs = processor(frames_list, return_tensors="pt")
    pixel_values = inputs["pixel_values"][0]
    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values.contiguous().cpu().numpy()
    return {
        "pixel_values": pixel_values,
        "labels": int(example["label"]),
        "eye": example.get("eye", "unknown"),
    }


def prepare_split(dataset, is_train, num_frames, processor, do_flip=True):
    drop_cols = [c for c in ("video", "path") if c in dataset.column_names]
    return dataset.map(
        lambda example: preprocess_video(
            example,
            is_train=is_train,
            num_frames=num_frames,
            processor=processor,
            do_flip=do_flip,
        ),
        remove_columns=drop_cols,
        desc="Preprocessing videos",
    )


def video_collate_fn(examples):
    return {
        "pixel_values": torch.stack([torch.as_tensor(ex["pixel_values"]) for ex in examples]),
        "labels": torch.tensor([ex["labels"] for ex in examples], dtype=torch.long),
    }


# --- Metrics ---


def classification_scores(y_true, y_pred):
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "count": int(len(y_true)),
    }


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    scores = classification_scores(eval_pred.label_ids, predictions)
    return {k: scores[k] for k in ("accuracy", "precision", "recall", "f1", "specificity")}


def metrics_from_predictions(labels, preds, eyes=None):
    report = {"overall": classification_scores(labels, preds)}
    if eyes is None:
        return report
    eyes = np.asarray(eyes)
    for side, nice in (("zuo", "left"), ("you", "right")):
        mask = eyes == side
        if mask.any():
            report[nice] = classification_scores(labels[mask], preds[mask])
            report[nice]["eye_code"] = side
            report[nice]["count"] = int(mask.sum())
    return report


# --- Trainer configuration ---


def make_training_args(output_dir, **overrides):
    kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=42,
    )
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        kwargs["bf16"] = True
    elif torch.cuda.is_available():
        kwargs["fp16"] = True
    kwargs.update(overrides)
    return TrainingArguments(**kwargs)


# --- Evaluation and prediction export ---


def evaluate_split(trainer, dataset):
    output = trainer.predict(dataset)
    preds = np.argmax(output.predictions, axis=-1)
    labels = np.asarray(output.label_ids)
    eyes = np.asarray(dataset["eye"])
    return metrics_from_predictions(labels, preds, eyes), labels, preds, eyes


def evaluate_per_eye(trainer, name, dataset, sides=("zuo", "you")):
    print(f"\n--- Detailed Per-Eye Results for {name} ---")
    report, _, _, _ = evaluate_split(trainer, dataset)
    overall = report["overall"]
    print(
        f"Overall | Count: {overall['count']} | Acc: {overall['accuracy']:.4f} | "
        f"P: {overall['precision']:.4f} | R: {overall['recall']:.4f} | "
        f"F1: {overall['f1']:.4f}"
    )
    for nice in ("left", "right"):
        if nice not in report:
            continue
        res = report[nice]
        print(
            f"Side: {res['eye_code']} ({nice}) | Count: {res['count']} | "
            f"Acc: {res['accuracy']:.4f} | P: {res['precision']:.4f} | "
            f"R: {res['recall']:.4f} | F1: {res['f1']:.4f}"
        )
    return report


def save_predictions_csv(trainer, dataset, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    preds = trainer.predict(dataset)
    import pandas as pd

    pd.DataFrame(
        {
            "True_Label": preds.label_ids,
            "Predicted_Label": np.argmax(preds.predictions, axis=-1),
            "Eye": dataset["eye"],
        }
    ).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
