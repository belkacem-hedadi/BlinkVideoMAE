import os
import sys
from pathlib import Path

from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, Trainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_utils import (
    compute_metrics,
    evaluate_per_eye,
    load_blink_dataset,
    make_training_args,
    prepare_split,
    save_predictions_csv,
    set_seed,
    video_collate_fn,
)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ID2LABEL = {0: "non_blink", 1: "blink"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def main():
    set_seed(42)
    num_frames = 16
    model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
    out_dir = Path(__file__).resolve().parent / "outputs" / "model1"

    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    dataset = load_blink_dataset("HUST-LBEW")
    dataset_2 = load_blink_dataset("Epan-EyeBlink2")

    train_dataset = prepare_split(dataset["train"], True, num_frames, processor, do_flip=True)
    test_dataset = prepare_split(dataset["test"], False, num_frames, processor)
    test_dataset_2 = prepare_split(dataset_2["test"], False, num_frames, processor)

    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    trainer = Trainer(
        model=model,
        args=make_training_args(out_dir),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=video_collate_fn,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    trainer.train()

    evaluate_per_eye(trainer, "HUST-LBEW", test_dataset)
    evaluate_per_eye(trainer, "Epan-EyeBlink2", test_dataset_2)

    results_dir = out_dir / "predictions"
    save_predictions_csv(trainer, test_dataset, results_dir / "Model1_HUST_Predictions.csv")
    save_predictions_csv(trainer, test_dataset_2, results_dir / "Model1_Epan_Predictions.csv")


if __name__ == "__main__":
    main()
