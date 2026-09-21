import os
import sys
from pathlib import Path

from transformers import Trainer, VideoMAEImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blink_model import BlinkVideoMAE
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


def main():
    set_seed(42)
    num_frames = 16
    model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
    out_dir = Path(__file__).resolve().parent / "outputs" / "model2"

    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    dataset = load_blink_dataset("HUST-LBEW")
    dataset_2 = load_blink_dataset("Epan-EyeBlink2")

    train_ds = prepare_split(dataset["train"], True, num_frames, processor, do_flip=True)
    test_ds = prepare_split(dataset["test"], False, num_frames, processor)
    test_ds2 = prepare_split(dataset_2["test"], False, num_frames, processor)

    model = BlinkVideoMAE(model_name, use_tdm=True, temporal_layers=2)
    trainer = Trainer(
        model=model,
        args=make_training_args(out_dir),
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=video_collate_fn,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    trainer.train()
    evaluate_per_eye(trainer, "HUST-LBEW", test_ds)
    evaluate_per_eye(trainer, "Epan-EyeBlink2", test_ds2)

    results_dir = out_dir / "predictions"
    save_predictions_csv(trainer, test_ds, results_dir / "Model2_HUST_Predictions.csv")
    save_predictions_csv(trainer, test_ds2, results_dir / "Model2_Epan_Predictions.csv")


if __name__ == "__main__":
    main()
