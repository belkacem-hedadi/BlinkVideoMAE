import argparse
import json
import os
import sys
from pathlib import Path

from transformers import Trainer, VideoMAEImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blink_model import build_model
from video_utils import (
    compute_metrics,
    evaluate_split,
    load_blink_dataset,
    make_training_args,
    prepare_split,
    save_predictions_csv,
    set_seed,
    video_collate_fn,
)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

CODE_DIR = Path(__file__).resolve().parent
OUT_ROOT = CODE_DIR / "outputs" / "experiments"
MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"

EXPERIMENTS = [
    {
        "name": "videomae_baseline",
        "kind": "baseline",
        "split": "full",
        "num_frames": 16,
        "do_flip": True,
    },
    {
        "name": "blinkvideomae",
        "kind": "blink",
        "split": "full",
        "num_frames": 16,
        "use_tdm": True,
        "temporal_layers": 2,
        "do_flip": True,
    },
    {
        "name": "ablate_tdm",
        "kind": "blink",
        "split": "full",
        "num_frames": 16,
        "use_tdm": False,
        "temporal_layers": 2,
        "do_flip": True,
    },
    {
        "name": "ablate_temporal",
        "kind": "blink",
        "split": "full",
        "num_frames": 16,
        "use_tdm": True,
        "temporal_layers": 0,
        "do_flip": True,
    },
    {
        "name": "r3d18_baseline",
        "kind": "r3d18",
        "split": "full",
        "num_frames": 16,
        "do_flip": True,
    },
    {
        "name": "blinkvideomae_span10",
        "kind": "blink",
        "split": "10",
        "num_frames": 10,
        "use_tdm": True,
        "temporal_layers": 2,
        "do_flip": True,
    },
    {
        "name": "blinkvideomae_span10_tdm",
        "kind": "blink",
        "split": "10",
        "num_frames": 10,
        "use_tdm": True,
        "temporal_layers": 0,
        "do_flip": True,
    },
]


def split_names(split_key):
    if split_key == "10":
        return "train_10", "test_10", "test_10"
    return "train", "test", "test"


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def run_one(cfg, hust, epan, processor):
    set_seed(42)
    name = cfg["name"]
    out_dir = OUT_ROOT / name
    pred_dir = out_dir / "predictions"
    train_split, hust_test_split, epan_test_split = split_names(cfg["split"])
    num_frames = cfg["num_frames"]

    train_ds = prepare_split(
        hust[train_split], True, num_frames, processor, do_flip=cfg.get("do_flip", True)
    )
    hust_test = prepare_split(hust[hust_test_split], False, num_frames, processor)
    epan_test = prepare_split(epan[epan_test_split], False, num_frames, processor)

    model = build_model(
        cfg["kind"],
        MODEL_NAME,
        use_tdm=cfg.get("use_tdm", True),
        temporal_layers=cfg.get("temporal_layers", 2),
        num_frames=num_frames,
    )
    trainer = Trainer(
        model=model,
        args=make_training_args(out_dir, save_total_limit=1),
        train_dataset=train_ds,
        eval_dataset=hust_test,
        data_collator=video_collate_fn,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    trainer.train()

    hust_metrics, _, _, _ = evaluate_split(trainer, hust_test)
    epan_metrics, _, _, _ = evaluate_split(trainer, epan_test)
    save_predictions_csv(trainer, hust_test, pred_dir / f"{name}_HUST.csv")
    save_predictions_csv(trainer, epan_test, pred_dir / f"{name}_Epan.csv")

    result = {
        "name": name,
        "config": {k: v for k, v in cfg.items()},
        "hust": hust_metrics,
        "epan": epan_metrics,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(result), handle, indent=2)
    print(json.dumps(to_jsonable(result), indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="*", default=None, help="Run named experiments only")
    args = parser.parse_args()

    selected = EXPERIMENTS
    if args.exp:
        wanted = set(args.exp)
        selected = [cfg for cfg in EXPERIMENTS if cfg["name"] in wanted]
        missing = wanted - {cfg["name"] for cfg in selected}
        if missing:
            raise SystemExit(f"Unknown experiments: {sorted(missing)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    hust = load_blink_dataset("HUST-LBEW")
    epan = load_blink_dataset("Epan-EyeBlink2")

    summary = []
    for cfg in selected:
        print(f"\n========== Running {cfg['name']} ==========")
        try:
            summary.append(run_one(cfg, hust, epan, processor))
        except Exception as exc:
            print(f"FAILED {cfg['name']}: {exc}")
            summary.append({"name": cfg["name"], "error": str(exc)})
            with open(OUT_ROOT / f"{cfg['name']}_error.txt", "w", encoding="utf-8") as handle:
                handle.write(repr(exc))

    with open(OUT_ROOT / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary), handle, indent=2)
    print(f"\nWrote {OUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
