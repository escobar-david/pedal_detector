from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import interactive
import predict
import split_dataset
import train


def test_train_parse_args_repro_flags():
    args = train.parse_args([
        "--epochs", "5",
        "--data", "custom.yaml",
        "--seed", "7",
        "--project", "runs/custom",
        "--name", "exp1",
    ])
    assert args.epochs == 5
    assert args.data == "custom.yaml"
    assert args.seed == 7
    assert args.project == "runs/custom"
    assert args.name == "exp1"


def test_split_dataset_parse_args_custom():
    args = split_dataset.parse_args([
        "--source-images", "img_dir",
        "--source-labels", "lbl_dir",
        "--output-dir", "out_dir",
        "--train-ratio", "0.9",
        "--seed", "9",
    ])
    assert args.source_images == "img_dir"
    assert args.source_labels == "lbl_dir"
    assert args.output_dir == "out_dir"
    assert args.train_ratio == 0.9
    assert args.seed == 9


def test_interactive_parse_args_device():
    args = interactive.parse_args(["--device", "cpu", "--weights", "best.pt"])
    assert args.device == "cpu"
    assert args.weights == "best.pt"


def test_predict_parse_args_and_missing_weights(tmp_path):
    source = tmp_path / "frame.jpg"
    source.write_text("stub")
    args = predict.parse_args(["--source", str(source), "--weights", str(tmp_path / "missing.pt")])
    assert args.source == str(source)
    assert predict.main(["--source", args.source, "--weights", args.weights]) == 1


def test_predict_validate_inputs_missing_source(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_text("stub")
    with pytest.raises(FileNotFoundError, match="Input source not found"):
        predict.validate_inputs(str(tmp_path / "missing.jpg"), str(weights))


def test_split_dataset_smoke(tmp_path):
    source_images = tmp_path / "images"
    source_labels = tmp_path / "labels"
    output_dir = tmp_path / "datasets"
    source_images.mkdir()
    source_labels.mkdir()

    (source_images / "a.jpg").write_text("a")
    (source_images / "b.jpg").write_text("b")
    (source_labels / "a.txt").write_text("0 0.5 0.5 0.1 0.1")
    (source_labels / "b.txt").write_text("0 0.5 0.5 0.2 0.2")

    stats = split_dataset.split_dataset(
        source_images=source_images,
        source_labels=source_labels,
        output_dir=output_dir,
        train_ratio=0.5,
        seed=42,
    )
    assert stats["total_pairs"] == 2
    assert stats["train_count"] + stats["val_count"] == 2
    assert len(list((output_dir / "train" / "images").iterdir())) == stats["train_count"]
    assert len(list((output_dir / "val" / "images").iterdir())) == stats["val_count"]
