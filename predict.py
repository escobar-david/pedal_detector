"""
Inference Script for Guitar Pedal Detection

Run predictions on images using the trained YOLOv8 model.
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Run inference with trained pedal detector')
    parser.add_argument('--source', type=str, required=True,
                        help='Image file, directory, or video to run inference on')
    parser.add_argument('--weights', type=str, default='runs/detect/pedal_detector/weights/best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device (0, 1, etc.) or "cpu"')
    parser.add_argument('--save', action='store_true',
                        help='Save results to runs/detect/predict')
    parser.add_argument('--show', action='store_true',
                        help='Display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results as .txt files')
    parser.add_argument('--save-crop', action='store_true',
                        help='Save cropped detection boxes')
    return parser.parse_args(argv)


def _is_non_file_source(source: str) -> bool:
    source_lower = source.lower()
    network_prefixes = ('http://', 'https://', 'rtsp://', 'rtmp://', 'tcp://')
    if source_lower.startswith(network_prefixes):
        return True
    if source.isdigit():
        return True
    return any(char in source for char in ('*', '?', '[', ']'))


def validate_inputs(source: str, weights: str) -> None:
    source_path = Path(source)
    weights_path = Path(weights)

    if not _is_non_file_source(source) and not source_path.exists():
        raise FileNotFoundError(
            f"Input source not found: {source_path}\n"
            "Provide an existing local file/directory, a valid glob, URL, or webcam index."
        )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            "Train first with `python train.py` or pass `--weights /path/to/best.pt`."
        )


def main(argv=None):
    args = parse_args(argv)

    try:
        validate_inputs(args.source, args.weights)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Load the trained model
    print(f"Loading model from: {args.weights}")
    model = YOLO(args.weights)

    # Run inference
    print(f"Running inference on: {args.source}")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save,
        show=args.show,
        save_txt=args.save_txt,
        save_crop=args.save_crop,
        project='runs/detect',
        name='predict',
        exist_ok=True,
    )

    # Print results summary
    print("\n" + "="*60)
    print("Inference Results")
    print("="*60)

    total_detections = 0
    for i, result in enumerate(results):
        num_detections = len(result.boxes)
        total_detections += num_detections

        if hasattr(result, 'path'):
            source_name = Path(result.path).name
        else:
            source_name = f"Image {i+1}"

        print(f"{source_name}: {num_detections} guitar pedal(s) detected")

        # Print detection details
        for j, box in enumerate(result.boxes):
            conf = box.conf[0].item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  [{j+1}] Confidence: {conf:.2%}, Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    print("="*60)
    print(f"Total detections: {total_detections}")

    if args.save:
        print(f"Results saved to: runs/detect/predict/")

    return 0


def export_model(weights_path: str = 'runs/detect/pedal_detector/weights/best.pt',
                 format: str = 'onnx'):
    """
    Export the trained model to different formats.

    Args:
        weights_path: Path to the trained model weights
        format: Export format ('onnx', 'torchscript', 'tflite', 'openvino', etc.)
    """
    model = YOLO(weights_path)
    model.export(format=format)
    print(f"Model exported to {format} format")


if __name__ == '__main__':
    raise SystemExit(main())
