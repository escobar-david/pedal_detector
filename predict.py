"""
Inference Script for Guitar Pedal Detection

Run predictions on images using the trained YOLOv8 model.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
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
    parser.add_argument('--device', type=str, default='cpu',
                        help='CUDA device (0, 1, etc.) or "cpu"')
    parser.add_argument('--save', action='store_true',
                        help='Save results to runs/detect/predict')
    parser.add_argument('--show', action='store_true',
                        help='Display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results as .txt files')
    parser.add_argument('--save-crop', action='store_true',
                        help='Save cropped detection boxes')
    return parser.parse_args()


def main():
    args = parse_args()

    # Check if weights exist
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Model weights not found at {weights_path}")
        print("Make sure you have trained the model first with: python train.py")
        return

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

    return results


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
    main()
