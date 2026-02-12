"""
YOLOv8 Nano Training Script for Guitar Pedal Detection

This script trains a YOLOv8 nano model on the guitar pedal dataset.
"""

import argparse
from ultralytics import YOLO


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train YOLOv8 for guitar pedal detection')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to dataset yaml file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--project', type=str, default='runs/detect', help='Project output directory')
    parser.add_argument('--name', type=str, default='pedal_detector', help='Run name')
    parser.add_argument('--device', type=str, default='0', help='CUDA device (0, 1, etc.) or "cpu"')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Load YOLOv8 nano pretrained model
    model = YOLO('yolov8n.pt')

    print("="*60)
    print("Starting YOLOv8 Nano Training for Guitar Pedal Detection")
    print("="*60)

    # Train the model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        seed=args.seed,

        # Augmentation settings (optimized for small dataset)
        hsv_h=0.015,      # Hue variation
        hsv_s=0.7,        # Saturation variation
        hsv_v=0.4,        # Value/brightness variation
        degrees=10,       # Rotation (moderate - pedals are usually upright)
        translate=0.1,    # Translation
        scale=0.5,        # Scale variation
        flipud=0.0,       # No vertical flip (pedals have orientation)
        fliplr=0.5,       # Horizontal flip
        mosaic=1.0,       # Mosaic augmentation
        mixup=0.1,        # Mix two images

        # Training settings
        optimizer='AdamW',
        lr0=0.001,        # Initial learning rate
        lrf=0.01,         # Final learning rate factor
        weight_decay=0.0005,
        warmup_epochs=3,

        # Output settings
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        verbose=True,

        # Resume from checkpoint if specified
        resume=args.resume,
    )

    # Validate the model
    print("\n" + "="*60)
    print("Validating trained model...")
    print("="*60)

    metrics = model.val()

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"\nBest model saved to: {args.project}/{args.name}/weights/best.pt")
    print("="*60)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
