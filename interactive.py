"""
Interactive Guitar Pedal Detection
Loads model once and allows multiple predictions without reloading weights.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Interactive YOLOv8 pedal detector')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device (0, 1, etc.) or "cpu"')
    parser.add_argument('--weights', type=str, default='',
                        help='Optional explicit path to model weights')
    return parser.parse_args(argv)


def find_weights(explicit_weights: str = ''):
    """Find the best.pt weights file."""
    if explicit_weights:
        explicit_path = Path(explicit_weights)
        if explicit_path.exists():
            return explicit_path
        return None

    possible_paths = (
        Path("best.pt"),
        Path("runs/detect/pedal_detector/weights/best.pt"),
    )
    for path in possible_paths:
        if path.exists():
            return path
    return None


def main(argv=None):
    args = parse_args(argv)

    print("=" * 60)
    print("Interactive Guitar Pedal Detector")
    print("=" * 60)

    # Find and load weights
    weights_path = find_weights(args.weights)
    if not weights_path:
        print("Error: Could not find model weights file.")
        if args.weights:
            print(f"Tried explicit --weights path: {args.weights}")
        print("Download from: https://github.com/escobar-david/pedal_detector/releases")
        return 1

    print(f"Loading model from: {weights_path}")
    model = YOLO(str(weights_path))
    print("Model loaded! Ready for predictions.\n")

    print("Commands:")
    print("  - Enter image path to detect pedals")
    print("  - 'save on' / 'save off' to toggle saving results")
    print("  - 'conf 0.5' to change confidence threshold")
    print("  - 'quit' or 'q' to exit\n")

    save_results = True
    conf_threshold = 0.25

    while True:
        try:
            user_input = input("Image path: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ("quit", "q", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "save on":
            save_results = True
            print("Saving results: ON")
            continue

        if user_input.lower() == "save off":
            save_results = False
            print("Saving results: OFF")
            continue

        if user_input.lower().startswith("conf "):
            try:
                conf_threshold = float(user_input.split()[1])
                print(f"Confidence threshold: {conf_threshold}")
            except (IndexError, ValueError):
                print("Usage: conf 0.5")
            continue

        # Check if file exists
        if not Path(user_input).exists():
            print(f"File not found: {user_input}")
            continue

        # Run prediction
        results = model.predict(
            source=user_input,
            conf=conf_threshold,
            save=save_results,
            device=args.device,
            verbose=False,
        )

        # Display results
        for result in results:
            boxes = result.boxes
            num_detections = len(boxes)
            filename = Path(user_input).name

            print(f"\n{filename}: {num_detections} pedal(s) detected")

            if num_detections > 0:
                for i, box in enumerate(boxes):
                    conf = box.conf[0].item()
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    print(
                        f"  [{i + 1}] {conf * 100:.1f}% at ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
                    )

            if save_results:
                print(f"Saved to: runs/detect/predict/")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
