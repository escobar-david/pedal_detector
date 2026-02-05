"""
Interactive Guitar Pedal Detection
Loads model once and allows multiple predictions without reloading weights.
"""

import os
from ultralytics import YOLO


def find_weights():
    """Find the best.pt weights file."""
    possible_paths = [
        "best.pt",
        "runs/detect/pedal_detector/weights/best.pt",
        "runs/detect/runs/detect/pedal_detector/weights/best.pt",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def main():
    print("=" * 60)
    print("Interactive Guitar Pedal Detector")
    print("=" * 60)

    # Find and load weights
    weights_path = find_weights()
    if not weights_path:
        print("Error: Could not find best.pt weights file.")
        print("Download from: https://github.com/escobar-david/pedal_detector/releases")
        return

    print(f"Loading model from: {weights_path}")
    model = YOLO(weights_path)
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
        if user_input.lower() in ('quit', 'q', 'exit'):
            print("Goodbye!")
            break

        if user_input.lower() == 'save on':
            save_results = True
            print("Saving results: ON")
            continue

        if user_input.lower() == 'save off':
            save_results = False
            print("Saving results: OFF")
            continue

        if user_input.lower().startswith('conf '):
            try:
                conf_threshold = float(user_input.split()[1])
                print(f"Confidence threshold: {conf_threshold}")
            except (IndexError, ValueError):
                print("Usage: conf 0.5")
            continue

        # Check if file exists
        if not os.path.exists(user_input):
            print(f"File not found: {user_input}")
            continue

        # Run prediction
        results = model.predict(
            source=user_input,
            conf=conf_threshold,
            save=save_results,
            device='cpu',
            verbose=False
        )

        # Display results
        for result in results:
            boxes = result.boxes
            num_detections = len(boxes)
            filename = os.path.basename(user_input)

            print(f"\n{filename}: {num_detections} pedal(s) detected")

            if num_detections > 0:
                for i, box in enumerate(boxes):
                    conf = box.conf[0].item()
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    print(f"  [{i+1}] {conf*100:.1f}% at ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")

            if save_results:
                print(f"Saved to: runs/detect/predict/")
        print()


if __name__ == "__main__":
    main()
