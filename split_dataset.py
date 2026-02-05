"""
Dataset Splitting Utility for Guitar Pedal Detection

Splits the dataset into train/val sets while ensuring image-label pairs
are kept together. Handles missing labels gracefully.
"""

import os
import shutil
import random
from pathlib import Path


def get_image_label_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    """Find matching image-label pairs."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    pairs = []
    orphan_images = []
    orphan_labels = []

    # Get all images
    images = {f.stem: f for f in images_dir.iterdir()
              if f.suffix.lower() in image_extensions}

    # Get all labels
    labels = {f.stem: f for f in labels_dir.iterdir()
              if f.suffix == '.txt'}

    # Find matching pairs
    for stem, image_path in images.items():
        if stem in labels:
            pairs.append((image_path, labels[stem]))
        else:
            orphan_images.append(image_path)

    # Find orphan labels
    for stem, label_path in labels.items():
        if stem not in images:
            orphan_labels.append(label_path)

    return pairs, orphan_images, orphan_labels


def split_dataset(
    source_images: Path,
    source_labels: Path,
    output_dir: Path,
    train_ratio: float = 0.85,
    seed: int = 42
) -> dict:
    """
    Split dataset into train/val sets.

    Args:
        source_images: Path to source images directory
        source_labels: Path to source labels directory
        output_dir: Path to output datasets directory
        train_ratio: Ratio of training data (default 0.85)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with split statistics
    """
    random.seed(seed)

    # Get image-label pairs
    pairs, orphan_images, orphan_labels = get_image_label_pairs(
        source_images, source_labels
    )

    print(f"Found {len(pairs)} image-label pairs")
    if orphan_images:
        print(f"Warning: {len(orphan_images)} images without labels:")
        for img in orphan_images:
            print(f"  - {img.name}")
    if orphan_labels:
        print(f"Warning: {len(orphan_labels)} labels without images:")
        for lbl in orphan_labels:
            print(f"  - {lbl.name}")

    # Shuffle pairs
    random.shuffle(pairs)

    # Calculate split index
    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    # Create output directories
    train_images_dir = output_dir / 'train' / 'images'
    train_labels_dir = output_dir / 'train' / 'labels'
    val_images_dir = output_dir / 'val' / 'images'
    val_labels_dir = output_dir / 'val' / 'labels'

    for dir_path in [train_images_dir, train_labels_dir,
                     val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Copy files
    def copy_pairs(pairs_list, images_dir, labels_dir):
        for img_path, lbl_path in pairs_list:
            shutil.copy2(img_path, images_dir / img_path.name)
            shutil.copy2(lbl_path, labels_dir / lbl_path.name)

    print("\nCopying training set...")
    copy_pairs(train_pairs, train_images_dir, train_labels_dir)

    print("Copying validation set...")
    copy_pairs(val_pairs, val_images_dir, val_labels_dir)

    # Statistics
    stats = {
        'total_pairs': len(pairs),
        'train_count': len(train_pairs),
        'val_count': len(val_pairs),
        'orphan_images': len(orphan_images),
        'orphan_labels': len(orphan_labels),
    }

    print("\n" + "="*50)
    print("Dataset Split Complete!")
    print("="*50)
    print(f"Total pairs:     {stats['total_pairs']}")
    print(f"Training set:    {stats['train_count']} ({stats['train_count']/stats['total_pairs']*100:.1f}%)")
    print(f"Validation set:  {stats['val_count']} ({stats['val_count']/stats['total_pairs']*100:.1f}%)")
    print("="*50)

    return stats


def main():
    # Define paths
    project_root = Path(__file__).parent
    source_images = project_root / 'images'
    source_labels = project_root / 'labels'
    output_dir = project_root / 'datasets'

    # Verify source directories exist
    if not source_images.exists():
        print(f"Error: Images directory not found: {source_images}")
        return
    if not source_labels.exists():
        print(f"Error: Labels directory not found: {source_labels}")
        return

    # Split dataset
    stats = split_dataset(
        source_images=source_images,
        source_labels=source_labels,
        output_dir=output_dir,
        train_ratio=0.85,
        seed=42
    )

    print(f"\nDataset ready at: {output_dir}")
    print("You can now run training with: python train.py")


if __name__ == '__main__':
    main()
