# Pedal Detector (YOLOv8)

Object detection pipeline for identifying guitar pedals in pedalboard photos using YOLOv8n.

This project focuses on a practical end-to-end ML workflow in Python: dataset preparation, reproducible training, CLI inference, and model export.

## Why This Project

- Real-world computer vision task on custom data.
- Complete workflow from raw labels to inference artifacts.
- Lightweight model choice for fast inference.

## Results Snapshot

Validation split: 25 images (trained on 139 images).

| Metric | Value |
|---|---:|
| mAP@50 | 0.975 |
| mAP@50-95 | 0.735 |
| Precision | 0.941 |
| Recall | 0.941 |

## Quick Start

```bash
git clone https://github.com/<your-user>/pedal_detector.git
cd pedal_detector
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download pretrained weights from GitHub Releases:

```bash
curl -L -o best.pt https://github.com/escobar-david/pedal_detector/releases/download/yolo-pedalboards/best.pt
python predict.py --source my_pedals.jpg --weights best.pt --save
```

## Reproduce Training

```bash
python split_dataset.py
python train.py --epochs 100 --batch 16 --imgsz 640 --device 0
python predict.py --source my_pedals.jpg --weights runs/detect/pedal_detector/weights/best.pt --save
```

CPU-only training:

```bash
python train.py --device cpu
```

## CLI Usage

```bash
python train.py --help
python predict.py --help
python interactive.py
```

## Project Structure

```text
pedal_detector/
|-- data.yaml
|-- split_dataset.py
|-- train.py
|-- predict.py
|-- interactive.py
|-- requirements.txt
|-- images/                # raw images (local/private by default)
|-- labels/                # raw labels (local/private by default)
|-- datasets/              # generated train/val split
`-- runs/                  # training and inference outputs
```

## Design Choices

- Model: YOLOv8n for speed and compact size.
- Transfer learning from pretrained weights.
- Augmentation tuned for small dataset:
- HSV jitter
- Horizontal flips
- Moderate rotation (+/-10 degrees)
- Mosaic and mixup

## Limitations

- Single class only: `guitar_pedal`.
- Small dataset size may reduce generalization.
- No instance segmentation or pedal-type classification yet.

## Roadmap

- Multi-class labels by pedal type.
- Structured error analysis and evaluation reports.
- Dockerized inference service.

## Dataset and Rights

Before publishing images/labels publicly, confirm you have redistribution rights for every photo and annotation.

## License

MIT. See `LICENSE`.
