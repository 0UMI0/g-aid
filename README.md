# G-AID (g-aid)

This project was made for my honours project, its for detecting AI-Generated Images.

## Quick summary

- Input: an image (png, jpg, jpeg, webp)
- Output: a prediction label ("Real" or "AI-generated") and a probability score
- Web UI: simple Flask app served from `frontend/app.py`
- Model checkpoint: `checkpoints/npr_model_augs.pth`


## Repository structure

- frontend/
  - app.py            — Flask server and routes (web UI, upload handler)
  - model.py          — model architecture (NPRResNet50 / custom ResNet-50 variant)
  - utils.py          — preprocessing and prediction helpers
  - requirements.txt  — Python dependencies for the front-end (Flask, PyTorch, etc.)
  - templates/index.html — simple upload UI
  - static/uploads/   — example images and where uploaded images are stored

- checkpoints/
  - npr_model_augs.pth — pretrained model checkpoint used by the app

- pre-processing/
  - chunks.py         — helper script for splitting/processing the dataset in the project
  - maketest.py       — utility creation of chunks for colab

- g_aid.ipynb         — notebook with experiments 


## Features

- Single-image upload from a browser.
- Model inference on CPU or GPU (PyTorch). The app will use CUDA if available, otherwise fall back to CPU.
- Small, self-contained front-end (Flask) for quick demonstrations.


## Requirements

- Python 3.8+ (3.10+ recommended)
- Windows / macOS / Linux supported
- Recommended: a virtual environment to isolate dependencies

Python packages required (listed in `frontend/requirements.txt`):
- flask
- torch
- torchvision
- pillow
- werkzeug

If you want to run with GPU acceleration, install a PyTorch binary with CUDA support (see https://pytorch.org/).



## Using the web UI

1. Python app.py and open port in your browser.
2. Use the upload control to select an image (supported types: png, jpg, jpeg, webp). Max upload size is 8 MB.
3. After upload the page will show the selected image and the model prediction (label + probability).
4. Uploaded files are saved to `frontend/static/uploads/`.


## Model details

- Architecture: custom ResNet-style model implemented in `frontend/model.py` (class `NPRResNet50`).
- Input: RGB images resized to 256x256 (see `IMG_SIZE` in `frontend/utils.py`).
- Preprocessing: normalization using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
- Output head: single-output binary score (logit). The code applies a sigmoid and thresholds at 0.5 to pick the label.
- Checkpoint: `checkpoints/npr_model_augs.pth` — the app expects a model checkpoint to be present and loadable by `torch.load`.
