# idtrackerai — Enhanced Segmentation

This project is based on [idtrackerai](https://gitlab.com/polavieja_lab/idtrackerai) by the Polavieja Lab. idtrackerai is an open-source multi-animal tracking system that uses artificial intelligence to track up to 100 unmarked animals from videos recorded in laboratory conditions.

## What this fork adds

This implementation enhances idtrackerai's segmentation pipeline by integrating modern deep-learning segmentation backends alongside the original threshold-based approach:

- **SAM 3 (Segment Anything Model 3)** — Text-prompted, zero-shot segmentation powered by [Ultralytics](https://github.com/ultralytics/ultralytics). Describe the animals you want to detect (e.g. "zebrafish", "ant") and SAM 3 segments them automatically — no manual intensity or area thresholds required.
- **Detectron2 (Instance Segmentation)** — Facebook's [Detectron2](https://github.com/facebookresearch/detectron2) framework for instance segmentation with pretrained or custom Mask R-CNN models. Enables pixel-accurate masks and per-instance class labels, ideal for complex scenes with overlapping or visually similar animals.

These additions allow users to choose the segmentation method best suited to their experimental setup, significantly improving accuracy in challenging conditions such as low contrast, cluttered backgrounds, or variable lighting — where the legacy threshold-based method struggles.

> **Acknowledgment:** This project builds upon the original [idtrackerai](https://gitlab.com/polavieja_lab/idtrackerai) developed by the Polavieja Lab. Please refer to the original repository for the upstream codebase, documentation, and citation information.

## Installation for developers

### 1. Using a Conda Environment (Recommended)

Conda is highly recommended as it simplifies the installation of Python, PyTorch with CUDA support, and system-level dependencies like PyQt6 and OpenCV.

1. **Create a new Conda environment** (Python 3.10 is recommended and fully supported):
   ```bash
   conda create --name idtrackerai_env python=3.10 -y
   ```

2. **Activate the environment**:
   ```bash
   conda activate idtrackerai_env
   ```

3. **Install PyTorch with CUDA support** (required for GPU acceleration):

   For most GPUs (RTX 20/30/40 series):
   ```bash
   conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
   ```

   For **newer GPUs** (RTX 50 series / Blackwell architecture) — requires PyTorch ≥ 2.7 with CUDA 12.8:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

4. **Install idtrackerai in editable developer mode**:
   Navigate to the repository root directory and run:
   ```bash
   pip install -e .
   ```
   *Note: If you need development tools (formatting, testing, etc.), install with the `dev` option: `pip install -e .[dev]`*

5. **Run the application**:
   ```bash
   idtrackerai
   ```

## Segmentation Methods

idtracker.ai supports three segmentation backends:

### 1. Threshold (legacy) — default

The original intensity-based segmentation with background subtraction. No extra installation needed.

### 2. SAM 3 (Segment Anything Model 3)

Text-prompted segmentation using [ultralytics](https://github.com/ultralytics/ultralytics). Included in the base install.

- Place the `sam3.pt` weights in the `weights/` directory
- Select **SAM 3** from the segmentation method dropdown in the Segmentation App and enter a text prompt describing your animals (e.g. "zebrafish")
- **Note**: You must request access to SAM3 weights at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)

### 3. Detectron2 (Instance Segmentation)

Uses Facebook's [Detectron2](https://github.com/facebookresearch/detectron2) for instance segmentation with pretrained or custom models.

**Installation (Windows — from source):**

1. Clone the repository:
   ```bash
   git clone https://github.com/facebookresearch/detectron2.git
   ```

2. **Disable C++ extensions** (recommended — avoids needing Visual C++ Build Tools):
   Open `detectron2/setup.py`, find the line `ext_modules=get_extensions()` inside the `setup(...)` call near the bottom, and change it to:
   ```python
   ext_modules=[],
   ```
   > Detectron2 works fully in pure-Python mode. The C++ extensions only provide minor speed-ups for a few custom ops.

3. Install in editable mode:
   ```bash
   pip install --no-build-isolation -e detectron2
   ```

> **Alternative:** If you have [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) installed, you can skip step 2 and the C++ extensions will compile automatically.

**Usage:**

All three fields (config, weights, class names) are **required**:

```toml
segmentation_method = "detectron2"
detectron2_config = "configs/detectron2/config.yaml"  # required
detectron2_weights = "weights/model_final.pth"        # required
detectron2_class_names = ["fish"]                     # required
detectron2_confidence_threshold = 0.5
```

The `detectron2_class_names` must match the class names in your model's training dataset (e.g. `["fish"]`, `["zebrafish", "medaka"]`). For models without named classes, use integer class IDs (e.g. `["0", "2"]`).
