# CafeRacer: Robot Learning Data Analysis and Augmentation

This repository contains tools for analyzing and augmenting robot learning datasets, with a focus on pick-and-place tasks for robot arms manipulating objects like Lego bricks and containers.

## Overview

CafeRacer helps improve robot learning by:

1. Analyzing training and evaluation data using vision models
2. Identifying biases and limitations in training datasets
3. Suggesting and applying data augmentations to improve policy robustness
4. Evaluating episode success

## Prerequisites

- Python 3.10+
- PyTorch
- OpenCV
- NumPy
- PIL
- Matplotlib
- Modal
- Google Gemini API

## Setup

1. Clone the repository and navigate to the project directory
2. Set up environment variables:
   ```
   export GOOGLE_API_KEY="your_gemini_api_key"
   ```
3. Install required packages:
   ```
   pip install torch opencv-python numpy pillow matplotlib modal google-generativeai
   ```

## Project Structure

- `01_explore_data.ipynb`: Main notebook for data exploration, analysis, and augmentation
- `scripts/`: Utility functions
  - `aug_utils.py`: Functions for data augmentation (flipping, color changing, masking)
  - `image_utils.py`: Utilities for image handling and visualization
  - `inpaint_utils.py`: Tools for image inpainting and region manipulation

## Usage Examples

### 1. Loading a Dataset

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load training dataset
train_repo = "user/repo_name"
dataset = LeRobotDataset(train_repo, episodes=list(range(8)))
```

### 2. Analyzing Training Data

```python
from lerobot.caferacer.scripts.image_utils import tensor_to_pil

# Analyze training dataset
train_results = analyze_training_data(train_repo, episodes=list(range(8)))
train_summary = summarize_training_data(train_results)
```

### 3. Data Augmentation

```python
# Simple frame flipping
from lerobot.caferacer.scripts.aug_utils import flip_frame

flipped_obs, flipped_action = flip_frame(obs, action, ["observation.images.phone", "observation.images.laptop"])

# Color augmentation
from lerobot.caferacer.scripts.aug_utils import apply_color, get_mask

masks = get_mask(gsam, obs, ["observation.images.phone"], object="container")
colored_obs = apply_color(obs, masks, ["observation.images.phone"], target_color="yellow")
```

### 4. Evaluating Episodes

```python
from lerobot.caferacer.scripts.image_utils import tensor_to_pil

# Evaluate success of an episode
last_frame = dataset[to_idx-1]["observation.images.phone"]
evaluation = evaluate_episode_success(last_frame)
```

## Common Workflows

1. **Dataset Analysis**: Use `analyze_training_data()` to understand the training data
2. **Identify Biases**: Generate a summary with `summarize_training_data()` to find biases
3. **Evaluate Performance**: Use `analyze_eval_data()` to assess how the model performs
4. **Generate Augmentations**: Get automatic augmentation suggestions with `get_augmentations()`
5. **Apply Augmentations**: Use the augmentation utilities to transform your dataset

## Troubleshooting

- If you encounter GPU memory issues, set `GPU_POOR=True` in analysis functions
- For Modal function errors, check that you have the correct environment name configured

For more details, refer to the example notebook `01_explore_data.ipynb`.
