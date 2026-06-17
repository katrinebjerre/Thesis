# Gaze Estimation Study

**Thesis (Spring 2026)**  
MSc Software Design  
IT University of Copenhagen  
  
**Authors**  
Katrine Bjerre (<katbj@itu.dk>)  
Kristine Emilie Risager Pedersen (<krep@itu.dk>)

<br/>

## Overview

This project investigates appearance-based gaze estimation using convolutional neural networks.  
Model generalization across subjects and targets is evaluated, along with the impact of transfer learning under limited subject-specific data.  
Model interpretability is further analyzed using Grad-CAM and B-cos visualizations.

### Table of Contents

- [Project Structure](#project-structure)
- [How to run](#how-to-run)
  - [1. Create and activate the environment](#1-create-and-activate-the-environment)
  - [2. Collect Data](#2-collect-data)
  - [3. Extract Frames](#3-extract-frames)
  - [4. Detect Pupils](#4-detect-pupils)
  - [5. Create Metadata and Process Tensors](#5-create-metadata-and-process-tensors)
  - [6. Run Grid Search](#6-run-grid-search)
  - [7. Run Experiments](#7-run-experiments)
  - [8. Analyze Results](#8-analyze-results)
  - [9. Generate Interpretability Visualizations](#9-generate-interpretability-visualizations)
- [Runtime](#runtime)
- [Acknowledgements](#acknowledgements)

<br/>

## Project Structure

Top-level overview of the repository structure:

```text
Thesis2026
├── data/                         # raw collected dataset (symlink)
├── data_processed/               # preprocessed tensors used for training
├── data_collection/              # data acquisition script
├── data_utils/                   # dataset loading, metadata files and data splits
├── experiments/                  # experiment configurations and execution scripts
├── models/                       # model architectures (CNN and B-cos)
├── notebooks/                    # analysis and visualization notebooks
├── preprocessing/                # frame extraction, pupil detection, metadata and tensor processing
├── results/                      # CSV results/loss history, interpretability visualizations and trained models
├── training/                     # training pipeline, evaluation, losses and metrics
├── environment*.yml              # environment setup files
└── README.md
```

> **Note:** Some utility scripts and 3D print files referenced in this repository are not included in the public version. See [Acknowledgements](#acknowledgements) for details.

<br/>

## How to run


### 1. Create and activate the environment

```bash
conda env create -f environment.yml
conda activate iml
```

<br/>

### 2. Collect Data

Due to privacy constraints, the dataset used in this project is not publicly available. Thus, to run the project, you must collect your own data using the provided script. For a full description of the experimental setup, including illustrations (measurements and angles), apparatus, and descriptions of the two fixation tasks, refer to Section 4.1 (Data Collection) in `thesis_censored.pdf`. The 3D models for the glint holder can be found in `data_collection/glintholder_3Dprint`.

Install dependency (one-time):

```bash
python -m pip install pypylon
```

Run the data collection script:

```bash
python data_collection/test_cam.py
```

This records video sessions with two cameras:
- **Camera A**: both eyes  
- **Camera B**: right eye only

Data is stored under `sessions/`.  

<br/>

### 3. Extract Frames

Extract frames from the Camera B video based on fixation timestamps.  
The script remaps fixation labels to a consistent grid order (`00` = top-left, `24` = bottom-right) and saves frames as `<fixation_index>_<timestamp>.png`.

```bash
python preprocessing/extract_frames_fixation.py sessions/<subject_id>/<date>_T<trial>/25_point_grid/session_cam1.mp4 \
  --ts sessions/<subject_id>/<date>_T<trial>/25_point_grid/CamB.ts.csv \
  --event-log sessions/<subject_id>/<date>_T<trial>/25_point_grid/event_log.csv \
  --fix-coords sessions/<subject_id>/<date>_T<trial>/25_point_grid/fixation_coords.csv
```

This creates a `frames/` folder in the `sessions/` directory.

After extracting frames, keep only `Camera B` data and organize it under `data/` instead:

```text
data/
├── 001/
│   ├── frames/
│   ├── CamB.ts.csv
│   ├── event_log.csv
│   └── fixation_coords.csv
├── 002/
...
```

<br/>

### 4. Detect Pupils

Run the pupil detection notebook:

```bash
preprocessing/pupil_detection/pupil_detection.ipynb
```

This detects pupil centers in the extracted frames and saves `pupil_coordinates.csv` for each subject.  
It also creates `failed_pupil_detection.csv`, which is used to exclude frames where detection failed.  
Additional frames can be excluded manually by adding their paths to `manual_bad_image_paths.csv`, which are then included in `failed_pupil_detection.csv`.

**Optional:** Explore the data before and after pupil detection in:
- `preprocessing/eda/eda_raw.ipynb`
- `preprocessing/eda/eda_clean.ipynb`

<br/>

### 5. Create Metadata and Process Tensors

Create the filtered metadata file by running:

```bash
python preprocessing/make_metadata.py
```

This creates `data_utils/metadata.csv`, which contains image paths, subject IDs, target labels and screen coordinates. Frames listed in `failed_pupil_detection.csv` are excluded.

Then convert the images to tensors:

```bash
python preprocessing/process_tensors.py
```

This script uses `metadata.csv` to load the images, resize them, convert them to grayscale, and save them as tensors in `data_processed/` to reduce computation during training.
It also creates `data_utils/metadata_tensors.csv`, which contains the same information as `metadata.csv`, but with updated paths pointing to the processed tensors.

This file is used to load the processed tensors for all training and experiments.

<br/>

### 6. Run Grid Search

> **Note:** If you have access to a GPU, use the GPU environment:
> ```bash
> conda env create -f environment-gpu.yml
> conda activate iml
> ```

Run the grid search to select the best hyperparameters within the predefined search space:

```bash
python experiments/grid_search.py
```

> **Note:** This step can take several days depending on hardware.

This evaluates all hyperparameter combinations in the search space defined in `experiments/grid_search.py` using `data_utils/metadata_tensors.csv`.

Grid search outputs are saved under `results/grid_search/`:

```text
results/grid_search/
├── best_config.json         # best hyperparameter setting from the search
├── grid_search_best.pth     # model trained with the selected setting
└── grid_search_results.csv  # results for all tested configurations
```
<br/>

### 7. Run Experiments

> **Note:** If you have access to a GPU, use the GPU environment:
> ```bash
> conda env create -f environment-gpu.yml
> conda activate iml
> ```

All experiment scripts are located in `experiments/`.

They follow one of two splitting strategies:
- **LOSO:** Train on all subjects except one → evaluate on the held-out subject  
- **LOTO:** Train on all targets except one → evaluate on the held-out target  

All experiments use the hyperparameters defined in `results/grid_search/best_config.json`.

#### General models
General models are trained on the full dataset.  
In LOTO, the 5 reserved subjects (defined in `data_utils/subject_groups.py`) are excluded.

| Script | Model | Setup | # Models |
|-------|------|------|---------|
| `gen_loso.py` | CNN | LOSO | #subjects |
| `gen_loto.py` | CNN | LOTO | #targets |
| `gen_loto_bcos.py` | BCOS | LOTO | #targets |
| `gen_loto_reserved_eval.py` | CNN | LOTO (eval) | – |


#### Subject-specific models
Models are trained per reserved subject using LOTO.  
Transfer models are initialized from the corresponding general LOTO model (trained without the target subject).  

Transfer setups specify which layers are fine-tuned while the rest are frozen:
- **Head**: only the regression head is trained.
- **Last conv**: the last convolutional layer and the head are trained.
- **Last N conv**: the last N convolutional layers and the head are trained.

| Script | Model | Setup | # Models |
|-------|------|------|---------|
| `subject_spec_scratch.py` | CNN | Scratch | 5 × #targets |
| `subject_spec_scratch_bcos.py` | BCOS | Scratch | 5 × #targets |
| `subject_spec_transfer_head_only.py` | CNN | Transfer (head) | 5 × #targets |
| `subject_spec_transfer_last_conv.py` | CNN | Transfer (last conv) | 5 × #targets |
| `subject_spec_transfer_last_two_conv.py` | CNN | Transfer (last 2 conv) | 5 × #targets |
| `subject_spec_transfer_last_two_conv_bcos.py` | BCOS | Transfer (last 2 conv) | 5 × #targets |
| `subject_spec_transfer_last_three_conv.py` | CNN | Transfer (last 3 conv) | 5 × #targets |


#### Run an experiment by executing the desired script. For example:

```bash
python experiments/gen_loto.py
```

Outputs are saved under `results/`, organized by experiment type. For example, running `gen_loto.py` produces:

```text
results/
├── generalized/
│   ├── gen_loto.csv           # evaluation results across all LOTO splits
│   ├── gen_loto_history.csv   # training and validation loss per split
├── models/
│   ├── gen_loto_0.pth         # model with target 0 held out
│   ├── gen_loto_1.pth
│   ├── gen_loto_2.pth
│   ├── ...
│   └── gen_loto_24.pth 
```
<br/>

### 8. Analyze Results

Use the analysis notebooks to summarize and compare model performance:

- `notebooks/stats_cnn.ipynb` and `notebooks/stats_bcos.ipynb` for per-model results, error distributions, summary statistics, and experiment comparisons
- `notebooks/train_val_loss_cnn.ipynb` and `notebooks/train_val_loss_bcos.ipynb` for training and validation loss curves

These notebooks read the CSV files saved under `results/`.

<br/>

### 9. Generate Interpretability Visualizations

Use the following notebooks to visualize model behavior:

- `notebooks/grad_cam.ipynb` generates Grad-CAM heatmaps for the x- and y-coordinate predictions of the CNN models
- `notebooks/b_cos.ipynb` generates B-cos explanations for the x- and y-coordinate predictions using a modified B-cos regression setup, enabling comparison with Grad-CAM

The visualizations are generated for the reserved subjects across targets and model types, and are saved under `results/`.

<br/>

## Runtime

The following runtimes are approximate and measured on the dataset used in this project:

- 108,136 grayscale images  
- resized to 96 × 128 pixels  

Experiments were conducted on two local machines, both using GPU acceleration.
Due to differences in performance, selected runtimes were re-measured on a single machine for comparability.
The runtimes listed below may therefore not directly correspond to the results stored in the repository.

Approximate runtime (selected scripts):

- `process_tensors.py`: ~90 min  
- `grid_search.py`: ~3 days total (864 combinations; measured before introducing preprocessed tensors — expected to be lower with the current pipeline)
- `gen_loso.py`: ~196 min  
- `gen_loto.py`: ~243 min
- `gen_loto_bcos.py`: ~466 min  
- `subject_spec_scratch.py`: ~55 min  
- `subject_spec_scratch_bcos.py`: ~56 min  
- `subject_spec_transfer_head_only.py`: ~31 min
- `subject_spec_transfer_last_conv.py`: ~33 min
- `subject_spec_transfer_last_two_conv.py`: ~34 min
- `subject_spec_transfer_last_two_conv_bcos.py`: ~50 min  
- `subject_spec_transfer_last_three_conv.py.py`: ~ 35 min

<br/>

## Acknowledgements

The following scripts and utilities were provided by Ingrid Jakobi Wollf Madsen (<inma@itu.dk>) and used during development of this project, but are not included in this public repository out of consideration for her PhD project.


- `environment.yml`
- `test_cam.py`
- `extract_frames_fixation.py`
- `filtering_util.py` and `iml_util.py` (used in `pupil_detection.ipynb`)
- `glint_holderv3.obj` and `new_glint v8.obj` (3D prints for data collection)

Please note that these files are **not included** in this GitHub repository out of consideration for her PhD project.
