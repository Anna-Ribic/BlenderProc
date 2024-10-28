
# 3D-FRONT Processing Pipeline

This repository provides Python scripts for automatic reconstruction of 3D-FRONT scenes with spiral camera trajectories, leveraging `BlenderProc`.


## Installation

### Via pip

The simplest way to install blenderproc is via pip:

```bash
pip install blenderproc
```

### Via git
To still make use of the blenderproc command and therefore use blenderproc anywhere on your system, make a local pip installation:

```bash
cd BlenderProc
pip install -e .
```


## Download Required Datasets
The following datasets are essential for this pipeline:

1. **3D-FRONT and 3D-FUTURE**  
   Download the 3D-FRONT dataset by following these steps:
   - Visit the official [3D-FRONT website](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) and register for an account.
   - After logging in, you can download the 3d-FRONT dataset, which includes 3D room layouts in `.json` format and unzip the files (e.g., `3D-FRONT/`).
   - Download the furniture models and unzip them (e.g., `3D-FUTURE-model/`).

2. **CCTextures**  
   Download high-quality textures from CCTextures using BlenderProc's `download_cc_textures.py` at  `BlenderProc/examples/scripts`:
   - Replace `path_to_cctextures/` with the directory where you want to save the textures.

    
## Reconstruction Pipeline


1. Automatically create camera trajectories for each room defined in a 3D-FRONT scene config `.json` file. 
   Generates spiral trajectories based on 3D bounding boxes from JSON scene files.
   - **Usage**: 
     ```bash
     python extract_traj_from_json.py --input_file json_file.txt --base_path 3D-FRONT/ --global_txt_file generated_configs.txt --output_dir configs
     ```
   - **Arguments**:
     - `--input_file`: Path to the JSON or text file with multiple JSON paths.
     - `--base_path`: Directory prefix for JSON paths.
     - `--global_txt_file`: Output text file to store generated paths.
     - `--output_dir`: Directory to save generated configs.

   
2. (Optional) For each 3D-FRONT scene extract geometry as `.ply` in order to use the `--visualize` option in the next steps.
   - **Usage**:
     ```bash
     python extract_geom.py --json_list_file json_files.txt --json_dir 3D-FRONT/ --future_model_dir 3D-FUTURE-model/ --resources_dir path_to_cctextures --blenderproc_script "examples/datasets/front_3d_with_improved_mat/geom.py" --output_dir "scenes"
     ```
   - **Arguments**:
     - `--json_list_file`: Path to the JSON or text file with multiple JSON paths.
     - `--json_dir`: Directory for JSON files.
     - `--future_model_dir`: Directory for 3D-FUTURE models.
     - `--resources_dir`: Path to CCTextures.
     - `--blenderproc_script`: BlenderProc script path.
     - `--output_dir`: Directory to save processed scenes.

    
3. Run reconstruction pipeline to render RGB and depth images for precomputed camera trajectory and create sparse reconstruction in COLMAP format.
   - **Usage**:
     ```bash
     python pipeline.py generated_configs.txt --front_path 3D-FRONT/ --future_path 3D-FUTURE-model/ --resources_path path_to_cctextures --scenes_path scenes
     ```
   - **Arguments**:
     - `--config_path`: Path to the JSON or text file with multiple JSON paths.
     - `--visualize`: Optional; add to enable visualization during spiral generation.
     - `--front_path`, `--future_path`, `--resources_path`, `--scenes_path`: Paths to respective 3D-FRONT, 3D-FUTURE, resources, and scenes directories.

    
4. Given the image sets rendered in the previous step, create a low quality reconstruction of the same room using an eight of the full image set.
   - **Usage**:
     ```bash
     python pipeline_oct.py scene_paths.txt --front_path 3D-FRONT/ --future_path 3D-FUTURE-model/ --resources_path path_to_cctextures --scenes_path scenes
     ```
   - **Arguments**:
     - `--scene_path`: Path to a single scene directory or a `.txt` file listing scene directories.
     - `--front_path`, `--future_path`, `--resources_path`, `--scenes_path`: Paths to respective 3D-FRONT, 3D-FUTURE, resources, and scenes directories.
