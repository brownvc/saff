# Semantic Attention Flow Fields for Monocular Dynamic Scene Decomposition (ICCV 2023)  <img src="https://visual.cs.brown.edu/projects/semantic-attention-flow-fields-webpage/BrownCSLogo.png"  width="10%" height="10%">

[Yiqing Liang](lynl7130.github.io), [Eliot Laidlaw](https://www.linkedin.com/in/eliot-laidlaw-472640197/), [Alexander Meyerowitz](https://www.linkedin.com/in/ameyerow/), [Srinath Sridhar](https://cs.brown.edu/people/ssrinath/), [James Tompkin](https://jamestompkin.com/)

Official Implementation for "Semantic Attention Flow Fields for Monocular Dynamic Scene Decomposition".

[[```Paper```](https://arxiv.org/abs/2303.01526)]  [[```Project```](https://visual.cs.brown.edu/projects/semantic-attention-flow-fields-webpage/)]

## Installation

```
virtualenv <env_name>
source <env_name>/bin/activate
pip install -r [path/to/repo]/requirements.txt
```

## Downloada Data, Checkpoints and Results

[NVIDIA dataset](https://drive.google.com/drive/folders/1JNhhRt9VAFXnQeoykpgTc35ByDZGkO6v?usp=drive_link)

```ours_1018_processed_crf.zip```: final results on NVIDIA dataset

```nvidia_data_full.zip```: processed NVIDIA dataset for training and testing

```gt_masks.zip```: segmentation annotation for NVIDIA dataset

```checkpoints.zip```: trained models for NIVIDIA dataset



## Data, Configurations and Checkpoints

| Scene  | Data | Config  |  Checkpoint  |
|---|---|---|---|
| Balloon1-2  | nvidia_data_full/Balloon1-2  | https://github.com/brownvc/NOF/blob/camera_ready/Neural-Scene-Flow-Fields/nsff_exp/configs/config_balloon1-2_4.txt  | checkpoints/Balloon1-2/360000.tar  |
| Balloon2-2  | nvidia_data_full/Balloon2-2  | https://github.com/brownvc/NOF/blob/camera_ready/Neural-Scene-Flow-Fields/nsff_exp/configs/config_balloon2-2_4.txt  | checkpoints/Balloon2-2/360000.tar  |
| DynamicFace-2  | nvidia_data_full/DynamicFace-2  | https://github.com/brownvc/NOF/blob/camera_ready/Neural-Scene-Flow-Fields/nsff_exp/configs/config_dynamicFace-2_4.txt  | checkpoints/DynamicFace-2/360000.tar  |
| Jumping  | nvidia_data_full/Jumping  | https://github.com/brownvc/NOF/blob/camera_ready/Neural-Scene-Flow-Fields/nsff_exp/configs/config_jumping_4.txt  | checkpoints/Jumping/360000.tar  |
| Playground  | nvidia_data_full/playground  | https://github.com/brownvc/NOF/blob/camera_ready/Neural-Scene-Flow-Fields/nsff_exp/configs/config_playground_4.txt  | checkpoints/Playground/360000.tar  |
| Skating-2  | nvidia_data_full/Skating-2  | https://github.com/brownvc/NOF/blob/camera_ready/Neural-Scene-Flow-Fields/nsff_exp/configs/config_skating-2_4.txt  | checkpoints/Skating-2/360000.tar  |
| Truck  | nvidia_data_full/Truck-2  | https://github.com/brownvc/NOF/blob/camera_ready/Neural-Scene-Flow-Fields/nsff_exp/configs/config_truck2_4.txt  | checkpoints/Truck/360000.tar  |
| Umbrella  | nvidia_data_full/Umbrella  | https://github.com/brownvc/NOF/blob/camera_ready/Neural-Scene-Flow-Fields/nsff_exp/configs/config_umbrella_4.txt  | checkpoints/Umbrella/360000.tar  |


## Workflow for NVIDIA Dataset

```cd [path/to/repo]/Neural-Scene-Flow-Fields/nsff_exp```

### Training 
```python run_nerf.py --config [path/to/config/file]```

For now, each scene(```[path/to/config/file]```)'s corresponding results is stored to:
```
├── [path/to/repo]/Neural-Scene-Flow-Fields/nsff_exp/logs/[your/exp/name]_F[start_frame]_[end_frame]/
│   ├── 000000.tar
│   ├── 010000.tar
│   ├── ...
│   ├── 360000.tar
│   ├── args.txt
│   ├── config.txt
```

### After Training, render per-view
```python run_nerf.py --config [path/to/config/file] --render_2D```

This would create a new folder under scene folder:
```
├── [path/to/repo]/Neural-Scene-Flow-Fields/nsff_exp/logs/[your/exp/name]_F[start_frame]_[end_frame]/
|   ├──render_2D-010_path_[last_ckpt_id+1]
|       ├──0_blend.png
|       ├──0_depth.png
|       ├──0_dino.pt
|       ├──0_rgb.png
|       ├──0_sal.png
|       ├──...
```

### After Training, cluster per-view
```python run_nerf.py --config [path/to/config/file] --cluster_2D```

```python run_nerf.py --config [path/to/config/file] --cluster_2D --render_mode```

This would create a new folder under scene folder:
```
├── [path/to/repo]/Neural-Scene-Flow-Fields/nsff_exp/logs/[your/exp/name]_F[start_frame]_[end_frame]/
|   ├──cluster_2D-010_path_[last_ckpt_id+1]
|       ├──final
|           ├──0.png
|           ├──1.png
|           ├──...
|       ├──no_merge_no_salient
|           ├──0.png
|           ├──1.png
|           ├──...
|       ├──no_salient
|           ├──0.png
|           ├──1.png
|           ├──...
|       ├──0_bld_full_beforemerge.png
|       ├──0_bld_full.png
|       ├──0_bld.pt
|       ├──0_clu_full_beforemerge.png
|       ├──0_clu_full.png
|       ├──0_clu.png
|       ├──...
```


### Collect Results

First organize all scene's final clustering result(```[path/to/repo]/Neural-Scene-Flow-Fields/nsff_exp/logs/[your/exp/name]_F[start_frame]_[end_frame]/cluster_2D-010_path_[last_ckpt_id+1]/final```) and final rgb rendering result(```[path/to/repo]/Neural-Scene-Flow-Fields/nsff_exp/logs/[your/exp/name]_F[start_frame]_[end_frame]/render_2D-010_path_[last_ckpt_id+1]/*_rgb.png```) under the same folder:
```
├── [path/to/results]
|   ├──Balloon1-2
|       ├──0.png # cluster result
|       ├──0_rgb.png # rgb result
|       ├──...
|   ├──Balloon2-2
|   ├──DynamicFace-2
|   ├──Jumping
|   ├──playground
|   ├──Skating-2
|   ├──Truck-2
|   ├──Umbrella-2
```


### PostProcessing

#### NOTE: 
for now only supports NVIDIA scene organization!

```python postprocess.py --root_dir [path/to/results]```

```python postprocess_crf.py --root_dir [path/to/results]_processed```

Now the results would be stored in folder ```[path/to/results]_processed_crf```.

### Evaluation
for now only supports NVIDIA scene organization!

```cd [path/to/repo]/benchmarks```

```python evaluate_fg_ours.py --vis_folder [path/to/results]_processed_crf --gt_folder [path/to/gt_masks```

```python evaluate_ours.py --vis_folder [path/to/results]_processed_crf --gt_folder [path/to/gt_masks```

## Reference 
* https://github.com/gaochen315/DynamicNeRF
* https://github.com/zhengqili/Neural-Scene-Flow-Fields/tree/main