# Semantic Attention Flow Fields for Monocular Dynamic Scene Decomposition (ICCV 2023)  <img src="https://visual.cs.brown.edu/projects/semantic-attention-flow-fields-webpage/BrownCSLogo.png"  width="10%" height="10%">

[Yiqing Liang](lynl7130.github.io), [Eliot Laidlaw](https://www.linkedin.com/in/eliot-laidlaw-472640197/), [Alexander Meyerowitz](https://www.linkedin.com/in/ameyerow/), [Srinath Sridhar](https://cs.brown.edu/people/ssrinath/), [James Tompkin](https://jamestompkin.com/)

Official Implementation for "Semantic Attention Flow Fields for Monocular Dynamic Scene Decomposition".

[[```Paper```](https://arxiv.org/abs/2303.01526)]  [[```Project```](https://visual.cs.brown.edu/projects/semantic-attention-flow-fields-webpage/)] [[```Data```](#download-data-checkpoints-and-results)]

## Installation

Tested with System Spec:

```
python==3.7.4
cmake==3.20.0
gcc==10.2
cuda==11.3.1
cudnn==8.2.0 
```

Steps:
```
python3 -m venv <env_name>
source <env_name>/bin/activate

pip install --upgrade pip

pip install -r [path/to/repo]/requirements.txt

# install eigen using apt-get if can; otherwise compile from source:
cd <folder to store tmp file>
wget https://gitlab.com/libeigen/eigen/-/archive/master/eigen-master.tar.gz
tar -xzf eigen-master.tar.gz
cd eigen-master
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<env_name>
make -j8
make install

#install pydensecrf
pip3 install --force-reinstall cython==0.29.36
pip install --no-build-isolation git+https://github.com/lucasb-eyer/pydensecrf.git

```

We tested our code with 1 single RTX3090 / A6000.

## Download Data, Checkpoints and Results

[NVIDIA dataset](https://drive.google.com/drive/folders/1JNhhRt9VAFXnQeoykpgTc35ByDZGkO6v?usp=drive_link)


```gt_masks.zip```: segmentation annotation for NVIDIA dataset

```ours_1018_processed_crf.zip```: final results on NVIDIA dataset

```nvidia_data_full.zip```: processed NVIDIA dataset for training and testing

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

For usage, Data and Checkpoints should be put to corresponding location according to config files.

```cd [path/to/repo]/Neural-Scene-Flow-Fields/``` 

Data folder should be renamed as ```datadir```.

Checkpoint should be put under ```basedir/expname```.




## Workflow 

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

#### Split (Shared by all sections):
* ```0-23```: training views

* ```24-47```: fixed camera 0, moving times

* ```48-59```: moving camera, fixed time = 0

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

### Oracle Vote Using GT mask

```python postprocess_oracle.py --raw_folder [path/to/repo]/Neural-Scene-Flow-Fields/nsff_exp/logs/[your/exp/name]_F[start_frame]_[end_frame]/cluster_2D-010_path_[last_ckpt_id+1]/no_salient --gt_folder [path/to/this/scene/mask] [--flip_fg]```

If use Black to denote background, need ```--flip_fg``` flag.

This would create a new folder ```oracle``` under ```no_salient``` folder.

### CRF postprocessing

Note: ```[path/to/your/final/result]``` could be your default clustering result folder ```[path/to/repo]/Neural-Scene-Flow-Fields/nsff_exp/logs/[your/exp/name]_F[start_frame]_[end_frame]/cluster_2D-010_path_[last_ckpt_id+1]/final``` or oracle processed folder ```[path/to/repo]/Neural-Scene-Flow-Fields/nsff_exp/logs/[your/exp/name]_F[start_frame]_[end_frame]/cluster_2D-010_path_[last_ckpt_id+1]/no_salient/oracle```.

```python postprocess_per_scene.py --root_dir [path/to/your/final/result]```

```python postprocess_crf_per_scene.py --root_dir [path/to/your/final/result]_processed --render_dir [path/to/repo]/Neural-Scene-Flow-Fields/nsff_exp/logs/[your/exp/name]_F[start_frame]_[end_frame]/render_2D-010_path_[last_ckpt_id+1]```



## Collect Results for NVIDIA dataset


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


### PostProcessing (NVIDIA dataset only)

#### NOTE: 
for now only supports NVIDIA scene organization!

```python postprocess.py --root_dir [path/to/results]```

```python postprocess_crf.py --root_dir [path/to/results]_processed```

Now the results would be stored in folder ```[path/to/results]_processed_crf```.

### Evaluation (NVIDIA dataset only)
for now only supports NVIDIA scene organization!

```cd [path/to/repo]/benchmarks```

```python evaluate_fg_ours.py --vis_folder [path/to/results]_processed_crf --gt_folder [path/to/gt_masks]```

```python evaluate_ours.py --vis_folder [path/to/results]_processed_crf --gt_folder [path/to/gt_masks]```

## Use SAFF on your own data

### Step 1. Data prepration
Collect rgb image sequence and corresponding dynamic masks, organize like:
```
├── [path/to/data]
|   ├──images
|       ├──00000.png 
|       ├──00001.png
|       ├──...
|   ├──colmap_masks
|       ├──00000.png.png
|       ├──00001.png.png
|       ├──...
```
Note: Naming has to follow ```%05d``` format!

```colmap_masks``` is where we store foreground masks. Dynamic Foreground -> black. Static Background -> white.

### Step 2. run COLMAP

Make sure you have colmap installed on your machine.

For example, on Mac, run
```sudo zsh colmap.sh [path/to/data]```

After running, the same data folder looks like: 
```
├── [path/to/data]
|   ├──images
|       ├──...
|   ├──colmap_masks
|       ├──...
|   ├──dense
|       ├──...
|   ├──sparse
|       ├──...
|   ├──database.db
|   ├──database.db-shm
|   ├──database.db-wal
```

```[path/to/data]/dense``` is the final data we want.

### Step 3. calculate pseudo depth and optical flow for supervision

```cd [path/to/repo]/Neural-Scene-Flow-Fields/nsff_scripts```

```python save_poses_nerf.py --data_path "[path/to/data]/dense/"```

Download single view depth prediction model ```model.pt``` from [link](https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV?usp=sharing), and put it under the folder ```nsff_scripts```.

```python run_midas.py --data_path "[path/to/data]/dense/" [--resize_height ???] # use resize_height if data is over big for SAFF ```

Download RAFT model ```raft-things.pth``` from [link](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing), and put it under the folder ```nsff_scripts/models```.

```python run_flows_video.py --model models/raft-things.pth --data_path [path/to/data]/dense/```


After running, the same data folder looks like: 
```
├── [path/to/data]
|   ├──images
|       ├──...
|   ├──colmap_masks
|       ├──...
|   ├──dense
|       ├──disp
|           ├──...
|       ├──flow_i1
|           ├──...
|       ├──images
|           ├──...
|       ├──images_306x400
|           ├──...
|       ├──motion_masks
|           ├──...
|       ├──sparse
|           ├──...
|       ├──stereo
|           ├──...
|       ├──poses_bounds.npy
|       ├──run-colmap-geometric.sh
|       ├──run-colmap-photometric.sh
|       ├──scene.json
|   ├──sparse
|       ├──...
|   ├──database.db
|   ├──database.db-shm
|   ├──database.db-wal
```

### Step 4. Create Config file 

copy config file [template](https://github.com/brownvc/NOF/blob/camera_ready/Neural-Scene-Flow-Fields/nsff_exp/configs/config_balloon1-2_4.txt), and change the value of field:
* ```expname```: to be your expname
* ```datadir```: to your [path/to/data]/dense
* ```final_height```: this must be same as --resize_height argument in run_midas.py
* ```start_frame```, ```end_frame```: which images would 
participate in training according to image id.

Note: if end_frame - start_frame == 48, would filter out half of the images for testing. (for DyCheck's sake)

#### Then you are all set! 

## Reference 
* https://github.com/gaochen315/DynamicNeRF
* https://github.com/zhengqili/Neural-Scene-Flow-Fields/tree/main