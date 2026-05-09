# Quick Start

### Set up a new virtual environment
```bash
conda create -n sparsedrive python=3.8 -y
conda activate sparsedrive
```

### Install dependency packpages
```bash
sparsedrive_path="path/to/sparsedrive"
cd ${sparsedrive_path}
pip3 install --upgrade pip
pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirement.txt
```

### Compile the deformable_aggregation CUDA op
```bash
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../
```

### Prepare the data
Download the [Bench2Drive dataset](https://github.com/Thinklab-SJTU/Bench2Drive#Dataset).

Pack the meta-information and labels of the dataset, and generate the required pkl files to data/infos. Note that we also generate map_annos in data_converter.
```bash
sh scripts/create_data.sh
```
You can also directly download the generated info from [here](https://huggingface.co/wenchaosun/SparseDriveV2).

### Generate anchors by K-means
Gnerated anchors are saved to data/kmeans and can be visualized in vis/kmeans.
```bash
sh scripts/kmeans.sh
```
You can also directly download the clustered anchor from [here](https://huggingface.co/wenchaosun/SparseDriveV2).


### Download pre-trained weights
Download the pretrained backbone:
```bash
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```
Download the pretrained weights from [here](https://huggingface.co/wenchaosun/SparseDriveV2).

### Check the folder structure
The folder structure after preparing everything should look like:

```
.
├── ckpt
│   ├── resnet50-19c8e357.pth
│   ├── sparsedrive_small_b2d_stage1.pth
│   └── sparsedrive_small_b2d_stage2.pth
├── data
│   ├── bench2drive
│   │   └── v1
│   ├── infos
│   │   ├── b2d_infos_train.pkl
│   │   └── b2d_infos_val.pkl
│   ├── kmeans
│   │   ├── kmeans_det_900.npy
│   │   ├── kmeans_map_100.npy
│   │   ├── kmeans_motion_6.npy
│   │   ├── path_1m_pts_15_1024_b2d_new_ego.npy
│   │   ├── trajectory_1024_256_b2d.npz
│   │   └── vel_seq_K256_t30.npy
├── leaderboard
├── projects
├── scenario_runner
├── scripts
├── tools
```

### Commence training and testing
```bash
# training.  note that the stage2 config is for training on 16 gpus, you need to tune some parameters (num_gpus, learning rate) if you want to train on different number of gpus.
sh scripts/train.sh

# test.
sh scripts/test.sh
```

### Visualization
```
sh scripts/visualize.sh
```

### Closed-loop evalution
```
python scripts/eval_b2d_multi.py
```
Results are saved in close_loop_log/ similar to [this](merged.json).
