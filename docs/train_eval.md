# Installation for SparseDriveV2

After successfully installing the NAVSIM environment, you should further proceed to install the following packages for SparseDriveV2:

```bash
conda activate navsim
cd navsim/agents/sparsedrive/ops
python setup.py develop
```

# Data and metric caching

```bash
sh scripts/cache/run_dataset_caching_navtrain.sh
sh scripts/cache/run_dataset_caching_navtest.sh

# for navsimv1
sh scripts/cache/run_metric_caching_navtrain_v1.sh
sh scripts/cache/run_metric_caching_navtest_v1.sh

# for navsimv2
sh scripts/cache/run_metric_caching_navtrain_v2.sh
sh scripts/cache/run_metric_caching_navtest_v2.sh
```

# Anchor preparation
you can download anchor files from [here]() and put to ckpt/kmeans/ or cluster by
```bash
mkdir -p ckpt/kmeans
sh scripts/cluster/cluster_anchor.py
```

# Download pretrained backbone and released model


# Training
```bash
# navsimv1
sh scripts/training/sparsedrive_navsimv1.sh
# navsimv2
sh scripts/training/sparsedrive_navsimv2.sh
```

# Evaluation
```bash
# navsimv1
sh scripts/evaluation/run_pdm_score_navtest_v1.sh
# navsimv2
sh scripts/evaluation/run_pdm_score_navtest_v2.sh
```

The PDM scores for navsimv2 both before and after the [fix]() will be reported.

