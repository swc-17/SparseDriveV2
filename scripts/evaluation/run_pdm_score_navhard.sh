export HYDRA_FULL_ERROR=1
# export CUDA_VISIBLE_DEVICES=0,

TRAIN_TEST_SPLIT=navhard_two_stage
CHECKPOINT=ckpt/id_102.ckpt
CACHE_PATH=exp/metric_cache_navhard
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_navhard_fast.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=sparsedrive_agent_v8 \
    worker=ray_distributed_no_torch \
    agent.checkpoint_path=$CHECKPOINT \
    experiment_name=sparsedrive_agent_v8 \
    metric_cache_path=$CACHE_PATH \
    +test_cache_path=${NAVSIM_EXP_ROOT}/metric_feature_cache_navhard/ \
    synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
    synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
    dataloader.params.batch_size=16 \
    +agent.config.velocity_filter_num=[64,20]


    # worker=ray_distributed_no_torch \