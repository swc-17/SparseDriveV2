export HYDRA_FULL_ERROR=1

TRAIN_TEST_SPLIT=navtest
CHECKPOINT=ckpt/sparsedrive_navsimv1.ckpt
CACHE_PATH=exp/metric_cache_navtestv1

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_navtest_v1_fast.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=sparsedrive_agent \
    agent.checkpoint_path=$CHECKPOINT \
    experiment_name=sparsedrive_agent \
    metric_cache_path=$CACHE_PATH \
    +test_cache_path=exp/data_cache_navtest \
    dataloader.params.batch_size=8 \
    +agent.config.dataset_version=v1 \
    +agent.config.metrics=["no_at_fault_collisions","drivable_area_compliance","driving_direction_compliance","time_to_collision_within_bound","comfort","ego_progress"] \
    +agent.config.velocity_filter_num=[64,20]