export HYDRA_FULL_ERROR=1

TRAIN_TEST_SPLIT=navtest
CHECKPOINT=ckpt/sparsedrive_navsimv2.ckpt
CACHE_PATH=exp/metric_cache_navtestv2

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_navtest_v2_fast.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=sparsedrive_agent \
    agent.checkpoint_path=$CHECKPOINT \
    experiment_name=sparsedrive_agent \
    metric_cache_path=$CACHE_PATH \
    +test_cache_path=exp/data_cache_navtest \
    dataloader.params.batch_size=8
