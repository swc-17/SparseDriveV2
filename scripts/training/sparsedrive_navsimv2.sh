export HYDRA_FULL_ERROR=1

config=default_training
agent=sparsedrive_agent

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    --config-name $config \
    agent=$agent \
    experiment_name=$agent \
    train_test_split=navtrain \
    use_cache_without_dataset=True  \
    force_cache_computation=False \
    cache_path=exp/data_cache_navtrain \
    dataloader.params.batch_size=16 \
    dataloader.params.num_workers=16 \
    dataloader.params.prefetch_factor=4 \
    trainer.params.max_epochs=10 \
    agent.lr=0.0001