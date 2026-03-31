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
    agent.lr=0.0001 \
    +agent.config.dataset_version=v1 \
    +agent.config.metrics=["no_at_fault_collisions","drivable_area_compliance","driving_direction_compliance","time_to_collision_within_bound","comfort","ego_progress"] \
    +agent.config.velocity_filter_num=[64,20]