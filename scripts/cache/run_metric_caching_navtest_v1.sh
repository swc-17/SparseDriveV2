TRAIN_TEST_SPLIT=navtest
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_navtestv1

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching_v1.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache.cache_path=$CACHE_PATH