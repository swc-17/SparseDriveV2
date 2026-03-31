TRAIN_TEST_SPLIT=navtrain
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_navtrainv1

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching_v1.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache.cache_path=$CACHE_PATH