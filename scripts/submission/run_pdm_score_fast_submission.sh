export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=7,

TEAM_NAME="SparseDriveV2"
AUTHORS="SunWenchao"
EMAIL="swc21@mails.tsinghua.edu.cn"
INSTITUTION="Tsinghua"
COUNTRY="China"

# CHECKPOINT="http://pfs-svcspawner.bcloud-bj-zone1.hobot.cc/user/homespace/wenchao01.sun/plat_gpu/2026-02-27/10-18/id_145_repro_v1_new-20260227-101833.787136/output/exp/sparsedrive_agent_v11/2026.02.27.22.57.26/periodic_pdm_ckpts/ep0010.ckpt"
CHECKPOINT=exp/sparsedrive_agent_v11/2026.03.10.10.47.11/periodic_pdm_ckpts/ep0001.ckpt
AGENT="sparsedrive_agent_v11"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_fast_navsimv1.py \
        agent=${AGENT} \
        agent.checkpoint_path=${CHECKPOINT} \
        +team_name=$TEAM_NAME \
        +authors=$AUTHORS \
        +email=$EMAIL \
        +institution=$INSTITUTION \
        +country=$COUNTRY \
        experiment_name=${AGENT} \
        train_test_split=navtest \
        +test_cache_path=${NAVSIM_EXP_ROOT}/metric_feature_cache/ \
        worker=ray_distributed_no_torch \
        worker.threads_per_node=64 \
        dataloader.params.batch_size=8 \
        +agent.config.velocity_filter_num=[64,20] \
        +agent.config.pdm_score_version="v1_new" \
        +agent.config.eval_version="v1" \
        +agent.config.select_traj_mode={"v1":"pdm_v1_ori","v2":"pdm_v2_ori","hard":"pdm_v2_ori"} \
        +agent.config.metrics=["no_at_fault_collisions","drivable_area_compliance","driving_direction_compliance","time_to_collision_within_bound","comfort","ego_progress"]