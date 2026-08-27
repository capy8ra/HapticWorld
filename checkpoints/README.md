# Checkpoints

Fetched from HuggingFace (`ShaotingPeng/HapticWorld`) by:

    python checkpoints/download_assets.py

Layout after download:

    checkpoints/wm/boxflip/{stage1,stage2,stage3}/checkpoints/last.ckpt   world model (3 stages)
    checkpoints/policies/<run>/                                                 ACT policies:
        image_only_sim_*    no-torque policy (i_sim)
        image_torque_sim_*  torque policy (it_sim)
        it_sim_wo_t_*       ablation: torque architecture, trained w/o torque loss
    data/wm_seed_live/episode_*.hdf5                                            rollout seed frames
    worldsim/.../metrics/i3d_torchscript.pt                                     FVD metric model (training only)
