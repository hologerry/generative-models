#!/bin/sh
{
python scripts/sampling/simple_video_sdedit.py \
    --input_path /data/Dynamics/gaussian_fluid_dynamics_log/gaussian_pbd_fluid_scalar_sim_rec_simple/gpf_basic_siminrec_simple_gascsprat_velnn_velgascsprat_l2gascs_lcurdist01_lexyz01_dur90_fut90_sol5_p01d5_emitfirstymoremore_decayp030/training_render_for_generative_svd/ \
    --output_folder /data/Dynamics/sgm_outputs/simple_video_sdedit_svd_xt_1_1/ \
    --experiment_name dur90_fut90 \
    --view_idx 4 \
    --num_frames 21 \
    --num_steps 30 \
    --img2img_strength 0.4 \
    --sigma_idx 0 \
    --motion_bucket_id 15 \
    --fps_id 6 \
    --extra_str _sigmax700 \
    --version svd_xt_1_1
exit
}


# --input_path /data/Dynamics/gaussian_fluid_dynamics_log/gaussian_pbd_fluid_scalar_sim_rec_simple/gpf_basic_siminrec_simple_gascsprat_velnn_velgascsprat_l2gascs_lcurdist01_lexyz01_dur90_fut90_sol5_p01d5_emitfirstymoremore_decayp030/training_render_for_generative_svd/ \

# --input_path /data/Dynamics/gaussian_fluid_dynamics_log/gaussian_pbd_fluid_scalar_simrec_level_two_future/emitrhid1d61vis2d1_color_scales_opacity_rotation_inherit_lcol10_lsca0_lopa8_lrot01_scalingregthr4l1_fut40_sol5_p01d5_emitfirst2ymoremore_decayp030/training_render_for_generative_svd/ \
