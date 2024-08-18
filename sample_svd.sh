#!/bin/sh
{
python scripts/sampling/simple_video_sample.py \
    --input_path /data/Dynamics/gaussian_fluid_dynamics_log/gaussian_pbd_fluid_scalar_sim_rec_simple/gpf_basic_siminrec_simple_gascsprat_velnn_velgascsprat_l2gascs_lcurdist01_lexyz01_dur90_fut90_sol5_p01d5_emitfirstymoremore_decayp030/training_render_for_generative_svd/render_frame090_train00_last.png \
    --output_folder /data/Dynamics/sgm_outputs/simple_video_sample_svd_xt_1_1/ \
    --num_frames 25 \
    --seed 23 \
    --motion_bucket_id 127 \
    --fps_id 6 \
    --version svd_xt_1_1
exit
}

# --extra_str leveltwosmooth \

# --input_path /data/Dynamics/gaussian_fluid_dynamics_log/gaussian_pbd_fluid_scalar_sim_rec_simple/gpf_basic_siminrec_simple_gascsprat_velnn_velgascsprat_l2gascs_lcurdist01_lexyz01_dur90_fut90_sol5_p01d5_emitfirstymoremore_decayp030/training_render_for_generative_svd/render_frame090_train00_last.png \

# --input_path /data/Dynamics/gaussian_fluid_dynamics_log/gaussian_pbd_fluid_scalar_simrec_level_two_future/emitrhid1d61vis2d1_color_scales_opacity_rotation_inherit_lcol10_lsca0_lopa8_lrot01_scalingregthr4l1_fut40_sol5_p01d5_emitfirst2ymoremore_decayp030/training_render_for_generative_svd/render_frame120_train02_last.png \

# --input_path /data/Dynamics/gaussian_fluid_dynamics_log/gaussian_pbd_fluid_scalar_simrec_level_two_future/emitrhid1d61vis2d1_color_scales_opacity_rotation_inherit_lcol10_lsca0_lopa8_lrot01_scalingregthr4l1_fut40_sol5_p01d5_emitfirst2ymoremore_decayp030_leveltwosmooth/training_render_for_generative_svd/render_frame120_train04_last.png \
