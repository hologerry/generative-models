import os
import sys

from typing import Optional

import cv2
import imageio
import lovely_tensors as lt
import numpy as np
import torch
import torchvision.transforms.functional as TF

from einops import rearrange, repeat
from fire import Fire
from PIL import Image


sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))

from scripts.demo.svd_helpers import (
    get_batch,
    get_unique_embedder_keys_from_conditioner,
    load_model,
)
from sgm.util import append_dims, default
from scripts.demo.discretization import Img2ImgDiscretizationWrapper

def load_frames(frame_dir, start_frame_idx=90, num_frames=90, view_idx=0, fps=30):
    frames = []
    frame_step = 30 // fps
    for i in range(start_frame_idx, start_frame_idx + num_frames * frame_step, frame_step):
        frame_path = os.path.join(frame_dir, f"render_frame{i:03d}_train{view_idx:02d}_last.png")
        assert os.path.exists(frame_path), f"Frame {frame_path} does not exist."
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = TF.to_tensor(frame)
        frame = frame * 2.0 - 1.0
        frames.append(frame)
    return frames


@torch.no_grad
@torch.autocast("cuda")
def sample(
    input_path: str,  # folder with image files
    num_frames: Optional[int] = None,  # 21 for SV3D
    num_steps: Optional[int] = None,
    version: str = "svd",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    input_aug: float = 0.0,
    img2img_strength: Optional[float] = None,
    offset_noise_level: float = 0.0,
    sigma_idx: int = 0,
    seed: int = 23,
    decoding_t: int = 7,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    view_idx: int = 0,
    start_idx: int = 90,
    device: str = "cuda",
    experiment_name: str = "experiment_name",
    output_folder: Optional[str] = None,
    verbose: Optional[bool] = True,
    extra_str: Optional[str] = None,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    if version == "svd":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "simple_video_sdedit_svd/")
        model_config = "scripts/sampling/configs/svd.yaml"
    elif version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "simple_video_sdedit_svd_xt/")
        model_config = "scripts/sampling/configs/svd_xt.yaml"
    elif version == "svd_xt_1_1":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "simple_video_sdedit_svd_xt_1_1/")
        model_config = "scripts/sampling/configs/svd_xt_1_1.yaml"

    else:
        raise ValueError(f"Version {version} does not exist.")

    print("Loading model...")
    model, _ = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
    )

    if img2img_strength is not None:
        print(f"Wrapping {model.sampler.__class__.__name__} with Img2ImgDiscretizationWrapper")
        model.sampler.discretization = Img2ImgDiscretizationWrapper(model.sampler.discretization, strength=img2img_strength)

    torch.manual_seed(seed)
    print("Model loaded.")

    print("Loading frames...")
    assert os.path.isdir(input_path), f"Input path {input_path} is not a directory."
    assert os.path.exists(input_path), f"Input path {input_path} does not exist."
    frames = load_frames(
        input_path, start_frame_idx=start_idx, num_frames=num_frames, view_idx=view_idx, fps=fps_id + 1
    )
    image = frames[0]
    input_image = TF.to_pil_image((image + 1.0) / 2.0)

    frames = torch.stack(frames, dim=0).to(device)
    image = image.unsqueeze(0).to(device)
    H, W = image.shape[2:]
    assert image.shape[1] == 3
    # F = 8
    # C = 4
    # shape = (num_frames, C, H // F, W // F)
    if (H, W) != (576, 1024):
        print(
            "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
        )
    if motion_bucket_id > 255:
        print("WARNING: High motion bucket! This may lead to suboptimal performance.")

    if fps_id < 5:
        print("WARNING: Small fps value! This may lead to suboptimal performance.")

    if fps_id > 30:
        print("WARNING: Large fps value! This may lead to suboptimal performance.")
    print("Loaded frames...")

    print("Sampling...")

    value_dict = {}
    value_dict["cond_frames_without_noise"] = image
    value_dict["motion_bucket_id"] = motion_bucket_id
    value_dict["fps_id"] = fps_id
    value_dict["cond_aug"] = cond_aug
    value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)

    batch, batch_uc = get_batch(
        get_unique_embedder_keys_from_conditioner(model.conditioner),
        value_dict,
        [1, num_frames],
        T=num_frames,
        device=device,
    )
    c, uc = model.conditioner.get_unconditional_conditioning(
        batch,
        batch_uc=batch_uc,
        force_uc_zero_embeddings=[
            "cond_frames",
            "cond_frames_without_noise",
        ],
    )

    for k in ["crossattn", "concat"]:
        uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
        uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
        c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
        c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

    ## Referenced from sgm/inference/helpers.py do_img2img
    if input_aug > 0.0:
        frames_aug = frames + input_aug * torch.randn_like(frames)
    else:
        frames_aug = frames

    frames_z = model.encode_first_stage(frames_aug)
    noise = torch.randn_like(frames_z)

    sigmas = model.sampler.discretization(model.sampler.num_steps)

    sigma = sigmas[sigma_idx].to(frames_z.device)

    if offset_noise_level > 0.0:
        noise = noise + offset_noise_level * append_dims(
            torch.randn(frames_z.shape[0], device=frames_z.device), frames_z.ndim
        )
    noised_z = frames_z + noise * append_dims(sigma, frames_z.ndim)
    # Note: hardcoded to DDPM-like scaling. need to generalize later.
    noised_z = noised_z / torch.sqrt(1.0 + sigmas[sigma_idx] ** 2.0)
    ####

    additional_model_inputs = {}
    additional_model_inputs["image_only_indicator"] = torch.zeros(2, num_frames).to(device)
    additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

    def denoiser(input, sigma, c):
        return model.denoiser(model.model, input, sigma, c, **additional_model_inputs)

    samples_z = model.sampler(denoiser, noised_z, cond=c, uc=uc)

    model.en_and_decode_n_samples_a_time = decoding_t
    samples_x = model.decode_first_stage(samples_z)

    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

    output_path = os.path.join(output_folder, experiment_name)
    os.makedirs(output_path, exist_ok=True)

    basename = f"view{view_idx}_start{start_idx:03d}_frames{num_frames}_fsteps{30//(fps_id+1)}"
    basename += f"_steps{num_steps}_fps{fps_id}_motion{motion_bucket_id}_condaug{cond_aug}"
    basename += f"_inputaug{input_aug}_offsetnoise{offset_noise_level}_seed{seed}"
    if img2img_strength is not None:
        basename += f"_img2img{img2img_strength}"
    basename += f"_sigma{sigma_idx}v{sigma:.3f}"
    if extra_str is not None:
        basename += f"{extra_str}"
    basename = basename.replace(".", "d").replace("-", "n")

    input_image_path = os.path.join(output_path, f"{basename}_input.jpg")
    input_image.save(input_image_path)

    video_path = os.path.join(output_path, f"{basename}.mp4")
    input_video_path = os.path.join(output_path, f"{basename}_input.mp4")

    vid = (rearrange(samples, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)
    frames = (frames + 1.0) / 2.0
    inp = (rearrange(frames, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)

    writer = imageio.get_writer(video_path, fps=fps_id + 1)
    input_writer = imageio.get_writer(input_video_path, fps=fps_id + 1)

    video_frame_path = os.path.join(output_path, f"{basename}_frames")
    os.makedirs(video_frame_path, exist_ok=True)
    for i in range(num_frames):
        imageio.imwrite(
            os.path.join(video_frame_path, f"{basename}_output_{i:02d}.jpg"),
            vid[i],
        )
        imageio.imwrite(
            os.path.join(video_frame_path, f"{basename}_input_{i:02d}.jpg"),
            inp[i],
        )
        writer.append_data(vid[i])
        input_writer.append_data(inp[i])
    writer.close()
    input_writer.close()

    print(f"Finisheds SDEDit sampling: {basename}")


if __name__ == "__main__":
    lt.monkey_patch()
    Fire(sample)
