import math
from collections import defaultdict
from typing import List

import einops
import numpy as np
import PIL
import safetensors.torch
import torch
import torch.nn as nn
import tqdm
from einops import repeat

from .autoencoder import AutoEncoder
from .conditioner import CLIPConditioner
from .geometry import get_default_intrinsics, get_plucker_coordinates, get_preset_pose_fov
from .sampling import DiscreteDenoiser, MultiviewCFG
from .transformer import (
    ATTENTION_LAYERS,
    Attention,
)
from .unet import Seva, SevaParams, SGMWrapper
from .utils import (
    append_dims,
    to_d,
    to_hom_pose,
    transform_img_and_K,
)

C = 4
T = 22
H = 576
W = 576
F = 8
B = 3

# H = 256
# W = 256

def load_pipeline(
    device="cuda",
    activate_layers=True,
    swapping_config=None,
    resume_from=None,
    manual_parameters=None,
):
    state_dict = safetensors.torch.load_file(
        "models/seva_model.safetensors", device=device
    )
    unet = Seva(SevaParams())
    missing, unexpected = unet.load_state_dict(state_dict, strict=False, assign=True)

    vae = AutoEncoder(chunk_size=1).to(device)
    image_encoder = CLIPConditioner().to(device)
    denoiser = DiscreteDenoiser(num_idx=1000, device=device)
    guider_mid = MultiviewCFG(3.0)
    # guider_mid = MultiviewCFG(2.0)
    # guider_mid = MultiviewCFG(1.2)
    guider = MultiviewCFG(1.2)

    pipeline = SEVAPipeline(
        unet=unet,
        vae=vae,
        image_encoder=image_encoder,
        denoiser=denoiser,
        guider=guider,
        guider_mid=guider_mid,
    )
    if activate_layers:
        assert swapping_config is not None or resume_from is not None or manual_parameters is not None, (
            "Either swapping_config or resume_from or manual_parameters must be provided when activate_layers is True."
        )
        pipeline.activate_layers(
            swapping_config=swapping_config, resume_from=resume_from, manual_parameters=manual_parameters
        )

    pipeline.to(device)
    return pipeline


class SEVAPipeline(nn.Module):
    def __init__(self, unet, vae, image_encoder, denoiser, guider, guider_mid):
        super().__init__()
        self.unet = SGMWrapper(unet)
        self.vae = vae
        self.image_encoder = image_encoder
        self.denoiser = denoiser
        self.guider = guider
        self.guider_mid = guider_mid
        self.swapping_config = None

        self.sigmas = self.denoiser.discretization(1000)
        self.sigma_counter_step = 0

    def load_vae(self, device):
        self.vae = self.vae.to(device=device)

    def load_image_encoder(self, device):
        self.image_encoder = self.image_encoder.to(device=device)

    def unload_vae(self):
        self.vae = self.vae.to("cpu")
        torch.cuda.empty_cache()

    def unload_image_encoder(self):
        self.image_encoder = self.image_encoder.to("cpu")
        torch.cuda.empty_cache()

    def _get_loaded_params_from_manual(self, manual_parameters):
        params = {}
        for attn_type in manual_parameters.keys():
            for v_id, name in enumerate(ATTENTION_LAYERS[attn_type]):
                sigma_lists = []
                for sig_val in manual_parameters[attn_type]:
                    v = sig_val[v_id]
                    sigma_lists.append(torch.tensor(v))
                params[name] = torch.stack(sigma_lists, dim=0)
                # params[name] = list(value)
        return params

    def activate_layers(self, swapping_config=None, resume_from=None, manual_parameters=None):
        if resume_from is not None:
            print(f"Loading swapping params from {resume_from}")
            loaded_params = torch.load(resume_from, weights_only=False)
            self.swapping_config = loaded_params["swapping_config"]

            if swapping_config is not None:
                print("Updating swapping config with provided config.")
                self.swapping_config.update(swapping_config)

        else:
            loaded_params = None
            if manual_parameters is not None:
                print("Activating swapping with provided manual parameters.")
                loaded_params = {"no_activation": self._get_loaded_params_from_manual(manual_parameters)}

            print("Activating swapping with provided config.")
            self.swapping_config = swapping_config

        if self.swapping_config is None:
            self.swapping_config = {}

        if "params_frames_setup" not in self.swapping_config:
            self.swapping_config["params_frames_setup"] = {
                "self_spatial": [list(range(22))],
                "self_temporal": [list(range(22))],
                "self_spatio_temporal": [[0]],
            }

        if isinstance(self.swapping_config.get("noise_separate_bins"), str):
            if self.swapping_config["noise_separate_bins"] == "sigmas_10_mid":
                sigmas = self.denoiser.discretization(10).cpu().numpy()
            elif self.swapping_config["noise_separate_bins"] == "sigmas_25_mid":
                sigmas = self.denoiser.discretization(25).cpu().numpy()
            elif self.swapping_config["noise_separate_bins"] == "sigmas_50_mid":
                sigmas = self.denoiser.discretization(50).cpu().numpy()
            sigmas_mids = (sigmas[:-1] + sigmas[1:]) / 2
            self.swapping_config["noise_separate_bins"] = list(reversed(sigmas_mids[:-1].tolist()))
            print(f"Using noise_separate_bins: {self.swapping_config['noise_separate_bins']}")

        for name, module in self.unet.named_modules():
            for attn_type, layers in ATTENTION_LAYERS.items():
                if name in layers and attn_type in self.swapping_config["params_frames_setup"].keys():
                    params = loaded_params["no_activation"][name] if loaded_params else None
                    module.activate_attention(swapping_config=self.swapping_config, params=params, attn_type=attn_type)
                    # module.to(device=self.unet.device)

    def set_swapping_active(self, enable=True):
        for name, module in self.unet.named_modules():
            if isinstance(module, Attention):
                module.swapping_active = enable

    def save_swap_params(self, path):
        swap_params = {
            "no_activation": {},
            "with_activation": {},
            "swapping_config": self.swapping_config,
        }
        for name, module in self.unet.named_modules():
            if isinstance(module, Attention) and module.layer_activated:
                swap_params["no_activation"][name] = module.get_swap_param()

        torch.save(swap_params, path)
        print(f"Saved swapping params to {path}")

    def get_swap_params(self, full=False):
        swap_params = {
            "no_activation": defaultdict(list),
            "with_activation": defaultdict(list),
        }
        modules_dict = dict(self.unet.named_modules())

        for name, layers in ATTENTION_LAYERS.items():
            for layer in layers:
                processor = modules_dict[layer]
                if isinstance(processor, Attention) and processor.layer_activated:
                    swap_params["no_activation"][name].append(
                        processor.resolve_param_frames_all(force_no_activation=True, full=full).float().detach().cpu()
                    )
                    swap_params["with_activation"][name].append(
                        processor.resolve_param_frames_all(force_no_activation=False, full=full).float().detach().cpu()
                    )

        return swap_params

    def _open_image(self, image_path, image_size=H, bg_color="white"):
        image = PIL.Image.open(image_path)
        if image.mode == "RGBA":
            r, g, b, a = image.split()
            rgb_image = PIL.Image.new("RGB", image.size, bg_color)
            rgb_image.paste(image, mask=a)
            image = rgb_image

        image = image.convert("RGB")
        image = image.resize((image_size, image_size))

        return image

    @torch.no_grad()
    def encode_latents(self, samples, chunk_size=5):
        self.vae.to("cuda")
        with torch.amp.autocast("cuda", dtype=torch.float32):
            latents = self.vae.encode(samples, chunk_size)
        self.vae.to("cpu")
        return latents

    def preprocess_videos(self, pil_videos: List[List[PIL.Image.Image]]) -> torch.Tensor:
        """B lists of N PIL images -> (B*N, C, H, W) tensor in [-1, 1]"""
        samples = []
        for video in pil_videos:
            for frame in video:
                np_img = np.array(frame, dtype=np.float32) / 255.0
                img = torch.as_tensor(np_img, dtype=torch.float32).permute(2, 0, 1)[None, ...]
                img = img * 2.0 - 1.0

                shorter = H
                shorter = round(shorter / 64) * 64
                img, _ = transform_img_and_K(img, shorter, K=None, size_stride=64)
                samples.append(img)

        samples = torch.cat(samples, dim=0)
        return samples

    def encode_pil_to_latents(self, pil_videos, chunk_size=5, device="cuda"):
        samples = self.preprocess_videos(pil_videos).to(device=device)
        latents = self.encode_latents(samples, chunk_size)
        return latents

    @torch.no_grad()
    def decode_latents(self, latents, chunk_size=5):
        with torch.amp.autocast("cuda", dtype=torch.float32):
            self.vae.to("cuda")
            samples = self.vae.decode(latents, chunk_size)
            self.vae.to("cpu")

        return samples

    def postprocess_videos(self, samples, reduce_first_frame=False, batch_size=B):
        """(B*N, C, H, W) tensor in [-1, 1] -> B lists of N PIL images"""
        samples = (samples.permute(0, 2, 3, 1) + 1) / 2.0
        samples = (samples * 255).clamp(0, 255).to(torch.uint8)

        samples = einops.rearrange(samples, "(b n) c h w -> b n c h w", b=batch_size)

        if reduce_first_frame:
            samples = samples[:, 1:]

        pil_videos = []
        for vid in samples:
            pil_vid = [PIL.Image.fromarray(frame.cpu().numpy()) for frame in vid]
            pil_videos.append(pil_vid)
        return pil_videos

    def decode_latents_to_pil(self, latents, reduce_first_frame=False, chunk_size=5, batch_size=B):
        samples = self.decode_latents(latents, chunk_size)
        pil_videos = self.postprocess_videos(samples, reduce_first_frame=reduce_first_frame, batch_size=batch_size)
        return pil_videos

    def prepare_sampling_loop(self, num_steps, device):
        num_samples = [1, T]
        shape = (math.prod(num_samples), C, H // F, W // F)
        x = torch.randn(shape).to(device=device)
        x = repeat(x, "n ... -> (b n) ...", b=B)
        sigmas = self.denoiser.discretization(num_steps, device=device)
        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas

    def prepare_inputs(self, x: torch.Tensor, s: torch.Tensor, c: dict, uc: dict):
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat", "replace", "dense_vector"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out

    def get_combined_conditioning(self, object_conditioning, reference_conditioning, main_stream, device):
        combined_cond = dict()
        combined_uncond = dict()

        keys_the_same_uncond = ["crossattn", "replace", "concat", "dense_vector"]
        keys_the_same_cond = ["concat", "dense_vector"]

        zeroing = ["crossattn", "replace"]

        main_cond_keys = ["crossattn", "replace"]  # otherwise main =  uncond

        for key in object_conditioning["cond"]:
            if key in keys_the_same_uncond:
                if key in zeroing:
                    obj_uncond = torch.zeros_like(object_conditioning["uncond"][key])
                    ref_uncond = torch.zeros_like(object_conditioning["uncond"][key])
                    main_uncond = torch.zeros_like(object_conditioning["uncond"][key])
                else:
                    obj_uncond = object_conditioning["uncond"][key]
                    ref_uncond = obj_uncond.clone()
                    main_uncond = obj_uncond.clone()
            else:
                if key in zeroing:
                    obj_uncond = torch.zeros_like(object_conditioning["uncond"][key])
                    ref_uncond = torch.zeros_like(object_conditioning["uncond"][key])
                    main_uncond = torch.zeros_like(object_conditioning["uncond"][key])
                else:
                    obj_uncond = object_conditioning["uncond"][key]
                    main_uncond = object_conditioning["uncond"][key]
                    ref_uncond = reference_conditioning["uncond"][key]

            if key in keys_the_same_cond:
                obj_cond = object_conditioning["cond"][key]
                ref_cond = reference_conditioning["cond"][key]
                main_cond = obj_cond.clone()
            else:
                obj_cond = object_conditioning["cond"][key]
                if key in zeroing:
                    main_cond = torch.zeros_like(object_conditioning["cond"][key])
                else:
                    main_cond = object_conditioning["uncond"][key]
                ref_cond = reference_conditioning["cond"][key]

            if key in main_cond_keys:
                if main_stream == "object":
                    main_cond = obj_cond.clone()
                elif main_stream == "reference":
                    main_cond = ref_cond.clone()
                elif main_stream == "mixed":
                    if key == "crossattn":
                        main_cond = ref_cond.clone()
                    elif key == "replace":
                        main_cond = obj_cond.clone()
                elif main_stream == "mixed2":
                    if key == "replace":
                        main_cond = obj_cond.clone()

                elif main_stream == "combined" and key == "crossattn":
                    main_cond = (ref_cond + obj_cond) / 2.0
                elif main_stream == "combined_object":
                    if key == "crossattn":
                        main_cond = (ref_cond + obj_cond) / 2.0
                    elif key == "replace":
                        main_cond = obj_cond.clone()

            combined_cond[key] = torch.cat((ref_cond, main_cond, obj_cond), 0).to(
                device=device,
            )
            combined_uncond[key] = torch.cat((ref_uncond, main_uncond, obj_uncond), 0).to(
                device=device,
            )

        c2w = repeat(object_conditioning["c2w"], "n ... -> (b n) ...", b=B).to(
            device=device,
        )
        k = repeat(object_conditioning["curr_Ks"], "n ... -> (b n) ...", b=B).to(
            device=device,
        )
        input_mask = repeat(object_conditioning["input_masks"], "n -> (b n)", b=B).to(device=device, dtype=torch.bool)

        if main_stream == "uncond" or main_stream == "combined":
            input_mask[T] = False

        return combined_cond, combined_uncond, c2w, k, input_mask

    @torch.no_grad()
    def encode_conditioning(
        self,
        image_or_path,
        camera_scale=2.0,
        camera_info_path=None,
        back_reference_image_path=None,
        elevation_deg=None,
        distance=None,
        fov=0.9424777960769379,  # 54 degrees by default
        device="cuda",
        bg_color="white",
        grayscale=False,
        add_front=True,
        add_back=False,
        camera_trajectory="orbit",
        zoom_factor=None,
        orbit_axis=None
    ):
        self.vae.to(device)
        self.image_encoder.to(device)

        if isinstance(image_or_path, str):
            image = self._open_image(image_or_path, H, bg_color)
        else:
            image = image_or_path

        if grayscale:
            image = image.convert("L").convert("RGB")

        np_img = np.array(image, dtype=np.float32) / 255.0
        input_img = torch.as_tensor(np_img, device=device)[None, ...]

        shorter = H
        shorter = round(shorter / 64) * 64
        input_img = transform_img_and_K(
            input_img.permute(0, 3, 1, 2),
            shorter,
            K=None,
            size_stride=64,
        )[0]

        if back_reference_image_path is not None:
            back_reference_image = self._open_image(back_reference_image_path, H, bg_color)
            np_img = np.array(back_reference_image, dtype=np.float32) / 255.0
            back_reference_image = torch.as_tensor(np_img, device=device)[None, ...]
            back_reference_image = transform_img_and_K(
                back_reference_image.permute(0, 3, 1, 2),
                shorter,
                K=None,
                size_stride=64,
            )[0]

        else:
            back_reference_image = None

        if camera_info_path is not None:
            camera_config = np.load(camera_info_path, allow_pickle=True).item()
            fov = camera_config["fov"]
            distance = camera_config["distance"]
            elevation_deg = camera_config["elevation"]
        else:
            assert elevation_deg is not None and distance is not None and fov is not None, (
                "Either camera_info_path or all of elevation_deg, distance and fov must be provided."
            )

        height = distance * math.sin(math.radians(elevation_deg))
        horizontal_distance = distance * math.cos(math.radians(elevation_deg))

        look_at = torch.tensor([0, 0, horizontal_distance])

        start_c2ws = torch.eye(4)
        start_c2ws[1, 3] = -height
        start_w2c = torch.linalg.inv(start_c2ws)

        up_direction = torch.tensor([0.0, -1.0, 0.0])

        all_c2ws, all_fovs = get_preset_pose_fov(
            option=camera_trajectory,
            num_frames=21,
            start_w2c=start_w2c,
            look_at=look_at,
            up_direction=up_direction,
            fov=fov,
            zoom_factor=zoom_factor,
            orbit_axis=orbit_axis,
        )

        all_c2ws = torch.as_tensor(all_c2ws, device=device)
        all_fovs = torch.as_tensor(all_fovs, device=device)

        if add_front:
            all_c2ws = torch.cat([all_c2ws[0:1], all_c2ws])
            all_fovs = torch.cat([all_fovs[0:1], all_fovs])
            t = T
        else:
            t = T - 1

        all_Ks = get_default_intrinsics(all_fovs, aspect_ratio=W / H).to(
            device=device,
        )

        all_imgs = input_img.new_zeros(t, *input_img.shape[1:])
        all_imgs[0] = input_img[0]

        input_masks = torch.zeros(t, dtype=torch.bool, device=device)
        input_masks[0] = True
        if back_reference_image is not None:
            all_imgs[10] = back_reference_image[0]
            input_masks[10] = True

        all_imgs = all_imgs * 2.0 - 1.0

        c2w = to_hom_pose(all_c2ws.float())
        w2c = torch.linalg.inv(c2w)

        ref_c2ws = all_c2ws
        camera_dist_2med = torch.norm(
            ref_c2ws[:, :3, 3] - ref_c2ws[:, :3, 3].median(0, keepdim=True).values,
            dim=-1,
        )
        valid_mask = camera_dist_2med <= torch.clamp(
            torch.quantile(camera_dist_2med, 0.97) * 10,
            max=1e6,
        )
        c2w[:, :3, 3] -= ref_c2ws[valid_mask, :3, 3].mean(0, keepdim=True)
        w2c = torch.linalg.inv(c2w)

        camera_dists = (
            c2w[:, :3, 3]
            .clone()
            .to(
                device=device,
            )
        )
        translation_scaling_factor = (
            camera_scale
            if torch.isclose(
                torch.norm(camera_dists[0]),
                torch.zeros(
                    1,
                    device=device,
                ),
                atol=1e-5,
            ).any()
            else (camera_scale / torch.norm(camera_dists[0]))
        )
        w2c[:, :3, 3] *= translation_scaling_factor
        c2w[:, :3, 3] *= translation_scaling_factor

        pluckers = get_plucker_coordinates(
            extrinsics_src=w2c[0],
            extrinsics=w2c,
            intrinsics=all_Ks.float().clone(),
            target_size=(H // F, W // F),
            fov_rad=fov,
        ).to(
            device=device,
        )

        latents = torch.nn.functional.pad(self.vae.encode(all_imgs[input_masks], 1), (0, 0, 0, 0, 0, 1), value=1.0)

        c_crossattn = repeat(self.image_encoder(all_imgs[input_masks]).mean(0), "d -> n 1 d", n=t).clone()
        uc_crossattn = torch.zeros_like(
            c_crossattn,
            device=device,
        )
        c_replace = latents.new_zeros(t, *latents.shape[1:])
        c_replace[input_masks] = latents
        uc_replace = torch.zeros_like(
            c_replace,
            device=device,
        )
        c_concat = torch.cat(
            [repeat(input_masks, "n -> n 1 h w", h=pluckers.shape[2], w=pluckers.shape[3]), pluckers], 1
        ).clone()
        uc_concat = torch.cat([pluckers.new_zeros(t, 1, *pluckers.shape[-2:]), pluckers.clone()], 1)
        c_dense_vec = pluckers.clone()
        uc_dense_vec = c_dense_vec

        cond = {"crossattn": c_crossattn, "replace": c_replace, "concat": c_concat, "dense_vector": c_dense_vec}
        uncond = {"crossattn": uc_crossattn, "replace": uc_replace, "concat": uc_concat, "dense_vector": uc_dense_vec}

        self.vae.to("cpu")
        self.image_encoder.to("cpu")
        return {"cond": cond, "uncond": uncond, "c2w": c2w, "curr_Ks": all_Ks, "input_masks": input_masks}

    @torch.no_grad()
    def infer_demo(
        self,
        input_image_path,
        reference_image_path,
        elevation,
        distance=2.0,
        fov=0.7,
        camera_trajectory="orbit",
        bg_color="white",
        zoom_factor=None,
        num_inference_steps=50,
        cfg_scale=3.0,
        cfg_scale_mid=3.0,
        main_stream="combined",
        orbit_axis=None,
    ):
        self.guider_mid = MultiviewCFG(cfg_scale)

        object_conditioning = self.encode_conditioning(
            input_image_path,
            elevation_deg=elevation,
            distance=distance,
            fov=fov,
            camera_trajectory=camera_trajectory,
            bg_color=bg_color,
            zoom_factor=zoom_factor,
            orbit_axis=orbit_axis,
        )

        reference_conditioning = self.encode_conditioning(
            reference_image_path,
            elevation_deg=elevation,
            distance=distance,
            fov=fov,
            camera_trajectory=camera_trajectory,
            bg_color=bg_color,
            zoom_factor=zoom_factor,
            orbit_axis=orbit_axis,
        )

        cond, uncond, c2w, k, input_mask = self.get_combined_conditioning(
            object_conditioning, reference_conditioning, main_stream, device=self.unet.device
        )

        x, s_in, sigmas, num_sigmas = self.prepare_sampling_loop(num_inference_steps, self.unet.device)

        for i in tqdm.tqdm(range(num_sigmas - 1), desc="Sampling", total=num_sigmas - 1):
            gamma = 0.0
            sigma = s_in * sigmas[i]
            next_sigma = s_in * sigmas[i + 1]
            sigma_hat = sigma * (gamma + 1.0) + 1e-6
            eps = torch.randn_like(x[T : 2 * T], device=self.unet.device)
            eps = repeat(eps, "n ... -> (b n) ...", b=B)
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

            denoised_cond = self.denoiser(self.unet, x, sigma_hat, cond, num_frames=T, batch_size=B)
            denoised_uncond = self.denoiser(self.unet, x, sigma_hat, uncond, num_frames=T, batch_size=B)
            denoised = torch.cat((denoised_uncond, denoised_cond), 0)

            denoised_mid = self.guider_mid(
                denoised, sigma_hat, cfg_scale_mid, c2w=c2w, K=k, input_frame_mask=input_mask
            )

            denoised = self.guider(denoised, sigma_hat, 2.0, c2w=c2w, K=k, input_frame_mask=input_mask)

            denoised[T : 2 * T] = denoised_mid[T : 2 * T]

            d = to_d(x, sigma_hat, denoised)
            dt = append_dims(next_sigma - sigma_hat, x.ndim)

            x = x + dt * d

        samples = self.decode_latents_to_pil(x)

        return samples
