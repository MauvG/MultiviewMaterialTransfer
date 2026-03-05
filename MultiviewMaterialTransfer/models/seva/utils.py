import math
import os

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def create_c2w2(distance, elevation, azimuth, look_at=torch.tensor([0.0, 0.0, 0.0])):
    """Creates a 4x4 C2W matrix from spherical coordinates."""

    elevation_rad = np.deg2rad(elevation)
    azimuth_rad = np.deg2rad(azimuth)

    x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    y = -distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    z = distance * np.sin(elevation_rad)

    # --- FIX 1: Ensure camera_position is float32 ---
    camera_position = torch.tensor([x, y, z], dtype=torch.float32) + look_at.to(torch.float32)

    forward_vec = F.normalize(look_at - camera_position, dim=-1)

    # --- FIX 2: Ensure up_vec is float32 ---
    up_vec = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    right_vec = F.normalize(torch.cross(forward_vec, up_vec, dim=-1), dim=-1)
    up_vec_new = F.normalize(torch.cross(right_vec, forward_vec, dim=-1), dim=-1)

    c2w = torch.eye(4, dtype=torch.float32)  # Also good practice to set dtype here
    c2w[:3, 0] = right_vec
    c2w[:3, 1] = up_vec_new
    c2w[:3, 2] = forward_vec
    c2w[:3, 3] = camera_position

    return c2w


def get_elevation_from_c2w(c2w):
    # np to torch
    c2w = torch.as_tensor(c2w, dtype=torch.float32)
    # forward vector (camera looks along -Z in camera space → column 2 of c2w)
    forward = -c2w[:3, 2]

    world_up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device=c2w.device)
    horiz = torch.tensor([forward[0], 0.0, forward[2]], dtype=torch.float32, device=c2w.device)

    # angle between forward and horizontal projection
    elevation = math.degrees(torch.atan2(forward.dot(world_up), horiz.norm()))
    return elevation


def debug_c2w(c2w):
    for x in c2w:
        print(f" location {x[0, 3]:.3f} {x[1, 3]:.3f} {x[2, 3]:.3f} elevation {get_elevation_from_c2w(x):.3f}")


def elevation_matrix(theta_deg: float):
    theta = torch.deg2rad(torch.tensor(theta_deg))
    R = torch.eye(4)
    R[1, 1] = torch.cos(theta)
    R[1, 2] = -torch.sin(theta)
    R[2, 1] = torch.sin(theta)
    R[2, 2] = torch.cos(theta)
    return R


def create_c2w(distance: float, elevation_deg: float, azimuth_deg: float = 0.0):
    theta = torch.deg2rad(torch.tensor(elevation_deg))  # elevation
    phi = torch.deg2rad(torch.tensor(azimuth_deg))  # azimuth

    # Camera position in world space
    cam_x = distance * torch.cos(theta) * torch.cos(phi)
    cam_y = distance * torch.sin(theta)  # height
    cam_z = distance * torch.cos(theta) * torch.sin(phi)
    cam_pos = torch.tensor([cam_x, cam_y, cam_z])

    # Build a look-at matrix (camera -> world)
    forward = torch.nn.functional.normalize(-cam_pos, dim=0)  # towards origin
    up = torch.tensor([0.0, 1.0, 0.0])
    right = torch.nn.functional.normalize(torch.cross(up, forward), dim=0)
    up = torch.cross(forward, right)

    c2w = torch.eye(4)
    c2w[:3, :3] = torch.stack([right, up, forward], dim=1)
    c2w[:3, 3] = cam_pos
    return c2w


def to_d(x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    return (x - denoised) / append_dims(sigma, x.ndim)


def to_hom_pose(pose):
    # get homogeneous coordinates of the input pose
    if pose.shape[-2:] == (3, 4):
        pose_hom = torch.eye(4, device=pose.device)[None].repeat(pose.shape[0], 1, 1)
        pose_hom[:, :3, :] = pose
        return pose_hom
    return pose


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def decode_output(
    samples,
    T,
    indices=None,
):
    # decode model output into dict if it is not
    if isinstance(samples, dict):
        # model with postprocessor and outputs dict
        for sample, value in samples.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            elif isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            else:
                value = torch.tensor(value)

            if indices is not None and value.shape[0] == T:
                value = value[indices]
            samples[sample] = value
    else:
        # model without postprocessor and outputs tensor (rgb)
        samples = samples.detach().cpu()

        if indices is not None and samples.shape[0] == T:
            samples = samples[indices]
        samples = {"samples-rgb/image": samples}

    return samples


def save_output(
    samples,
    save_path,
    video_save_fps=2,
):
    os.makedirs(save_path, exist_ok=True)
    for sample in samples:
        media_type = "video"
        if "/" in sample:
            sample_, media_type = sample.split("/")
        else:
            sample_ = sample

        value = samples[sample]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        elif isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        else:
            value = torch.tensor(value)

        if media_type == "image":
            value = (value.permute(0, 2, 3, 1) + 1) / 2.0
            value = (value * 255).clamp(0, 255).to(torch.uint8)
            iio.imwrite(
                os.path.join(save_path, f"{sample_}.mp4") if sample_ else f"{save_path}.mp4",
                value,
                fps=video_save_fps,
                macro_block_size=1,
                ffmpeg_log_level="error",
            )
            os.makedirs(os.path.join(save_path, sample_), exist_ok=True)
            for i, s in enumerate(value):
                iio.imwrite(
                    os.path.join(save_path, sample_, f"{i:03d}.png"),
                    s,
                )
        elif media_type == "video":
            value = (value.permute(0, 2, 3, 1) + 1) / 2.0
            value = (value * 255).clamp(0, 255).to(torch.uint8)
            iio.imwrite(
                os.path.join(save_path, f"{sample_}.mp4"),
                value,
                fps=video_save_fps,
                macro_block_size=1,
                ffmpeg_log_level="error",
            )
        elif media_type == "raw":
            torch.save(
                value,
                os.path.join(save_path, f"{sample_}.pt"),
            )
        else:
            pass


def assemble(
    input,
    test,
    input_maps,
    test_maps,
):
    T = len(input_maps)
    assembled = torch.zeros_like(test[-1:]).repeat_interleave(T, dim=0)
    assembled[input_maps != -1] = input[input_maps[input_maps != -1]]
    assembled[test_maps != -1] = test[test_maps[test_maps != -1]]
    assert np.logical_xor(input_maps != -1, test_maps != -1).all()
    return assembled


def load_img_and_K(
    image_path_or_size,
    size,
    scale=1.0,
    center=(0.5, 0.5),
    K=None,
    size_stride=1,
    center_crop=False,
    image_as_tensor=True,
    context_rgb=None,
    device="cuda",
):
    if isinstance(image_path_or_size, torch.Size):
        image = Image.new("RGBA", image_path_or_size[::-1])
    else:
        image = Image.open(image_path_or_size).convert("RGBA")

    w, h = image.size
    if size is None:
        size = (w, h)

    image = np.array(image).astype(np.float32) / 255
    if image.shape[-1] == 4:
        rgb, alpha = image[:, :, :3], image[:, :, 3:]
        if context_rgb is not None:
            image = rgb * alpha + context_rgb * (1 - alpha)
        else:
            image = rgb * alpha + (1 - alpha)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).to(dtype=torch.float32)
    image = image.unsqueeze(0)

    if isinstance(size, (tuple, list)):
        # => if size is a tuple or list, we first rescale to fully cover the `size`
        # area and then crop the `size` area from the rescale image
        W, H = size
    else:
        # => if size is int, we rescale the image to fit the shortest side to size
        # => if size is None, no rescaling is applied
        W, H = get_wh_with_fixed_shortest_side(w, h, size)
    W, H = (
        math.floor(W / size_stride + 0.5) * size_stride,
        math.floor(H / size_stride + 0.5) * size_stride,
    )

    rfs = get_resizing_factor((math.floor(H * scale), math.floor(W * scale)), (h, w))
    resize_size = rh, rw = [int(np.ceil(rfs * s)) for s in (h, w)]
    image = torch.nn.functional.interpolate(image, resize_size, mode="area", antialias=False)
    if scale < 1.0:
        pw = math.ceil((W - resize_size[1]) * 0.5)
        ph = math.ceil((H - resize_size[0]) * 0.5)
        image = F.pad(image, (pw, pw, ph, ph), "constant", 1.0)

    cy_center = int(center[1] * image.shape[-2])
    cx_center = int(center[0] * image.shape[-1])
    if center_crop:
        side = min(H, W)
        ct = max(0, cy_center - side // 2)
        cl = max(0, cx_center - side // 2)
        ct = min(ct, image.shape[-2] - side)
        cl = min(cl, image.shape[-1] - side)
        image = TF.crop(image, top=ct, left=cl, height=side, width=side)
    else:
        ct = max(0, cy_center - H // 2)
        cl = max(0, cx_center - W // 2)
        ct = min(ct, image.shape[-2] - H)
        cl = min(cl, image.shape[-1] - W)
        image = TF.crop(image, top=ct, left=cl, height=H, width=W)

    if K is not None:
        K = K.clone()
        if torch.all(K[:2, -1] >= 0) and torch.all(K[:2, -1] <= 1):
            K[:2] *= K.new_tensor([rw, rh])[:, None]  # normalized K
        else:
            K[:2] *= K.new_tensor([rw / w, rh / h])[:, None]  # unnormalized K
        K[:2, 2] -= K.new_tensor([cl, ct])

    if image_as_tensor:
        # tensor of shape (1, 3, H, W) with values ranging from (-1, 1)
        image = image.to(device) * 2.0 - 1.0
    else:
        # PIL Image with values ranging from (0, 255)
        image = image.permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).astype(np.uint8))
    return image, K


def get_resizing_factor(
    target_shape,  # H, W
    current_shape,  # H, W
    cover_target: bool = True,
    # If True, the output shape will fully cover the target shape.
    # If No, the target shape will fully cover the output shape.
) -> float:
    r_bound = target_shape[1] / target_shape[0]
    aspect_r = current_shape[1] / current_shape[0]
    if r_bound >= 1.0:
        if cover_target:
            if aspect_r >= r_bound:
                factor = min(target_shape) / min(current_shape)
            elif aspect_r < 1.0:
                factor = max(target_shape) / min(current_shape)
            else:
                factor = max(target_shape) / max(current_shape)
        else:
            if aspect_r >= r_bound:
                factor = max(target_shape) / max(current_shape)
            elif aspect_r < 1.0:
                factor = min(target_shape) / max(current_shape)
            else:
                factor = min(target_shape) / min(current_shape)
    else:
        if cover_target:
            if aspect_r <= r_bound:
                factor = min(target_shape) / min(current_shape)
            elif aspect_r > 1.0:
                factor = max(target_shape) / min(current_shape)
            else:
                factor = max(target_shape) / max(current_shape)
        else:
            if aspect_r <= r_bound:
                factor = max(target_shape) / max(current_shape)
            elif aspect_r > 1.0:
                factor = min(target_shape) / max(current_shape)
            else:
                factor = min(target_shape) / min(current_shape)
    return factor


def get_wh_with_fixed_shortest_side(w, h, size):
    # size is smaller or equal to zero, we return original w h
    if size is None or size <= 0:
        return w, h
    if w < h:
        new_w = size
        new_h = int(size * h / w)
    else:
        new_h = size
        new_w = int(size * w / h)
    return new_w, new_h


def transform_img_and_K(
    image,
    size,
    scale=1.0,
    center=(0.5, 0.5),
    K=None,
    size_stride=1,
    mode="crop",
):
    assert mode in [
        "crop",
        "pad",
        "stretch",
    ], f"mode should be one of ['crop', 'pad', 'stretch'], got {mode}"

    h, w = image.shape[-2:]
    if isinstance(size, (tuple, list)):
        # => if size is a tuple or list, we first rescale to fully cover the `size`
        # area and then crop the `size` area from the rescale image
        W, H = size
    else:
        # => if size is int, we rescale the image to fit the shortest side to size
        # => if size is None, no rescaling is applied
        W, H = get_wh_with_fixed_shortest_side(w, h, size)
    W, H = (
        math.floor(W / size_stride + 0.5) * size_stride,
        math.floor(H / size_stride + 0.5) * size_stride,
    )

    if mode == "stretch":
        rh, rw = H, W
    else:
        rfs = get_resizing_factor(
            (H, W),
            (h, w),
            cover_target=mode != "pad",
        )
        (rh, rw) = [int(np.ceil(rfs * s)) for s in (h, w)]

    rh, rw = int(rh / scale), int(rw / scale)
    image = torch.nn.functional.interpolate(image, (rh, rw), mode="area", antialias=False)

    cy_center = int(center[1] * image.shape[-2])
    cx_center = int(center[0] * image.shape[-1])
    if mode != "pad":
        ct = max(0, cy_center - H // 2)
        cl = max(0, cx_center - W // 2)
        ct = min(ct, image.shape[-2] - H)
        cl = min(cl, image.shape[-1] - W)
        image = TF.crop(image, top=ct, left=cl, height=H, width=W)
        pl, pt = 0, 0
    else:
        pt = max(0, H // 2 - cy_center)
        pl = max(0, W // 2 - cx_center)
        pb = max(0, H - pt - image.shape[-2])
        pr = max(0, W - pl - image.shape[-1])
        image = TF.pad(
            image,
            [pl, pt, pr, pb],
        )
        cl, ct = 0, 0

    if K is not None:
        K = K.clone()
        # K[:, :2, 2] += K.new_tensor([pl, pt])
        if torch.all(K[:, :2, -1] >= 0) and torch.all(K[:, :2, -1] <= 1):
            K[:, :2] *= K.new_tensor([rw, rh])[None, :, None]  # normalized K
        else:
            K[:, :2] *= K.new_tensor([rw / w, rh / h])[None, :, None]  # unnormalized K
        K[:, :2, 2] += K.new_tensor([pl - cl, pt - ct])

    return image, K


def infer_prior_stats(
    T,
    num_input_frames,
    num_total_frames,
):
    t = T

    chunk_strategy = "nearest"
    T_first_pass = T[0] if isinstance(T, (list, tuple)) else T
    T_second_pass = T[1] if isinstance(T, (list, tuple)) else T
    # get traj_prior_c2ws for 2-pass sampling
    if chunk_strategy.startswith("interp"):
        # Start and end have alreay taken up two slots
        # +1 means we need X + 1 prior frames to bound X times forwards for all test frames

        # Tuning up `num_prior_frames_ratio` is helpful when you observe sudden jump in the
        # generated frames due to insufficient prior frames. This option is effective for
        # complicated trajectory and when `interp` strategy is used (usually semi-dense-view
        # regime). Recommended range is [1.0 (default), 1.5].
        if num_input_frames >= 9:
            num_prior_frames = math.ceil(num_total_frames / (T_second_pass - 2) * 1.0) + 1

            if num_prior_frames + num_input_frames < T_first_pass:
                num_prior_frames = T_first_pass - num_input_frames

            num_prior_frames = max(
                num_prior_frames,
                0,
            )

            T_first_pass = num_prior_frames + num_input_frames

            if "gt" in chunk_strategy:
                T_second_pass = T_second_pass + num_input_frames

            # Dynamically update context window length.
            t = [T_first_pass, T_second_pass]

        else:
            num_prior_frames = (
                math.ceil(
                    num_total_frames / (T_second_pass - 2 - (num_input_frames if "gt" in chunk_strategy else 0)) * 1.0
                )
                + 1
            )

            if num_prior_frames + num_input_frames < T_first_pass:
                num_prior_frames = T_first_pass - num_input_frames

            num_prior_frames = max(
                num_prior_frames,
                0,
            )
    else:
        num_prior_frames = max(
            T_first_pass - num_input_frames,
            0,
        )

        if num_input_frames >= 9:
            T_first_pass = num_prior_frames + num_input_frames

            # Dynamically update context window length.
            t = [T_first_pass, T_second_pass]

    return num_prior_frames, t


def generate_orbital_path(num_frames: int, elevation_deg: float = 10.0, distance: float = 2.0) -> torch.Tensor:
    """
    Generates a series of camera-to-world (c2w) matrices for a circular orbit.
    This creates a path similar to your Blender rendering script.

    Args:
        num_frames: The number of camera poses to generate for the orbit.
        elevation_deg: The elevation angle of the camera in degrees.
        distance: The distance of the camera from the origin (0,0,0).

    Returns:
        A tensor of c2w matrices of shape [num_frames, 4, 4].
    """
    c2ws = []
    elevation_rad = np.deg2rad(elevation_deg)

    # We will assume a standard Y-up world coordinate system.
    origin = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    for i in range(num_frames):
        azimuth_deg = i * (360.0 / num_frames)
        azimuth_rad = np.deg2rad(azimuth_deg)

        # Calculate camera position using spherical coordinates in a Y-up system
        x = distance * np.cos(azimuth_rad) * np.cos(elevation_rad)
        y = distance * np.sin(elevation_rad)
        z = distance * np.sin(azimuth_rad) * np.cos(elevation_rad)
        location = torch.tensor([x, y, z], dtype=torch.float32)

        # The camera's -Z axis should point towards the origin
        look_dir = location - origin

        # Define the camera's coordinate system axes in world space
        camera_z = look_dir / torch.linalg.norm(look_dir)
        camera_x = torch.cross(world_up, camera_z, dim=0)
        camera_x = camera_x / torch.linalg.norm(camera_x)
        camera_y = torch.cross(camera_z, camera_x, dim=0)

        # Create the 4x4 camera-to-world matrix
        c2w = torch.eye(4)
        c2w[:3, 0] = camera_x
        c2w[:3, 1] = camera_y
        c2w[:3, 2] = camera_z
        c2w[:3, 3] = location
        c2ws.append(c2w)

    return torch.stack(c2ws)
