import bisect

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        self.net = nn.Sequential(GEGLU(dim, inner_dim), nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwappingAttentionProcessor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_activated = False
        self.swapping_active = False
        self.num_streams = 3
        self.num_frames = 22

    def activate_attention(self, swapping_config, attn_type, params=None):
        self.attn_type = attn_type
        self.softmax_temperature = swapping_config.get("softmax_temperature", 1.0)
        self.layer_activated = True
        self.swapping_active = True
        self.param_initialization = swapping_config.get("param_initialization", "zeros")
        self.frames_setup = swapping_config["params_frames_setup"][attn_type]
        self.noise_separate_bins = swapping_config.get("noise_separate_bins", [])
        self.swapping_type = swapping_config.get("swapping_control_type", "param")
        self.activation_type = swapping_config.get("swapping_activation_type", "softmax")

        if self.swapping_type == "param":
            param_shape = (len(self.noise_separate_bins) + 1, self.num_streams, len(self.frames_setup))

            if params is not None:
                init_param = params
                if isinstance(init_param, list):
                    init_param = torch.tensor(init_param)
                init_param = init_param.view(param_shape).clone()
            elif self.param_initialization == "zeros":
                init_param = torch.zeros(param_shape)
            elif self.param_initialization == "random":
                init_param = torch.randn(param_shape)
            elif self.param_initialization == "mid":
                init_param = torch.tensor([-1., 1., -1.])
                # init_param = torch.tensor([0.1, 0.8, 0.1])
                init_param = init_param.view(1, 3, 1).repeat(param_shape[0], 1, param_shape[2])
            elif self.param_initialization == "ref":
                init_param = torch.tensor([1., -1., -1.])
                init_param = init_param.view(1, 3, 1).repeat(param_shape[0], 1, param_shape[2])
            elif self.param_initialization == "obj":
                init_param = torch.tensor([-1., -1., 1.])
                init_param = init_param.view(1, 3, 1).repeat(param_shape[0], 1, param_shape[2])
                    
            else:
                raise ValueError(f"Unknown param_initialization {self.param_initialization}")

            # self.swap_param = nn.Parameter(init_param).float()
            self.swap_param = nn.Parameter(init_param)

        elif self.swapping_type == "mlp":
            param_shape = (self.num_streams, len(self.frames_setup))

            self.swap_param = nn.Sequential(
                nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, self.num_streams * len(self.frames_setup))
            )
            if params is not None:
                self.swap_param.load_state_dict(params)
        else:
            raise ValueError(f"Unknown swapping_control_type {self.swapping_type}")

        self.frame_setup_idx = torch.repeat_interleave(
            torch.arange(len(self.frames_setup)), torch.tensor([len(p) for p in self.frames_setup])
        )
        self.frame_setup_flat = torch.tensor([p for group in self.frames_setup for p in group])

    def set_swap_param(self, param: torch.Tensor):
        self.swap_param.data = param

    def get_swap_param(self):
        if self.swapping_type == "param":
            return self.swap_param
        else:
            return self.swap_param.state_dict()

    def resolve_param_frames(self, force_no_activation=False, sigma_val=None, sig_id=None):
        if self.swapping_type == "param":
            swap_param = self.get_swap_param()
            if sig_id is not None:
                param_id = sig_id
            elif sigma_val is not None:
                param_id = bisect.bisect_right(self.noise_separate_bins, sigma_val)
            else:
                raise ValueError("Either sig_id or sigma_val must be provided")
            swap_param = swap_param[param_id]
        else:
            device = next(self.swap_param.parameters()).device
            dtype = next(self.swap_param.parameters()).dtype

            if sig_id is not None:
                param_id = sig_id
            elif sigma_val is not None:
                param_id = bisect.bisect_right(self.noise_separate_bins, sigma_val)
            else:
                raise ValueError("Either sig_id or sigma_val must be provided")

            # 0-len(noise_seperate_bins) + 1 -> -1 - 1.0
            param_id_normalized = param_id / (len(self.noise_separate_bins) + 1) * 2 - 1.0
            param_id_normalized = torch.tensor([[param_id_normalized]], device=device, dtype=dtype)

            # logsnr = -(torch.log(torch.tensor([sigma_val], device=device, dtype=dtype)) ** 2)
            # logsnr = torch.tensor([sigma_val], device=device, dtype=dtype)
            swap_param = self.swap_param(param_id_normalized)
            swap_param = swap_param.view(self.num_streams, len(self.frames_setup))

        if not force_no_activation:
            if self.activation_type.startswith("softmax"):
                swap_param = F.softmax(swap_param / self.softmax_temperature, dim=0)
            elif self.activation_type.startswith("sigmoid"):
                swap_param = F.sigmoid(swap_param) 
                swap_param = swap_param / swap_param.sum(dim=0, keepdim=True)
           
            if self.activation_type.endswith("one_hot_frame"):
                max_idx = torch.argmax(swap_param, dim=0)  # shape [21]
                swap_param = F.one_hot(max_idx, num_classes=swap_param.shape[0]).T.float()  # shape [3, 21]
            elif self.activation_type.endswith("one_hot_layer"):
                avg_vals = swap_param.mean(dim=1)  # [3]
                max_idx = torch.argmax(avg_vals)  # scalar
                one_hot = F.one_hot(max_idx, num_classes=swap_param.shape[0]).float()  # [3]
                swap_param = one_hot.unsqueeze(1).repeat(1, swap_param.shape[1])  # [3, 21]
            elif "one_hot_frame_threshold" in self.activation_type:
                threshold = float(self.activation_type.split("one_hot_frame_threshold_")[-1])
                max_idx = torch.argmax(swap_param, dim=0)  # shape [21]
                preference_of_second = 0.0
                for i in range(swap_param.shape[1]):
                    if swap_param[1, i] < threshold:
                        if swap_param[0, i] > swap_param[2, i] + preference_of_second:
                            max_idx[i] = 0
                        else:
                            max_idx[i] = 2
                swap_param = F.one_hot(max_idx, num_classes=swap_param.shape[0]).T.float()  # shape [3, 21]

            elif "one_hot_layer_limited_threshold" in self.activation_type:
                threshold = float(self.activation_type.split("one_hot_layer_limited_threshold_")[-1])

                # avg only over [2:] frames, keep 0  1 separatly as one hot frame
                swap_params_first_two = swap_param[:, :1]  # [3, 1]
                swap_params_rest = swap_param[:, 1:]  # [3, 20]

                
                max_idx = torch.argmax(swap_params_first_two, dim=0)  # shape [1]
                preference_of_second = 0.0
                for i in range(swap_params_first_two.shape[1]):
                    if swap_params_first_two[1, i] < threshold:
                        if swap_params_first_two[0, i] > swap_params_first_two[2, i] + preference_of_second:
                            max_idx[i] = 0
                        else:
                            max_idx[i] = 2 

                one_hot_first_two = F.one_hot(max_idx, num_classes=swap_param.shape[0]).T.float()  # shape [3, 2]


                avg_vals = swap_params_rest.mean(dim=1)  # [3]
                max_idx = torch.argmax(avg_vals)  # scalar
                preference_of_second = 0.0
                if avg_vals[max_idx] < threshold:
                    if avg_vals[0] > avg_vals[2] + preference_of_second:
                        max_idx = torch.tensor(0, device=avg_vals.device)
                    else:
                        max_idx = torch.tensor(2, device=avg_vals.device)
                one_hot = F.one_hot(max_idx, num_classes=swap_params_rest.shape[0]).float()  # [3]
                swap_params_rest = one_hot.unsqueeze(1).repeat(1, swap_params_rest.shape[1])  # [3, 19]

                swap_param = torch.cat([one_hot_first_two, swap_params_rest], dim=1)  # [3, 21]

                if self.attn_type == "self_temporal":
                    swap_param[0, :] = 0.0
                    swap_param[1, :] = 1.0
                    swap_param[2, :] = 0.0


            # temporal not layere
            elif "one_hot_layer_threshold" in self.activation_type:
                threshold = float(self.activation_type.split("one_hot_layer_threshold_")[-1])

                avg_vals = swap_param.mean(dim=1)  # [3]
                max_idx = torch.argmax(avg_vals)  # scalar
                preference_of_second = 0.0
                if avg_vals[max_idx] < threshold:
                    if avg_vals[0] > avg_vals[2] + preference_of_second:
                        max_idx = torch.tensor(0, device=avg_vals.device)
                    else:
                        max_idx = torch.tensor(2, device=avg_vals.device)
                one_hot = F.one_hot(max_idx, num_classes=swap_param.shape[0]).float()  # [3]
                swap_param = one_hot.unsqueeze(1).repeat(1, swap_param.shape[1])  # [3, 21]


                if self.attn_type == "self_temporal":
                    swap_param[0, :] = 0.0
                    swap_param[1, :] = 1.0
                    swap_param[2, :] = 0.0

                                               

        out = torch.zeros(
            self.num_streams, self.frame_setup_flat.max() + 1, dtype=swap_param.dtype, device=swap_param.device
        )
        out[:, self.frame_setup_flat] = swap_param[:, self.frame_setup_idx]
        return out

    def resolve_param_frames_all(self, force_no_activation=False, full=False):
        all_params = []
        for sig_id in range(0, len(self.noise_separate_bins) + 1):
            all_params.append(self.resolve_param_frames(force_no_activation=force_no_activation, sig_id=sig_id))
        return torch.stack(all_params, dim=0)

    def swap_attention(self, key, value, sigma_val=None):
        # print(key.shape)

        if self.layer_activated and self.swapping_active:
            swap_param = self.resolve_param_frames(sigma_val=sigma_val)  # [3, 21]
            swap_param = swap_param.to(key.dtype)

            b = self.num_streams
            n = swap_param.shape[1]

            # [63, 5184, 320]    swap_param = [3,21]
            if self.attn_type in ["self_spatial", "self_spatio_temporal"]:
                key_cur = rearrange(key, "(b n) ... -> b n ...", b=b, n=n)  # [3, 21, 5184, 320]
                val_cur = rearrange(value, "(b n) ... -> b n ...", b=b, n=n)

                w = rearrange(swap_param, "b n -> b n 1 1", b=b, n=n)  # [3, 21, 1, 1]

                key_new_main = (key_cur * w).sum(dim=0, keepdim=True)  # [1, 21, 5184, 320]
                val_new_main = (val_cur * w).sum(dim=0, keepdim=True)

                key_combo = torch.cat([key_cur[0:1], key_new_main, key_cur[2:3]], dim=0)  # [3, 21, 5184, 320]
                val_combo = torch.cat([val_cur[0:1], val_new_main, val_cur[2:3]], dim=0)

                key_combo = rearrange(key_combo, "b n ... -> (b n) ...", b=b, n=n)  # [63, 5184, 320]
                val_combo = rearrange(val_combo, "b n ... -> (b n) ...", b=b, n=n)
                return key_combo, val_combo

            # [15552, 21, 320]  swap_param = [3,21]
            elif self.attn_type == "self_temporal":
                key_cur = rearrange(key, "(b x) n ... -> b x n ...", b=b)  # [3, 5184, 21, 320]
                val_cur = rearrange(value, "(b x) n ... -> b x n ...", b=b)

                w = rearrange(swap_param, "b n -> b 1 n 1")  # [3, 1, 21, 1]

                key_new_main = (key_cur * w).sum(dim=0, keepdim=True)  # [1, 5184, 21, 320]
                val_new_main = (val_cur * w).sum(dim=0, keepdim=True)

                key_combo = torch.cat([key_cur[0:1], key_new_main, key_cur[2:3]], dim=0)  # [3, 5184, 21, 320]
                val_combo = torch.cat([val_cur[0:1], val_new_main, val_cur[2:3]], dim=0)

                key_combo = rearrange(key_combo, "b x n ... -> (b x) n ...")  # [15552, 21, 320]
                val_combo = rearrange(val_combo, "b x n ... -> (b x) n ...")
                return key_combo, val_combo
            else:
                raise ValueError(f"Unknown attn_type {self.attn_type}")

        return key, value


class Attention(SwappingAttentionProcessor):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     context: torch.Tensor | None = None,
    #     sigma_val: float | None = None,
    # ) -> torch.Tensor:
    #     return torch.utils.checkpoint.checkpoint(self._forward, x, context, sigma_val, use_reentrant=False)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None, sigma_val: float | None = None
    ) -> torch.Tensor:
        # print(x.shape, context.shape if context is not None else "None", end="")
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        k, v = self.swap_attention(k, v, sigma_val=sigma_val)

        q, k, v = map(
            lambda t: rearrange(t, "b l (h d) -> b h l d", h=self.heads),
            (q, k, v),
        )

        # print(k.shape)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h l d -> b l (h d)")
        out = self.to_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn1 = Attention(
            query_dim=dim,
            context_dim=None,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.ff = FeedForward(dim, dropout=dropout)
        self.attn2 = Attention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor, sigma_val: float | None = None) -> torch.Tensor:
        x = self.attn1(self.norm1(x), sigma_val=sigma_val) + x
        x = self.attn2(self.norm2(x), context=context, sigma_val=sigma_val) + x
        x = self.ff(self.norm3(x)) + x
        return x


class TransformerBlockTimeMix(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        context_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = n_heads * d_head
        self.norm_in = nn.LayerNorm(dim)
        self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout)
        self.attn1 = Attention(
            query_dim=inner_dim,
            context_dim=None,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout)
        self.attn2 = Attention(
            query_dim=inner_dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, num_frames: int, sigma_val: float | None = None
    ) -> torch.Tensor:
        _, s, _ = x.shape
        x = rearrange(x, "(b t) s c -> (b s) t c", t=num_frames)
        x = self.ff_in(self.norm_in(x)) + x
        x = self.attn1(self.norm1(x), context=None, sigma_val=sigma_val) + x
        x = self.attn2(self.norm2(x), context=context, sigma_val=sigma_val) + x
        x = self.ff(self.norm3(x))
        x = rearrange(x, "(b s) t c -> (b t) s c", s=s)
        return x


class SkipConnect(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_spatial: torch.Tensor, x_temporal: torch.Tensor) -> torch.Tensor:
        return x_spatial + x_temporal


class MultiviewTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        name: str,
        unflatten_names: list[str] = [],
        depth: int = 1,
        context_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.name = name
        self.unflatten_names = unflatten_names

        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    context_dim=context_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)
        self.time_mixer = SkipConnect()
        self.time_mix_blocks = nn.ModuleList(
            [
                TransformerBlockTimeMix(
                    inner_dim,
                    n_heads,
                    d_head,
                    context_dim=context_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, num_frames: int, sigma_val: float | None = None
    ) -> torch.Tensor:
        assert context.ndim == 3
        _, _, h, w = x.shape
        x_in = x

        time_context = context
        time_context_first_timestep = time_context[::num_frames]
        time_context = repeat(time_context_first_timestep, "b ... -> (b n) ...", n=h * w)

        if self.name in self.unflatten_names:
            context = context[::num_frames]

        x = self.norm(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.proj_in(x)

        for block, mix_block in zip(self.transformer_blocks, self.time_mix_blocks):
            if self.name in self.unflatten_names:
                x = rearrange(x, "(b t) (h w) c -> b (t h w) c", t=num_frames, h=h, w=w)
            x = block(x, context=context, sigma_val=sigma_val)
            if self.name in self.unflatten_names:
                x = rearrange(x, "b (t h w) c -> (b t) (h w) c", t=num_frames, h=h, w=w)
            x_mix = mix_block(x, context=time_context, num_frames=num_frames, sigma_val=sigma_val)
            x = self.time_mixer(x_spatial=x, x_temporal=x_mix)

        x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        out = x + x_in
        return out


ATTENTION_LAYERS = {
    "self_spatial": [
        "module.input_blocks.1.1.transformer_blocks.0.attn1",
        "module.input_blocks.2.1.transformer_blocks.0.attn1",
        "module.input_blocks.4.1.transformer_blocks.0.attn1",
        "module.input_blocks.5.1.transformer_blocks.0.attn1",
        "module.input_blocks.7.1.transformer_blocks.0.attn1",
        "module.input_blocks.8.1.transformer_blocks.0.attn1",
        "module.output_blocks.9.1.transformer_blocks.0.attn1",
        "module.output_blocks.10.1.transformer_blocks.0.attn1",
        "module.output_blocks.11.1.transformer_blocks.0.attn1",
    ],
    "self_spatio_temporal": [
        "module.middle_block.1.transformer_blocks.0.attn1",
        "module.output_blocks.3.1.transformer_blocks.0.attn1",
        "module.output_blocks.4.1.transformer_blocks.0.attn1",
        "module.output_blocks.5.1.transformer_blocks.0.attn1",
        "module.output_blocks.6.1.transformer_blocks.0.attn1",
        "module.output_blocks.7.1.transformer_blocks.0.attn1",
        "module.output_blocks.8.1.transformer_blocks.0.attn1",
    ],
    "self_temporal": [
        "module.input_blocks.1.1.time_mix_blocks.0.attn1",
        "module.input_blocks.2.1.time_mix_blocks.0.attn1",
        "module.input_blocks.4.1.time_mix_blocks.0.attn1",
        "module.input_blocks.5.1.time_mix_blocks.0.attn1",
        "module.input_blocks.7.1.time_mix_blocks.0.attn1",
        "module.input_blocks.8.1.time_mix_blocks.0.attn1",
        "module.middle_block.1.time_mix_blocks.0.attn1",
        "module.output_blocks.3.1.time_mix_blocks.0.attn1",
        "module.output_blocks.4.1.time_mix_blocks.0.attn1",
        "module.output_blocks.5.1.time_mix_blocks.0.attn1",
        "module.output_blocks.6.1.time_mix_blocks.0.attn1",
        "module.output_blocks.7.1.time_mix_blocks.0.attn1",
        "module.output_blocks.8.1.time_mix_blocks.0.attn1",
        "module.output_blocks.9.1.time_mix_blocks.0.attn1",
        "module.output_blocks.10.1.time_mix_blocks.0.attn1",
        "module.output_blocks.11.1.time_mix_blocks.0.attn1",
    ],
}


CROSS_SPATIAL_ATTENTION_LAYERS = [
    "module.input_blocks.1.1.transformer_blocks.0.attn2",
    "module.input_blocks.2.1.transformer_blocks.0.attn2",
    "module.input_blocks.4.1.transformer_blocks.0.attn2",
    "module.input_blocks.5.1.transformer_blocks.0.attn2",
    "module.input_blocks.7.1.transformer_blocks.0.attn2",
    "module.input_blocks.8.1.transformer_blocks.0.attn2",
    "module.middle_block.1.transformer_blocks.0.attn2",
    "module.output_blocks.3.1.transformer_blocks.0.attn2",
    "module.output_blocks.4.1.transformer_blocks.0.attn2",
    "module.output_blocks.5.1.transformer_blocks.0.attn2",
    "module.output_blocks.6.1.transformer_blocks.0.attn2",
    "module.output_blocks.7.1.transformer_blocks.0.attn2",
    "module.output_blocks.8.1.transformer_blocks.0.attn2",
    "module.output_blocks.9.1.transformer_blocks.0.attn2",
    "module.output_blocks.10.1.transformer_blocks.0.attn2",
    "module.output_blocks.11.1.transformer_blocks.0.attn2",
]

CROSS_TEMPORAL_ATTENTION_LAYERS = [
    "module.input_blocks.1.1.time_mix_blocks.0.attn2",
    "module.input_blocks.2.1.time_mix_blocks.0.attn2",
    "module.input_blocks.4.1.time_mix_blocks.0.attn2",
    "module.input_blocks.5.1.time_mix_blocks.0.attn2",
    "module.input_blocks.7.1.time_mix_blocks.0.attn2",
    "module.input_blocks.8.1.time_mix_blocks.0.attn2",
    "module.middle_block.1.time_mix_blocks.0.attn2",
    "module.output_blocks.3.1.time_mix_blocks.0.attn2",
    "module.output_blocks.4.1.time_mix_blocks.0.attn2",
    "module.output_blocks.5.1.time_mix_blocks.0.attn2",
    "module.output_blocks.6.1.time_mix_blocks.0.attn2",
    "module.output_blocks.7.1.time_mix_blocks.0.attn2",
    "module.output_blocks.8.1.time_mix_blocks.0.attn2",
    "module.output_blocks.9.1.time_mix_blocks.0.attn2",
    "module.output_blocks.10.1.time_mix_blocks.0.attn2",
    "module.output_blocks.11.1.time_mix_blocks.0.attn2",
]
