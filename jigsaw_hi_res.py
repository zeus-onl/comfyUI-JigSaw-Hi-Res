import torch
import torch.nn as nn
import math
from contextlib import contextmanager

import comfy.ldm.flux.math
import comfy.patcher_extension

from .rope import rope_dype, dype_axis_rope

try:
    from comfy.ldm.moss_image.model import MOSSImage
except ImportError:
    MOSSImage = None

try:
    from comfy.ldm.fibo2.model import Fibo2
except ImportError:
    Fibo2 = None


class JigsawPosEmbed(nn.Module):

    def __init__(
        self,
        axes_dim: list[int],
        theta: float = 10000.0,
        method: str = "yarn",
        rope_percentage: float = 1.0,
        dype: bool = True,
        ori_max_pe_len_spatial: int = 64,
        yarn_ramp_type: str = "linear",
        yarn_ratio: float = 1.0,
        yarn_beta_fast: int = 32,
        yarn_beta_slow: int = 1,
        timestep_modulation: bool = False,
        theta_period_min: float = 1.0,
        theta_period_max: float = 10000.0,
        attn_ratio: float = 1.0
    ):
        super().__init__()

        self.n_axes_init = len(axes_dim)

        for dim in axes_dim:
            assert dim % 2 == 0, f"Dimension {dim} muss gerade sein"

        if self.n_axes_init == 4:
            # MOSS-Image: (r, f, h, w)
            self.dim_axis0 = axes_dim[0] + axes_dim[1]
            self.dim_spatial = axes_dim[2] + axes_dim[3]

        elif self.n_axes_init == 3:
            self.dim_axis0 = axes_dim[0]
            self.dim_spatial = axes_dim[1] + axes_dim[2]

        elif self.n_axes_init == 2:
            self.dim_spatial = axes_dim[0] + axes_dim[1]

        elif self.n_axes_init == 1:
            self.dim_axis0 = axes_dim[0]

        else:
            raise ValueError(
                "Nur 1, 2, 3 oder 4 Achsen unterstützt"
            )

        self.axes_dim = axes_dim
        self.theta = theta
        self.method = method
        self.rope_percentage = rope_percentage
        self.dype = dype

        self.ori_max_pe_len_spatial = ori_max_pe_len_spatial

        self.yarn_ramp_type = yarn_ramp_type
        self.yarn_ratio = yarn_ratio
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow

        self.timestep_modulation = timestep_modulation

        self.theta_period_min = theta_period_min
        self.theta_period_max = theta_period_max

        self.current_timestep = 0.0
        self.attn_ratio = attn_ratio


    def set_timestep(self, timestep):
        self.current_timestep = timestep


    def rope(self, pos, dim, theta):
        # Drop-in for comfy.ldm.flux.math.rope(pos, dim, theta) with this embedder's scaling
        return rope_dype(
            pos,
            dim=dim,
            theta=theta,
            method=self.method,
            rope_percentage=self.rope_percentage,
            dype=self.dype,
            yarn_ramp_type=self.yarn_ramp_type,
            yarn_ratio=self.yarn_ratio,
            yarn_beta_fast=self.yarn_beta_fast,
            yarn_beta_slow=self.yarn_beta_slow,
            current_timestep=self.current_timestep,
            timestep_modulation=self.timestep_modulation,
            theta_period_min=self.theta_period_min,
            theta_period_max=self.theta_period_max,
            attn_ratio=self.attn_ratio,
        )


    def forward(self, ids):
        pos = ids.float()

        if pos.ndim != 3:
            raise RuntimeError(
                f"JigsawPosEmbed erwartet [B,N,Axes], bekam {pos.shape}"
            )

        n_axes = pos.shape[-1]
        if n_axes != len(self.axes_dim):
            raise RuntimeError(
                f"JigsawPosEmbed: ids hat {n_axes} Achsen, axes_dim hat aber "
                f"{len(self.axes_dim)} Eintraege ({self.axes_dim})"
            )

        parts = [self.rope(pos[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)]

        emb = torch.cat(parts, dim=-3)  # [B, N, headdim/2, 2, 2]
        return emb.unsqueeze(1).to(ids.device)  # [B, 1, N, headdim/2, 2, 2]


class DyPEAxisEmbed:
    """
    Per-axis Dy-YaRN / Dy-NTK (official DyPE schedule) for the models that
    build their pe inline. axis_scales maps the spatial axis index to its
    target / trained extent; the other axes keep the trained frequencies.
    """

    def __init__(self, axis_scales, method="yarn", dype=True, timestep_modulation=True, rope_percentage=1.0, ori_len=64):
        self.axis_scales = axis_scales
        self.method = method if dype else "base"
        self.timestep_modulation = timestep_modulation
        self.rope_percentage = rope_percentage
        self.ori_len = ori_len
        self.noise_level = 1.0

    def set_timestep(self, timestep):
        # same input as JigsawPosEmbed (0 = start of sampling); DyPE wants t with 1 = pure noise
        self.noise_level = 1.0 - timestep

    def rope(self, pos, dim, theta, axis):
        return dype_axis_rope(
            pos,
            dim,
            theta,
            scale=self.axis_scales[axis],
            ori_len=self.ori_len,
            noise_level=self.noise_level,
            method=self.method,
            dype=self.timestep_modulation,
            rope_percentage=self.rope_percentage,
        )


@contextmanager
def _flux_rope_override(embedder, n_axes):
    """
    Routes comfy.ldm.flux.math.rope() through the embedder for its spatial axes.
    For models that build their pe inline (MOSS-Image, Fibo 2.0): they call it
    once per axis, in axis order. Only the first n_axes calls are touched, so
    the text/plane axes and any later call keep the trained frequencies.
    """
    orig = comfy.ldm.flux.math.rope
    calls = [0]

    def rope(pos, dim, theta):
        i = calls[0]
        calls[0] += 1
        if i < n_axes and i in embedder.axis_scales:
            return embedder.rope(pos, dim, theta, i)
        return orig(pos, dim, theta)

    comfy.ldm.flux.math.rope = rope
    try:
        yield
    finally:
        comfy.ldm.flux.math.rope = orig


def _patch_moss_image(m, embedder):
    # axes (r, f, h, w): f carries the text positions, only h and w are scaled
    orig_rope = m.get_model_object("diffusion_model.rope")

    def moss_rope(h, w, *args, **kwargs):
        if h * w == 0:
            # preprocess_refs: the references keep the trained frequencies
            return orig_rope(h, w, *args, **kwargs)
        # the first four calls are the main pe; the references' second pass stays untouched
        with _flux_rope_override(embedder, n_axes=4):
            return orig_rope(h, w, *args, **kwargs)

    m.add_object_patch("diffusion_model.rope", moss_rope)


def _patch_fibo2(m, embedder):
    # axes (plane, h, w): the plane axis orders gist tokens and images, only h and w are scaled
    def fibo2_wrapper(executor, *args, **kwargs):
        with _flux_rope_override(embedder, n_axes=3):
            return executor(*args, **kwargs)

    m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "jigsaw_hi_res", fibo2_wrapper)


class JigsawHiRes:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "method": (
                    ["yarn", "ntk", "base"],
                    {"default": "yarn"}
                ),
                "rope_percentage": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "dype_enabled": (
                    "BOOLEAN",
                    {"default": True}
                ),
                "max_pe_length": (
                    "INT",
                    {"default": 64, "min": 1, "max": 4096, "tooltip": "MOSS-Image / Fibo 2.0: the trained "
                     "image size in latent tokens per side (64 = 1024px); the DyPE scale per axis is "
                     "target / this. Unused for Krea2/Flux."}
                ),
                "timestep_modulation": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "The 'Dy' in DyPE: blend the YaRN ratio from 0 "
                     "(fully extrapolated/untouched) at the start of sampling up to full strength "
                     "as sampling progresses, instead of applying one fixed scale for every step. "
                     "This follows the coarse-structure-first, fine-detail-later spectral progression "
                     "that the DyPE paper is built around. Off = old static-scale behavior. "
                     "MOSS-Image / Fibo 2.0 use the official DyPE schedule instead: strongest at the "
                     "start of sampling (kappa = t^2), fading to the trained frequencies at the end."}
                )
            },
            "optional": {
                "latent": ("LATENT",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_jigsaw_patch"
    CATEGORY = "Jigsaw/Patches"

    def apply_jigsaw_patch(
        self,
        model,
        width,
        height,
        method,
        rope_percentage,
        dype_enabled,
        max_pe_length,
        timestep_modulation=True,
        latent=None
    ):
        m = model.clone()

        # Ziel-Auflösung ermitteln
        target_w = width
        target_h = height
        
        if latent is not None:
            # Latent Shape ist [B, C, H, W] - Faktor 8 (Krea2/Flux/SD) oder 16 (Flux2-VAE: MOSS-Image, Fibo 2.0)
            downscale = getattr(m.model.latent_format, "spacial_downscale_ratio", 8)
            target_h = latent["samples"].shape[-2] * downscale
            target_w = latent["samples"].shape[-1] * downscale

        diffusion_model = m.get_model_object("diffusion_model")
        is_moss = MOSSImage is not None and isinstance(diffusion_model, MOSSImage)
        is_fibo2 = Fibo2 is not None and isinstance(diffusion_model, Fibo2)

        if is_moss or is_fibo2:
            # one token per latent pixel; max_pe_length = trained extent in tokens (64 = 1024px)
            downscale = getattr(m.model.latent_format, "spacial_downscale_ratio", 16)
            scale_h = max(1.0, target_h / downscale / max_pe_length)
            scale_w = max(1.0, target_w / downscale / max_pe_length)
            h_axis, w_axis = (2, 3) if is_moss else (1, 2)
            new_embedder = DyPEAxisEmbed(
                axis_scales={h_axis: scale_h, w_axis: scale_w},
                method=method,
                dype=dype_enabled,
                timestep_modulation=timestep_modulation,
                rope_percentage=rope_percentage,
                ori_len=max_pe_length,
            )
            print(f"[Jigsaw] {'MOSS-Image' if is_moss else 'Fibo 2.0'} | Target Res: {target_w}x{target_h} | "
                  f"DyPE scale h {scale_h:.4f} / w {scale_w:.4f}")
            if is_moss:
                _patch_moss_image(m, new_embedder)
            else:
                _patch_fibo2(m, new_embedder)

        else:
            # --- NEUE DIAGONAL-DÄMPFUNG GEGEN DEN GRAUSCHLEIER ---
            max_dim = max(target_w, target_h)
            linear_ratio = float(max_dim / 1024.0)
            area_ratio = float(((target_w * target_h) / (1024.0 * 1024.0)) ** 0.5)

            # Mittelwert fängt extreme Seitenverhältnisse ab
            calculated_ratio = (linear_ratio + area_ratio) / 2.0
            # Sanfter Deckel bei 1.25, damit der Kontrast stabil bleibt
            yarn_ratio = max(1.0, min(calculated_ratio, 1.25))

            print(f"[Jigsaw] Target Res: {target_w}x{target_h} | Adjusted Max-Dim YaRN Ratio: {yarn_ratio:.4f}")
            # -----------------------------------------------------

            try:
                orig = diffusion_model.pe_embedder
                theta = orig.theta
                axes_dim = orig.axes_dim
            except Exception:
                raise ValueError(
                    "Jigsaw Hi-Res: kein kompatibles Modell gefunden (Krea2/Flux pe_embedder, MOSS-Image oder Fibo 2.0)"
                )

            new_embedder = JigsawPosEmbed(
                axes_dim=axes_dim,
                theta=theta,
                method=method,
                rope_percentage=rope_percentage,
                dype=dype_enabled,
                ori_max_pe_len_spatial=max_pe_length,
                yarn_ratio=yarn_ratio,
                timestep_modulation=timestep_modulation
            )
            m.add_object_patch("diffusion_model.pe_embedder", new_embedder)

        sigma_max = m.model.model_sampling.sigma_max.item()

        def jigsaw_wrapper(model_function, args_dict):
            timestep = args_dict.get("timestep")

            if timestep is not None and timestep.numel() > 0:
                # cond and uncond (or a batch) share one sigma
                sigma = timestep.max().item()
                if sigma_max > 0:
                    value = max(0.0, min(1.0, 1.0 - (sigma / sigma_max)))
                    new_embedder.set_timestep(value)

            return model_function(
                args_dict.get("input"),
                args_dict.get("timestep"),
                **args_dict.get("c", {})
            )

        m.set_model_unet_function_wrapper(jigsaw_wrapper)

        return (m,)


NODE_CLASS_MAPPINGS = {
    "JigsawHiRes": JigsawHiRes
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JigsawHiRes": "Jigsaw Hi-Res"
}
