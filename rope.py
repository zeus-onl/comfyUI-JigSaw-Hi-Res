import math
import torch
from einops import rearrange

import comfy.model_management


def _scaled_omega(
    dim,
    theta,
    device,
    method: str = "yarn",
    rope_percentage: float = 1.0,
    dype: bool = True,
    yarn_ramp_type: str = "linear",
    yarn_ratio: float = 1.0,
    yarn_beta_fast: int = 32,
    yarn_beta_slow: int = 1,
    current_timestep: float = 0.0,
    timestep_modulation: bool = False,
    theta_period_min: float = 1.0,
    theta_period_max: float = 10000.0,
    attn_ratio: float = 1.0,
    **kwargs
):
    """
    Baut omega (= inv_freq) exakt wie comfy.ldm.flux.math.rope(), aber mit
    optionaler YaRN/NTK-Skalierung und Timestep-Blend. Nutzt float64 wie das
    Original, damit die Praezision fuer hohe Aufloesungen erhalten bleibt.
    """
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device)
    base_omega = 1.0 / (theta ** scale)

    if not dype or method == "base":
        return base_omega * rope_percentage

    if timestep_modulation:
        blend = max(0.0, min(1.0, float(current_timestep)))
    else:
        blend = 1.0

    effective_ratio = 1.0 + (yarn_ratio - 1.0) * blend
    if effective_ratio <= 1.0 + 1e-6:
        return base_omega * rope_percentage

    if method == "ntk":
        scaled_theta = theta * (effective_ratio ** (dim / max(dim - 2, 1)))
        scaled_theta = max(theta_period_min, min(theta_period_max, scaled_theta))
        omega = 1.0 / (scaled_theta ** scale)
        return omega * rope_percentage

    # method == "yarn": pro-Frequenz unterschiedliche Interpolation
    omega_extrapolated = base_omega
    omega_interpolated = base_omega / effective_ratio

    n_freqs = base_omega.shape[0]
    idx = torch.arange(n_freqs, dtype=torch.float64, device=device)

    beta_fast = max(1, int(yarn_beta_fast))
    beta_slow = max(1, int(yarn_beta_slow))
    low = min(beta_slow, beta_fast)
    high = max(beta_slow, beta_fast)
    high = max(high, low + 1)

    if yarn_ramp_type == "linear":
        ramp = (idx - low) / (high - low)
    else:
        ramp = 0.5 - 0.5 * torch.cos(math.pi * (idx - low) / (high - low))

    ramp = torch.clamp(ramp, 0.0, 1.0)
    mask = 1.0 - ramp  # 1 = voll interpoliert (gedehnt), 0 = unveraendert (extrapoliert)

    omega = omega_interpolated * mask + omega_extrapolated * (1.0 - mask)
    return omega * rope_percentage


def rope_dype(pos: torch.Tensor, dim: int, theta: int, **kwargs) -> torch.Tensor:
    """
    Drop-in-Ersatz fuer comfy.ldm.flux.math.rope() mit identischem Ausgabeformat
    ([..., N, dim/2, 2, 2] - volle 2x2-Rotationsmatrizen), aber mit YaRN/NTK/
    Timestep-skaliertem omega statt starrem 1/theta**scale.

    WICHTIG: Format muss exakt zu comfy.ldm.flux.math.rope() passen, da Krea2s
    nativer comfy_kitchen-RoPE-Kernel dieses Format hart erwartet (siehe
    apply_rope1 in math.py). Kein eigenes cos/sin-Paar-Format mehr wie zuvor.
    """
    assert dim % 2 == 0
    if not comfy.model_management.supports_fp64(pos.device):
        device = torch.device("cpu")
    else:
        device = pos.device

    omega = _scaled_omega(dim, theta, device, **kwargs)

    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def _correction_factor(num_rotations, dim, theta, ori_len):
    # frequency index whose wavelength fits num_rotations times into ori_len
    return (dim * math.log(ori_len / (num_rotations * 2 * math.pi))) / (2 * math.log(theta))


def _ramp_mask(low_ratio, high_ratio, dim, theta, ori_len, device):
    # clamped on both ends: past dim // 2 the ramp would flip and interpolate every band
    low = min(max(math.floor(_correction_factor(low_ratio, dim, theta, ori_len)), 0), dim // 2)
    high = min(math.ceil(_correction_factor(high_ratio, dim, theta, ori_len)), dim // 2)
    if low == high:
        high += 0.001
    ramp = (torch.arange(dim // 2, dtype=torch.float64, device=device) - low) / (high - low)
    return 1.0 - torch.clamp(ramp, 0.0, 1.0)  # 1 = keep the higher-precision branch


def dype_axis_rope(
    pos: torch.Tensor,
    dim: int,
    theta: float,
    scale: float,
    ori_len: int = 64,
    noise_level: float = 1.0,
    method: str = "yarn",
    dype: bool = True,
    rope_percentage: float = 1.0,
) -> torch.Tensor:
    """
    Dy-YaRN / Dy-NTK from the official DyPE code (github.com/guyyariv/DyPE,
    flux/transformer_flux.py), in the comfy.ldm.flux.math.rope() output format.

    scale: target / trained extent of this axis (in tokens), ori_len: trained
    extent. noise_level: t in [0, 1], 1 = pure noise; with dype the scaling is
    strongest there (kappa(t) = t^2) and fades to the trained frequencies at t = 0.
    """
    assert dim % 2 == 0
    if not comfy.model_management.supports_fp64(pos.device):
        device = torch.device("cpu")
    else:
        device = pos.device

    exps = torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim
    omega = 1.0 / (theta ** exps)
    mscale = 1.0

    if method != "base" and scale > 1.0:
        kappa = max(float(noise_level), 0.0) ** 2.0 if dype else 1.0
        if method == "ntk":
            base_ntk = scale ** (dim / (dim - 2))
            ntk_factor = max(1.0, base_ntk ** (2.0 * kappa) if dype else base_ntk)
            omega = 1.0 / ((theta * ntk_factor) ** exps)
        else:
            kappa = max(kappa, 1e-6)  # kappa = 0 would put the thresholds at infinite wavelength
            omega_linear = omega / scale
            omega_ntk = 1.0 / ((theta * scale ** (dim / (dim - 2))) ** exps)
            mask = _ramp_mask(1.25 * kappa, 0.75 * kappa, dim, theta, ori_len, device)
            blended = omega_linear * (1 - mask) + omega_ntk * mask
            mask = _ramp_mask(16 * kappa, 2 * kappa, dim, theta, ori_len, device)
            omega = blended * (1 - mask) + omega * mask
            mscale = 0.1 * math.log(scale) + 1.0

    omega = omega * rope_percentage
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1) * mscale
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)
