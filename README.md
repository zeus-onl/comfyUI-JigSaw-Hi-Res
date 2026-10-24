# 🧩 Jigsaw Hi-Res

**Generate beyond your model's native resolution — without the repeating-pattern artifacts.**

A ComfyUI node that patches a Flux/Krea2-family model's rotary positional
embedder (RoPE) with **YaRN and NTK-aware frequency scaling**, the same
technique used to extend context length in LLMs — applied here to 2D image
positions instead of 1D token sequences.

## The problem

Flux/Krea2-style diffusion transformers encode spatial position with RoPE,
tuned for a specific training resolution (commonly ~1024×1024). Ask for a
significantly larger image and the position embeddings extrapolate past what
the model has ever seen — the usual symptom is duplicated limbs/faces, tiled
repetition, or a hazy "gray veil" wash over the image.

## What Jigsaw Hi-Res does

At apply time, it:
1. Reads your target resolution (explicit `width`/`height`, or auto-detected
   from a connected `latent`).
2. Computes an adaptive **YaRN ratio** from how far the target resolution
   sits above the ~1024px reference (blending a linear and an area-based
   ratio, softly capped at 1.25× to keep contrast stable at extreme
   resolutions/aspect ratios — this cap is what keeps the classic
   "gray veil" oversaturation from creeping in on the more aggressive high-res
   attempts).
3. Swaps the model's `pe_embedder` for a custom RoPE implementation that
   reproduces `comfy.ldm.flux.math.rope()`'s exact output format
   (`[..., N, dim/2, 2, 2]` rotation matrices), but computes each frequency
   band's scaling individually — near-DC frequencies stay untouched
   (extrapolated), high frequencies get interpolated/compressed, following
   the standard YaRN "NTK-by-parts" ramp (`beta_fast`/`beta_slow` cutoffs).
4. Does all of this **without retraining or fine-tuning** — it's a pure
   inference-time patch on the position embedder.

## Installation

Drop the folder into `ComfyUI/custom_nodes/`, restart ComfyUI. The node
appears under **Jigsaw/Patches → 🧩 Jigsaw Hi-Res**.

## Usage

Connect your `MODEL` in, plug a `LATENT` in (optional — auto-detects the
real target resolution from it) or set `width`/`height` explicitly, and
sample as usual at your target resolution.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `width` / `height` | 1024 / 1024 | Target resolution. Overridden automatically if `latent` is connected. |
| `method` | `yarn` | `yarn` — per-frequency-band interpolation (recommended, best quality at large upscales). `ntk` — single global frequency-base rescale (cheaper, less precise). `base` — disables scaling entirely (pass-through RoPE). |
| `rope_percentage` | 1.0 | Blend factor for how much of the computed scaling to actually apply (1.0 = full effect). |
| `dype_enabled` | `True` | Master switch for the whole dynamic scaling path. Off = falls back to the model's original, untouched RoPE. |
| `max_pe_length` | 64 | MOSS-Image / Fibo 2.0: trained size in latent tokens per side (64 = 1024px). Unused for Krea2/Flux. |
| `latent` *(optional)* | — | If connected, its shape is used to auto-derive the true target width/height (×8, or ×16 for Flux2-VAE models) instead of the manual `width`/`height` inputs. |

## Compatibility

Requires a model whose diffusion backbone exposes a `pe_embedder` module
with `.theta` and `.axes_dim` attributes, and whose native RoPE kernel
consumes the same `[..., dim/2, 2, 2]` rotation-matrix format as
`comfy.ldm.flux.math.rope()` (Flux.1 and Krea2/Flux.2-Klein-family models).
**MOSS-Image** and **Fibo 2.0** are supported too (needs a ComfyUI recent
enough to ship `comfy/ldm/moss_image` and `comfy/ldm/fibo2`). They build their
RoPE inline instead of through a `pe_embedder`, so the node hooks their
`comfy.ldm.flux.math.rope()` calls during the forward pass and scales only the
spatial (height/width) axes; the text / gist-plane axes and MOSS-Image's
reference images keep their trained frequencies. A connected `latent` is
converted with the model's own VAE factor (16 for their Flux2 VAE).

For these two models the scaling follows the official
[DyPE](https://github.com/guyyariv/DyPE) Dy-YaRN / Dy-NTK code rather than the
Krea2-tuned ratio above: each axis is scaled by its real target / trained
extent (`max_pe_length` = trained size in tokens, 64 = 1024px; no 1.25 cap),
YaRN picks the bands by rotation count and adds its attention temperature, and
`timestep_modulation` uses DyPE's κ(t) = t² schedule: strongest at the start
of sampling, where the layout (and any duplication) is decided, fading to the
trained frequencies at the end. Krea2/Flux behave exactly as before.

Models using a different RoPE convention are not supported and will raise a
clear error on load rather than silently producing garbage.

## Credits

By Jigsaw & Zeus — [zeus.onl](https://zeus.onl)
