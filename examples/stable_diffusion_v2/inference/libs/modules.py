import math
import numpy as np
import mindspore as ms
from mindspore import ops, nn
from mindspore._extends import cell_attr_register


class VAEEncoder(nn.Cell):
    def __init__(self, vae, scaling_factor, scheduler):
        super(VAEEncoder, self).__init__()
        self.vae = vae
        self.encode = self.vae.encode
        self.scaling_factor = scaling_factor
        self.result_add_noise = scheduler.add_noise
        self.alphas_cumprod = scheduler.alphas_cumprod

    def construct(self, x, noise, ts):
        image_latents = self.encode(x)
        image_latents = image_latents * self.scaling_factor
        latents = self.result_add_noise(image_latents, noise, self.alphas_cumprod[ts])
        return latents, image_latents


class VAEDecoder(nn.Cell):
    def __init__(self, vae, scaling_factor):
        super(VAEDecoder, self).__init__()
        self.vae = vae
        self.decode = self.vae.decode
        self.scaling_factor = scaling_factor

    def construct(self, x):
        y = self.decode(x / self.scaling_factor)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)
        return y


class MaskLatent(nn.Cell):
    def __init__(self, scheduler):
        super(MaskLatent, self).__init__()
        self.add_noise = scheduler.add_noise
        self.alphas_cumprod = scheduler.alphas_cumprod

    def construct(self, image_latents, latents, noise, mask, ts, not_last):
        init_latents_proper = image_latents[:1]
        init_mask = mask[:1]
        add_noise_latents = self.add_noise(init_latents_proper, noise, self.alphas_cumprod[ts])
        init_latents_proper = ops.select(not_last, add_noise_latents, init_latents_proper)
        latents = (1 - init_mask) * init_latents_proper + init_mask * latents
        return latents


class DiffusionModel(nn.Cell):
    """
    Create a wrapper function for the noise prediction model.
    """

    def __init__(self, model, guidance_scale=7.5, guidance_rescale=0.0):
        super(DiffusionModel, self).__init__()
        self.model = model
        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale

    def rescale_noise_cfg(self, noise_pred, noise_pred_text, guidance_rescale=0.0):
        """
        Rescale `noise_pred` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = ops.std(noise_pred_text, axis=tuple(range(1, len(noise_pred_text.shape))), keepdims=True)
        std_cfg = ops.std(noise_pred, axis=tuple(range(1, len(noise_pred.shape))), keepdims=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_pred * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_pred = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_pred
        return noise_pred

    def construct(self, x, t_continuous, condition, unconditional_condition):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        t_continuous = ops.tile(t_continuous.reshape(1), (x.shape[0],))
        if self.guidance_scale == 1.0:
            return self.model(x, t_continuous, c_crossattn=condition)
        x_in = ops.concat([x] * 2, axis=0)
        t_in = ops.concat([t_continuous] * 2, axis=0)
        c_in = ops.concat([unconditional_condition, condition], axis=0)
        noise_pred = self.model(x_in, t_in, c_crossattn=c_in)
        noise_pred_uncond, noise_pred_text = ops.split(noise_pred, split_size_or_sections=noise_pred.shape[0] // 2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        if self.guidance_rescale > 0:
            noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
        return noise_pred
