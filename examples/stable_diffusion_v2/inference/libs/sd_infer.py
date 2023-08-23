from abc import ABC

from tqdm import tqdm

import mindspore as ms
from mindspore import nn

from .modules import MaskLatent


class SDBaseInfer(ABC):
    def __init__(
        self,
        text_encoder: nn.Cell,
        unet: nn.Cell,
        scheduler,
        vae_decoder: nn.Cell,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
    ):
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler = scheduler
        self.vae_decoder = vae_decoder
        self.do_classifier_free_guidance = guidance_scale > 1.0
        self.num_inference_steps = ms.Tensor(num_inference_steps, ms.int32)

    def _infer(
        self, prompts, negative_prompts, latents, timesteps: ms.Tensor, image_latents=None, noise=None, mask=None
    ):
        prompt_embeds = self.text_encoder(prompts)
        negative_prompt_embeds = self.text_encoder(negative_prompts)

        iterator = tqdm(timesteps, desc="Stable Diffusion Sampling", total=len(timesteps))
        for i, ts in enumerate(iterator):
            noise_pred = self.unet(latents, ts, prompt_embeds, negative_prompt_embeds)
            latents = self.scheduler(noise_pred, ts, latents, self.num_inference_steps)

            if image_latents is not None:
                noise_timestep = timesteps[min(i + 1, len(timesteps) - 1)]
                not_last = ms.Tensor(i < (len(timesteps) - 1))
                latents = self.mask_latent(image_latents, latents, noise, mask, noise_timestep, not_last)  # NOQA

        image = self.vae_decoder(latents)
        return image


class SDText2Img(SDBaseInfer):
    def __init__(
        self,
        text_encoder,
        unet,
        scheduler,
        vae_decoder,
        guidance_scale=7.5,
        num_inference_steps=50,
    ):
        super().__init__(text_encoder, unet, scheduler, vae_decoder, guidance_scale, num_inference_steps)

    def __call__(self, prompts, negative_prompts, start_sample, timesteps):
        timesteps = ms.Tensor(timesteps, ms.int32)
        self._infer(prompts, negative_prompts, start_sample, timesteps)


class SDImg2Img(SDBaseInfer):
    def __init__(
        self,
        text_encoder,
        vae_encode,
        unet,
        scheduler,
        vae_decoder,
        guidance_scale=7.5,
        num_inference_steps=50,
    ):
        super().__init__(text_encoder, unet, scheduler, vae_decoder, guidance_scale, num_inference_steps)
        self.vae_encode = vae_encode

    def __call__(self, prompts, negative_prompts, img, noise, timesteps):
        timesteps = ms.Tensor(timesteps, ms.int32)
        latents, _ = self.vae_encode(img, noise, timesteps[0])
        self._infer(prompts, negative_prompts, latents, timesteps)


class SDInpaint(SDBaseInfer):
    def __init__(
        self,
        text_encoder,
        vae_encode,
        unet,
        scheduler,
        vae_decoder,
        guidance_scale=7.5,
        num_inference_steps=50,
    ):
        super().__init__(text_encoder, unet, scheduler, vae_decoder, guidance_scale, num_inference_steps)
        self.vae_encode = vae_encode
        self.init_noise_sigma = self.scheduler.init_noise_sigma
        self.mask_latent = MaskLatent(scheduler)

    def __call__(self, prompts, negative_prompts, img, noise, mask, timesteps):
        timesteps = ms.Tensor(timesteps, ms.int32)
        latents, image_latents = self.vae_encode(img, noise, timesteps[0])
        self._infer(prompts, negative_prompts, latents, timesteps, image_latents, noise, mask)
