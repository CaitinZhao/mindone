from time import time
from tqdm import tqdm
import mindspore as ms
from .modules import MaskLatent

class SD_Text2Img:
    def __init__(self, text_encoder, unet, scheduler, vae_decoder, guidance_scale=7.5, num_inference_steps=50):
        super(SD_Text2Img, self).__init__()
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler = scheduler
        self.vae_decoder = vae_decoder
        self.do_classifier_free_guidance = guidance_scale > 1.0
        self.num_inference_steps = ms.Tensor(num_inference_steps, ms.int32)

    def __call__(self, prompts, negative_prompts, start_sample, timesteps):
        prompt_embeds = self.text_encoder(prompts)
        negative_prompt_embeds = self.text_encoder(negative_prompts)
        latents = start_sample
        iterator = tqdm(timesteps, desc="Stable Diffusion Sampling", total=len(timesteps))
        for i, t in enumerate(iterator):
            ts = ms.Tensor(t, ms.int32)
            noise_pred = self.unet(latents, ts, prompt_embeds, negative_prompt_embeds)
            latents = self.scheduler(noise_pred, ts, latents, self.num_inference_steps)
        image = self.vae_decoder(latents)
        return image


class SD_Img2Img:
    def __init__(self, text_encoder, vae_encode, unet, scheduler, vae_decoder, guidance_scale=7.5, num_inference_steps=50):
        super(SD_Img2Img, self).__init__()
        self.text_encoder = text_encoder
        self.vae_encode = vae_encode
        self.unet = unet
        self.scheduler = scheduler
        self.vae_decoder = vae_decoder
        self.do_classifier_free_guidance = guidance_scale > 1.0
        self.num_inference_steps = ms.Tensor(num_inference_steps, ms.int32)

    def __call__(self, prompts, negative_prompts, img, noise, timesteps):
        prompt_embeds = self.text_encoder(prompts)
        negative_prompt_embeds = self.text_encoder(negative_prompts)
        t0 = ms.Tensor(timesteps[0], ms.int32)
        latents, _ = self.vae_encode(img, noise, t0)
        iterator = tqdm(timesteps, desc="Stable Diffusion Sampling", total=len(timesteps))
        for i, t in enumerate(iterator):
            ts = ms.Tensor(t, ms.int32)
            noise_pred = self.unet(latents, ts, prompt_embeds, negative_prompt_embeds)
            latents = self.scheduler(noise_pred, ts, latents, self.num_inference_steps)
        image = self.vae_decoder(latents)
        return image


class SD_Inpaint:
    def __init__(self, text_encoder, vae_encode, unet, scheduler, vae_decoder,
                 guidance_scale=7.5, num_inference_steps=50):
        super(SD_Inpaint, self).__init__()
        self.text_encoder = text_encoder
        self.vae_encode = vae_encode
        self.unet = unet
        self.scheduler = scheduler
        self.vae_decoder = vae_decoder
        self.do_classifier_free_guidance = guidance_scale > 1.0
        self.num_inference_steps = ms.Tensor(num_inference_steps, ms.int32)
        self.init_noise_sigma = self.scheduler.init_noise_sigma
        self.mask_latent = MaskLatent(scheduler)

    def __call__(self, prompts, negative_prompts, img, noise, mask, timesteps):
        prompt_embeds = self.text_encoder(prompts)
        negative_prompt_embeds = self.text_encoder(negative_prompts)
        t0 = ms.Tensor(timesteps[0], ms.int32)
        latents, image_latents = self.vae_encode(img, noise, t0)
        iterator = tqdm(timesteps, desc="Stable Diffusion Sampling", total=len(timesteps))
        for i, t in enumerate(iterator):
            ts = ms.Tensor(t, ms.int32)
            noise_pred = self.unet(latents, ts, prompt_embeds, negative_prompt_embeds)
            latents = self.scheduler(noise_pred, ts, latents, self.num_inference_steps)
            noise_timestep = ms.Tensor(timesteps[min(i + 1, len(timesteps) - 1)], ms.int32)
            not_last = ms.Tensor(i < len(timesteps) - 1)
            latents = self.mask_latent(image_latents, latents, noise, mask, noise_timestep, not_last)

        image = self.vae_decoder(latents)
        return image
