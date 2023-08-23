from tqdm import tqdm
import numpy as np
import mindspore_lite as mslite
from .model_base import ModelBase


class SDLiteText2Img(ModelBase):
    def __init__(
        self,
        text_encoder,
        unet,
        scheduler,
        vae_decoder,
        device_target="ascend",
        device_id=0,
        guidance_scale=7.5,
        num_inference_steps=50,
    ):
        super().__init__(device_target, device_id)
        self.text_encoder = self._init_model(text_encoder)
        self.unet = self._init_model(unet)
        self.scheduler = self._init_model(scheduler)
        self.vae_decoder = self._init_model(vae_decoder)
        self.do_classifier_free_guidance = guidance_scale > 1.0
        n_infer_steps = mslite.Tensor()
        n_infer_steps.shape = []
        n_infer_steps.dtype = mslite.DataType.INT32
        n_infer_steps.set_data_from_numpy(np.array(num_inference_steps, np.int32))
        self.num_inference_steps = n_infer_steps

        # get input
        self.text_encoder_input = self.text_encoder.get_inputs()
        self.unet_input = self.unet.get_inputs()
        self.scheduler_input = self.scheduler.get_inputs()
        self.vae_decoder_input = self.vae_decoder.get_inputs()

        # resize
        # self.resize(batch_size)

    def resize(self, batch_size):
        self.text_encoder.resize(
            self.text_encoder_input,
            [
                [batch_size, *self.text_encoder_input[0].shape[1:]],
            ],
        )
        self.unet.resize(
            self.unet_input,
            [
                [batch_size, *self.unet_input[0].shape[1:]],
                self.unet_input[1].shape,
                [batch_size, *self.unet_input[2].shape[1:]],
                [batch_size, *self.unet_input[3].shape[1:]],
            ],
        )
        self.scheduler.resize(
            self.scheduler_input,
            [
                [batch_size, *self.scheduler_input[0].shape[1:]],
                self.unet_input[1].shape,
                [batch_size, *self.scheduler_input[2].shape[1:]],
                self.unet_input[3].shape,
            ],
        )
        self.vae_decoder.resize(
            self.vae_decoder_input,
            [
                [batch_size, *self.vae_decoder[0].shape[1:]],
            ],
        )

    def __call__(self, prompts, negative_prompts, start_sample, timesteps):
        self.text_encoder_input[0].set_data_from_numpy(prompts)
        prompt_embeds = self.text_encoder.predict(self.text_encoder_input)[0]
        self.text_encoder_input[0].set_data_from_numpy(negative_prompts)
        negative_prompt_embeds = self.text_encoder.predict(self.text_encoder_input)[0]
        latents = self.unet_input[0]
        latents.set_data_from_numpy(start_sample)
        iterator = tqdm(timesteps, desc="Stable Diffusion Sampling", total=len(timesteps))
        for i, t in enumerate(iterator):
            # predict the noise residual
            ts = self.unet_input[1]
            ts.set_data_from_numpy(np.array(t).astype(np.int32))
            noise_pred = self.unet.predict([latents, ts, prompt_embeds, negative_prompt_embeds])[0]
            latents = self.scheduler.predict([noise_pred, ts, latents, self.num_inference_steps])[0]
        image = self.vae_decoder.predict([latents])[0]
        image = image.get_data_to_numpy()
        return image
