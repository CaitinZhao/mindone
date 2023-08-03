import numpy as np
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from ldm.util import extract_into_tensor
from tqdm import tqdm

import mindspore as ms
from mindspore import ops, nn


class DDIMSamplerInfer(nn.Cell):
    def __init__(self, model, ddim_num_steps, batch_size, shape, ddim_discretize="uniform", ddim_eta=0.0, verbose=True,
                 temperature=1.0, noise_dropout=0.0, unconditional_guidance_scale=1.0, log_every_t=100,
                 schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.split = ops.Split(0, 2)

        # make_schedule
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, "alphas have to be defined for each timestep"
        self.batch_size = batch_size
        self.shape = shape
        self.temperature = temperature
        self.noise_dropout = noise_dropout
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.log_every_t = log_every_t
        self.betas = self.model.betas
        self.alphas_cumprod = self.model.alphas_cumprod
        self.alphas_cumprod_prev = self.model.alphas_cumprod_prev

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = ops.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = ops.sqrt(1.0 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = ops.log(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = ops.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = ops.sqrt(1.0 / alphas_cumprod - 1)

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod, ddim_timesteps=self.ddim_timesteps, eta=ddim_eta, verbose=verbose
        )

        self.ddim_sigmas = ddim_sigmas
        self.ddim_alphas = ddim_alphas
        self.ddim_alphas_prev = ddim_alphas_prev
        self.ddim_sqrt_one_minus_alphas = ops.sqrt(1.0 - ddim_alphas)
        sigmas_for_original_sampling_steps = ddim_eta * ops.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.ddim_sigmas_for_original_num_steps = sigmas_for_original_sampling_steps

    def sample(self, conditioning=None, unconditional_conditioning=None, x_T=None):
        # sampling
        C, H, W = self.shape
        size = (self.batch_size, C, H, W)
        samples, intermediates = self.ddim_sampling(
            conditioning,
            size,
            noise_dropout=self.noise_dropout,
            temperature=self.temperature,
            log_every_t=self.log_every_t,
            unconditional_guidance_scale=self.unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            x_T=x_T
        )
        return samples, intermediates

    def ddim_sampling(
        self,
        cond,
        shape,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        x_T=None
    ):
        b = shape[0]
        if x_T is None:
            img = ops.standard_normal(shape)
        else:
            img = x_T
        timesteps = self.ddim_timesteps

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = ms.numpy.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = time_range

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = ms.numpy.full((b,), step, dtype=ms.int64)

            outs = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                temperature=temperature,
                noise_dropout=noise_dropout,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
            img, pred_x0 = outs

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)

        return img, intermediates

    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        temperature=1.0,
        noise_dropout=0.0,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):
        b = x.shape[0]

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            model_output = self.model.apply_model(x, t, c_crossattn=c)
        else:
            x_in = ops.concat((x, x), axis=0)
            t_in = ops.concat((t, t), axis=0)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [
                            ops.concat([unconditional_conditioning[k][i], c[k][i]]) for i in range(len(c[k]), axis=0)
                        ]
                    else:
                        c_in[k] = ops.concat([unconditional_conditioning[k], c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(ops.concat([unconditional_conditioning[i], c[i]], axis=0))
            else:
                c_in = ops.concat([unconditional_conditioning, c], axis=0)
            model_uncond, model_t = self.split(self.model.apply_model(x_in, t_in, c_crossattn=c_in))
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = ms.numpy.full((b, 1, 1, 1), alphas[index])
        a_prev = ms.numpy.full((b, 1, 1, 1), alphas_prev[index])
        sigma_t = ms.numpy.full((b, 1, 1, 1), sigmas[index])
        sqrt_one_minus_at = ms.numpy.full((b, 1, 1, 1), sqrt_one_minus_alphas[index])

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise, _ = ops.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0

    def encode(
        self,
        x0,
        c,
        t_enc,
        use_original_steps=False,
        return_intermediates=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        callback=None,
    ):
        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = self.ddim_alphas_prev[:num_steps]

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in range(num_steps):
            t = ms.numpy.full((x0.shape[0],), i, dtype=ms.int64)
            if unconditional_guidance_scale == 1.0:
                noise_pred = self.model.apply_model(x_next, t, c_crossattn=c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond = self.model.apply_model(x_next, t, c_crossattn=unconditional_conditioning)
                noise_pred = self.model.apply_model(x_next, t, c_crossattn=c)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = (
                alphas_next[i].sqrt() * ((1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            )
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback:
                callback(i)

        out = {"x_encoded": x_next, "intermediate_steps": inter_steps}
        if return_intermediates:
            out.update({"intermediates": intermediates})
        return x_next, out

    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = ops.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = ms.numpy.randn(x0.shape)
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def decode(
        self,
        x_latent,
        cond,
        t_start,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):
        timesteps = self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc="Decoding image", total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = ms.numpy.full((x_latent.shape[0],), step, dtype=ms.int64)
            x_dec, _ = self.p_sample_ddim(
                x_dec,
                cond,
                ts,
                index=index,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
        return x_dec
