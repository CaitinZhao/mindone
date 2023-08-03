from mindspore import nn, ops
from .diffusion import DDIMSamplerInfer, PLMSSamplerInfer


class SD_Infer(nn.Cell):
    def __init__(self, args, model):
        super(SD_Infer, self).__init__()
        # create sampler
        shape = [4, args.H // 8, args.W // 8]
        if args.ddim:
            sampler = DDIMSamplerInfer(model, args.sampling_steps, args.n_samples,
                                       shape, ddim_discretize="uniform", ddim_eta=args.ddim_eta,
                                       verbose=False, temperature=1.0, noise_dropout=0.0,
                                       unconditional_guidance_scale=args.scale, log_every_t=100,
                 )
            self.sname = "ddim"
        else:
            sampler = PLMSSamplerInfer(model, args.sampling_steps, args.n_samples,
                                       shape, ddim_discretize="uniform", ddim_eta=args.ddim_eta,
                                       verbose=False, temperature=1.0, noise_dropout=0.0,
                                       unconditional_guidance_scale=args.scale, log_every_t=100,
                 )
            self.sname = "plms"
        self.sampler = sampler
        self.get_learned_conditioning = sampler.model.get_learned_conditioning
        self.decode_first_stage = sampler.model.decode_first_stage
        self.sample = sampler.sample
        self.scale = args.scale
        self.n_samples = args.n_samples
        self.H = args.H
        self.W = args.W
        self.sampling_steps = args.sampling_steps
        self.ddim_eta = args.ddim_eta

    def construct(self, prompts, negative_prompts, start_code=None):
        if start_code is None:
            start_code = ops.standard_normal((self.n_samples, 4, self.H // 8, self.W // 8))
        uc = None
        if self.scale != 1.0:
            uc = self.get_learned_conditioning(negative_prompts)
        c = self.get_learned_conditioning(prompts)
        samples_ddim, _ = self.sample(
            conditioning=c,
            unconditional_conditioning=uc,
            x_T=start_code,
        )
        x_samples_ddim = self.decode_first_stage(samples_ddim)
        x_samples_ddim = ops.clip_by_value((x_samples_ddim + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)
        return x_samples_ddim
