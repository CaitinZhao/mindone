"""
Text to image generation
"""
import argparse
import logging
import os
import sys
import time
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import ops

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(workspace))
from ldm.modules.logger import set_logger
from ldm.util import str2bool, instantiate_from_config
from libs.helper import load_model_from_config, set_env, VaeImageProcessor
from libs.modules import VAEDecoder, DiffusionModel
from libs.sd_infer import SD_Text2Img

logger = logging.getLogger("text to image speed up")


def main(args):
    # set logger
    set_logger(name="", output_dir=args.output_path, rank=0, log_level=args.log_level)
    set_env(args)
    # create model
    config = OmegaConf.load(f"{args.config}")
    model = load_model_from_config(
        config,
        ckpt=args.ckpt_path,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_only_ckpt=args.lora_ckpt_path,
    )
    sampler_config = OmegaConf.load(args.sampler)
    scheduler = instantiate_from_config(sampler_config)
    timesteps = scheduler.set_timesteps(args.sampling_steps)
    text_encoder = model.cond_stage_model  # FrozenCLIPEmbedder
    unet = DiffusionModel(model.model, guidance_scale=args.scale)
    vae_decoder = VAEDecoder(model.first_stage_model, model.scale_factor)
    img_processor = VaeImageProcessor()
    sd_infer = SD_Text2Img(
        text_encoder, unet, scheduler, vae_decoder, guidance_scale=args.scale, num_inference_steps=args.sampling_steps
    )
    # read prompts
    batch_size = args.n_samples
    if not args.data_path:
        prompt = args.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        logger.info(f"Reading prompts from {args.data_path}")
        with open(args.data_path, "r") as f:
            prompts = f.read().splitlines()
            # TODO: try to put different prompts in a batch
            data = [batch_size * [prompt] for prompt in prompts]

    # read negative prompts
    if not args.negative_data_path:
        negative_prompt = args.negative_prompt
        assert negative_prompt is not None
        negative_data = [batch_size * [negative_prompt]]
    else:
        logger.info(f"Reading negative prompts from {args.negative_data_path}")
        with open(args.negative_data_path, "r") as f:
            negative_prompts = f.read().splitlines()
            # TODO: try to put different prompts in a batch
            negative_data = [batch_size * [negative_prompt] for negative_prompt in negative_prompts]

    # post-process negative prompts
    assert len(negative_data) <= len(data), "Negative prompts should be shorter than positive prompts"
    if len(negative_data) < len(data):
        logger.info("Negative prompts are shorter than positive prompts, padding blank prompts")
        blank_negative_prompt = batch_size * [""]
        for _ in range(len(data) - len(negative_data)):
            negative_data.append(blank_negative_prompt)

    # log
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.ms_mode}",
            "Distributed mode: False",
            f"Number of input prompts: {len(data)}",
            f"Number of input negative prompts: {len(negative_data)}",
            f"Number of trials for each prompt: {args.n_iter}",
            f"Number of samples in each trial: {args.n_samples}",
            f"Model: StableDiffusion v{args.version}",
            f"Precision: {model.model.diffusion_model.dtype}",
            f"Pretrained ckpt path: {args.ckpt_path}",
            f"Lora ckpt path: {args.lora_ckpt_path if args.use_lora else None}",
            f"Sampler: {args.sampler}",
            f"Sampling steps: {args.sampling_steps}",
            f"Uncondition guidance scale: {args.scale}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    for i, prompts in enumerate(data):
        negative_prompts = negative_data[i]
        logger.info(
            "[{}/{}] Generating images with conditions:\nPrompt(s): {}\nNegative prompt(s): {}".format(
                i + 1, len(data), prompts[0], negative_prompts[0]
            )
        )
        for n in range(args.n_iter):
            start_time = time.time()
            if isinstance(negative_prompts, tuple):
                negative_prompts = list(negative_prompts)
            tokenized_negative_prompts = model.tokenize(negative_prompts)
            if isinstance(prompts, tuple):
                prompts = list(prompts)
            tokenized_prompts = model.tokenize(prompts)
            start_sample = ops.standard_normal((args.n_samples, 4, args.H // 8, args.W // 8)).astype(ms.float16)
            x_samples = sd_infer(tokenized_prompts, tokenized_negative_prompts, start_sample, timesteps)
            x_samples = img_processor.postprocess(x_samples)

            for sample in x_samples:
                sample.save(os.path.join(args.sample_path, f"{args.base_count:05}.png"))
                args.base_count += 1

            end_time = time.time()
            logger.info(
                "{}/{} images generated, time cost for current trial: {:.3f}s".format(
                    batch_size * (n + 1), batch_size * args.n_iter, end_time - start_time
                )
            )

    logger.info(f"Done! All generated images are saved in: {args.output_path}/samples" f"\nEnjoy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument(
        "--device_target", type=str, default="Ascend", help="Device target, should be in [Ascend, GPU, CPU]"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        nargs="?",
        default="",
        help="path to a file containing prompt list (each line in the file corresponds to a prompt to render).",
    )
    parser.add_argument(
        "--negative_data_path",
        type=str,
        nargs="?",
        default="",
        help="path to a file containing negative prompt list "
             "(each line in the file corresponds to a prompt not to render).",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        nargs="?",
        default="2.0",
        help="Stable diffusion version, 1.x or 2.0. 1.x support Chinese prompts. 2.0 support English prompts.",
    )
    parser.add_argument(
        "--prompt", type=str, nargs="?", default="A cute wolf in winter forest", help="the prompt to render"
    )
    parser.add_argument("--negative_prompt", type=str, nargs="?", default="", help="the negative prompt not to render")
    parser.add_argument("--output_path", type=str, nargs="?", default="output", help="dir to write results to")
    parser.add_argument("--sampling_steps", type=int, default=50, help="number of ddim sampling steps")
    parser.add_argument("--n_iter", type=int, default=1, help="number of iterations or trials. sample this often")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt in an iteration. A.k.a. batch size",
    )
    parser.add_argument("--H", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--W", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--sampler", type=str, default="config/ddim.yaml", help="infer sampler yaml path")
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="unconditional guidance scale: eps = eps(x, uncond) + scale * (eps(x, cond) - eps(x, uncond)). "
             "Simplified: `uc + scale * (uc - prompt)`",
    )
    parser.add_argument("--from-file", type=str, help="if specified, load prompts from this file")
    parser.add_argument(
        "--config", type=str, default=None, help="path to config which constructs model. If None, select by version"
    )
    parser.add_argument(
        "--use_lora",
        default=False,
        type=str2bool,
        help="whether the checkpoint used for inference is finetuned from LoRA",
    )
    parser.add_argument(
        "--lora_rank",
        default=None,
        type=int,
        help="LoRA rank. If None, lora checkpoint should contain the value for lora rank in its append_dict.",
    )
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to checkpoint of model")
    parser.add_argument(
        "--lora_ckpt_path", type=str, default=None, help="path to lora only checkpoint. Set it if use_lora is not None"
    )
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
    parser.add_argument("--log_level", type=str, default="INFO", help="log level, options: DEBUG, INFO, WARNING, ERROR")
    args = parser.parse_args()

    # overwrite env var by parsed arg
    if args.version:
        os.environ["SD_VERSION"] = args.version
    if args.ckpt_path is None:
        args.ckpt_path = (
            "models/wukong-huahua-ms.ckpt" if args.version.startswith("1.") else "models/sd_v2_base-57526ee4.ckpt"
        )
    if args.config is None:
        args.config = (
            "config/v1-inference-chinese.yaml" if args.version.startswith("1.") else "config/v2-inference.yaml"
        )
    if args.scale is None:
        args.scale = 7.5 if args.version.startswith("1.") else 9.0
    if not os.path.isabs(args.config):
        args.config = os.path.join(workspace, args.config)
    main(args)
