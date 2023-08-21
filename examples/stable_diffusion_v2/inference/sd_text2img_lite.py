"""
Text to image generation
"""
import argparse
import logging
import os
import sys
import time
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import mindspore as ms

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(workspace))

from ldm.models.clip.simple_tokenizer import get_tokenizer
from ldm.modules.logger import set_logger
from libs.infer_engine.sd_lite_text2img import SDLiteText2Img
from ldm.util import instantiate_from_config
from libs.helper import VaeImageProcessor

logger = logging.getLogger("text to image speed up")


def tokenize(tokenizer, texts):
    SOT_TEXT = tokenizer.sot_text
    EOT_TEXT = tokenizer.eot_text
    CONTEXT_LEN = 77

    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder[SOT_TEXT]
    eot_token = tokenizer.encoder[EOT_TEXT]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = np.zeros((len(all_tokens), CONTEXT_LEN), np.int64)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > CONTEXT_LEN:
            tokens = tokens[: CONTEXT_LEN - 1] + [eot_token]

        result[i, : len(tokens)] = np.array(tokens, np.int64)

    return result.astype(np.int32)

def main(args):
    # set logger
    set_logger(name="", output_dir=args.output_path, rank=0, log_level=args.log_level)
    ms.set_context(device_target="CPU")
    args.sample_path = os.path.join(args.output_path, "samples")
    args.base_count = len(os.listdir(args.sample_path))
    model_save_path = os.path.join(args.output_path, args.model_save_path)
    text_encoder_mindir = os.path.join(model_save_path, args.text_encoder)
    unet_mindir = os.path.join(model_save_path, args.unet)
    scheduler_mindir = os.path.join(model_save_path, args.scheduler)
    sampler_config = OmegaConf.load(args.sampler)
    scheduler = instantiate_from_config(sampler_config)
    timesteps = scheduler.set_timesteps(args.sampling_steps)
    vae_decoder_mindir = os.path.join(model_save_path, args.vae_decoder)
    sd_infer = SDLiteText2Img(text_encoder_mindir,
                              unet_mindir,
                              scheduler_mindir,
                              vae_decoder_mindir,
                              device_target=args.device_target,
                              device_id=int(os.getenv("DEVICE_ID", 0)),
                              guidance_scale=args.scale,
                              num_inference_steps=args.sampling_steps)
    img_processor = VaeImageProcessor()
    # read prompts
    batch_size = 1
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
            f"MindSpore Lite inference",
            f"Number of input prompts: {len(data)}",
            f"Number of input negative prompts: {len(negative_data)}",
            f"Number of trials for each prompt: {args.n_iter}",
            f"Number of samples in each trial: 1",
            f"Model: StableDiffusion v{args.version}",
            f"text_encoder_mindir: {os.path.join(model_save_path, args.text_encoder)}",
            f"unet_encoder_mindir: {os.path.join(model_save_path, args.unet)}",
            f"scheduler_mindir: {os.path.join(model_save_path, args.scheduler)}",
            f"vae_decoder_mindir: {os.path.join(model_save_path, args.vae_decoder)}",
            f"Sampling steps: {args.sampling_steps}",
            f"Uncondition guidance scale: {args.scale}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)
    tokenizer = get_tokenizer(args.version)
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
            tokenized_negative_prompts = tokenize(tokenizer, negative_prompts)
            if isinstance(prompts, tuple):
                prompts = list(prompts)
            tokenized_prompts = tokenize(tokenizer, prompts)
            start_sample = np.random.standard_normal(
                size=(1, 4, args.H // 8, args.W // 8)).astype(np.float16)
            x_samples = sd_infer(tokenized_prompts, tokenized_negative_prompts, start_sample, timesteps)
            images = img_processor.postprocess(x_samples)

            for img in images:
                img.save(os.path.join(args.sample_path, f"{args.base_count:05}.png"))
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
        "--device_target", type=str, default="Ascend", help="Device target, should be in [Ascend, GPU, CPU]"
    )
    parser.add_argument("--text_encoder", type=str, default="text_encoder_lite.mindir",
                        help="The path of MindSpore Lite MindIR for text_encoder.")
    parser.add_argument("--unet", type=str, default="unet_lite_graph.mindir",
                        help="The path of MindSpore Lite MindIR for unet.")
    parser.add_argument("--scheduler", type=str, default="ddim_scheduler_lite.mindir",
                        help="The path of MindSpore Lite MindIR for sampler scheduler.")
    parser.add_argument("--vae_decoder", type=str, default="vae_decoder_lite.mindir",
                        help="The path of MindSpore Lite MindIR for vae_decoder.")
    parser.add_argument("--sampler", type=str, default="config/ddim.yaml", help="infer sampler yaml path")
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
        "--prompt", type=str, nargs="?", default="A cute wolf in winter forest", help="the prompt to render"
    )
    parser.add_argument("--negative_prompt", type=str, nargs="?", default="", help="the negative prompt not to render")
    parser.add_argument("--model_save_path", type=str, nargs="?", default="models", help="dir to write mindir")
    parser.add_argument("--output_path", type=str, nargs="?", default="output", help="dir to write results to")
    parser.add_argument("--sampling_steps", type=int, default=50, help="number of ddim sampling steps")
    parser.add_argument("--n_iter", type=int, default=1, help="number of iterations or trials. sample this often, ")
    parser.add_argument("--H", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--W", type=int, default=512, help="image width, in pixel space")
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="unconditional guidance scale: eps = eps(x, uncond) + scale * (eps(x, cond) - eps(x, uncond)). "
             "Simplified: `uc + scale * (uc - prompt)`",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        nargs="?",
        default="2.0",
        help="Stable diffusion version, 1.x or 2.0. " "1.x support Chinese prompts. 2.0 support English prompts.",
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="log level, options: DEBUG, INFO, WARNING, ERROR")
    args = parser.parse_args()
    if args.scale is None:
        args.scale = 7.5 if args.version.startswith("1.") else 9.0
    main(args)
