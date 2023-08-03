"""
Text to image generation
"""
import argparse
import logging
import os
import sys
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import ops

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(workspace))

from ldm.modules.logger import set_logger
from ldm.modules.lora import inject_trainable_lora
from ldm.modules.train.tools import set_random_seed
from ldm.util import instantiate_from_config, str2bool
from utils import model_utils
from speedup_ldm.sd_infer import SD_Infer

logger = logging.getLogger("text_to_image")


def load_model_from_config(config, ckpt, use_lora=False, lora_rank=4, lora_fp16=True, lora_only_ckpt=None):
    model = instantiate_from_config(config.model)

    def _load_model(_model, ckpt_fp, verbose=True, filter=None):
        if os.path.exists(ckpt_fp):
            param_dict = ms.load_checkpoint(ckpt_fp)
            if param_dict:
                param_not_load, ckpt_not_load = model_utils.load_param_into_net_with_filter(
                    _model, param_dict, filter=filter
                )
                if verbose:
                    if len(param_not_load) > 0:
                        logger.info(
                            "Net params not loaded: {}".format([p for p in param_not_load if not p.startswith("adam")])
                        )
        else:
            logger.error(f"!!!Error!!!: {ckpt_fp} doesn't exist")
            raise FileNotFoundError(f"{ckpt_fp} doesn't exist")

    if use_lora:
        load_lora_only = True if lora_only_ckpt is not None else False
        if not load_lora_only:
            logger.info(f"Loading model from {ckpt}")
            _load_model(model, ckpt)
        else:
            if os.path.exists(lora_only_ckpt):
                lora_param_dict = ms.load_checkpoint(lora_only_ckpt)
                if "lora_rank" in lora_param_dict.keys():
                    lora_rank = int(lora_param_dict["lora_rank"].value())
                    logger.info(f"Lora rank is set to {lora_rank} according to the found value in lora checkpoint.")
            else:
                raise ValueError(f"{lora_only_ckpt} doesn't exist")
            # load the main pretrained model
            logger.info(f"Loading pretrained model from {ckpt}")
            _load_model(model, ckpt, verbose=True, filter=ms.load_checkpoint(ckpt).keys())
            # inject lora params
            injected_attns, injected_trainable_params = inject_trainable_lora(
                model,
                rank=lora_rank,
                use_fp16=(model.model.diffusion_model.dtype == ms.float16),
            )
            # load fine-tuned lora params
            logger.info(f"Loading LoRA params from {lora_only_ckpt}")
            _load_model(model, lora_only_ckpt, verbose=True, filter=injected_trainable_params.keys())
    else:
        logger.info(f"Loading model from {ckpt}")
        _load_model(model, ckpt)

    model.set_train(False)
    for param in model.trainable_params():
        param.requires_grad = False

    return model


def main(args):
    # set logger
    set_logger(
        name="",
        output_dir=args.output_path,
        rank=0,
        log_level=eval(args.log_level),
    )

    work_dir = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"WORK DIR:{work_dir}")
    os.makedirs(args.output_path, exist_ok=True)
    outpath = args.output_path

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

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    # set ms context
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.context.set_context(mode=args.ms_mode, device_target="Ascend", device_id=device_id)

    set_random_seed(args.seed)

    # create model
    if not os.path.isabs(args.config):
        args.config = os.path.join(work_dir, args.config)
    config = OmegaConf.load(f"{args.config}")
    model = load_model_from_config(
        config,
        ckpt=args.ckpt_path,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_only_ckpt=args.lora_ckpt_path,
    )

    sd_infer = SD_Infer(args, model)
    stdnormal = ops.StandardNormal()
    start_code = stdnormal((args.n_samples, 4, args.H // 8, args.W // 8))
    tokenized_negative_prompts = model.tokenize([""] * args.n_samples)
    tokenized_prompts = model.tokenize(["A cute wolf"] * args.n_samples)
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
            f"Sampler: {sd_infer.sname}",
            f"Sampling steps: {args.sampling_steps}",
            f"Uncondition guidance scale: {args.scale}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)
    ms.export(sd_infer, tokenized_prompts, tokenized_negative_prompts, start_code,
              file_name=f'sdv{args.version}_{sd_infer.sname}_bs_{args.n_samples}_'
                        f'samplestep_{args.sampling_steps}_scale{args.scale}',
              file_format='MINDIR')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
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
        help="path to a file containing negative prompt list (each line in the file corresponds to a prompt not to "
        "render).",
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
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action="store_true",
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="number of iterations or trials. sample this often, ",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="how many samples to produce for each given prompt in an iteration. A.k.a. batch size",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        help="use ddim sampling",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="unconditional guidance scale: eps = eps(x, uncond) + scale * (eps(x, cond) - eps(x, uncond)). "
        "Simplified: `uc + scale * (uc - prompt)`",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to config which constructs model. If None, select by version",
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
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--lora_ckpt_path",
        type=str,
        default=None,
        help="path to lora only checkpoint. Set it if use_lora is not None",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )
    args = parser.parse_args()

    # overwrite env var by parsed arg
    if args.version:
        os.environ["SD_VERSION"] = args.version
    if args.ckpt_path is None:
        args.ckpt_path = (
            "../models/wukong-huahua-ms.ckpt" if args.version.startswith("1.") else "../models/sd_v2_base-57526ee4.ckpt"
        )
    if args.config is None:
        args.config = (
            "../configs/v1-inference-chinese.yaml" if args.version.startswith("1.") else "../configs/v2-inference.yaml"
        )
    if args.scale is None:
        args.scale = 7.5 if args.version.startswith("1.") else 9.0

    # core task
    main(args)
