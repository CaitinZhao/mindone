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
from ldm.util import str2bool, instantiate_from_config
from libs.helper import load_model_from_config, set_env
from libs.modules import VAEDecoder, DiffusionModel

logger = logging.getLogger("text_to_image")


def main(args):
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
    text_encoder = model.cond_stage_model  # FrozenCLIPEmbedder
    unet = DiffusionModel(model.model, guidance_scale=args.scale)
    vae_decoder = VAEDecoder(model.first_stage_model, model.scale_factor)
    tokenized_prompts = ops.ones((1, 77), ms.int32)
    output_dim = 768 if args.version.startswith("1.") else 1024
    prompt_embeds = ops.ones((1, 77, output_dim), ms.float16)
    latents = ops.ones((1, 4, args.H // 8, args.W // 8), ms.float16)
    ts = ops.ones((), ms.int32)
    model_save_path = os.path.join(args.output_path, args.model_save_path)
    os.makedirs(model_save_path, exist_ok=True)
    ms.export(
        text_encoder,
        tokenized_prompts,
        file_name=os.path.join(model_save_path, "text_encoder.mindir"),
        file_format="MINDIR",
    )
    logger.info("export text_encoder mindir done")
    ms.export(
        unet,
        latents,
        ts,
        prompt_embeds,
        prompt_embeds,
        file_name=os.path.join(model_save_path, "unet.mindir"),
        file_format="MINDIR",
    )
    logger.info("export unet mindir done")
    ms.export(
        scheduler,
        latents,
        ts,
        latents,
        ts,
        file_name=os.path.join(model_save_path, f"{sampler_config.type}_scheduler.mindir"),
        file_format="MINDIR",
    )
    logger.info("export scheduler mindir done")
    ms.export(vae_decoder, latents, file_name=os.path.join(model_save_path, "vae_decoder.mindir"), file_format="MINDIR")
    logger.info("export vae_decoder mindir done")
    model_save_path = os.path.join(args.output_path, args.model_save_path)
    if args.converte_lite:
        import mindspore_lite as mslite
        optimize_dict = {"ascend": "ascend_oriented", "gpu": "gpu_oriented", "cpu": "general"}
        converter = mslite.Converter()
        converter.save_type = mslite.ModelType.MINDIR
        converter.optimize = optimize_dict[args.device_target.lower()]
        text_encoder_mindir = os.path.join(model_save_path, "text_encoder.mindir")
        if not os.path.isfile(text_encoder_mindir):
            text_encoder_mindir = os.path.join(model_save_path, "text_encoder_graph.mindir")
        converter.convert(
            fmk_type=mslite.FmkType.MINDIR,
            model_file=text_encoder_mindir,
            output_file=os.path.join(model_save_path, "text_encoder_lite"),
        )
        logger.info("convert text_encoder lite mindir done")
        unet_mindir = os.path.join(model_save_path, "unet.mindir")
        if not os.path.isfile(unet_mindir):
            unet_mindir = os.path.join(model_save_path, "unet_graph.mindir")
        converter.convert(
            fmk_type=mslite.FmkType.MINDIR,
            model_file=unet_mindir,
            output_file=os.path.join(model_save_path, "unet_lite"),
        )
        logger.info("convert unet lite mindir done")
        converter.convert(
            fmk_type=mslite.FmkType.MINDIR,
            model_file=os.path.join(model_save_path, f"{sampler_config.type}_scheduler.mindir"),
            output_file=os.path.join(model_save_path, f"{sampler_config.type}_scheduler_lite"),
        )
        logger.info("convert scheduler lite mindir done")
        vae_decoder_mindir = os.path.join(model_save_path, "vae_decoder.mindir")
        if not os.path.isfile(unet_mindir):
            vae_decoder_mindir = os.path.join(model_save_path, "vae_decoder_graph.mindir")
        converter.convert(
            fmk_type=mslite.FmkType.MINDIR,
            model_file=vae_decoder_mindir,
            output_file=os.path.join(model_save_path, "vae_decoder_lite"),
        )
        logger.info("convert vae_decoder lite mindir done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument(
        "--device_target", type=str, default="Ascend", help="Device target, should be in [Ascend, GPU, CPU]"
    )
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
    parser.add_argument("--converte_lite", default=True, type=str2bool, help="whether convert lite mindir")
    parser.add_argument("--model_save_path", type=str, nargs="?", default="models", help="dir to write mindir")
    parser.add_argument("--output_path", type=str, nargs="?", default="output", help="dir to write results to")
    parser.add_argument("--H", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--W", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--sampler", type=str, default="config/ddim.yaml", help="infer sampler yaml path")
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="unconditional guidance scale: "
        "eps = eps(x, uncond) + scale * (eps(x, cond) - eps(x, uncond)). "
        "Simplified: `uc + scale * (uc - prompt)`",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="path to config which constructs model. If None, select by version"
    )
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to checkpoint of model")
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
        help="LoRA rank. If None, " "lora checkpoint should contain the value for lora rank in its append_dict.",
    )
    parser.add_argument(
        "--lora_ckpt_path", type=str, default=None, help="path to lora only checkpoint. Set it if use_lora is not None"
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="log level, options: DEBUG, INFO, WARNING, ERROR")
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        nargs="?",
        default="2.0",
        help="Stable diffusion version, 1.x or 2.0. " "1.x support Chinese prompts. 2.0 support English prompts.",
    )
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
