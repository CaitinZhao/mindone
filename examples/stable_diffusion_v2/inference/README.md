# Stabel Diffusion Inference

## Pretrained Weights

<!---
<details close>
  <summary>Pre-trained SD weights that are compatible with MindSpore: </summary>
-->

Currently, we provide pre-trained stable diffusion model weights that are compatible with MindSpore as follows.

| **Version name** |**Task** |  **MindSpore Checkpoint**  | **Ref. Official Model** | **Resolution**|
|-----------------|---------------|---------------|------------|--------|
| 2.0            | text-to-image | [sd_v2_base-57526ee4.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_base-57526ee4.ckpt) |  [stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base) | 512x512 |
| 2.0-v768      | text-to-image | [sd_v2_768_v-e12e3a9b.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_768_v-e12e3a9b.ckpt) |  [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) | 768x768 |
| 2.0-inpaint      | image inpainting | [sd_v2_inpaint-f694d5cf.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v2_inpaint-f694d5cf.ckpt) | [stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) | 512x512|
| 1.5       | text-to-image | [sd_v1.5-d0ab7146.ckpt](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v1.5-d0ab7146.ckpt) | [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) | 512x512 |
| wukong    | text-to-image |  [wukong-huahua-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-ms.ckpt) |  | 512x512 |
| wukong-inpaint    | image |  [wukong-huahua-inpaint-ms.ckpt](https://download.mindspore.cn/toolkits/minddiffusion/wukong-huahua/wukong-huahua-inpaint-ms.ckpt) |  | 512x512 |

<!---
</details>
-->

To transfer other Stable Diffusion models to MindSpore, please refer to [model conversion](tools/model_conversion/README.md).

## Text2Image

### Online Inference

Run `sd_text2img.py` to generate images for the prompt of your interest.

```shell
python sd_text2img.py --prompt='A cute wolf in winter forest'
```

Please run `python sd_text2img.py -h` for details of command parameters.

**Note: must set `--ms_mode=1` when run on Ascend 910B.**

### Offline Inference

#### MindSpore Lite Install

<details close> 

<summary> Details of Install MindSpore Lite </summary>>

   Refer to [Lite install](https://mindspore.cn/lite/docs/zh-CN/r2.1/use/downloads.html)

   Note: MindSpore Lite applyed python3.7. Please prepare the environment for Python 3.7 before installing Lite.

   1. Download the supporting tar.gz and whl packages according to the environment.
   2. Unzip the tar.gz package and install the corresponding version of the WHL package.

   ```shell
   tar -zxvf mindspore-lite-2.1.0-*.tar.gz
   pip install mindspore_lite-2.1.0-*.whl
   ```

   3. Configure Lite's environment variables

   `LITE_HOME` is the folder path extracted from tar.gz, and it is recommended to use an absolute path.

   ```shell
   export LITE_HOME=/path/to/mindspore-lite-{version}-{os}-{platform}
   export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
   export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH
   ```

</details>

#### Inference

Step 1. Export：

```shell
python export.py --ckpt_path=[CKPT_PATH]
```

Step 2. Run `sd_text2img_lite.py` to generate images for the prompt of your interest.

```shell
python sd_text2img_lite.py --prompt='A cute wolf in winter forest' --n_iter=[ITER_COUNT]
```

Please run `python sd_text2img_lite.py -h` for details of command parameters.

## Image2Image

### Online Inference

Run `sd_img2img.py` to generate images for the prompt of your interest.

```shell
python sd_img2img.py
```

The default image is `imgs/sketch-mountains-input.jpg`, default prompt is "A fantasy landscape, trending on artstation".

Please run `python sd_text2img.py -h` for details of command parameters.

**Note: must set `--ms_mode=1` when run on Ascend 910B.**

## Inpaint

### Online Inference

Run `sd_img2img.py` to generate images for the prompt of your interest.

```shell
python sd_inpaint.py --ckpt_path=./models/sd_v2_inpaint-f694d5cf.ckpt
```

The default image is `imgs/overture-creations-5sI6fQgYIuo.png`, default mask is `imgs/overture-creations-5sI6fQgYIuo_mask.png`, 
default prompt is "Face of a yellow cat, high resolution, sitting on a park bench".

Please run `python sd_text2img.py -h` for details of command parameters.

**Note: must set `--ms_mode=1` when run on Ascend 910B.**
