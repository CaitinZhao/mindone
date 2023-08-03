模型导出：

```shell
python export.py --ckpt_path=[CKPT_PATH] --ddim(可选)
```

由于大sampling_steps图编译极慢，测试时可以适当减小`sampling_steps`的值，但是生成效果有影响。

图模式推理验证：

```shell
python text_to_image_speedup.py --ckpt_path=[CKPT_PATH] --ddim(可选) --ms_mode 0
```

lite推理验证流程同`text_to_image_speedup.py`
