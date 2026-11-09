# Qwen2.5-0.5B base checkpoint

This directory holds the official pretrained base model from
[Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B).

- **Model:** 0.49B parameters, BF16 weights, pretrained base (not instruction-tuned)
- **License:** Apache-2.0; see `LICENSE`
- **Pinned revision:** `060db6499f32faf8b98477b0a26969ef7d8b9987`
- **Weight file:** `model.safetensors`, 988,097,824 bytes
- **Expected SHA-256:** `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`

The model card advises post-training (such as supervised fine-tuning or
continued pretraining) for task-specific use. The weights are excluded from
Git and deployment bundles; run the download script again if they are absent.

From the repository root:

```sh
bash scripts/download-qwen-base.sh
```

The script downloads the model configuration, tokenizer files, license, and
weights from the pinned Hugging Face revision, then verifies the weight hash.
