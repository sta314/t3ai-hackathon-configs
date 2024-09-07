---
library_name: transformers
license: other
base_model: pt-model
tags:
- llama-factory
- full
- generated_from_trainer
model-index:
- name: train_2024-09-06-08-14-32
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# train_2024-09-06-08-14-32

This model is a fine-tuned version of [pt-model](https://huggingface.co/pt-model) on the turkish_law dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- gradient_accumulation_steps: 8
- total_train_batch_size: 512
- total_eval_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- num_epochs: 10.0

### Training results

### Framework versions

- Transformers 4.44.2
- Pytorch 2.4.1+cu121
- Datasets 2.21.0
- Tokenizers 0.19.1
