#!/bin/zsh

accelerate launch --config_file $1 src/train_conditional.py \
  --train_data_dir="path/to/data" \
  --resolution=128 \
  --output_dir="/path/to/output/dir" \
  --train_batch_size=64 \
  --eval_batch_size=64 \
  --num_epochs=3000 \
  --gradient_accumulation_steps=2 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=fp16 \
  --enable_xformers_memory_efficient_attention \
  --logger=wandb \
  --save_model_epochs=100 \
  --save_images_epochs=100 \
  --nb_generated_images=10000 \
  --nb_classes=2 \
  --ddim_num_inference_steps=100 \
  --random_flip \
  --use_pytorch_loader \
  --resume_from_checkpoint="latest" \
  --checkpoints_total_limit=5 \
  --checkpointing_steps=1000 \
  --ddim_beta_schedule="cosine" \

# Call `train_conditional.py --help` to get the full list of args!
