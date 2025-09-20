# coding=utf-8
# LoRA finetuning for Stable Diffusion using CSV dataset

import argparse
import logging
import os
import math
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model, TaskType
from safetensors.torch import save_file
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",  # 시간 포함
    datefmt="%Y-%m-%d %H:%M:%S"  # 시간 포맷 (원하는 대로 수정 가능)
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA training for Stable Diffusion")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--caption_column", type=str, default="text")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=2500)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    # LoRA 설정
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    logger.info("Loading models...")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # LoRA 주입
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_v"],  # 주로 attention q,v에 적용
        lora_dropout=args.lora_dropout,
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)

    # Dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("csv", data_files={"train": args.train_file})
    logger.info(f"Loaded dataset with {len(dataset['train'])} samples")
    print("DEBUG >>> Dataset size:", len(dataset["train"]))

    # Preprocess
    preprocess = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def transform_fn(examples):
        images = []
        for path in examples[args.image_column]:
            image = Image.open(path).convert("RGB")
            images.append(preprocess(image))

        tokens = tokenizer(
            examples[args.caption_column],
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        return {"pixel_values": images, "input_ids": tokens.input_ids}

    dataset = dataset["train"].with_transform(transform_fn)
    train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    print("DEBUG >>> Length of train_dataloader:", len(train_dataloader))

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    # Training loop
    logger.info("Starting training...")
    global_step = 0
    unet.train()

    for epoch in range(999999):
        for step, batch in enumerate(train_dataloader):
            global_step += 1

            pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float32)
            input_ids = batch["input_ids"].to(accelerator.device)

            # 1. Encode images to latent space
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215

            # 2. Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=latents.device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 3. Encode text
            encoder_hidden_states = text_encoder(input_ids)[0]

            # 4. Predict noise with UNet
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # 5. Loss
            loss = torch.nn.functional.mse_loss(model_pred, noise)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if global_step % 50 == 0:
                logger.info(f"Step {global_step} - Loss {loss.item():.4f}")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    # Save LoRA weights
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        # safetensors 저장
        save_path = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
        save_file(unet.state_dict(), save_path)
        logger.info(f"LoRA weights saved to {save_path}")


if __name__ == "__main__":
    main()
