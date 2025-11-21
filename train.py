import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import math

from data_utils import OCRDataset, load_all_data
from model_utils import load_model
from logger import get_logger

def collate_fn(batch, processor):
    # Filter None
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    # Prepare lists
    input_ids_list = []
    labels_list = []
    pixel_values_list = []
    images_crop_list = []
    images_spatial_crop_list = []
    images_seq_mask_list = []
    
    for item in batch:
        image = item['image']
        prompt = item['prompt']
        ground_truth = item['ground_truth']
        
        # Ensure prompt has <image> tag
        if '<image>' not in prompt:
            prompt = '<image>\n' + prompt
            
        full_text = prompt + ground_truth
        
        # Tokenize prompt only (to find length for masking)
        # We use eos=False to avoid EOS token at the end of prompt
        prompt_outputs = processor.tokenize_with_images([image], prompt=prompt, bos=True, eos=False)
        # prompt_outputs is [[input_ids, pixel_values, ...]]
        prompt_ids = prompt_outputs[0][0].squeeze(0) # (seq_len)
        prompt_len = prompt_ids.size(0)
        
        # Tokenize full text
        full_outputs = processor.tokenize_with_images([image], prompt=full_text, bos=True, eos=True)
        input_ids = full_outputs[0][0].squeeze(0) # (seq_len)
        pixel_values = full_outputs[0][1] # (n_images, 3, H, W) or (1, 3, H, W)
        images_crop = full_outputs[0][2]
        images_seq_mask = full_outputs[0][3]
        images_spatial_crop = full_outputs[0][4]
        
        # Create labels
        labels = input_ids.clone()
        # Mask prompt
        if prompt_len < labels.size(0):
            labels[:prompt_len] = -100
        else:
            # Should not happen if ground_truth is not empty
            print("Warning: Prompt length >= Full length")
            labels[:] = -100
            
        # Mask padding/image tokens (already handled by processor in target_ids? No, processor returns input_ids and target_ids but we only unpacked input_ids)
        # Wait, processor.tokenize_with_images returns a list.
        # Let's check `tokenize_with_images` return value in `model_utils.py`.
        # It returns `[[input_ids, pixel_values, images_crop, images_seq_mask, images_spatial_crop, num_image_tokens, image_shapes]]`
        # It does NOT return `target_ids`.
        # But inside `tokenize_with_images`, it calculates `target_ids`.
        # "input_ids = torch.LongTensor(tokenized_str)"
        # "target_ids = torch.LongTensor(masked_tokenized_str)"
        # But it doesn't return `target_ids`!
        # I missed that in `model_utils.py`.
        # I should verify `model_utils.py` content.
        
        # In `model_utils.py`:
        # return [[input_ids, pixel_values, images_crop, images_seq_mask, images_spatial_crop, num_image_tokens, image_shapes]]
        
        # So I need to manually mask image tokens in `labels`.
        image_token_id = processor.image_token_id
        labels[input_ids == image_token_id] = -100
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        pixel_values_list.append(pixel_values)
        images_crop_list.append(images_crop)
        images_spatial_crop_list.append(images_spatial_crop)
        images_seq_mask_list.append(images_seq_mask)

    # Pad sequences
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=processor.pad_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
    
    # Stack other tensors
    # pixel_values: (B, N_images, 3, H, W) -> (B*N_images, 3, H, W) ?
    # The model expects `pixel_values` as a list or stacked?
    # In `deepseek_ocr.py`: `pixel_values = kwargs.pop("pixel_values", None)`
    # `_pixel_values_to_embedding` takes `pixel_values` as `[n_image, batch_size, 3, height, width]`?
    # Wait, `deepseek_ocr.py` line 371: `# Pixel_values (global view): [n_image, batch_size, 3, height, width]`
    # This comment seems to imply `n_image` is first dimension?
    # But `process_one` returns `pixel_values` as `(n_images, 3, H, W)`.
    # If we have a batch, we usually stack them.
    # Let's assume the model expects `(Total_Images, 3, H, W)` or `(B, N_images, 3, H, W)`.
    # `deepseek_ocr.py` line 475: `pixel_values = image_input[0].to(torch.bfloat16)`
    # `image_input` comes from `_parse_and_validate_image_input`.
    # It expects `pixel_values` to be passed in kwargs.
    
    # Let's look at `_pixel_values_to_embedding` again.
    # `for jdx in range(images_spatial_crop.size(0)):`
    # `image_ori = pixel_values[jdx]`
    # This implies `pixel_values` has same first dim as `images_spatial_crop`.
    # `images_spatial_crop` usually has shape `(B, N_images, 2)`?
    # In `tokenize_with_images`, `images_spatial_crop` is `(N_images, 2)`.
    # So if we stack them, we get `(B, N_images, 2)`.
    # Then `pixel_values` should be `(B, N_images, 3, H, W)`.
    # But `deepseek_ocr.py` iterates `range(images_spatial_crop.size(0))`.
    # If `images_spatial_crop` is `(B, ...)`, then it iterates over batch.
    # So `pixel_values` should be `(B, ...)` too.
    
    # However, `tokenize_with_images` returns `pixel_values` as `(N_images, 3, H, W)`.
    # If N_images=1, it's `(1, 3, H, W)`.
    # So we can stack to `(B, 1, 3, H, W)`.
    # But `deepseek_ocr.py` might expect flattened images if `n_image` is large?
    # "Pixel_values (global view): [n_image, batch_size, 3, height, width]" comment is confusing.
    # But code `image_ori = pixel_values[jdx]` suggests `pixel_values` is `(B, N_images, 3, H, W)` if `jdx` is batch index.
    # Let's verify `images_spatial_crop` shape.
    # `images_spatial_crop` is `(N_images, 2)` from processor.
    # Stacked: `(B, N_images, 2)`.
    # So `jdx` is batch index.
    
    # So we stack `pixel_values` to `(B, N_images, 3, H, W)`.
    # But wait, `pixel_values` from processor is `(N_images, 3, H, W)`.
    # If different samples have different `N_images`, we can't stack.
    # But here we assume 1 image per sample.
    
    pixel_values_stacked = torch.stack(pixel_values_list, dim=0) # (B, N, 3, H, W)
    images_crop_stacked = torch.stack(images_crop_list, dim=0)
    images_spatial_crop_stacked = torch.stack(images_spatial_crop_list, dim=0)
    # images_seq_mask is list of bools?
    # Processor returns `torch.tensor(images_seq_mask, dtype=torch.bool)`.
    # So we pad it.
    images_seq_mask_padded = torch.nn.utils.rnn.pad_sequence(images_seq_mask_list, batch_first=True, padding_value=False)
    
    # Flatten pixel_values if needed?
    # The model seems to handle `(B, N, ...)` if `images_spatial_crop` is `(B, N, 2)`.
    # Actually `deepseek_ocr.py` line 388: `image_ori = pixel_values[jdx]` -> `(N, 3, H, W)`.
    # line 404: `global_features_1 = self.sam_model(image_ori)`
    # `sam_model` likely expects `(Batch, 3, H, W)`. Here `Batch` is `N_images`.
    # So yes, `pixel_values` should be `(B, N, 3, H, W)`.
    
    # But wait, `images_crop` is `(B, N, num_patches, 3, h, w)`.
    # `patches = images_crop[jdx][0]` -> `(num_patches, 3, h, w)`.
    # This assumes `N=1` (index 0).
    # If N > 1, the code in `deepseek_ocr.py` might fail or only process first image?
    # `for jdx in range(images_spatial_crop.size(0)):` iterates batch.
    # `patches = images_crop[jdx][0]` -> Hardcoded 0?
    # Yes, `deepseek_ocr.py` line 387: `patches = images_crop[jdx][0]`.
    # This implies the model (or at least this version) only supports 1 image per sample?
    # Or `images_crop` has shape `(B, 1, ...)`?
    # Processor returns `images_crop` as `(1, 3, H, W).unsqueeze(0)` -> `(1, 1, 3, H, W)`?
    # No, `images_crop = torch.stack(images_crop_list, dim=0).unsqueeze(0)`.
    # If `images_crop_list` has `P` patches. `stack` -> `(P, 3, H, W)`. `unsqueeze` -> `(1, P, 3, H, W)`.
    # So `images_crop` is `(1, P, 3, H, W)`.
    # Stacked batch: `(B, 1, P, 3, H, W)`.
    # `images_crop[jdx]` -> `(1, P, 3, H, W)`.
    # `images_crop[jdx][0]` -> `(P, 3, H, W)`.
    # So yes, it supports 1 image per sample (or at least the crop logic does).
    # Since we have 1 image per sample, this is fine.
    
    return {
        'input_ids': input_ids_padded,
        'labels': labels_padded,
        'pixel_values': pixel_values_stacked,
        'images_crop': images_crop_stacked,
        'images_spatial_crop': images_spatial_crop_stacked,
        'images_seq_mask': images_seq_mask_padded
    }

def train(args):
    # Load data
    with open(args.train_file, 'r') as f:
        train_data = json.load(f)
    
    # Load model
    model, processor, tokenizer = load_model(args.model_path)
    model.train()
    
    # Logger
    logger = get_logger(args.output_dir)
    
    # Dataset
    train_dataset = OCRDataset(train_data, args.data_root, tokenizer, processor)
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, processor),
        num_workers=4
    )
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            if batch is None:
                continue
                
            input_ids = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()
            pixel_values = batch['pixel_values'].cuda().to(torch.bfloat16)
            images_crop = batch['images_crop'].cuda().to(torch.bfloat16)
            images_spatial_crop = batch['images_spatial_crop'].cuda()
            # images_seq_mask = batch['images_seq_mask'].cuda() # Not used in forward?
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                pixel_values=pixel_values,
                images_crop=images_crop,
                images_spatial_crop=images_spatial_crop
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        logger.log(f"Epoch {epoch+1} Average Loss: {avg_loss}")
        
        # Save checkpoint
        save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.log(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
