import os
import argparse
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from Levenshtein import distance as levenshtein_distance

from data_utils import OCRDataset
from model_utils import load_model
from logger import get_logger

def collate_fn_test(batch, processor):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    # For testing, we process one by one or batch?
    # Generation usually works better with batching if padding is handled correctly.
    # But `tokenize_with_images` handles padding.
    
    input_ids_list = []
    pixel_values_list = []
    images_crop_list = []
    images_spatial_crop_list = []
    images_seq_mask_list = []
    ground_truths = []
    image_paths = []
    
    for item in batch:
        image = item['image']
        prompt = item['prompt']
        ground_truths.append(item['ground_truth'])
        image_paths.append(item['image_path'])
        
        if '<image>' not in prompt:
            prompt = '<image>\n' + prompt
            
        # Tokenize prompt only for generation input
        outputs = processor.tokenize_with_images([image], prompt=prompt, bos=True, eos=False)
        
        input_ids = outputs[0][0].squeeze(0)
        pixel_values = outputs[0][1]
        images_crop = outputs[0][2]
        images_seq_mask = outputs[0][3]
        images_spatial_crop = outputs[0][4]
        
        input_ids_list.append(input_ids)
        pixel_values_list.append(pixel_values)
        images_crop_list.append(images_crop)
        images_spatial_crop_list.append(images_spatial_crop)
        images_seq_mask_list.append(images_seq_mask)
        
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=processor.pad_id)
    
    pixel_values_stacked = torch.stack(pixel_values_list, dim=0)
    images_crop_stacked = torch.stack(images_crop_list, dim=0)
    images_spatial_crop_stacked = torch.stack(images_spatial_crop_list, dim=0)
    images_seq_mask_padded = torch.nn.utils.rnn.pad_sequence(images_seq_mask_list, batch_first=True, padding_value=False)
    
    return {
        'input_ids': input_ids_padded,
        'pixel_values': pixel_values_stacked,
        'images_crop': images_crop_stacked,
        'images_spatial_crop': images_spatial_crop_stacked,
        'images_seq_mask': images_seq_mask_padded,
        'ground_truths': ground_truths,
        'image_paths': image_paths
    }

def test(args):
    with open(args.test_file, 'r') as f:
        test_data = json.load(f)
        
    model, processor, tokenizer = load_model(args.model_path)
    model.eval()
    
    # Logger
    logger = get_logger(args.output_dir)
    
    test_dataset = OCRDataset(test_data, args.data_root, tokenizer, processor)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn_test(b, processor),
        num_workers=4
    )
    
    results = []
    total_distance = 0
    total_chars = 0
    
    logger.log("Starting inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if batch is None:
                continue
                
            input_ids = batch['input_ids'].cuda()
            pixel_values = batch['pixel_values'].cuda().to(torch.bfloat16)
            images_crop = batch['images_crop'].cuda().to(torch.bfloat16)
            images_spatial_crop = batch['images_spatial_crop'].cuda()
            # images_seq_mask = batch['images_seq_mask'].cuda()
            
            # Generate
            generated_ids = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                images_crop=images_crop,
                images_spatial_crop=images_spatial_crop,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=processor.pad_id,
                eos_token_id=processor.eos_id
            )
            
            # Decode
            # generated_ids contains input_ids + new_tokens. We need to slice.
            # But model.generate usually returns full sequence?
            # Yes.
            
            input_len = input_ids.shape[1]
            new_tokens = generated_ids[:, input_len:]
            pred_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            
            for i, pred in enumerate(pred_texts):
                gt = batch['ground_truths'][i]
                dist = levenshtein_distance(pred, gt)
                total_distance += dist
                total_chars += len(gt)
                
                results.append({
                    'image_path': batch['image_paths'][i],
                    'ground_truth': gt,
                    'prediction': pred,
                    'levenshtein_distance': dist
                })
                
    avg_distance = total_distance / len(results) if results else 0
    norm_distance = total_distance / total_chars if total_chars > 0 else 0
    
    logger.log(f"Average Levenshtein Distance: {avg_distance:.2f}")
    logger.log(f"Normalized Distance (CER): {norm_distance:.4f}")
    
    output_file = os.path.join(args.output_dir, 'test_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': {
                'avg_levenshtein_distance': avg_distance,
                'cer': norm_distance
            },
            'results': results
        }, f, ensure_ascii=False, indent=2)
        
    logger.log(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    test(args)
