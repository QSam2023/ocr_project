import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Tuple, Optional

class OCRDataset(Dataset):
    def __init__(self, data_list: List[Dict], root_dir: str, tokenizer=None, image_processor=None, max_length=1024):
        """
        Args:
            data_list: List of data items (from json).
            root_dir: Root directory for images.
            tokenizer: Tokenizer for the model.
            image_processor: Image processor for the model.
            max_length: Max token length.
        """
        self.data_list = data_list
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image_path = os.path.join(self.root_dir, item['image_path'])
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy item or skip (handling skip in Dataset is tricky, usually return None and collate_fn handles it)
            return None

        prompt = item.get('prompt', '')
        # Ensure result is a string
        result = item.get('result', '')
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)

        # Prepare inputs for the model
        # Note: This part depends on the specific model's expected input format.
        # DeepSeek-OCR likely expects pixel_values and input_ids.
        
        return {
            'image': image,
            'prompt': prompt,
            'ground_truth': result,
            'image_path': image_path
        }

def load_all_data(data_root: str) -> List[Dict]:
    """
    Traverse the data_root to find all json annotation files and load data.
    Assumes structure: data_root/category/batch/batch_ocr.json
    """
    all_data = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.json') and 'ocr' in file:
                json_path = os.path.join(root, file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    if 'results' in data:
                        # Adjust image paths to be relative to data_root if they are not already absolute
                        # The json usually contains relative paths like "stamp_01/stamp_0001.png"
                        # We need to make sure we can find the image.
                        # If the json is in data_root/stamp_data_example/stamp_01/stamp_ocr_01.json
                        # and image_path is "stamp_01/stamp_0001.png", then relative to what?
                        # Usually relative to the directory containing the json or the parent of that.
                        # Let's check the example file content again.
                        # "image_path": "stamp_01/stamp_0001.png"
                        # File location: .../stamp_data_example/stamp_01/stamp_ocr_01.json
                        # Image location: .../stamp_data_example/stamp_01/stamp_0001.png
                        # So "stamp_01/stamp_0001.png" seems to be relative to `stamp_data_example`.
                        
                        # We will store the absolute path or path relative to data_root.
                        # Let's try to resolve the path.
                        
                        # Determine the base directory for the images in this json
                        # If json is in .../A/B/x.json and image is A/B/img.png, then base is .../
                        
                        # Heuristic: check if the image path exists relative to the json's parent's parent
                        # or just relative to the json's directory.
                        
                        # In the observed case:
                        # Json: .../stamp_data_example/stamp_01/stamp_ocr_01.json
                        # Image entry: "stamp_01/stamp_0001.png"
                        # Actual Image: .../stamp_data_example/stamp_01/stamp_0001.png
                        # So the image path is relative to `.../stamp_data_example`.
                        
                        # We can pass the `data_root` as `.../ocr_data`.
                        # Then `stamp_data_example` is a subdir.
                        # But the json is inside `stamp_data_example/stamp_01`.
                        
                        # Let's just store the full absolute path in the data item to be safe.
                        
                        # Find the "category" dir (e.g. stamp_data_example)
                        # The json is at root/file
                        
                        # Let's assume the user passes `ocr_data` as data_root.
                        # Then we look for files.
                        
                        results = data['results']
                        for item in results:
                            # Try to resolve image path
                            rel_path = item['image_path']
                            
                            # Check if it exists relative to the json file's directory
                            # Case 1: Json in `dir/subdir`, image in `dir/subdir` but path is `subdir/image`? No that would be weird.
                            # Case 2: Json in `dir/subdir`, image in `dir/subdir`, path is `subdir/image`.
                            # Then `dir` is the base.
                            
                            # Let's try to find the image.
                            # We know where the json is: `root`
                            # We can try to join `root` and `rel_path`?
                            # If root is `.../stamp_01` and rel_path is `stamp_01/stamp_0001.png`, then `.../stamp_01/stamp_01/stamp_0001.png` -> Wrong.
                            
                            # If root is `.../stamp_01`, and we go up one level: `.../` (which is `stamp_data_example`)
                            # Then join with `stamp_01/stamp_0001.png` -> Correct.
                            
                            parent_dir = os.path.dirname(root)
                            candidate_path = os.path.join(parent_dir, rel_path)
                            
                            if os.path.exists(candidate_path):
                                item['image_path'] = candidate_path
                                all_data.append(item)
                            else:
                                # Try other possibilities
                                candidate_path_2 = os.path.join(root, os.path.basename(rel_path))
                                if os.path.exists(candidate_path_2):
                                    item['image_path'] = candidate_path_2
                                    all_data.append(item)
                                else:
                                    print(f"Warning: Could not find image for {rel_path} in {root} or {parent_dir}")
                                    
                except Exception as e:
                    print(f"Error reading {json_path}: {e}")
                    
    return all_data

def split_dataset(data_root: str, output_dir: str, train_ratio: float = 0.8, seed: int = 42):
    """
    Load all data, split into train/test, and save to json files.
    """
    random.seed(seed)
    
    all_data = load_all_data(data_root)
    print(f"Total samples found: {len(all_data)}")
    
    random.shuffle(all_data)
    
    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, 'train.json')
    test_file = os.path.join(output_dir, 'test.json')
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
        
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(train_data)} training samples to {train_file}")
    print(f"Saved {len(test_data)} testing samples to {test_file}")

if __name__ == "__main__":
    # Test the splitting
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    split_dataset(args.data_root, args.output_dir)
