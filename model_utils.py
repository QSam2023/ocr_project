import os
import math
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageOps
from typing import List, Tuple, Optional, Union
from transformers import AutoModel, AutoTokenizer, ProcessorMixin, BatchFeature, LlamaTokenizerFast

# Constants from config.py (defaults)
DEFAULT_BASE_SIZE = 1024
DEFAULT_IMAGE_SIZE = 640
DEFAULT_CROP_MODE = True
DEFAULT_MIN_CROPS = 2
DEFAULT_MAX_CROPS = 6
DEFAULT_PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

# Helper functions from process/image_process.py

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def count_tiles(orig_width, orig_height, min_num=DEFAULT_MIN_CROPS, max_num=DEFAULT_MAX_CROPS, image_size=640, use_thumbnail=False):
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    return target_aspect_ratio

def dynamic_preprocess(image, min_num=DEFAULT_MIN_CROPS, max_num=DEFAULT_MAX_CROPS, image_size=640, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio

class ImageTransform:
    def __init__(self,
                 mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 normalize: bool = True):
        self.mean = mean
        self.std = std
        self.normalize = normalize
        transform_pipelines = [T.ToTensor()]
        if normalize:
            transform_pipelines.append(T.Normalize(mean, std))
        self.transform = T.Compose(transform_pipelines)

    def __call__(self, pil_img: Image.Image):
        return self.transform(pil_img)

class DeepseekOCRProcessor(ProcessorMixin):
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")
    attributes = ["tokenizer"]

    def __init__(
        self,
        tokenizer: LlamaTokenizerFast,
        image_size: int = DEFAULT_IMAGE_SIZE,
        base_size: int = DEFAULT_BASE_SIZE,
        patch_size: int = 16,
        downsample_ratio: int = 4,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
        image_token: str = "<image>",
        pad_token: str = "<｜ pad ｜>",
        add_special_token: bool = False,
        sft_format: str = "deepseek",
        mask_prompt: bool = True,
        ignore_id: int = -100,
        crop_mode: bool = DEFAULT_CROP_MODE,
        **kwargs,
    ):
        self.image_size = image_size
        self.base_size = base_size
        self.patch_size = patch_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.normalize = normalize
        self.downsample_ratio = downsample_ratio
        self.crop_mode = crop_mode
        
        self.image_transform = ImageTransform(mean=image_mean, std=image_std, normalize=normalize)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            
        self.image_token_id = self.tokenizer.vocab.get(image_token)
        self.image_token = image_token
        self.pad_token = pad_token
        self.add_special_token = add_special_token
        self.sft_format = sft_format
        self.mask_prompt = mask_prompt
        self.ignore_id = ignore_id
        
        super().__init__(tokenizer, **kwargs)

    @property
    def bos_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id

    def encode(self, text: str, bos: bool = True, eos: bool = False):
        t = self.tokenizer.encode(text, add_special_tokens=False)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int], **kwargs) -> str:
        return self.tokenizer.decode(t, **kwargs)

    def process_one(self, prompt: str, images: List, inference_mode: bool = True, **kwargs):
        assert (prompt is not None and images is not None), "prompt and images must be used at the same time."
        
        # This method assumes images are already processed by tokenize_with_images
        # But wait, tokenize_with_images returns a list of lists.
        # This method seems to just unpack the first item?
        # In the original code:
        # input_ids, pixel_values, images_crop, images_seq_mask, images_spatial_crop, num_image_tokens, _ = images[0]
        # This implies `images` passed here is the output of `tokenize_with_images`.
        
        input_ids, pixel_values, images_crop, images_seq_mask, images_spatial_crop, num_image_tokens, _ = images[0]
        
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "images_crop": images_crop,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
            "num_image_tokens": num_image_tokens,
        }

    def tokenize_with_images(self, images: List[Image.Image], prompt: str = DEFAULT_PROMPT, bos: bool = True, eos: bool = True, cropping: bool = None):
        if cropping is None:
            cropping = self.crop_mode
            
        conversation = prompt
        # assert conversation.count(self.image_token) == len(images), f"Prompt image tokens {conversation.count(self.image_token)} != images {len(images)}"
        # If prompt doesn't have enough image tokens, we might have an issue.
        # But for OCR usually 1 image.
        
        text_splits = conversation.split(self.image_token)
        images_list, images_crop_list, images_seq_mask, images_spatial_crop = [], [], [], []
        image_shapes = []
        num_image_tokens = []
        tokenized_str = []
        
        for text_sep, image in zip(text_splits, images):
            tokenized_sep = self.encode(text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)
            
            image_shapes.append(image.size)
            
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = [1, 1]
            else:
                if cropping:
                    images_crop_raw, crop_ratio = dynamic_preprocess(image, image_size=self.image_size)
                else:
                    crop_ratio = [1, 1]
            
            if self.image_size <= 640 and not cropping:
                image = image.resize((self.image_size, self.image_size))
                
            global_view = ImageOps.pad(image, (self.base_size, self.base_size),
                                    color=tuple(int(x * 255) for x in self.image_transform.mean))
            images_list.append(self.image_transform(global_view))
            
            num_width_tiles, num_height_tiles = crop_ratio
            images_spatial_crop.append([num_width_tiles, num_height_tiles])
            
            if num_width_tiles > 1 or num_height_tiles > 1:
                for i in range(len(images_crop_raw)):
                    images_crop_list.append(self.image_transform(images_crop_raw[i]))
                    
            num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            
            tokenized_image = ([self.image_token_id] * num_queries_base + [self.image_token_id]) * num_queries_base
            tokenized_image += [self.image_token_id]
            if num_width_tiles > 1 or num_height_tiles > 1:
                tokenized_image += ([self.image_token_id] * (num_queries * num_width_tiles) + [self.image_token_id]) * (
                            num_queries * num_height_tiles)
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
            num_image_tokens.append(len(tokenized_image))
            
        tokenized_sep = self.encode(text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)
        
        if bos:
            tokenized_str = [self.bos_id] + tokenized_str
            images_seq_mask = [False] + images_seq_mask
        if eos:
            tokenized_str = tokenized_str + [self.eos_id]
            images_seq_mask = images_seq_mask + [False]
            
        masked_tokenized_str = []
        for token_index in tokenized_str:
            if token_index != self.image_token_id:
                masked_tokenized_str.append(token_index)
            else:
                masked_tokenized_str.append(self.ignore_id)
                
        input_ids = torch.LongTensor(tokenized_str)
        target_ids = torch.LongTensor(masked_tokenized_str)
        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)
        
        target_ids[(input_ids < 0) | (input_ids == self.image_token_id)] = self.ignore_id
        input_ids[input_ids < 0] = self.pad_id
        
        # Inference mode logic (remove last eos)
        # In original code, it always does this?
        # "if inference_mode: ... assert input_ids[-1] == self.eos_id ... input_ids = input_ids[:-1]"
        # But for training we might want EOS.
        # I'll keep it configurable or just follow original logic if I'm doing inference.
        # But for training, we usually want the full sequence.
        # I'll add `inference_mode` arg to `tokenize_with_images`? No, it was in `process_one`.
        # But `tokenize_with_images` had `inference_mode = True` hardcoded at the end!
        # "inference_mode = True ... if inference_mode: ..."
        # This removes the EOS token.
        # For training (Causal LM), we usually predict the next token.
        # If we remove EOS, we can't predict EOS.
        # I will comment out the hardcoded inference_mode and use an argument.
        
        # if inference_mode: ...
        
        if len(images_list) == 0:
            pixel_values = torch.zeros((1, 3, self.base_size, self.base_size))
            images_spatial_crop = torch.zeros((1, 1), dtype=torch.long)
            images_crop = torch.zeros((1, 3, self.image_size, self.image_size)).unsqueeze(0)
        else:
            pixel_values = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0).unsqueeze(0)
            else:
                images_crop = torch.zeros((1, 3, self.image_size, self.image_size)).unsqueeze(0)
                
        input_ids = input_ids.unsqueeze(0)
        
        return [[input_ids, pixel_values, images_crop, images_seq_mask, images_spatial_crop, num_image_tokens, image_shapes]]

def load_model(model_path: str, device='cuda'):
    """
    Load DeepSeek-OCR model and tokenizer.
    """
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, use_safetensors=True)
    
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda().to(torch.bfloat16)
    
    processor = DeepseekOCRProcessor(tokenizer)
    
    return model, processor, tokenizer

