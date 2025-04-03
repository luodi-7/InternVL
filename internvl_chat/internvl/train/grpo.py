# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import logging
import math
import os
import random
import sys
import traceback
import warnings
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Literal, Optional
from FlagEmbedding import FlagModel
from matplotlib import pyplot as plt
from transformers import AutoTokenizer
from sentence_transformers import util
from transformers import pipeline  # 用于流畅性评估
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

import numpy as np

try:
    import orjson as json
except:
    import json

import torch
import torch.distributed as dist
import transformers
from internvl.dist_utils import init_dist
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig,
                                          InternVLChatModel)
from internvl.patch import (concat_pad_data_collator,
                            replace_internlm2_attention_class,
                            replace_llama_attention_class,
                            replace_llama_rmsnorm_with_fused_rmsnorm,
                            replace_phi3_attention_class,
                            replace_qwen2_attention_class,
                            replace_train_dataloader, replace_train_sampler)
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)
from internvl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    check_conversations_repetition,
                                    dynamic_preprocess, preprocess,
                                    preprocess_internlm,
                                    preprocess_internvl2_5, preprocess_mpt,
                                    preprocess_phi3)
from internvl.train.dataset_packed import PackedDataset, packed_collate_fn
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments,
                          set_seed, TrainerCallback, GenerationConfig)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)
from sentence_transformers import SentenceTransformer, util, models

# Try to import petrel_client for image loading, fallback to PIL if unavailable
try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    ref_model_path: str = field(
        metadata={"help": "参考模型路径"}
    )
    num_generations: int = field(
        default=5, 
        metadata={"help": "每个prompt生成数"}
    )
    beta: float = field(
        default=0.1, 
        metadata={"help": "KL散度权重"}
    )
    generation_config: dict = field(
        default_factory=lambda: {
        'max_new_tokens': 1024,
        'do_sample': True,
        'temperature': 1.0
    })
    use_grpo: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM. Default is False.'},
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM. Default is False.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the ViT. Default is False.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP. Default is False.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the ViT. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the head of LLM. Default is False.'},
    )
    grad_checkpoint: bool = field(
        default=True,
        metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT. Default is 0.'},
    )
    ps_version: Literal['v1', 'v2'] = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is v2.'}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the fast mode of the tokenizer.'}
    )
    use_liger: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the liger kernel.'}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: int = field(
        default=8192,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: int = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'},
    )
    pad2square: bool = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True. Default is False.'},
    )
    conv_style: str = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    meta_path_eval: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the meta file of eval datasets.'},
    )
    # load_best_model_at_end: Optional[bool] = field(
    #     default=False,
    #     metadata={'help': 'Whether or not to load the best model found during training at the end of training. Default is False.'},
    # )
    # metric_for_best_model: Optional[str] = field(
    #     default=None,
    #     metadata={'help': 'Use in conjunction with load_best_model_at_end to specify the metric to use to compare two different models. '},
    # )
    use_data_resampling: bool = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling. Default is False.'},
    )
    dynamic_image_size: bool = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic high resolution strategy. Default is False.'},
    )
    use_thumbnail: bool = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image. Default is False.'},
    )
    min_dynamic_patch: int = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: int = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 12.'},
    )
    min_num_frame: int = field(
        default=8,
        metadata={'help': 'The minimum number of frames for video data. Default is 8.'},
    )
    max_num_frame: int = field(
        default=32,
        metadata={'help': 'The maximum number of frames for video data. Default is 32.'},
    )
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = field(
        default='imagenet',
        metadata={'help': 'The normalization type for the image. Default is imagenet.'},
    )
    use_packed_ds: bool = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset for efficient training. Default is False.'},
    )
    num_images_expected: int = field(
        default=40,
        metadata={'help': 'The maximum number of images per packed sample. Default is 40.'},
    )
    max_packed_tokens: int = field(
        default=8192,
        metadata={'help': 'The required token length of per packed sample. Default is 8192.'},
    )
    max_buffer_size: int = field(
        default=20,
        metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    )
    log_freq: int = field(
        default=1000,
        metadata={'help': 'The log frequency of the packed dataset. Default is 1000.'},
    )
    strict_mode: bool = field(
        default=True,
        metadata={'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'},
    )
    replacement: bool = field(
        default=False,
        metadata={'help': 'Whether to restart the dataset after it is exhausted. Default is False.'},
    )
    allow_overflow: bool = field(
        default=False,
        metadata={'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'},
    )
    loss_reduction: str = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is token.'},
    )
    loss_reduction_all_gather: bool = field(
        default=False,
        metadata={'help': 'Whether to gather all during loss reduction. Default is False.'},
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=448,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        min_num_frame=8,  # for video data
        max_num_frame=32,  # for video data
        sampling_method='rand',  # for video data
        repeat_time=1,
        normalize_type='imagenet',
        # hyperparameters for packed training
        use_packed_ds=False,
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,
        force_shuffle=False,
        random_seed=0,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        # hyperparameters for distributed training
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        # hyperparameters for packed dataset
        self.dataset_type = 'pair'
        self.max_num_images = 1
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle
        # TODO: quick resume
        self._state_dict = {}

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        if self.force_shuffle:
            self.rng.shuffle(self.raw_data)

        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    try:

                        # Compute token length using the tokenizer
                        conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    except KeyError as e:
                        print(f"KeyError occurred with data_item: {data_item}")
                        raise  
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                    max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        elif self.template_name == 'internvl2_5':
            preprocess_function = preprocess_internvl2_5
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        # Merge the image path
        image_path = self.get_image_path(data_item['image'])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()


        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == 1, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=max(1, self.max_dynamic_patch // num_image),
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_image)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def video_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains a video placeholder
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>\n' + data_item['conversations'][0]['value']

        # Get the video file path
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)

        # Load the video frames using tcs_loader
        # TODO: Load videos without using tcsloader.
        image_list = self.tcs_loader(
            video_path,
            image_type='video',
            max_num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method,
            clip=data_item.get('clip', None))

        # Generate special tokens for each video frame
        special_tokens = '\n'.join(['Frame-{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
            '<video>\n', special_tokens + '\n')

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_patches)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, use_packed_ds=self.use_packed_ds,
                                  ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        return ret

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id::self.num_workers]
            logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            if self.use_packed_ds:
                raise NotImplementedError
            else:
                i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = json.loads(self.raw_data[i])
                # conversations = data_item['conversations']
                # check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10)
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                print(e, self.ds_name, flush=True)
                if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0

        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']

            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] '
                f'begin to iter with {start_idx=}'
            )

        for i in range(start_idx, len(self)):
            yield self[i]


def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    min_num_frame=8,
    max_num_frame=32,
    normalize_type='imagenet',
    train = True
):
    datasets = []
    lengths = []
    if train:
        ds_collections = json.loads(open(data_args.meta_path).read())
    else:
        ds_collections = json.loads(open(data_args.meta_path_eval).read())
    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style, ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length and not data_args.use_packed_ds,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            min_num_frame=min_num_frame,
            max_num_frame=max_num_frame,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            # hyperparameters for packed training
            use_packed_ds=data_args.use_packed_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds,
            random_seed=ds_idx,
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))

    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False,
        )
    elif data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)

class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True

import torch
from transformers import AutoTokenizer
from sentence_transformers import util
from transformers import pipeline  # 用于流畅性评估
def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

class SimilarityModel():
    def __init__(self):
        model_path = '/fs-computility/ai-shen/shared/dilab/model/bge-base-zh-v1.5'
        self.model = FlagModel(model_path, 
                              query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                              use_fp16=True)

    def compare_similarity(self, output_str, label_str):
        try:
            # 检查输入是否为空
            if not output_str.strip() or not label_str.strip():
                return 0.0  # 返回默认相似度得分
            
            # 计算相似度
            embeddings_output = self.model.encode(output_str)
            embeddings_label = self.model.encode(label_str)
            cosine_scores = util.cos_sim(embeddings_output, embeddings_label).item()
            return cosine_scores
        except Exception as e:
            print(f"Error comparing similarity for output: '{output_str}' and label: '{label_str}'")
            print(f"Error message: {e}")
            return 0.0  # 返回默认相似度得分

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class FluencyModel():
    def __init__(self):
        # 加载 GPT-2 模型和分词器
        model_path = '/fs-computility/ai-shen/shared/dilab/model/gpt2'
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model.eval()  # 设置为评估模式
        self.target_percentiles = {  
            0.25: 0.25,  # 25th percentile  
            0.50: 0.50,  # 50th percentile (median)  
            0.75: 0.75   # 75th percentile  
        } 

    def evaluate_fluency(self, text):
        if not text.strip():  # 如果文本为空或只包含空白字符
            return 0.0  # 返回一个默认的流畅性得分（例如 0.0）
        # 使用 GPT-2 计算困惑度
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss  # 交叉熵损失
            perplexity = torch.exp(loss).item()  # 困惑度 = exp(loss)
        
        # 将困惑度归一化到 0 到 1 的范围
        normalized_score = self.normalize_fluency_score(perplexity)
        return normalized_score

    def normalize_fluency_score(self, perplexity):
        # 假设困惑度的范围是 10 到 1000
        min_perplexity = 10
        max_perplexity = 1000
        
        # 将困惑度映射到 0 到 1 的范围
        normalized_score = 1 - (perplexity - min_perplexity) / (max_perplexity - min_perplexity)
        
        # 确保得分在 0 到 1 之间
        normalized_score = max(0, min(1, normalized_score))
        return normalized_score
    # 计算百分位数距离
    def calculate_percentile_distance(self, gen_scores):
        gen_percentiles = {
            0.25: np.percentile(gen_scores, 25),
            0.50: np.percentile(gen_scores, 50),
            0.75: np.percentile(gen_scores, 75)
        }
        distance = 0
        for p in [0.25, 0.50, 0.75]:
            distance += (gen_percentiles[p] - self.target_percentiles[p]) ** 2
        return distance / 3

    def calculate_percentile_percentage(self, gen_scores):
        count=0
        for s in gen_scores:
            if s >= self.target_percentiles[0.25] and s <= self.target_percentiles[0.75]:
                count += 1
        return count / len(gen_scores)

class HumorModel():
    def __init__(self):
        # 手动加载模型和分词器
        model_path = "/fs-computility/ai-shen/xueyingyi/humor-detection-comb-23"  
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", max_length=512, truncation=True)

    def evaluate_humor(self, text):
        try:
            # 检查输入文本是否为空
            if not text.strip():
                return 0.0  # 返回默认幽默得分
            
            # 使用分词器对输入文本进行编码
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            
            # 使用模型进行预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)  # 将 logits 转换为概率
            
            # 获取预测标签和置信度分数
            label_id = torch.argmax(probs, dim=-1).item()  # 预测的标签 ID (0 或 1)
            score = probs[0][label_id].item()  # 预测标签的置信度分数

            # 计算幽默得分
            if label_id == 1:  # LABEL_1 表示幽默
                humor_score = score
            else:  # LABEL_0 表示非幽默
                humor_score = 1 - score

            return humor_score
        except Exception as e:
            print(f"Error evaluating humor for text: '{text}'")
            print(f"Error message: {e}")
            return 0.0  # 返回默认幽默得分

import re  

def extract_meme_text(text):  
    # 检查 Text on the Meme: 后面是否有换行符  
      
    pattern = r'Text on the Meme:\s*\n(.*)'  
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None  

def compute_json_metric(eval_preds, model_path: str):
    # 初始化相似度模型、流畅性模型和幽默感模型
    simmodel = SimilarityModel()
    fluency_model = FluencyModel()
    humor_model = HumorModel()
    
    # 从 eval_preds 中获取 predictions 和 labels
    predictions, labels = eval_preds
    
    with torch.no_grad():
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
        result = predictions[0]
        
        if 'InternVL2_5-4B' in model_path:
            eos_token_id = 151645
        else:
            eos_token_id = 2
        
        for id, pred in enumerate(labels):
            result[id][result[id] == -100] = eos_token_id
            result[id][labels[id] == -100] = eos_token_id
            labels[id][labels[id] == -100] = eos_token_id
        result = tokenizer.batch_decode(result, skip_special_tokens=True)
        label = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 计算相似度、流畅性和幽默感
        similarity_scores = []
        fluency_scores = []
        humor_scores = []
        text_similarity_scores = []
        # 定义正则表达式提取 meme text
    
        for r, l in zip(result, label):
            # write_to_file(l,r,"/fs-computility/ai-shen/xueyingyi/cot_unfreeze_mlp/output_cot.txt")
            try:
                similarity_score = simmodel.compare_similarity(r, l)
            except Exception as e:
                print(f"Error comparing similarity for output: '{r}' and label: '{l}'")
                print(f"Error message: {e}")
                similarity_score = 0.0  # 返回默认相似度得分
            
            try:
                fluency_score = fluency_model.evaluate_fluency(r)
            except Exception as e:
                print(f"Error evaluating fluency for text: '{r}'")
                print(f"Error message: {e}")
                fluency_score = 0.0  # 返回默认流畅性得分
            
            try:
                humor_score = humor_model.evaluate_humor(r)
            except Exception as e:
                print(f"Error evaluating humor for text: '{r}'")
                print(f"Error message: {e}")
                humor_score = 0.0  # 返回默认幽默得分
            try:


            # 新增提取 meme text 逻辑
                r_meme = extract_meme_text(r)
                l_meme = extract_meme_text(l)
  
                write_to_file(l_meme,r_meme)
                
            
                if not r_meme or not l_meme:
                    print(f"Warning: Missing meme text in prediction or label:\nPred: {r}\nLabel: {l}")
                    text_score = 0.0
                else:
                    text_score = simmodel.compare_similarity(r_meme, l_meme)
            
                text_similarity_scores.append(text_score)
            except Exception as e:
                print(f"Error calculating text similarity: {e}")
                text_similarity_scores.append(0.0)
            similarity_scores.append(similarity_score)
            fluency_scores.append(fluency_score)
            humor_scores.append(humor_score)
        try:
            fluency_distance = fluency_model.calculate_percentile_distance(fluency_scores)
        except Exception as e:
            print(f"Error calculating fluency distance")
            print(f"Error message: {e}")
            fluency_distance = 100  # 返回默认流畅性得分    
        try:
            fluency_percentage = fluency_model.calculate_percentile_percentage(fluency_scores)
        except Exception as e:
            print(f"Error calculating fluency percentage")
            print(f"Error message: {e}")
            fluency_percentage = 0.0  # 返回默认流畅性得分
        # 计算平均相似度、流畅性和幽默感
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        avg_fluency = sum(fluency_scores) / len(fluency_scores)
        avg_humor = sum(humor_scores) / len(humor_scores)
        avg_text_similarity = sum(text_similarity_scores) / len(text_similarity_scores) if text_similarity_scores else 0.0
        
        # 返回一个字典，Trainer 会将其记录到日志中
        return {
            "avg_similarity": avg_similarity,
            "avg_text_similarity": avg_text_similarity,
            "avg_fluency": avg_fluency,
            "avg_humor": avg_humor,
            "fluency_distance": fluency_distance,
            "fluency_percentage": fluency_percentage,
        }

def write_to_file(label, result, filename="/fs-computility/ai-shen/xueyingyi/grpo_test/output.txt"):
    with open(filename, "a") as file:  # 使用 "a" 模式追加写入文件
        # 获取当前文件的行数，用于生成序号
        with open(filename, "r") as f:
            lines = f.readlines()
            index = len(lines) // 2 + 1  # 每对 label 和 result 占两行

        # 写入序号、label 和 result
        file.write(f"/////////////// Entry {index} ///////////////\n")
        file.write(f"Label: {label}\n")
        file.write(f"Result: {result}\n")
        file.write("\n")  # 添加空行以便区分不同条目


import torch
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
from PIL import Image
import numpy as np
from transformers import AutoProcessor, CLIPModel

class MemeRewardFunction:
    def __init__(self, device='cuda'):
        # 加载CLIP模型评估图文相关性
        self.clip_model = AutoModel.from_pretrained("/fs-computility/ai-shen/shared/dilab/model/clip-vit-base-patch32")
        self.clip_tokenizer = AutoTokenizer.from_pretrained("/fs-computility/ai-shen/shared/dilab/model/clip-vit-base-patch32")
        self.clip_processor = AutoProcessor.from_pretrained("/fs-computility/ai-shen/shared/dilab/model/clip-vit-base-patch32")
        
        # 加载GPT-2模型评估文本流畅度
        self.gpt2_model = AutoModelForCausalLM.from_pretrained("/fs-computility/ai-shen/shared/dilab/model/gpt2")
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained("/fs-computility/ai-shen/shared/dilab/model/gpt2")
        
        self.device = device
        self._setup_models()
        
    def _setup_models(self):
        """将模型移动到指定设备并设置为评估模式"""
        self.clip_model = self.clip_model.to(self.device).eval()
        self.gpt2_model = self.gpt2_model.to(self.device).eval()
    
    def __call__(self,completions):
        """
        综合奖励计算函数
        Args:
            prompts: List[str] 原始提示文本列表
            completions: List[List[str]] 生成的文本列表（每个prompt对应多个completion）
            images: List[PIL.Image] 对应的输入图像列表
        Returns:
            rewards: List[torch.Tensor] 每个生成样本的奖励值
        """
        batch_rewards = []
        prompts = ["Generate a funny caption for this meme"]
        images = [Image.open("/fs-computility/ai-shen/xueyingyi/Eimages_inpainting/image_ (0).jpg")] 
        
        # 并行处理所有样本
        with torch.no_grad():
            # 计算CLIP图文相似度
            clip_scores = self._compute_clip_similarity(completions, images)
            
            # 计算文本流畅度
            fluency_scores = self._compute_fluency(completions)
            
            # 计算多样性惩罚
            diversity_penalty = self._compute_diversity(completions)
            
            # 组合奖励（权重可调）
            total_scores = (
                0.6 * clip_scores + 
                0.3 * fluency_scores - 
                0.1 * diversity_penalty
            )
            
            # 按prompt分组归一化
            for i in range(len(prompts)):
                group_scores = total_scores[i*4 : (i+1)*4]  # 假设每个prompt生成4个样本
                normalized = (group_scores - group_scores.mean()) / (group_scores.std() + 1e-8)
                batch_rewards.append(normalized)
                
        return torch.cat(batch_rewards)
    
    def _compute_clip_similarity(self, texts, images):
        """计算图文语义相似度"""
        # 处理文本
        text_inputs = self.clip_tokenizer(
            texts, padding=True, return_tensors="pt"
        ).to(self.device)
        
        # 处理图像
        image_inputs = self.clip_processor(
            images=images, return_tensors="pt", padding=True
        ).to(self.device)
        
        # 获取特征
        text_features = self.clip_model.get_text_features(**text_inputs)
        image_features = self.clip_model.get_image_features(**image_inputs)
        
        # 计算余弦相似度
        similarity = torch.cosine_similarity(text_features, image_features, dim=-1)
        return similarity.cpu()
    
    def _compute_fluency(self, texts):
        """基于语言模型的困惑度评估流畅度"""
        scores = []
        for text in texts:
            inputs = self.gpt2_tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.gpt2_model(**inputs, labels=inputs.input_ids)
            perplexity = torch.exp(outputs.loss)
            scores.append(1 / perplexity.item())  # 困惑度越低越好
        return torch.tensor(scores)
    
    def _compute_diversity(self, texts, n_gram=3):
        """基于重复n-gram的多样性惩罚"""
        penalties = []
        for text in texts:
            tokens = text.lower().split()
            ngrams = [tuple(tokens[i:i+n_gram]) for i in range(len(tokens)-n_gram+1)]
            unique = set(ngrams)
            penalty = 1 - len(unique) / (len(ngrams) + 1e-8)
            penalties.append(penalty)
        return torch.tensor(penalties)

class GRPOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 解包数据
        pixel_values = inputs['pixel_values']
        input_ids = inputs['input_ids']
        raw_prompts = inputs['raw_prompts']
        
        # Step 1: 生成候选答案
        generation_output = model.grpo_generate(
            pixel_values=pixel_values,
            prompts=raw_prompts,
            generation_config=model.generation_config,
            num_generations=model.args.num_generations
        )
        
        # Step 2: 计算logits（示例）
        logits = model.compute_completion_logits(
            pixel_values=pixel_values,
            input_ids=generation_output['input_ids'],
            completion_ids=generation_output['completion_ids']
        )
        
        # Step 3: 计算参考模型logits
        with torch.no_grad():
            ref_logits = model.ref_model.compute_completion_logits(
                pixel_values=pixel_values,
                input_ids=generation_output['input_ids'],
                completion_ids=generation_output['completion_ids']
            )
        
        # Step 4: 模拟奖励计算
        rewards = torch.rand(len(raw_prompts) * model.args.num_generations).to(model.device)
        
        # Step 5: 计算GRPO损失
        loss = model.grpo_loss(
            logits=logits,
            ref_logits=ref_logits,
            rewards=rewards,
            mask=generation_output['completion_mask']
        )
        
        # 添加指标记录
        self.log({"train/reward": rewards.mean().item()})
        self.log({"train/kl_div": loss_details['kl_div']}) 
        
        return (loss, outputs) if return_outputs else loss

import json
from torch.utils.data import Dataset
from PIL import Image
import torch

class GRPODataset(Dataset):
    def __init__(self, data_path, tokenizer, image_size=448, max_prompts=512):
        """
        Args:
            data_path: jsonl文件路径
            tokenizer: 文本分词器
            image_size: 图像处理尺寸
            max_prompts: 单个样本最大prompt长度
        """
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_prompts = max_prompts
        self.img_start_token = IMG_START_TOKEN
        self.img_end_token = IMG_END_TOKEN
        self.img_context_token = IMG_CONTEXT_TOKEN

    def __len__(self):
        return len(self.data)

    def _process_image(self, image_path):
        """图像处理流水线"""
        image = Image.open(image_path).convert('RGB')
        return load_image_vlm(image, self.image_size)  # 使用您现有的图像处理函数

    def _build_prompt(self, conversations):
        """构建多轮对话prompt"""
        template = get_conv_template(self.tokenizer.template)
        for msg in conversations:
            if msg['from'] == 'human':
                template.append_message(template.roles[0], msg['value'])
            elif msg['from'] == 'gpt':
                template.append_message(template.roles[1], msg['value'])
        return template.get_prompt()

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 图像处理
        pixel_values = self._process_image(item['image'])  # [num_patches, C, H, W]
        
        # 文本处理
        prompt = self._build_prompt(item['conversations'])
        
        # 添加图像占位符
        num_patches = pixel_values.size(0)
        image_tokens = f"{self.img_start_token}{self.img_context_token*num_patches}{self.img_end_token}"
        prompt = prompt.replace('<image>', image_tokens, 1)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_prompts,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'pixel_values': pixel_values,      # [num_patches, C, H, W]
            'input_ids': inputs['input_ids'],  # [1, seq_len]
            'attention_mask': inputs['attention_mask'],
            'raw_prompt': prompt               # 用于后续生成
        }

class GRPODataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        """
        输入batch结构:
        [
            {
                'pixel_values': tensor[num_patches, C, H, W],
                'input_ids': tensor[1, seq_len],
                'attention_mask': tensor[1, seq_len],
                'raw_prompt': str
            },
            ...
        ]
        """
        # 合并图像特征
        pixel_values = [item['pixel_values'] for item in batch]
        
        # 合并文本输入
        input_ids = torch.cat([item['input_ids'] for item in batch])
        attention_mask = torch.cat([item['attention_mask'] for item in batch])
        
        # 保留原始prompt用于生成
        raw_prompts = [item['raw_prompt'] for item in batch]

        return {
            'pixel_values': torch.cat(pixel_values),  # [total_patches, C, H, W]
            'input_ids': input_ids,                   # [batch_size, seq_len]
            'attention_mask': attention_mask,
            'raw_prompts': raw_prompts
        }  
def main():
    # Apply necessary patches for the transformers library
    replace_llama_rmsnorm_with_fused_rmsnorm()
    replace_train_sampler()
    replace_train_dataloader()

    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.use_packed_ds = data_args.use_packed_ds

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=model_args.use_fast_tokenizer)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tcs_loader = TCSLoader('~/petreloss.conf') if has_tcs_loader else None

    if data_args.use_packed_ds:
        replace_internlm2_attention_class()
        replace_qwen2_attention_class()
        replace_phi3_attention_class()
        replace_llama_attention_class()

    if model_args.use_liger:
        from internvl.patch import apply_liger_kernel_to_internvit
        from liger_kernel.transformers import (apply_liger_kernel_to_llama,
                                               apply_liger_kernel_to_qwen2)
        apply_liger_kernel_to_llama()
        apply_liger_kernel_to_qwen2()
        # apply_liger_kernel_to_internvit()

    if model_args.model_name_or_path is not None:
        logger.info('Loading InternVLChatModel...')
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        model = InternVLChatModel.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config)
    else:
        logger.info('Loading ViT-6B...')
        vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
        vision_config.drop_path_rate = model_args.drop_path_rate
        vision_model = InternVisionModel.from_pretrained(
            model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config)
        logger.info('Loading LLaMA...')
        llm_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
        if llm_config.model_type == 'internlm2':
            model_type = InternLM2ForCausalLM
            llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            model_type = AutoModelForCausalLM
            llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        llm = model_type.from_pretrained(
            model_args.llm_path, torch_dtype=torch.bfloat16,
            config=llm_config, trust_remote_code=True)
        logger.info('Building InternVLChatConfig...')
        internvl_chat_config = InternVLChatConfig(
            vision_config.to_dict(), llm_config.to_dict(), downsample_ratio=data_args.down_sample_ratio,
            pad2square=data_args.pad2square, template=data_args.conv_style,
            select_layer=model_args.vision_select_layer, dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail, ps_version=model_args.ps_version,
            min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch)
        internvl_chat_config.force_image_size = data_args.force_image_size
        logger.info('Building InternVLChatModel...')
        model = InternVLChatModel(internvl_chat_config, vision_model, llm)
    model.img_context_token_id = img_context_token_id

    assert model.config.downsample_ratio == data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info('Finished')

    patch_size = model.config.vision_config.patch_size
    logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'model.config.vision_config.image_size: {model.config.vision_config.image_size}')
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                 new_size=data_args.force_image_size,
                                                 patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()


    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)
        
    if model_args.freeze_backbone:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    for name, param in model.vision_model.named_parameters():  
        print(f'{name}: requires_grad={param.requires_grad}')  


    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    # 加载参考模型
    if model_args.use_grpo:
        model.prepare_grpo(model_args.ref_model_path)
        ref_model = model.ref_model
    else:
        ref_model = None
    

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    for name, param in model.mlp1.named_parameters():  
        print(f'{name}: requires_grad={param.requires_grad}')  

    # set seed for torch dataloaders
    set_seed(training_args.seed)
    training_args.remove_unused_columns = False
    training_args.per_device_eval_batch_size=1

    
    collator = GRPODataCollator(
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length  # 例如2048
    )
    # 修改为
    ref_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)  # 加载参考模型
    train_dataset = GRPODataset(data_args.train_data)
    eval_dataset = GRPODataset(data_args.eval_data)
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=GRPODataCollator(tokenizer),
    )
    trainer.train()
    
    

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()


if __name__ == '__main__':
    main()
