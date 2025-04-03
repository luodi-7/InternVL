# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import torch

IGNORE_INDEX = -100


def pad_data_collator(features, pad_id=0):

    first = features[0]
    batch = {}

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids') and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    return batch


def concat_pad_data_collator(features, max_item_length=None, pad_id=0):

    first = features[0]
    batch = {}
    prompts_list = []
    image_list = []

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max_item_length or max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)

        if 'position_ids' in feat:
            temp_position_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_position_ids[:feat['position_ids'].shape[0]] = feat['position_ids']
            feat['position_ids'] = temp_position_ids

        if 'loss_weight' in feat:
            temp_loss_weight = torch.FloatTensor([pad_id] * max_item_length)
            temp_loss_weight[:feat['loss_weight'].shape[0]] = feat['loss_weight']
            feat['loss_weight'] = temp_loss_weight
        #收集 prompts
        if "raw_prompts" in feat:
            prompts_list.append(feat["raw_prompts"])
        if "image_path" in feat:
            image_list.append(feat["image_path"])

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)
        # 保留原始 prompt 文本
    if prompts_list:
        batch["raw_prompts"] = prompts_list
    if image_list:
        batch["image_path"] = image_list

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids', 'pixel_values', 'image_flags','raw_prompts','image_path') and \
                v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ('pixel_values', 'image_flags'):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])
        
    return batch

# def concat_pad_data_collator(features, pad_id=0):
#     if not features:
#         return {}

#     first = features[0]
#     batch = {}

#     # 确定批次内最大序列长度
#     input_ids_lengths = [len(feat["input_ids"]) for feat in features]
#     max_length = max(input_ids_lengths)

#     # 初始化存储容器
#     padded_input_ids = []
#     padded_attention_mask = []
#     prompts_list = []

#     # 对每个样本进行填充
#     for feat in features:
#         # 处理 input_ids
#         input_ids = feat["input_ids"]
#         pad_len = max_length - len(input_ids)
#         padded_input = np.pad(input_ids, (0, pad_len), mode="constant", constant_values=pad_id)
#         padded_input_ids.append(padded_input)

#         # 处理 attention_mask
#         attention_mask = feat.get("attention_mask", np.ones_like(input_ids))  # 默认全1
#         padded_attention = np.pad(attention_mask, (0, pad_len), mode="constant", constant_values=0)
#         padded_attention_mask.append(padded_attention)

#         # 收集 prompts
#         if "raw_prompts" in feat:
#             prompts_list.append(feat["raw_prompts"])

#     # 转换为张量
#     batch["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)
#     batch["attention_mask"] = torch.tensor(padded_attention_mask, dtype=torch.long)

#     # 保留原始 prompt 文本
#     if prompts_list:
#         batch["raw_prompts"] = prompts_list

#     # 处理多模态字段（如 pixel_values、image_flags）
#     for key in first.keys():
#         if key not in ["input_ids", "attention_mask", "raw_prompts"]:
#             # 张量直接堆叠
#             if isinstance(first[key], torch.Tensor):
#                 batch[key] = torch.stack([f[key] for f in features])
#             # numpy数组转为张量后堆叠
#             elif isinstance(first[key], np.ndarray):
#                 batch[key] = torch.tensor(np.stack([f[key] for f in features]))
#             # 其他类型保持为列表
#             else:
#                 batch[key] = [f[key] for f in features]

#     return batch
    
def dpo_concat_pad_data_collator(features, pad_id=0):

    first = features[0]
    batch = {}

    for prefix in ['chosen_', 'rejected_']:
        batch_lens = [feat[f'{prefix}input_ids'].shape[0] for feat in features]
        max_item_length = max(batch_lens)
        for idx in range(len(features)):
            feat = features[idx]
            temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_input_ids[:feat[f'{prefix}input_ids'].shape[0]] = feat[f'{prefix}input_ids']
            feat[f'{prefix}input_ids'] = temp_input_ids
            temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
            temp_labels[:feat[f'{prefix}labels'].shape[0]] = feat[f'{prefix}labels']
            feat[f'{prefix}labels'] = temp_labels
            feat[f'{prefix}attention_mask'] = feat[f'{prefix}input_ids'].ne(pad_id)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('pixel_values', 'image_flags') and \
                v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ('pixel_values', 'image_flags'):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])
    return batch
