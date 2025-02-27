import argparse

import torch
import torch.nn as nn
from internvl.model.internvl_chat import InternVLChatModel
from transformers import AutoTokenizer
from transformers import AutoModel
from internvl.train.pause_class import (ExtendedEmbedding,ExtendedLinear)
def merge_embeddings(extended_embedding):
    original_embedding = extended_embedding.original_embedding
    new_embedding = extended_embedding.new_embedding

    # 合并权重
    merged_weight = torch.cat([original_embedding.weight, new_embedding.weight], dim=0)
    merged_embedding = nn.Embedding.from_pretrained(merged_weight, padding_idx=original_embedding.padding_idx)

    return merged_embedding

def merge_linear(extended_linear):
    original_linear = extended_linear.original_linear
    new_linear = extended_linear.new_linear

    # 合并权重
    merged_weight = torch.cat([original_linear.weight, new_linear.weight], dim=0)
    merged_linear = nn.Linear(
        original_linear.in_features,
        original_linear.out_features + new_linear.out_features,
        bias=False
    )
    merged_linear.weight.data = merged_weight

    return merged_linear
argparse = argparse.ArgumentParser()
argparse.add_argument('--input_path', type=str, help='Path to the input model')
argparse.add_argument('--output_path', type=str, help='Path to the output model')
args = argparse.parse_args()

print('Loading model...')
# 加载模型
model = AutoModel.from_pretrained(args.input_path, torch_dtype=torch.bfloat16)
print(model)
breakpoint()

# 替换 embed_tokens 和 lm_head
if hasattr(model, 'language_model'):
    original_embedding = model.language_model.get_input_embeddings()
    extended_embedding = ExtendedEmbedding(original_embedding, new_embedding)
    model.language_model.set_input_embeddings(extended_embedding)

    original_lm_head = model.language_model.get_output_embeddings()
    extended_lm_head = ExtendedLinear(original_lm_head, new_lm_head)
    model.language_model.set_output_embeddings(extended_lm_head)
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(args.input_path, trust_remote_code=True)

if model.config.use_backbone_lora:
    model.vision_model.merge_and_unload()
    model.vision_model = model.vision_model.model
    model.config.use_backbone_lora = 0
if model.config.use_llm_lora:
    model.language_model.merge_and_unload()
    model.language_model = model.language_model.model
    model.config.use_llm_lora = 0

extended_embedding = model.language_model.get_input_embeddings()
merged_embedding = merge_embeddings(extended_embedding)
model.language_model.set_input_embeddings(merged_embedding)

extended_lm_head = model.language_model.get_output_embeddings()
merged_lm_head = merge_linear(extended_lm_head)
model.language_model.set_output_embeddings(merged_lm_head)

print('Saving model...')
model.save_pretrained(args.output_path)
print('Saving tokenizer...')
tokenizer.save_pretrained(args.output_path)
print('Done!')
