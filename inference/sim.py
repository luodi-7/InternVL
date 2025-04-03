import json
import re
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from FlagEmbedding import FlagModel
import matplotlib.pyplot as plt

import json
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
# 测试代码
from transformers import GenerationConfig
# ================= 配置参数 =================
TEST_JSONL_PATH = '/fs-computility/ai-shen/xueyingyi/cot_picture/annotations/eval_data.jsonl'
MODEL_PATH = '/fs-computility/ai-shen/xueyingyi/model/cot_box_text'
SIMILARITY_MODEL_PATH = '/fs-computility/ai-shen/shared/dilab/model/bge-base-zh-v1.5'
OUTPUT_JSONL_PATH = '/fs-computility/ai-shen/xueyingyi/metric/output.jsonl'
SIMILARITY_PLOT_PATH = '/fs-computility/ai-shen/xueyingyi/metric/similarity_distribution.png'
# 定义图像预处理相关的参数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    print(f"Processed {len(images)} blocks for image {image_file}")
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
# ================= 辅助函数 =================
def extract_meme_text(text):
    """从文本中提取'Text on the Meme:'之后的内容"""
    match = re.search(r'Text on the Meme:\s*\n(.*?)(?=\n\n|\Z)', text, re.DOTALL)
    return match.group(1).strip() if match else ""

# ================= 相似度计算模型 =================
class SimilarityCalculator:
    def __init__(self):
        self.model = FlagModel(
            SIMILARITY_MODEL_PATH,
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
            use_fp16=True
        )
    
    def calculate(self, text1, text2):
        if not text1.strip() or not text2.strip():
            return 0.0
        emb1 = self.model.encode(text1)
        emb2 = self.model.encode(text2)
        return torch.nn.functional.cosine_similarity(
            torch.tensor(emb1).unsqueeze(0),
            torch.tensor(emb2).unsqueeze(0)
        ).item()

# ================= 主处理流程 =================
def main():
    # 初始化组件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    similarity_model = SimilarityCalculator()
    
    # 加载生成模型
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 结果存储
    all_scores = []
    results = []

    # 处理测试数据
    with open(TEST_JSONL_PATH, 'r') as test_file, \
         open(OUTPUT_JSONL_PATH, 'w') as output_file:

        for line in test_file:
            data = json.loads(line)
            
            try:
                # 解析输入数据
                image_path = data['image']
                human_input = next(c['value'] for c in data['conversations'] if c['from'] == 'human')
                label_text = next(c['value'] for c in data['conversations'] if c['from'] == 'gpt')
                
                # 加载图像并进行预处理
                pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
                assert pixel_values.numel() > 0, "Pixel values are empty!"

                # 生成文本
                # 在生成前确保输入梯度
                pixel_values = pixel_values.clone().requires_grad_(True)

                # 调整生成配置
                generation_config = {
                    'max_new_tokens': 1024,
                    'do_sample': True,
                    'num_beams': 1,
                }

                response = model.chat(
                    tokenizer,
                    pixel_values,
                    human_input,
                    generation_config
                )
                print(response)
                

                # 提取关键文本
                generated_meme = extract_meme_text(response)
                label_meme = extract_meme_text(label_text)
                print(generated_meme)
                print(label_meme)
                # 计算相似度
                similarity = similarity_model.calculate(generated_meme, label_meme)
                all_scores.append(similarity)

                # 保存结果
                result = {
                    'image_path': image_path,
                    'human_input': human_input,
                    'generated_response': response,
                    'generated_meme': generated_meme,
                    'label_meme': label_meme,
                    'similarity': similarity
                }
                results.append(result)
                output_file.write(json.dumps(result) + '\n')

            except Exception as e:
                print(f"Error processing sample {data.get('id', 'unknown')}: {str(e)}")
                continue

if __name__ == "__main__":
    main()