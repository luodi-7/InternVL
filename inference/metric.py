# ================= 新增导入 =================
from pathlib import Path  # 添加在文件开头的import部分
import numpy as np
import torch

from transformers import AutoModel, AutoTokenizer
from FlagEmbedding import FlagModel
import matplotlib.pyplot as plt
# 在文件开头的import部分添加
from transformers import CLIPModel, AutoProcessor
import json
import re
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from FlagEmbedding import FlagModel



import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
# ================= 新增配置参数 =================
USER_INPUT_JSONL_PATH = '/fs-computility/ai-shen/xueyingyi/meme/meme/generate/quickmeme/user_input.jsonl'
TEST_JSONL_PATH = '/fs-computility/ai-shen/xueyingyi/select_group/annotations/test_data.jsonl'
MODEL_PATH = '/fs-computility/ai-shen/xueyingyi/model/quickmeme_cot'
SIMILARITY_MODEL_PATH = '/fs-computility/ai-shen/shared/dilab/model/bge-base-zh-v1.5'
OUTPUT_JSONL_PATH = '/fs-computility/ai-shen/xueyingyi/metric4/output.jsonl'
SIMILARITY_PLOT_PATH = '/fs-computility/ai-shen/xueyingyi/metric4/similarity_distribution.png'
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

# ================= 新增数据处理类 =================
class UserInputLoader:
    def __init__(self, jsonl_path):
        self.data = {}
        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line)
                # 将字典值转换为规范化字符串
                input_str = self._format_input(entry["user_input"])
                self.data[entry["file_name"]] = input_str
    
    def _format_input(self, user_input):
        """将用户输入字典转换为文本描述"""
        parts = []
        for key, value in user_input.items():
            if isinstance(value, list):
                value = ", ".join(value)
            parts.append(f"{key}: {value}")
        return "; ".join(parts)
    
    def get_input(self, image_path):
        """从图像路径提取文件名并获取用户输入"""
        file_name = Path(image_path).name
        return self.data.get(file_name, "")

# ================= 修改主函数 =================
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
    # 初始化CLIP模型
    clip_model = CLIPModel.from_pretrained(
        "/fs-computility/ai-shen/shared/dilab/model/clip-vit-base-patch32"
    ).to(device)
    clip_processor = AutoProcessor.from_pretrained(
        "/fs-computility/ai-shen/shared/dilab/model/clip-vit-base-patch32"
    )
    
    # 新增初始化组件
    user_input_loader = UserInputLoader(USER_INPUT_JSONL_PATH)
    all_user_scores_gen = []
    all_user_scores_label = []
    all_clip_scores_gen = []
    all_clip_scores_label = []
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
                generation_config = {
                    'max_new_tokens': 1024,
                    'do_sample': False,
                    'num_beams': 1
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
                breakpoint()
                # 新增用户输入获取
                user_input_text = user_input_loader.get_input(image_path)
                if user_input_text:
                    user_sim_gen = similarity_model.calculate(generated_meme, user_input_text)
                    user_sim_label = similarity_model.calculate(label_meme, user_input_text)
                else:
                    user_sim_gen = user_sim_label = 0.0
                    print(f"Warning: No user input for {Path(image_path).name}")

                breakpoint()
                # 修改CLIP计算部分以包含标签
                clip_score_gen = 0.0
                clip_score_label = 0.0
                try:
                    clip_image = Image.open(image_path).convert("RGB")
                    
                    # 计算生成文本的CLIP分数
                    inputs_gen = clip_processor(
                        text=[generated_meme], 
                        images=clip_image,
                        return_tensors="pt", 
                        padding=True
                    ).to(device)
                    
                    # 计算标签文本的CLIP分数
                    inputs_label = clip_processor(
                        text=[label_meme], 
                        images=clip_image,
                        return_tensors="pt", 
                        padding=True
                    ).to(device)
                    
                    with torch.no_grad():
                        clip_score_gen = clip_model(**inputs_gen).logits_per_image[0].item()
                        clip_score_label = clip_model(**inputs_label).logits_per_image[0].item()
                        
                except Exception as e:
                    print(f"CLIP处理失败: {str(e)}")

                # 保存结果
                result = {
                    'image_path': image_path,
                    'human_input': human_input,
                    'generated_response': response,
                    'generated_meme': generated_meme,
                    'label_meme': label_meme,
                    'text_similarity': similarity
                }

                # 更新结果存储
                result.update({
                    'user_similarity_gen': user_sim_gen,
                    'user_similarity_label': user_sim_label,
                    'clip_score_gen': clip_score_gen,
                    'clip_score_label': clip_score_label
                })
                breakpoint()
                results.append(result)
                output_file.write(json.dumps(result) + '\n')
                
                
                # 收集所有分数
                all_user_scores_gen.append(user_sim_gen)
                all_user_scores_label.append(user_sim_label)
                all_clip_scores_gen.append(clip_score_gen)
                all_clip_scores_label.append(clip_score_label)

            except Exception as e:
                print(f"Error processing sample {data.get('id', 'unknown')}: {str(e)}")
                continue
    # ================= 结果分析 =================
    # 计算统计指标
    # 文本相似度统计
    if all_scores:
        avg_score = np.nanmean(all_scores)
        max_score = np.nanmax(all_scores) if not np.isnan(all_scores).all() else 0
        min_score = np.nanmin(all_scores) if not np.isnan(all_scores).all() else 0
    else:
        avg_score = max_score = min_score = 0.0
    print(f"Total Samples: {len(all_scores)}")
    # 用户相似度统计
    user_gen_avg = np.nanmean(all_user_scores_gen) if all_user_scores_gen else 0.0
    user_label_avg = np.nanmean(all_user_scores_label) if all_user_scores_label else 0.0

    # CLIP统计
    clip_gen_avg = np.nanmean(all_clip_scores_gen) if all_clip_scores_gen else 0.0
    clip_label_avg = np.nanmean(all_clip_scores_label) if all_clip_scores_label else 0.0
    # 绘制分布图
    if all_scores:
        plt.figure(figsize=(10, 6))
        plt.hist(all_scores, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Similarity Score Distribution')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig(SIMILARITY_PLOT_PATH)
        plt.close()    
    

    # ================= 新增可视化函数 =================
    def plot_comparison(scores_a, scores_b, labels, ylabel, filename):
        plt.figure(figsize=(15, 6))
        indices = np.arange(len(scores_a))
        
        plt.plot(indices, scores_a, 'b-', label=labels[0], alpha=0.7)
        plt.plot(indices, scores_b, 'r--', label=labels[1], alpha=0.7)
        
        plt.xticks(indices, [f"Sample {i+1}" for i in indices], rotation=45)
        plt.ylabel(ylabel)
        plt.title(f"Comparison of {ylabel}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # 生成对比图表
    if all_user_scores_gen and all_user_scores_label:
        plot_comparison(
            all_user_scores_gen, all_user_scores_label,
            labels=['Generated vs User', 'Label vs User'],
            ylabel='Text Similarity',
            filename='/fs-computility/ai-shen/xueyingyi/metric4/text_user_comparison.png'
        )
    if all_clip_scores_gen and all_clip_scores_label:
        plot_comparison(
            all_clip_scores_gen, all_clip_scores_label,
            labels=['Generated CLIP', 'Label CLIP'],
            ylabel='CLIP Score',
            filename='/fs-computility/ai-shen/xueyingyi/metric4/clip_comparison.png'
        )

    # ================= 更新统计输出 =================
    with open('/fs-computility/ai-shen/xueyingyi/metric4/output_sim.txt', 'a') as f:
        f.write("=== Text Similarity ===")
        f.write(f"Average Similarity: {avg_score:.4f}\n")
        f.write(f"Max Similarity: {max(all_scores):.4f}\n")
        f.write(f"Min Similarity: {min(all_scores):.4f}\n")
        f.write("\n\n=== User Input Similarity ===")
        f.write(f"\nGenerated Average: {np.mean(all_user_scores_gen):.4f}")
        f.write(f"\nLabel Average: {np.mean(all_user_scores_label):.4f}")
        
        f.write("\n\n=== CLIP Scores ===")
        f.write(f"\nGenerated Average: {np.mean(all_clip_scores_gen):.4f}")
        f.write(f"\nLabel Average: {np.mean(all_clip_scores_label):.4f}")
if __name__ == "__main__":
    main()