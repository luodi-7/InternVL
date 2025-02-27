import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import json
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

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








# 定义图像路径和模型路径
image_base_path = '/mnt/afs/xueyingyi/image_vague/inpainting_demo/'
jsonl_path = '/mnt/afs/xueyingyi/meme/pause_data/three_item/C_generate_pause.jsonl'#这里是需要的用户输入，已经插入pause token在需要的地方（注意缺失不同的地方用不同pause token)
model_path = '/mnt/afs/xueyingyi/model/add_pause'
output_json_file = '/mnt/afs/xueyingyi/meme/generate/pause/inference/three_item/C_generate_pause_100_inference.jsonl'#输出文件
example_image_path = '/mnt/afs/xueyingyi/vl2.5/InternVL/inference/image_ (100).jpg'# few-shot的exmaple图片，image_ (100)作为示例

# 加载模型和tokenizer
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

# 读取提示文本
with open('/mnt/afs/xueyingyi/vl2.5/InternVL/inference/text_new.txt', 'r') as prompt_file:
    PROMPT = prompt_file.read()

with open('/mnt/afs/xueyingyi/vl2.5/InternVL/inference/text_example_100.txt', 'r') as prompt_file:# image_ (100)的分析过程
    PROMPT_example = prompt_file.read()
    
with open(output_json_file, 'w') as output_file:
    # 读取jsonl文件并处理每一行
    with open(jsonl_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            file_name = data['file_name']
            user_input = data['user_input']
            
            # 构建完整的图像路径
            image_path = f"{image_base_path}{file_name}"
            # 设置生成配置
            generation_config = dict(max_new_tokens=1024, do_sample=False, num_beams=1)
            # 加载示例图像和用户推理图像
            pixel_values_example = load_image(example_image_path, max_num=12).to(torch.bfloat16).cuda()
            pixel_values_user = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            
            # 拼接两张图像的像素值
            pixel_values = torch.cat((pixel_values_example, pixel_values_user), dim=0)
            num_patches_list = [pixel_values_example.size(0), pixel_values_user.size(0)]
            
            # 构建问题
            question = f'{PROMPT}<image>\n{PROMPT_example}\n<image>\n{user_input}'
            
            # 尝试生成响应
            try:
                response = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list)
                print(f'image: {image_path}\nAssistant: {response}') #resoponse是模型生成的新文本
                
                # 构建输出数据
                output_data = {
                    "image": image_path,  # 图片路径
                    "conversations": [
                        {"from": "human", "value": user_input},  
                        {"from": "inference", "value": response}  # 模型生成的文本
                    ]
                }
                
                # 写入输出文件
                output_file.write(json.dumps(output_data) + '\n')
            except RuntimeError as e:
                print(f"Error processing image {image_path}: {e}")