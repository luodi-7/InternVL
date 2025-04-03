import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import json 
import os  
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
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
def filter_detections(origin_text, detections):  
    # If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
    path = '/fs-computility/ai-shen/shared/dilab/model/InternVL2_5-4B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    # # set the max number of tiles in `max_num`
    # pixel_values = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    # pure-text conversation (纯文本对话)
    system_prompt = """你是一个专业的文本匹配处理器，需要根据原始文本内容筛选和修正OCR检测结果。请严格遵循以下规则：  

1. **匹配规则**：比较每个检测框中的`text`是否出现在原始文本中（不区分大小写、换行符和标点符号）。允许模糊匹配，即考虑单词部分字母识别错误的情况（例如，原句是`"or we play..."`，识别为`"on we play..."`，仍然视为匹配）。  
2. **替换规则**：如果检测框中的`text`与原始文本匹配，保留该检测框，并将`text`替换为原始文本中的准确样式（保留原始大小写和标点）。  
3. **删除规则**：完全删除没有匹配项的检测框。  
4. **切分规则**：将原始文本合理切分到各个检测框中，确保每个检测框中的`text`仅包含与其匹配的部分，且原始文本中的每个词只能被匹配一次。切勿将整段原始文本分配给单个检测框。  
5. **补充规则**：每个检测框中的`text`只能包含与其匹配的原始文本部分，不得将其他检测框的内容补充到当前检测框中。如果检测框中的`text`已经与原始文本的部分内容匹配，则仅需修正识别错误的单词，不得额外补充其他内容。  
6. **输出格式**：始终返回目标格式的JSON数组，不要添加任何解释。  

示例输入1：  
origin_text: "That moment after you throw up and your friend asks you \"YOU GOOD BRO?\" I'M FUCKIN LIT\n"  
detections: [{'bbox': [1, 0, 151, 496], 'text': 'that moment after you throw up and your friend asks you "you good bro?"'}, {'bbox': [417, 138, 470, 373], 'text': "i'm fuckin lit"}, {'bbox': [481, 407, 499, 499], 'text': 'irunny.co'}]  

示例输出1：
```json  
[{'bbox': [1, 0, 151, 496], 'text': 'That moment after you throw up and your friend asks you \"YOU GOOD BRO?\"'}, {'bbox': [417, 138, 470, 373], 'text': "I\'M FUCKIN LIT"}]  
```
示例输入2：  
origin_text: "me\nfood at a potluck"  
detections: [{"bbox": [212, 103, 243, 164], "text": "good at a potluck"}, {"bbox": [131, 55, 146, 79], "text": "me"}]  

示例输出2：  
```json
[{"bbox": [212, 103, 243, 164], "text": "food at a potluck"}, {"bbox": [131, 55, 146, 79], "text": "me"}]  
```
现在：
"""  

    user_prompt = f"""origin_text: {json.dumps(origin_text)}  
    detections: {json.dumps(detections)}  

    请根据规则处理上述检测结果，直接返回处理后json数组："""
    question=system_prompt+user_prompt
    response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
    
    try:  
        print(f'User: {question}\nAssistant: {response}') 
        return json.loads(response)  
    except:  
        # 异常处理  
        print("error")  
        return []

def process_files(image_detections_file, text_file, output_file):  
    # 读取text文件  
    with open(text_file, 'r') as f:  
        texts = [json.loads(line) for line in f]  

    # 创建一个字典，以file_name为键，text为值  
    text_dict = {item['file_name']: item['text'] for item in texts}  

    # 打开输出文件准备写入  
    with open(output_file, 'w') as output_f:  
        # 读取image_detections文件并逐行处理  
        with open(image_detections_file, 'r') as input_f:  
            for line in input_f:  
                item = json.loads(line)  
                file_name = item['image_path'].split('/')[-1]  
                if file_name in text_dict:  
                    origin_text = text_dict[file_name]  
                    detections = item['detections']  
                    filtered_detections = filter_detections(origin_text, detections)  
                    item['detections'] = filtered_detections  
                
                # 将处理后的结果写入到输出文件  
                output_f.write(json.dumps(item) + '\n')

# 文件路径  
image_detections_file = '/fs-computility/ai-shen/xueyingyi/meme/image_vague/loc_dection/Eimages/processed_dections_quickmeme.jsonl'  
text_file = '/fs-computility/ai-shen/xueyingyi/meme/meme/generate/quickmeme/text.jsonl'  
output_file = '/fs-computility/ai-shen/xueyingyi/meme/image_vague/loc_dection/Eimages/processed_dections_quickmeme_update.jsonl'  

# 处理文件  
process_files(image_detections_file, text_file, output_file)

# question = 'Can you tell me a story?'
# response, history = model.chat(tokenizer, None, question, generation_config, history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# # single-image single-round conversation (单图单轮对话)
# question = '<image>\nPlease describe the image shortly.'
# response = model.chat(tokenizer, pixel_values, question, generation_config)
# print(f'User: {question}\nAssistant: {response}')

# # single-image multi-round conversation (单图多轮对话)
# question = '<image>\nPlease describe the image in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'Please write a poem according to the image.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# # multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
# pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

# question = '<image>\nDescribe the two images in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'What are the similarities and differences between these two images.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# # multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
# pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
# num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

# question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list,
#                                history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'What are the similarities and differences between these two images.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list,
#                                history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# # batch inference, single image per sample (单图批处理)
# pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
# num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

# questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
# responses = model.batch_chat(tokenizer, pixel_values,
#                              num_patches_list=num_patches_list,
#                              questions=questions,
#                              generation_config=generation_config)
# for question, response in zip(questions, responses):
#     print(f'User: {question}\nAssistant: {response}')

# # video multi-round conversation (视频多轮对话)
# def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
#     if bound:
#         start, end = bound[0], bound[1]
#     else:
#         start, end = -100000, 100000
#     start_idx = max(first_idx, round(start * fps))
#     end_idx = min(round(end * fps), max_frame)
#     seg_size = float(end_idx - start_idx) / num_segments
#     frame_indices = np.array([
#         int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
#         for idx in range(num_segments)
#     ])
#     return frame_indices

# def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
#     vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
#     max_frame = len(vr) - 1
#     fps = float(vr.get_avg_fps())

#     pixel_values_list, num_patches_list = [], []
#     transform = build_transform(input_size=input_size)
#     frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
#     for frame_index in frame_indices:
#         img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
#         img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
#         pixel_values = [transform(tile) for tile in img]
#         pixel_values = torch.stack(pixel_values)
#         num_patches_list.append(pixel_values.shape[0])
#         pixel_values_list.append(pixel_values)
#     pixel_values = torch.cat(pixel_values_list)
#     return pixel_values, num_patches_list

# video_path = './examples/red-panda.mp4'
# pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
# pixel_values = pixel_values.to(torch.bfloat16).cuda()
# video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
# question = video_prefix + 'What is the red panda doing?'
# # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list, history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'Describe this video in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list, history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')