import json
import re
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from tqdm import tqdm
from torchvision import transforms
from FlagEmbedding import FlagModel
import matplotlib.pyplot as plt
import torchvision.transforms as T
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageDraw, ImageFont,ImageColor  
import os  
import uuid  
from typing import List, Tuple, Optional  

# ================= 配置参数 =================
TEST_JSONL_PATH = '/fs-computility/ai-shen/xueyingyi/cot_picture/annotations/sample.jsonl'
MODEL_PATH = '/fs-computility/ai-shen/xueyingyi/model/cot_box_text'
SIMILARITY_MODEL_PATH = '/fs-computility/ai-shen/shared/dilab/model/bge-base-zh-v1.5'
OUTPUT_JSONL_PATH = '/fs-computility/ai-shen/xueyingyi/metric/output.jsonl'
IMG_OUTPUT_DIR = '/fs-computility/ai-shen/xueyingyi/cot_picture/quickmeme_drawn_reward'
SIMILARITY_PLOT_PATH = '/fs-computility/ai-shen/xueyingyi/metric/similarity_distribution.png'
# ================= 绘图相关 =================
def is_color_close_to_black(color, threshold=0.5):  
    """  
    判断颜色是否接近黑色  

    Args:  
        color: 颜色，可以是颜色名称字符串，也可以是 RGB 元组  
        threshold: 亮度阈值，0 到 1 之间，值越小越接近黑色  

    Returns:  
        True 如果颜色接近黑色，否则 False  
    """  
    try:  
        # 将颜色转换为 RGB 元组  
        rgb = color 
    except ValueError:  
        print(f"Invalid color format: {color}")  
        return False  

    # 计算颜色的亮度 (Luma)  
    # 亮度计算公式: Y = 0.299 * R + 0.587 * G + 0.114 * B  
    luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]  

    # 将亮度值归一化到 0 到 1 之间  
    normalized_luma = luma / 255.0  

    # 如果亮度低于阈值，则认为颜色接近黑色  
    return normalized_luma < threshold


def draw_multiline_text(draw, position, text, font, max_width, fill, line_spacing=5):  
    """  
    在图像上绘制多行文本  

    Args:  
        draw: ImageDraw 对象  
        position: 文本起始位置 (x, y)  
        text: 要绘制的文本  
        font: 使用的字体  
        max_width: 最大行宽  
        fill: 字体颜色  
        line_spacing: 行间距  
    """  
    lines = []  
    words = text.split()  
    current_line = ""  

    for word in words:  
        # 检查添加下一个单词后行的宽度  
        test_line = f"{current_line} {word}".strip()  # 包含空格  
        if draw.textsize(test_line, font=font)[0] <= max_width:  
            current_line = test_line  
        else:  
            if current_line:  
                lines.append(current_line)  
            current_line = word  

    if current_line:  
        lines.append(current_line)  

    # 在图像上绘制每一行文字  
    y_offset = 0  
    for line in lines:  
        draw.text((position[0], position[1] + y_offset), line, font=font, fill=fill) 
        print(font.size)

        y_offset += font.getsize(line)[1] + line_spacing  # 获取行高并增加行间距

def draw_multiline_text_with_outline(draw, position, text, font, max_width, fill,  
                                     outline_color="black", outline_width=2, line_spacing=5,  
                                     alignment="center"):  # 默认居中 
    """  
    绘制带描边的多行文本，支持左对齐、右对齐和居中对齐。  
    """  
    lines = []  
    words = text.split()  
    current_line = ""  

    for word in words:  
        test_line = f"{current_line} {word}".strip()  
        if draw.textsize(test_line, font=font)[0] <= max_width:  
            current_line = test_line  
        else:  
            if current_line:  
                lines.append(current_line)  
            current_line = word  

    if current_line:  
        lines.append(current_line)  

    x, y = position  
    y_offset = 0  
    for line in lines:  
        line_width = draw.textsize(line, font=font)[0]  
        if alignment == "center":  
            x_offset = (max_width - line_width) / 2  
        elif alignment == "right":  
            x_offset = max_width - line_width  
        else:  # 默认或 "left"  
            x_offset = 0  

        x_position = x + x_offset  # 计算实际的 x 坐标  

        # 绘制描边  
        for dx, dy in [(0, -outline_width), (0, outline_width),  
                       (-outline_width, 0), (outline_width, 0),  
                       (-outline_width, -outline_width), (-outline_width, outline_width),  
                       (outline_width, -outline_width), (outline_width, outline_width)]:  
            draw.text((x_position + dx, y + y_offset + dy), line, font=font, fill=outline_color)  

        # 绘制文本  
        draw.text((x_position, y + y_offset), line, font=font, fill=fill)  
        y_offset += font.getsize(line)[1] + line_spacing  


def get_contrasting_color(color):  
    """  
    根据给定的背景颜色计算反色，并进一步增强与背景颜色的对比度。  
    Args:  
        color: RGB 元组，例如 (255, 255, 255)  
    Returns:  
        选择的颜色元组。  
    """  
    # 计算颜色的亮度（luminance）  
    def calculate_luminance(color):  
        r, g, b = color  
        r = r / 255.0  
        g = g / 255.0  
        b = b / 255.0  
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4  
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4  
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4  
        return 0.2126 * r + 0.7152 * g + 0.0722 * b  

    # 计算对比度  
    def calculate_contrast(color1, color2):  
        luminance1 = calculate_luminance(color1)  
        luminance2 = calculate_luminance(color2)  
        if luminance1 > luminance2:  
            return (luminance1 + 0.05) / (luminance2 + 0.05)  
        else:  
            return (luminance2 + 0.05) / (luminance1 + 0.05)  

    # 计算反色  
    inverted_color = tuple(255 - c for c in color[:3])  # 只处理 RGB  

    # 计算反色与背景颜色的对比度  
    contrast = calculate_contrast(color, inverted_color)  

    # 如果对比度不足，调整反色的亮度以增强对比度  
    min_contrast = 4.5  # WCAG 标准的最小对比度  
    if contrast < min_contrast:  
        background_luminance = calculate_luminance(color)  
        if background_luminance > 0.5:  # 背景较亮，使用黑色  
            inverted_color = (0, 0, 0)  
        else:  # 背景较暗，使用白色  
            inverted_color = (255, 255, 255)  

    return inverted_color  





def generate_image_with_text(  
    uid: str,  
    base_image: str,  
    font_type: str,  
    detections: List[Tuple[int, int, int, int]],  
    texts: List[str],  
    output_dir: str = "输出目录",  
    font_sizes: Optional[List[int]] = None,  
    font_colors: Optional[List[Tuple[int, int, int]]] = None,
    outline_colors: Optional[List[Tuple[int, int, int]]] = None,  
    outline_width: Optional[int] = 2,
    alignments: Optional[List[str]] = None,  
    bold: bool = False,  
    italic: bool = False,  
):  
    """  
    在底图上添加文本并保存生成的图片。支持用户自定义字体大小、颜色、对齐方式等。  
    """  
    # 确保输出目录存在  
    os.makedirs(output_dir, exist_ok=True)  

    # 加载底图（假设底图路径是根据 base_image 生成的）  
    image_path = base_image  
    if not os.path.exists(image_path):  
        raise FileNotFoundError(f"Base image not found: {image_path}")  

    image = Image.open(image_path)  

    draw = ImageDraw.Draw(image)  

    # 加载字体（假设字体文件在 fonts 目录下）  
    font_path = os.path.join("/usr/share/fonts/truetype/dejavu", font_type)  
    if not os.path.exists(font_path):  
        raise FileNotFoundError(f"Font not found: {font_path}")  

    # 初始化默认值  
    if font_sizes is None:  
        font_sizes = [None] * len(texts)  # 动态调整字体大小  
    if font_colors is None:  
        font_colors = [(255,255,255)] * len(texts)  # 使用反色  
    if outline_colors is None:  
        outline_colors = [None] * len(texts)  # 使用反色 
    if alignments is None:  
        alignments = ["center"] * len(texts)  # 默认居中  

    # 遍历检测框和文本  
    for i, (detection, text) in enumerate(zip(detections, texts)):  
        (startY, startX, endY, endX) = detection  

        # 计算文本框的宽度和高度
          

        box_width = endX - startX  
        box_height = endY - startY
         # 检查并调整 x 坐标  
        if startX < 3:  
            startX = 3  
            if endX <= startX:  # 确保框存在  
                endX = startX + 3  
        elif endX > image.width - 3:  
            endX = image.width - 3  
            if startX >= endX:  # 确保框存在  
                startX = endX - 3  

        # 检查并调整 y 坐标  
        if startY < 3:  
            startY = 3  
            if endY <= startY:  # 确保框存在  
                endY = startY + 3  
        elif endY > image.height - 3:  
            endY = image.height - 3  
            if startY >= endY:  # 确保框存在  
                startY = endY - 3 

        # draw.rectangle([startX, startY, endX, endY], outline="red", width=2)  

        # 动态调整字体大小（如果未指定字体大小）  
        if font_sizes[i] is None:  
            font_size = 1  # 初始字体大小  
            max_font_size = min(box_width, box_height) * 2  # 最大字体大小（基于文本框尺寸）  

            # 逐步增加字体大小，直到文本超出文本框或达到最大字体大小  
            while font_size < max_font_size:  
                font = ImageFont.truetype(font_path, font_size)  
                lines = []  
                current_line = ""  
                words = text.split()  
                
                for word in words:  
                    test_line = f"{current_line} {word}".strip()  
                    if draw.textsize(test_line, font=font)[0] <= box_width:  
                        current_line = test_line  
                    else:  
                        if current_line:  
                            lines.append(current_line)  
                        current_line = word  

                if current_line:  
                    lines.append(current_line)  

                # 计算文本的总高度和每行最大宽度
                text_width = max(draw.textsize(line, font=font)[0] for line in lines)
                text_height = sum(font.getsize(line)[1] for line in lines)

                if text_width > box_width or text_height > box_height:  
                    break  

                font_size += 1  

            # 退回到最后一个合适的字体大小  
            font_size -= int(font_size/5)
        else:  
            font_size = font_sizes[i]  

        # 加载字体（支持加粗和斜体）  
        try:  
            if bold and italic:  
                font_path_variant = os.path.join("fonts", font_type.replace(".ttf", "-BoldItalic.ttf"))  
            elif bold:  
                font_path_variant = os.path.join("fonts", font_type.replace(".ttf", "-Bold.ttf"))  
            elif italic:  
                font_path_variant = os.path.join("fonts", font_type.replace(".ttf", "-Italic.ttf"))  
            else:  
                font_path_variant = font_path  

            font = ImageFont.truetype(font_path_variant, font_size)  
        except Exception as e:  
            print(f"加载字体失败: {e}")  
            font = ImageFont.load_default()  

        # 计算文本位置并绘制文本  
        if font_colors[i] is None:  
            # 获取文本框区域的平均颜色  
            box_region = image.crop((startX, startY, endX, endY))  
            average_color = box_region.resize((1, 1)).getpixel((0, 0))  
            # 获取与背景颜色对比的字体颜色  
            font_color = get_contrasting_color(average_color)  
        else:  
            font_color = font_colors[i]
        #描边颜色
        if outline_colors[i] is None:
            # 判断 font_color 是否更接近黑色
            if is_color_close_to_black(font_color):  
                outline_color = (255,255,255)  
            else:  
                outline_color = (0,0,0)
        else:  
            outline_color = outline_colors[i]

        # 重新计算文本并缩小字体直到适应文本框
        lines = []  
        current_line = ""  
        words = text.split()  
        for word in words:  
            test_line = f"{current_line} {word}".strip()  
            if draw.textsize(test_line, font=font)[0] <= box_width:  
                current_line = test_line  
            else:  
                if current_line:  
                    lines.append(current_line)  
                current_line = word  

        if current_line:  
            lines.append(current_line)  

        # 设定最大字体大小为32  
        max_font_size = 32  

        # 计算每行文本的最大宽度和总高度  
        text_width = max(draw.textsize(line, font=font)[0] for line in lines)  
        text_height = sum(font.getsize(line)[1] for line in lines)  

        while text_width > box_width or text_height > box_height:  
            if font_size > max_font_size:  
                font_size = max_font_size  # 强制设置为最大字体大小  
                font = ImageFont.truetype(font_path_variant, font_size)  
                break  # 停止调整，避免进一步减小  

            font_size -= 1  # 缩小字体  
            font = ImageFont.truetype(font_path_variant, font_size)  
            lines = []  
            current_line = ""  
            for word in words:  
                test_line = f"{current_line} {word}".strip()  
                if draw.textsize(test_line, font=font)[0] <= box_width:  
                    current_line = test_line  
                else:  
                    if current_line:  
                        lines.append(current_line)  
                    current_line = word  
            if current_line:  
                lines.append(current_line)  

            text_width = max(draw.textsize(line, font=font)[0] for line in lines)  
            text_height = sum(font.getsize(line)[1] for line in lines)  

        # 在给定文本框内绘制多行文本  
        draw_multiline_text_with_outline(draw, (startX, startY), text, font, box_width, font_color, outline_color=outline_color, outline_width=outline_width, alignment=alignments[i])  

    # 生成唯一的文件名  
    output_filename = f"output_image_{uid}_{uuid.uuid4().hex}.jpg"  
    output_path = os.path.join(output_dir, output_filename)  
    image.save(output_path)  

    print(f"图片已保存到: {output_path}")  
    return output_path  


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
        self.detections_db = self._load_detections()
    def _load_detections(self) -> dict[str, list]:
        """加载检测框数据库"""
        detections_path = "/fs-computility/ai-shen/xueyingyi/cot_picture/dections_quickmeme_sort.jsonl"
        db = {}
        with open(detections_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                db[data['image_path']] = [
                    {
                        'bbox': [int(x) for x in d['bbox']],  # 转换为整数
                        'text': d['text']
                    } for d in data['detections']
                ]
        return db
    def calculate(self, text1, text2):
        if not text1.strip() or not text2.strip():
            return 0.0
        emb1 = self.model.encode(text1)
        emb2 = self.model.encode(text2)
        return torch.nn.functional.cosine_similarity(
            torch.tensor(emb1).unsqueeze(0),
            torch.tensor(emb2).unsqueeze(0)
        ).item()
    #===============之前的单图评分=====================
    # def draw_scores(self, gen_text: str, image_path: str, 
    #                group_id: str, answer_id: str) -> dict:
    #     """处理图文合成和评分"""
    #     # 创建输出目录
    #     output_dir = os.path.join(IMG_OUTPUT_DIR, os.path.basename(os.path.dirname(image_path)))
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     # 生成两种图片并获取路径
    #     gen_image_path = self._process_single_image(
    #         image_path, gen_text, output_dir, 
    #         f"gen_{group_id}_{answer_id}"
    #     )
        
    #     # 模拟评分（后续可替换为真实评分模型）
    #     gen_score = np.random.rand()
        
    #     return {
    #         'gen_image': gen_image_path,
    #         'gen_score': float(gen_score),
    #     }
    #==================================================

    def generate_image(self, gen_text: str, image_path: str,
                    group_id: str, answer_id: str) -> str:
        """处理图文合成，返回生成的图片路径"""
        # 创建输出目录
        output_dir = os.path.join(IMG_OUTPUT_DIR, os.path.basename(os.path.dirname(image_path)))
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成图片并获取路径
        gen_image_path = self._process_single_image(
            image_path, gen_text, output_dir,
            f"gen_{group_id}_{answer_id}"
        )
        return gen_image_path

    def score_group(self, image_paths: list) -> list:
        """对同一组所有生成的图片进行评分"""
        # 替换为真实的评分模型
        # 输入是所有图片路径的列表，输出是对应的分数列表
        return [float(np.random.rand()) for _ in image_paths]

    def _process_single_image(self, image_path: str, text: str, 
                             output_dir: str, prefix: str) -> str:
        """处理单个图文合成"""
        # 获取检测框
        new_image_path = image_path.replace('cot_picture/quickmeme_drawn/', 'quickmeme_inpainting/')  
        

        detections = self.detections_db.get(new_image_path, [])

        
        # 解析生成文本
        text_blocks = [t.split(":")[1] for t in text.split("\n") if ":" in t]
        
        
        # 构建检测框和文本列表
        formatted_detections = []
        formatted_texts = []
        last_box_text = []
        
        for i, d in enumerate(detections):
            if i < len(text_blocks):
                formatted_detections.append(d['bbox'])
                formatted_texts.append(text_blocks[i])
            else:
                last_box_text.append(text_blocks[i])
        
        # 处理剩余文本
        if len(text_blocks) > len(detections):
            remaining_text = "\n".join(text_blocks[len(detections):])
            if detections:
                formatted_texts[-1] += "\n" + remaining_text
            else:
                # 如果没有检测框，创建默认区域
                formatted_detections = [[0, 0, 100, 100]]  # 默认检测框
                formatted_texts = [remaining_text]
        
        # 生成唯一文件名
        base_name = os.path.basename(image_path).split('.')[0]
        output_path = os.path.join(
            output_dir, 
            f"{prefix}_{base_name}_{uuid.uuid4().hex[:6]}.jpg"
        )
        detections = [tuple(det) for det in formatted_detections]
        
        
        # 调用绘图函数
        generate_image_with_text(
            uid=prefix,
            base_image=new_image_path,
            font_type="DejaVuSans.ttf",
            detections=detections,
            texts=formatted_texts,
            output_dir=output_dir,
            outline_width=4,
            bold=True
        )
        return output_path

class BatchProcessor:
    def __init__(self, model, tokenizer, similarity_model, batch_size=4, num_generations=16):
        self.model = model
        self.tokenizer = tokenizer
        self.similarity_model = similarity_model
        self.batch_size = batch_size
        self.num_generations = num_generations

    def process_batch(self, batch_data):
        # 准备批量数据
        pixel_values_list = []
        num_patches_list = []
        questions = []
        original_data = []
        
        # 预处理图像和问题
        for data in batch_data:
            image_path = data['image']
            human_input = next(c['value'] for c in data['conversations'] if c['from'] == 'human')
            
            # 加载图像
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            num_patches = pixel_values.size(0)
            
            # 扩展为多个生成样本
            pixel_values_list.extend([pixel_values] * self.num_generations)
            num_patches_list.extend([num_patches] * self.num_generations)
            questions.extend([human_input] * self.num_generations)
            original_data.append(data)

        # 合并图像特征 (总batch_size = len(batch_data)*num_generations)
        pixel_values = torch.cat(pixel_values_list, dim=0)
        
        # 配置生成参数
        generation_config = {
            'max_new_tokens': 1024,
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9,
            'num_beams': 1,
        }

        # 批量生成
        
        responses, generation_output,input_ids = self.model.batch_chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            questions=questions,
            num_patches_list=num_patches_list,
            generation_config=generation_config
        )#generation_output为实际answer_ids和logits
        completion_ids = generation_output.sequences
        
        tokenizer=self.tokenizer
        is_eos = completion_ids == tokenizer.eos_token_id  # 布尔张量，形状为 (batch_size, seq_len)
        # 初始化张量，用于存储每个序列中第一个EOS的索引。如果没有找到EOS，默认使用序列长度。
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
        
        # 找到包含至少一个EOS的序列。
        mask_exists = is_eos.any(dim=1)
        # 对于包含EOS的序列，更新eos_idx为第一个EOS的位置索引。
        eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
        
        # 创建张量，包含每个序列位置的索引 [0, 1, 2, ..., seq_len-1]。
        # 并将其扩展到与序列数量一致。
        sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
        
        # 构建掩码：位置索引小于等于第一个EOS位置的标记为1。
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        


        # 重组结果为 [batch_size, num_generations]
        grouped_responses = []
        for i in range(len(batch_data)):
            start = i * self.num_generations
            end = (i+1) * self.num_generations
            batch_responses = responses[start:end]
            
            # 提取文本和token ids
            text_list = [extract_meme_text(r) for r in batch_responses]
   
            grouped_responses.append(text_list)



        return {
            'original_data':original_data,
            'grouped_responses':grouped_responses,
            'expanded_input_ids':input_ids,
            'completion_ids':generation_output.sequences,
            'completion_mask':completion_mask,
            'expanded_pixel_values':pixel_values,
            'completion_logits':generation_output.logits
        }

def main():
    # 初始化组件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    similarity_model = SimilarityCalculator()
    
    # 加载模型
    model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 初始化批量处理器
    processor = BatchProcessor(model, tokenizer, similarity_model, 
                              batch_size=2, num_generations=3)

    # 读取数据
    with open(TEST_JSONL_PATH, 'r') as f:
        all_data = [json.loads(line) for line in f]

    # 批量处理
    results = []
    for i in tqdm(range(0, len(all_data), processor.batch_size)):
        batch_data = all_data[i:i+processor.batch_size]
        
        try:
            # 处理一个批次
            gen_output = processor.process_batch(batch_data)
            # 处理绘图和评分
            
            gen_all_scores=[]
            original_data=gen_output['original_data']
            grouped_responses=gen_output['grouped_responses']
            for data_idx, data in enumerate(original_data):
                image_path = data['image']
                label_text = next(c['value'] for c in data['conversations'] if c['from'] == 'gpt')

                label_meme = extract_meme_text(label_text)
                
                scores = []
                group_images = []
                group_outputs = []
                for ans_idx, gen_text in enumerate(grouped_responses[data_idx]):
                    gen_image_path = similarity_model.generate_image(
                        gen_text=gen_text,
                        image_path=image_path,
                        group_id=f"batch{i}_data{data_idx}",
                        answer_id=f"ans{ans_idx}"
                    )
                    group_images.append(gen_image_path)
                    group_outputs.append({'gen_image': gen_image_path})
                
                # 统一进行评分
                gen_scores = similarity_model.score_group(group_images)
                # 将评分结果合并到输出中
                for output, score in zip(group_outputs, gen_scores):
                    output['gen_score'] = score
                    scores.append(output)
                    gen_all_scores.append(score)
                
                
                
                
            list_logits = list(gen_output['completion_logits'])
            
            result = torch.stack(list_logits, dim=1)
             
            # 处理每个样本的结果

            generations={
                'expanded_input_ids':gen_output['expanded_input_ids'],
                'completion_ids':gen_output['completion_ids'],
                'completion_mask':gen_output['completion_mask'],
                'expanded_pixel_values':gen_output['expanded_pixel_values'],
                'completion_logits':result,
                'scores':torch.tensor(gen_all_scores)
            }

            
            for data_idx, data in enumerate(original_data):
                # 获取标注文本
                label_text = next(c['value'] for c in data['conversations'] if c['from'] == 'gpt')
                image_path = data['image']
                label_meme = extract_meme_text(label_text)
                
                # 计算相似度
                similarities = [
                    similarity_model.calculate(gen_text, label_meme)
                    for gen_text in grouped_responses[data_idx]
                ]
                
                
                # 保存结果
                result = {
                    'image_path': data['image'],
                    'human_input': data['conversations'][0]['value'],
                    'generated_responses': grouped_responses[data_idx],
                    'score_info': [
                            {
                                'gen_image': score['gen_image'],
                                'gen_score': score['gen_score'],
                            } for score in scores
                        ],
                    'label_meme': label_meme,
                    'similarities': similarities
                }
                results.append(result)
                
                # 写入文件
                with open(OUTPUT_JSONL_PATH, 'a') as f:
                    f.write(json.dumps(result) + '\n')
                    
        except Exception as e:
            print(f"Error processing batch {i//processor.batch_size}: {str(e)}")
            continue

if __name__ == "__main__":
    main()