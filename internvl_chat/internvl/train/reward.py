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
    
    def __call__(self, prompts, completions, images):
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

# 使用示例
reward_fn = MemeRewardFunction()

# 模拟输入
prompts = ["Generate a funny caption for this meme"]
images = [Image.open("/fs-computility/ai-shen/xueyingyi/Eimages_inpainting/image_ (0).jpg")] 
completions = [
    "When you finally understand the joke",
    "My face when the code works",
    "Monday mornings be like...",
    "That moment when you send the wrong message"
]

rewards = reward_fn(prompts, completions, images)
print(f"Rewards: {rewards}")