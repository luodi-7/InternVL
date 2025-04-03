import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from modeling_internvl_chat import InternVLChatModel, InternVLChatConfig
from transformers import AutoTokenizer

# 配置参数
class TrainingConfig:
    lr = 1e-5
    batch_size = 2
    num_epochs = 3
    max_seq_len = 2048
    num_generations = 5
    beta = 0.1
    save_dir = "./checkpoints"

# 数据加载
class GRPODataset(Dataset):
    def __init__(self, file_path, tokenizer, image_size=448):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.image_size = image_size
        
    def __len__(self):
        return len(self.data)
    
    def load_image(image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        print(f"Processed {len(images)} blocks for image {image_file}")
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def __getitem__(self, idx):
        item = self.data[idx]
        pixel_values = self.load_image(item['image'])
        prompt = next(c['value'] for c in item['conversations'] if c['from'] == 'human')
        return {
            'pixel_values': pixel_values,
            'prompt': prompt,
            'image_path': item['image']
        }

# 初始化模型
def initialize_model(config_path, device):
    config = InternVLChatConfig.from_pretrained(config_path)
    model = InternVLChatModel(config).to(device)
    return model

# 训练循环
def train():
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = TrainingConfig()
    tokenizer = AutoTokenizer.from_pretrained(
        "/fs-computility/ai-shen/shared/dilab/model/InternVL2_5-4B", add_eos_token=False, trust_remote_code=True, use_fast=model_args.use_fast_tokenizer)
    
    # 加载模型
    model = initialize_model('path/to/config', device)
    ref_model = initialize_model('path/to/config', device).eval()
    
    # 加载数据
    train_dataset = GRPODataset('path/to/train.jsonl', tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.lr)
    
    # 训练循环
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 数据准备
            pixel_values = batch['pixel_values'].to(device)
            prompts = batch['prompt']
            
            # 生成阶段
            input_ids, completion_ids, mask = model.grpo_generate(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                prompts=prompts,
                generation_config={
                    'max_new_tokens': 1024,
                    'do_sample': True,
                    'temperature': 1.0
                },
                num_generations=config.num_generations
            )
            
            # 计算当前模型logits
            logits = model.compute_completion_logits(
                pixel_values=pixel_values,
                input_ids=input_ids,
                completion_ids=completion_ids
            )
            
            # 计算参考模型logits
            with torch.no_grad():
                ref_logits = ref_model.compute_completion_logits(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    completion_ids=completion_ids
                )
            
            # 模拟评分（替换为真实评分）
            batch_size = len(prompts)
            rewards = torch.rand(batch_size * config.num_generations).to(device)  # 随机评分
            
            # 计算损失
            loss = model.grpo_loss(
                logits=logits,
                ref_logits=ref_logits,
                rewards=rewards,
                mask=mask,
                beta=config.beta
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 打印进度
            if batch_idx % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f'Epoch {epoch+1} | Batch {batch_idx} | Loss: {avg_loss:.4f}')
        
        # 保存检查点
        torch.save(model.state_dict(), f"{config.save_dir}/epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()