import torch.nn as nn
import torch
class ExtendedEmbedding(nn.Module):
    def __init__(self, original_embedding: nn.Embedding, new_embedding: nn.Embedding):
        super(ExtendedEmbedding, self).__init__()
        self.original_embedding = original_embedding
        self.new_embedding = new_embedding

        # 冻结 original_embedding 的参数
        for param in self.original_embedding.parameters():
            param.requires_grad = False

        # 解冻 new_embedding 的参数
        for param in self.new_embedding.parameters():
            param.requires_grad = True

        # 确保 embedding_dim 一致
        assert self.original_embedding.embedding_dim == self.new_embedding.embedding_dim, (
            f"Embedding dimension mismatch: original_embedding has {self.original_embedding.embedding_dim}, "
            f"new_embedding has {self.new_embedding.embedding_dim}"
        )

        # 设置 num_embeddings 和 embedding_dim
        self.num_embeddings = self.original_embedding.num_embeddings + self.new_embedding.num_embeddings
        self.embedding_dim = self.original_embedding.embedding_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        is_new_token = input >= self.original_embedding.num_embeddings
        original_tokens = input[~is_new_token]
        original_embeddings = self.original_embedding(original_tokens)

        combined_embeddings = (
            torch.zeros(input.shape + (self.embedding_dim,))
            .to(original_embeddings.device)
            .to(original_embeddings.dtype)
        )
        combined_embeddings[~is_new_token] = original_embeddings

        new_tokens = input[is_new_token] - self.original_embedding.num_embeddings
        if len(new_tokens) > 0:
            combined_embeddings[is_new_token] = self.new_embedding(new_tokens).to(
                original_embeddings.device
            )

        return combined_embeddings

    @property
    def weight(self) -> torch.Tensor:
        # 返回原始 embedding 和新 embedding 的权重的拼接结果
        return torch.cat([self.original_embedding.weight, self.new_embedding.weight], dim=0)

    def reset_parameters(self) -> None:
        # 重置 original_embedding 和 new_embedding 的参数
        self.original_embedding.reset_parameters()
        self.new_embedding.reset_parameters()

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"padding_idx={self.original_embedding.padding_idx}"
        )

class ExtendedLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, new_linear: nn.Linear):
        super(ExtendedLinear, self).__init__()
        self.original_linear = original_linear
        self.new_linear = new_linear

        # 冻结 original_linear 的参数
        for param in self.original_linear.parameters():
            param.requires_grad = False

        # 解冻 new_linear 的参数
        for param in self.new_linear.parameters():
            param.requires_grad = True

        # 确保输入维度一致
        assert self.original_linear.in_features == self.new_linear.in_features, (
            f"Input features mismatch: original_linear has {self.original_linear.in_features}, "
            f"new_linear has {self.new_linear.in_features}"
        )

        # 设置 in_features 和 out_features
        self.in_features = self.original_linear.in_features
        self.out_features = self.original_linear.out_features + self.new_linear.out_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 原始 lm_head 的输出
        original_output = self.original_linear(input)

        # 新 token 的输出
        new_output = self.new_linear(input)

        # 拼接原始输出和新 token 的输出
        combined_output = torch.cat([original_output, new_output], dim=-1)
        return combined_output

    @property
    def weight(self) -> torch.Tensor:
        # 返回原始 lm_head 和新 lm_head 的权重的拼接结果
        return torch.cat([self.original_linear.weight, self.new_linear.weight], dim=0)

    @property
    def bias(self) -> torch.Tensor:
        # 返回原始 lm_head 和新 lm_head 的偏置的拼接结果
        if self.original_linear.bias is not None and self.new_linear.bias is not None:
            return torch.cat([self.original_linear.bias, self.new_linear.bias], dim=0)
        else:
            return None

    def reset_parameters(self) -> None:
        # 重置 original_linear 和 new_linear 的参数
        self.original_linear.reset_parameters()
        self.new_linear.reset_parameters()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )