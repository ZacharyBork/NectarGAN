import math

import torch

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        '''Scaled dot product attention mechanism.
        
        Ref: https://arxiv.org/pdf/1706.03762 (3.2.1)
        
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
        '''
        super(ScaledDotProductAttention, self).__init__(*args, **kwargs)
        self.weights: torch.Tensor = None

    def softmax(self, x: torch.Tensor) -> torch.Tensor:
        '''Overflow guarded softmax activation.
        
        Args:
            x : The input tensor to apply the activation function to.
            
        Returns:
            torch.Tensor : The resulting tensor from the softmax operation.
        '''
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdims=True)[0])
        return exp_x / torch.sum(exp_x, dim=-1, keepdims=True)
    
    def forward(
        self, 
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        '''Scaled dot product attention forward function.
        
        Args:
            Q : The queries tensor.
            K : The keys tensor.
            V : The values tensor.
            mask : (optional) The mask to apply to the attention weights.
            
        Returns:
            torch.Tensor : The resulting tensor from the attention mechanism.
        '''
        K_t = K.transpose(-2, -1)
        scores = torch.matmul(Q, K_t)
        
        d_K = K.shape[-1]
        scaled_scores = scores / math.sqrt(d_K)
        if not mask is None: scaled_scores += (mask * -1e9)
        
        self.weights = self.softmax(scaled_scores)
        return torch.matmul(self.weights, V)
    
class MultiheadAttention(ScaledDotProductAttention):
    def __init__(self, num_heads: int = 8, *args, **kwargs) -> None:
        '''Multi-head attention mechanism.
        
        Ref: https://arxiv.org/pdf/1706.03762 (3.2.2)
        
        Args:
            num_heads: The number of heads to project the queries, keys, and 
                values to before applying attention.
        '''
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        
    def _get_projection_dim(self, K: torch.Tensor) -> int:
        d_K = K.shape[-1]
        assert d_K % self.num_heads == 0
        return int(d_K / self.num_heads)
        
    def _get_projections(
        self, 
        start: int,
        end: int,
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return Q[:, :, start:end], K[:, :, start:end], V[:, :, start:end] 
        
    def forward(
        self, 
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        '''Multi-head attention forward function.
                
        Args:
            Q : The queries tensor.
            K : The keys tensor.
            V : The values tensor.
            mask : (optional) The mask to apply to the attention weights.
            
        Returns:
            torch.Tensor : The resulting tensor from the attention mechanism.
        '''
        projection_dim = self._get_projection_dim(K)
        
        _values: list[torch.Tensor] = []
        _weights: list[torch.Tensor] = []
        
        for i in range(self.num_heads):
            start = projection_dim * i
            end = projection_dim * (i + 1)
            
            Q_h, K_h, V_h = self._get_projections(start, end, Q, K, V) 
        
            value = super().forward(Q_h, K_h, V_h, mask=mask)
            _values.append(value)
            _weights.append(self.weights)
            
        out_value = torch.cat(_values, dim=-1)
        self.weights = torch.mean(torch.stack(_weights, dim=0), dim=0)
        return out_value
    
