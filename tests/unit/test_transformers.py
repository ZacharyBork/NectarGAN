import torch

from nectargan.transformers.attention import \
    ScaledDotProductAttention, MultiheadAttention

def test_attention_mechanisms() -> None:
    parameters = [
        {
            'model': ScaledDotProductAttention,
            'shape': (2, 4, 8, 8)
        },
        {
            'model': MultiheadAttention,
            'shape': (2, 10, 512, 512)
        }
    ]
    
    for x in parameters:
        model = x['model']()
        shape = x['shape']
        
        Q = torch.randn(shape[0], shape[1], shape[2])
        K = torch.randn(shape[0], shape[1], shape[2])
        V = torch.randn(shape[0], shape[1], shape[3])
    
        output = model(Q, K, V)
        weights = model.weights
        
        assert output.shape == (shape[0], shape[1], shape[3])
        assert weights.shape == (shape[0], shape[1], shape[1])  
        assert torch.allclose(weights.sum(axis=-1), torch.ones(1))
