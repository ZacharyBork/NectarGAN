import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer    

class TextEncoder(nn.Module):
    def __init__(
            self, 
            device: str,
            model_name: str='openai/clip-vit-large-patch14',
            max_length: int=77,
            freeze: bool=True
        ) -> None:
        super().__init__()
        self.device = device
        self.max_length = max_length

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        if freeze: self._freeze_model()
            
    def _freeze_model(self) -> None:
        for p in self.model.parameters(): 
            p.requires_grad = False

    @torch.no_grad()
    def forward(
            self, 
            texts: list[str]
        ) -> torch.Tensor:
        tokens = self.tokenizer(
            texts, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt')

        outputs = self.model(
            input_ids=tokens.input_ids.to(self.device),
            attention_mask=tokens.attention_mask.to(self.device))
        return outputs.last_hidden_state, outputs.pooler_output
    
