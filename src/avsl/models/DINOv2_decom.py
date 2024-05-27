import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

class DINOv2Decom(nn.Module):
    def __init__(self, pretrained=True, bn_freeze = True):
        super(DINOv2Decom, self).__init__()

        check_point = '/home/zbr/.cache/huggingface/hub/facebook-dinov2-small'
        self.processor = AutoImageProcessor.from_pretrained(check_point)
        self.model = AutoModel.from_pretrained(check_point)
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        if bn_freeze: # freeze layer normalization
            for m in self.model.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def forward(self, x):
        x_shape =  x.shape
        x = x.view(x_shape[0], x_shape[1], -1)
        min_vals = x.min(dim=-1, keepdim=True)[0]
        max_vals = x.max(dim=-1, keepdim=True)[0]
        x = (x - min_vals) / (max_vals - min_vals)
        x = x.view(x_shape)

        x = self.processor(images=x, return_tensors="pt", do_rescale=False)
        x = x["pixel_values"].to(torch.device("cuda:0"))
        outputs = self.model(x, output_hidden_states=True)
        indices = [4, 8, 12]
        output_list = []
        for i in indices:
            feature_map = outputs.hidden_states[i]
            batch_size, hidden_size = feature_map.shape[0], feature_map.shape[2]
            feature_map = feature_map[:, 1:, :]
            feature_map = feature_map.view(batch_size, 16, 16, hidden_size)
            feature_map = feature_map.permute(0, 3, 1, 2)
            output_list.append(feature_map)
        '''
        B * 768 * 16 * 16 x 3
        or 384 
        '''
        return output_list 
    
if __name__ == "__main__":
    model = DINOv2Decom()
    image = torch.randn(5,3,224,224)
    image = image - image.min()
    image = image / image.max() * 255.0
    image = image.to(torch.uint8)
    out = model(image)
    for i in out:
        print(i.shape)
