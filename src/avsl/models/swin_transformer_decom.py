import torch
import torch.nn as nn
from torchviz import make_dot
import timm
import torch.nn.init as init

class swinTDecom(nn.Module):
    def __init__(self, pretrained=True, bn_freeze = True):
        super(swinTDecom, self).__init__()

        pretrained_cfg_overlay = {'file' : r"/home/zbr/.cache/huggingface/hub/models--timm--swin_tiny_patch4_window7_224.ms_in1k/pytorch_model.bin"}
        self.model = timm.models.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, pretrained_cfg_overlay=pretrained_cfg_overlay)
        self.num_ftrs = self.model.head.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)
        self.feature_dim_list = [192, 384, 768]
        self.output_dim = [512, 1024, 2048]

        if bn_freeze: # freeze layer normalization
            for m in self.model.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)
        
        for idx, dim in enumerate(self.feature_dim_list):
            # 1x1 conv
            conv = nn.Conv2d(dim, self.output_dim[idx], kernel_size=(1, 1), stride=(1, 1))
            init.kaiming_normal_(conv.weight, mode="fan_out")
            init.constant_(conv.bias, 0)
            setattr(
                self,
                f"conv1x1_{idx}",
                conv
            )

    def forward(self, x): # FIXME: use resnet50 to do experiments
        x = self.model.patch_embed(x) # B x 56 x 56 x 96
        output_list = []
        for idx, layer in enumerate(self.model.layers):
            x = layer(x)
            if idx:
                output_list.append(x)    
        for i in range(3):
            conv = getattr(self, f"conv1x1_{i}")
            output_list[i] = output_list[i].permute(0, 3, 1, 2) 
            output_list[i] = conv(output_list[i])
        return output_list 
        # B x 192 x 28 x 28 / B x 384 x 14 x 14/ B x 768 x 7 x 7
        # now: B x 512 x 28 x 28 / B x 1024 x 14 x 14/ B x 2048 x 7 x 7

if __name__ == "__main__":
    model = swinTDecom()
    data = torch.randn(3, 3, 224, 224)
    data.requires_grad = True
    out = model(data)
    loss = torch.sum(out[2])
    dot = make_dot(loss, params=dict(model.named_parameters()))
    dot.render("model")
    pass