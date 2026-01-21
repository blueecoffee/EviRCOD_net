import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pvtv2 import pvt_v2_b4

from models.rgde import MultiScaleDeformableEncoder
from models.uaed import MultiScaleEvidentialDecoder,EnhancedEvidenceLoss
from models.bar import BoundaryAwareRefinement

class BasicConv2d(nn.Module):
    def __init__(self, in_cha, out_cha, kernel_size, stride, padding):
        super(BasicConv2d, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_cha, out_cha, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_cha),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.block(x)
        return out


class LevelAttention(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.feat_quality = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  
            nn.Conv2d(dim, dim//4, kernel_size=1),  
            nn.ReLU(),
            nn.Conv2d(dim//4, 1, kernel_size=1)  
        )
        self.ref_proj = nn.Conv2d(dim, 1, kernel_size=1)  # ref_x：(B,64,1,1)→(B,1,1,1)

    def forward(self, feat_conv, ref_x):      
        feat_quality = self.feat_quality(feat_conv)  # (B,1,1,1)
        feat_quality = F.sigmoid(feat_quality) 

        ref_proj = self.ref_proj(ref_x)  
        feat_proj = F.conv2d(feat_conv, weight=torch.ones(1,64,1,1).to(feat_conv.device), bias=None)
        sim = F.cosine_similarity(feat_proj, ref_proj.expand_as(feat_proj), dim=1).unsqueeze(1) 
        sim = F.sigmoid(sim)  
        weight = (feat_quality * sim).clamp(0.3, 1.0) 
        return weight 
    
class Network(nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()
        self.imgsize = opt.imgsize
        self.backbone = pvt_v2_b4()
        path = '/home/coffee/projects/datasets/models/pvtv2/pvt_v2_b4.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.dim_out = opt.dim
        self.sigmoid = nn.Sigmoid()
        self.soft_split = nn.Unfold(kernel_size=(4, 4), stride=(4, 4), padding=(0, 0))
        self.soft_fuse = nn.Fold(output_size=(self.imgsize // 4, self.imgsize // 4), kernel_size=(4, 4),
                                  stride=(4, 4), padding=(0, 0))

        self.conv_ref = BasicConv2d(2048, self.dim_out, kernel_size=1, stride=1, padding=0)
        self.conv0 = BasicConv2d(64, self.dim_out, kernel_size=3, stride=1, padding=1)
        self.conv1 = BasicConv2d(128, self.dim_out, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv2d(320, self.dim_out, kernel_size=3, stride=1, padding=1)
        self.conv3 = BasicConv2d(512, self.dim_out, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(self.dim_out, 1, kernel_size=1, stride=1, padding=0)

        self.level_attn = nn.ModuleList([
            LevelAttention(),  # x3（11×11）
            LevelAttention(),  # x2（22×22）
            LevelAttention(),  # x1（44×44）
            LevelAttention()   # x0（88×88）
        ])

        self.MSDE = MultiScaleDeformableEncoder(
            dim=1024,
            num_heads=16, 
            depths=[2, 2, 3, 4],
            mlp_ratio=4.
        )


        self.EvidentialDecoder = MultiScaleEvidentialDecoder(
            dim=self.dim_out*4*4,  # 1024
            num_heads=8,
            depth=4, 
            imgsize=self.imgsize,  # 352
            mlp_ratio=4.0
        )
        self.EvidentialGuidedDecoder = MultiScaleEvidentialDecoder(
            dim=self.dim_out*4*4,  # 1024
            num_heads=8,
            depth=4,
            imgsize=self.imgsize,  # 352
            mlp_ratio=4.0
        )
        
        self.evidence_loss = EnhancedEvidenceLoss(
            lam=0.1, 
            boundary_weight=2.0,
            focal_gamma=2.0  
        )

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)


        self.bar_module = BoundaryAwareRefinement(in_channels=1, img_channels=3, hidden_channels=32)

    
    def forward(self, x, ref_x, y=None, training=True):

        B, _, _, _ = x.shape
        pvt = self.backbone(x)
        x0, x1, x2, x3 = pvt[0], pvt[1], pvt[2], pvt[3]
        #---------------- RGDE ------------------#
        ref_x = self.sigmoid(self.conv_ref(ref_x))
        
        x3_conv = self.conv3(x3) 
        x2_conv = self.conv2(x2) 
        x1_conv = self.conv1(x1) 
        x0_conv = self.conv0(x0) 

        weight_x3 = self.level_attn[0](x3_conv, ref_x) 
        weight_x2 = self.level_attn[1](x2_conv, ref_x)
        weight_x1 = self.level_attn[2](x1_conv, ref_x)
        weight_x0 = self.level_attn[3](x0_conv, ref_x) 

        x3 = torch.mul(x3_conv, weight_x3 * ref_x) 
        x2 = torch.mul(x2_conv, weight_x2 * ref_x)  
        x1 = torch.mul(x1_conv, weight_x1 * ref_x)  
        x0 = torch.mul(x0_conv, weight_x0 * ref_x)  

        x3 = self.soft_split(self.upsample8(x3)).transpose(-2, -1)
        x2 = self.soft_split(self.upsample4(x2)).transpose(-2, -1)
        x1 = self.soft_split(self.upsample2(x1)).transpose(-2, -1)
        x0 = self.soft_split(x0).transpose(-2, -1)

        s0, s1, s2, s3 = self.MSDE(x0, x1, x2, x3)     

        #---------------- UAED ------------------#
        prob, evidential_query, s3 = self.EvidentialDecoder(s3, evidential_guide_map=None, uncertainty=True)
        s2 = s2 + s3
        s2 = self.EvidentialGuidedDecoder(s2, evidential_guide_map=evidential_query, uncertainty=False)

        s1 = s1 + s2
        s1 = self.EvidentialGuidedDecoder(s1, evidential_guide_map=evidential_query, uncertainty=False)

        s0 = s0 + s1
        s0 = self.EvidentialGuidedDecoder(s0, evidential_guide_map=evidential_query, uncertainty=False)

        s3 = self.upsample4(self.conv_out(self.soft_fuse(s3.transpose(-2, -1))))
        s2 = self.upsample4(self.conv_out(self.soft_fuse(s2.transpose(-2, -1))))
        s1 = self.upsample4(self.conv_out(self.soft_fuse(s1.transpose(-2, -1))))
        s0 = self.upsample4(self.conv_out(self.soft_fuse(s0.transpose(-2, -1))))

        prob = self.upsample4(prob)

        #---------------- BAR ------------------#
        s0_refined = self.bar_module(s0, x)  
        s1_refined = self.bar_module(s1, x)  


        if training:
            loss_prob = 1 * self.evidence_loss(prob, y)
            return s3, s2, s1, s0, s1_refined,s0_refined,loss_prob
        else:
            return s3, s2, s1, s0,s0_refined
