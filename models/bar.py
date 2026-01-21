
import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryAwareRefinement(nn.Module):
    def __init__(self, in_channels=1, img_channels=3, hidden_channels=32):
        super().__init__()
        
        # 边缘特征提取
        self.edge_conv = nn.Sequential(
            nn.Conv2d(img_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)  # 输出边缘特征图
        )
        
        # 边界注意力模块
        self.boundary_attention = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid()  # 输出边界注意力权重 [0,1]
        )
        
        # 边界细化卷积
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1)
        )
        
    def forward(self, pred_mask, original_img):
        """
        Args:
            pred_mask: (B, 1, H, W) - TPD输出的预测掩码
            original_img: (B, 3, H, W) - 原始输入图像
        Returns:
            refined_mask: (B, 1, H, W) - 边界优化后的掩码
        """
        # 1. 提取边缘特征
        edge_features = self.edge_conv(original_img)  # (B, 1, H, W)
        
        # 2. 计算边界注意力权重
        boundary_input = torch.cat([pred_mask, edge_features], dim=1)  # (B, 2, H, W)
        boundary_weights = self.boundary_attention(boundary_input)  # (B, 1, H, W)
        
        # 3. 边界区域细化
        refine_input = torch.cat([pred_mask, edge_features], dim=1)  # (B, 2, H, W)
        boundary_refinement = self.refine_conv(refine_input)  # (B, 1, H, W)
        
        # 4. 应用边界优化：只在边界区域进行细化
        refined_mask = pred_mask + boundary_weights * boundary_refinement
        refined_mask = torch.sigmoid(refined_mask)  # 确保输出在[0,1]范围内
        
        return refined_mask