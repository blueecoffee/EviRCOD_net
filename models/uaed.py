import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

class MultiScaleEvidentialDecoder(nn.Module):
    def __init__(self, dim, num_heads, depth, imgsize, mlp_ratio=4.0):
        super(MultiScaleEvidentialDecoder, self).__init__()
        self.depth = depth 
        self.blocks = nn.ModuleList([
            AdaptiveEvidentialDecoderBlock(dim, num_heads, imgsize, mlp_ratio) for _ in range(depth)
        ])
        
        self.uncertainty_fusion = nn.Linear(dim * 2, dim)

    def forward(self, fea, evidential_guide_map=None, uncertainty=True):
        if uncertainty:
            all_evidential_features = []
            prob, evidential_query = None, None
            
            for i, block in enumerate(self.blocks):
                if i == 0:
                    prob, evidential_query, fea = block(fea, None, uncertainty=True)
                    all_evidential_features.append(evidential_query)
                else:
                    if len(all_evidential_features) > 0:
                        fused_uncertainty = all_evidential_features[0]
                        for j in range(1, len(all_evidential_features)):
                            fused_input = torch.cat([fused_uncertainty, all_evidential_features[j]], dim=-1)
                            fused_uncertainty = self.uncertainty_fusion(fused_input)
                        evidential_guide_map = fused_uncertainty
                    
                    fea = block(fea, evidential_guide_map, uncertainty=False)
                    all_evidential_features.append(evidential_guide_map)
            
            return prob, evidential_query, fea
        else:
            for block in self.blocks:
                fea = block(fea, evidential_guide_map, uncertainty=False)
            return fea
        
class AdaptiveEvidentialDecoderBlock(nn.Module):     
    def __init__(self, dim, num_heads, imgsize, mlp_ratio=4.0, qkv_bias=False, 
                 drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = HierarchicalEvidentialAttention(dim=dim, num_heads=num_heads, imgsize=imgsize, qkv_bias=qkv_bias,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        self.gate = nn.Linear(dim * 2, dim)
        
    def forward(self, fea, evidential_guide_map=None, uncertainty=True):
        residual = fea
        
        if uncertainty:
            prob, evidential_query, attn = self.attn(self.norm1(fea), None, uncertainty=True)
            
            gate_input = torch.cat([fea, attn], dim=-1)
            gate_weight = torch.sigmoid(self.gate(gate_input))
            fea = fea + self.drop_path(gate_weight * attn)
            
            fea = fea + self.drop_path(self.mlp(self.norm2(fea)))
            return prob, evidential_query, fea
        else:
            attn = self.attn(self.norm1(fea), evidential_guide_map, uncertainty=False)
            
            gate_input = torch.cat([fea, attn], dim=-1)
            gate_weight = torch.sigmoid(self.gate(gate_input))
            fea = fea + self.drop_path(gate_weight * attn)
            
            fea = fea + self.drop_path(self.mlp(self.norm2(fea)))
            return fea
        
class HierarchicalEvidentialAttention(nn.Module):    
    def __init__(self, dim, num_heads, imgsize, qkv_bias=False):
        super().__init__()
        self.scale = dim ** -0.5
        self.num_heads = num_heads
        self.evidence_uncertainty = EnhancedDirichletEvidenceGeneration(dim, imgsize)
        
        self.fea_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.fea_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.fea_v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.uncertainty_proj = nn.Linear(dim, num_heads)
 
        self.proj = nn.Linear(dim, dim)
        
        self.residual_alpha = nn.Parameter(torch.tensor(0.1))

    def uncertainty_guided_attention(self, Q, K, V, evidential_query):
        B, N, C = Q.shape
        H = self.num_heads
        
        Q = Q.reshape(B, N, H, C // H).permute(0, 2, 1, 3)
        K = K.reshape(B, N, H, C // H).permute(0, 2, 1, 3)
        V = V.reshape(B, N, H, C // H).permute(0, 2, 1, 3)
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        uncertainty_weights = self.uncertainty_proj(evidential_query)  # (B, N, H)
        uncertainty_weights = uncertainty_weights.permute(0, 2, 1).unsqueeze(-1)  # (B, H, N, 1)
    
        attn = attn * (1 + uncertainty_weights)
        
        attn = attn.softmax(dim=-1)
        attn_out = torch.matmul(attn, V).transpose(1, 2).reshape(B, N, C)
        attn_out = self.proj(attn_out)
        
        return attn_out

    def forward(self, fea, evidential_guide_map=None, uncertainty=True):
        B, N, C = fea.shape
        
        if uncertainty:
            prob, evidential_query = self.evidence_uncertainty(fea)
            Q = self.fea_q(evidential_query)
            K = self.fea_k(fea)
            V = self.fea_v(fea)
            
            attn = self.uncertainty_guided_attention(Q, K, V, evidential_query)
            attn = self.residual_alpha * attn + (1 - self.residual_alpha) * fea
            
            return prob, evidential_query, attn
        else:
            Q = self.fea_q(evidential_guide_map)
            K = self.fea_k(fea)
            V = self.fea_v(fea)
            
            attn = self.uncertainty_guided_attention(Q, K, V, evidential_guide_map)
            attn = self.residual_alpha * attn + (1 - self.residual_alpha) * fea
            
            return attn

class EnhancedDirichletEvidenceGeneration(nn.Module):    
    def __init__(self, dim, imgsize):
        super(EnhancedDirichletEvidenceGeneration, self).__init__()
        self.imgsize = imgsize

        self.soft_split = nn.Unfold(kernel_size=(4, 4), stride=(4, 4), padding=(0, 0))
        self.soft_fuse = nn.Fold(output_size=(self.imgsize // 4, self.imgsize // 4), 
                                kernel_size=(4, 4), stride=(4, 4), padding=(0, 0))
        
        self.evidence_pyramid = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 32, 1)
            ) for _ in range(3)
        ])
        
        self.boundary_evidence = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 1)
        )
        self.evidence_fusion = nn.Sequential(
            nn.Conv2d(98, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1)
        )
        
        self.uncertainty_weight = nn.Parameter(torch.tensor([0.6, 0.4]))
        
        self.boundary_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))
        self.sigma = nn.Parameter(torch.tensor(1.0))
        self.scale = nn.Parameter(torch.tensor(0.55)) 

    def _create_gaussian_kernel(self, x_feat, kernel_size=3):
        sigma = self.sigma.clamp(min=0.1, max=2.0)
        kernel = torch.zeros((kernel_size, kernel_size), device=x_feat.device)
        center = kernel_size // 2
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                distance = ((i - center) **2 + (j - center)** 2) / (2 * sigma ** 2)
                kernel[i, j] = torch.exp(-distance)
        
        kernel /= kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def compute_gradient_consistency(self, *gradients):
        if len(gradients) < 2:
            return torch.ones_like(gradients[0])
        
        main_grad_x, main_grad_y = gradients[0], gradients[1]
        main_orientation = torch.atan2(main_grad_y, main_grad_x)
        
        consistency_scores = []
        for i in range(2, len(gradients), 2):
            if i + 1 >= len(gradients):
                break
            curr_grad_x, curr_grad_y = gradients[i], gradients[i + 1]
            curr_orientation = torch.atan2(curr_grad_y, curr_grad_x)
            orientation_diff = torch.cos(main_orientation - curr_orientation)
            consistency_scores.append(orientation_diff)
        
        if consistency_scores:
            avg_consistency = torch.stack(consistency_scores, dim=0).mean(dim=0)
            return torch.clamp(avg_consistency, min=0.5, max=1.0)
        return torch.ones_like(main_grad_x)

    def enhanced_boundary_detection(self, x_feat):
        B, C, H, W = x_feat.shape
        
        boundary_maps = []
        
        for b in range(B):
            sample_feat = x_feat[b:b+1]
            channel_response = F.adaptive_avg_pool2d(sample_feat, output_size=1).squeeze()
            k = min(12, C)
            _, indices = torch.topk(channel_response, k=k, largest=True)
            
            sample_boundaries = []
            for c in indices:
                channel_feat = sample_feat[:, c:c+1]
                sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], 
                                     dtype=torch.float32, device=sample_feat.device).view(1,1,3,3)
                sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], 
                                     dtype=torch.float32, device=sample_feat.device).view(1,1,3,3)
                
                grad_x = F.conv2d(channel_feat, sobel_x, padding=1)
                grad_y = F.conv2d(channel_feat, sobel_y, padding=1)
                
                magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
                
                max_val = magnitude.max() + 1e-8
                magnitude = magnitude / (max_val * 1.2)
                
                sample_boundaries.append(magnitude)
            
            if sample_boundaries:
                sample_boundary = torch.cat(sample_boundaries, dim=1).mean(dim=1, keepdim=True)
            else:
                sample_boundary = torch.zeros(1, 1, H, W, device=sample_feat.device)
            
            boundary_maps.append(sample_boundary)
        
        boundary_magnitude = torch.cat(boundary_maps, dim=0)
        enhanced_input = boundary_magnitude * self.scale
        enhanced_input = torch.clamp(enhanced_input, max=4.0)
        boundary_magnitude = torch.sigmoid(enhanced_input)
        
        min_val = boundary_magnitude.min()
        max_val = boundary_magnitude.max()
        dynamic_range = max_val - min_val
        
        if dynamic_range < 0.3:
            stretch_factor = 0.3 / (dynamic_range + 1e-8)
            stretch_factor = min(stretch_factor, 3.0)
            
            stretched = (boundary_magnitude - min_val) / (dynamic_range + 1e-8)
            blend_ratio = min(0.8, 0.3 + (0.3 - dynamic_range) * 2)
            boundary_magnitude = (1 - blend_ratio) * boundary_magnitude + blend_ratio * stretched
        
        boundary_magnitude = boundary_magnitude * 0.82 
        boundary_magnitude = torch.pow(boundary_magnitude, 0.88)
           
        return boundary_magnitude

    def boundary_aware_evidence(self, x_feat):
        boundary_magnitude = self.enhanced_boundary_detection(x_feat)
        boundary_weight = 1.0 + boundary_magnitude * 1.5
        
        evidence = F.softplus(self.boundary_evidence(x_feat))
        return evidence * boundary_weight

    def multi_scale_evidence_extraction(self, x_feat):
        evidences = []
        for i, conv in enumerate(self.evidence_pyramid):
            scale_factor = 2 ** i
            if scale_factor > 1:
                scaled_feat = F.interpolate(x_feat, scale_factor=1/scale_factor, mode='bilinear')
            else:
                scaled_feat = x_feat
            evidence = F.softplus(conv(scaled_feat))
            if scale_factor > 1:
                evidence = F.interpolate(evidence, size=x_feat.shape[2:], mode='bilinear')
            evidences.append(evidence)
        return torch.cat(evidences, dim=1)

    def enhanced_dirichlet_uncertainty(self, evidence):
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        prob = alpha[:, 1:2] / S
        aleatoric_uncertainty = 2.0 / S
        epistemic_uncertainty = torch.var(alpha, dim=1, keepdim=True) / (S ** 2)
        
        uncertainty = (self.uncertainty_weight[0] * aleatoric_uncertainty + 
                     self.uncertainty_weight[1] * epistemic_uncertainty)
        
        return prob, uncertainty

    def forward(self, x):
        x_feat = self.soft_fuse(x.transpose(-2, -1))
        
        multi_scale_evidence = self.multi_scale_evidence_extraction(x_feat)
        boundary_evidence = self.boundary_aware_evidence(x_feat)
        
        fused_evidence = torch.cat([multi_scale_evidence, boundary_evidence], dim=1)
        final_evidence = F.softplus(self.evidence_fusion(fused_evidence))
        
        prob, uncertainty = self.enhanced_dirichlet_uncertainty(final_evidence)
        
        confidence_map = 1 - uncertainty
        weighted_feat = confidence_map * x_feat

        uncertainty_token = self.soft_split(weighted_feat).transpose(-2, -1)
        
        return prob, uncertainty_token

class EnhancedEvidenceLoss(nn.Module):    
    def __init__(self, lam=0.1, boundary_weight=2.0, focal_gamma=2.0):
        super(EnhancedEvidenceLoss, self).__init__()
        self.lam = lam
        self.boundary_weight = boundary_weight
        self.focal_gamma = focal_gamma
        
    def compute_boundary_weights(self, targets):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(targets.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(targets.device)
        
        grad_x = F.conv2d(targets, sobel_x, padding=1)
        grad_y = F.conv2d(targets, sobel_y, padding=1)
        boundary = torch.sqrt(grad_x**2 + grad_y**2)
        
        weights = torch.ones_like(targets)
        weights[boundary > 0.1] = self.boundary_weight
        return weights

    def forward(self, evidence, targets):
        B, _, H, W = evidence.shape
        targets_onehot = torch.cat([1-targets, targets], dim=1)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        loss_nll = torch.sum(targets_onehot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
        
        boundary_weights = self.compute_boundary_weights(targets)
        loss_nll = loss_nll * boundary_weights.squeeze(1)
        loss_nll = loss_nll.mean()
        
        p = alpha / S
        pt = targets_onehot * p
        focal_weight = (1 - pt) ** self.focal_gamma
        loss_reg = torch.sum(focal_weight * (alpha - 1), dim=1) / S.squeeze(1)
        loss_reg = loss_reg.mean()
        
        total_loss = loss_nll + self.lam * loss_reg
        return total_loss
