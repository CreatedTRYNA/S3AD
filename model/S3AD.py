from timm.models.resnet import Bottleneck

from model import get_model
from model import MODEL

import math
from util.util import add_jitter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import trunc_normal_
from model.selected_modules.CEN.semantic_segmentation.models.modules import Exchange


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=1,dilation=1):
        super().__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,padding=padding,stride=stride, groups=in_channels,dilation=1),
            nn.InstanceNorm2d(in_channels),nn.SiLU())
        self.pconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels),nn.SiLU())
 
    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class Mlp(nn.Module):
    def __init__(self, 
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.SiLU,
                 drop=0.):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.functional import linear


class GPPE(nn.Module):
    def __init__(self, feature_size, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.feature_size = feature_size
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        # 注意：这里的实现假设 feature_size 是固定的
        # 为了更灵活，可以动态获取 tensor 的 H, W
        if tensor.dim() == 4:
            # B, C, H, W -> h=H, w=W
            # B, H, W, C -> h=H, w=W
            h, w = tensor.shape[-2:] if tensor.shape[1] > 1 else tensor.shape[1:3]
        else:
            raise ValueError("Expected a 4D tensor as input")
        not_mask = torch.ones((h, w), device=tensor.device)
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=tensor.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)
        return pos

def build_position_embedding(pos_embed_type, feature_size, hidden_dim):
    if pos_embed_type in ("v2", "sine"):
        pos_embed = GPPE(feature_size, hidden_dim // 2, normalize=True)
    else:
        raise ValueError(f"not supported {pos_embed_type}")
    return pos_embed

class En_Adaptive_MHSA(nn.Module):
    """Enhanced Adaptive Multi-Head Self-Attention with stabilized token sampling and progressive drop rate"""
    
    def __init__(self, dim, num_heads, feature_size=(32,32), dropout=0.0, pos_embed_type='sine'):
        super(En_Adaptive_MHSA, self).__init__()
        self.dim_LIN = dim
        self.num_heads = num_heads
        self.pos_embed = build_position_embedding(pos_embed_type, feature_size, dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        self. dropout = nn.Dropout(dropout)
        self.out_proj = NonDynamicallyQuantizableLinear(dim, dim, bias=True)
        
        # 添加用于渐进式采样的参数
        self.current_epoch = 0
        self.warmup_epochs = 10
        self.min_drop_rate = 0.3  # 最小采样率
        self.max_drop_rate = 0.8  # 最大采样率（训练初期）
        
        # 添加用于稳定化采样的参数
        self.noise_scale = 0.01
        self.temperature = 0.1
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.constant_(self.k_proj.bias, 0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.constant_(self.v_proj.bias, 0)
        nn.init.constant_(self.out_proj.bias, 0)

    def set_epoch(self, epoch):
        """设置当前epoch，用于渐进式采样"""
        self.current_epoch = epoch

    def _get_progressive_drop_rate(self, base_drop):
        """计算渐进式drop率"""
        if not self.training:
            return base_drop
            
        if self.current_epoch < self.warmup_epochs:
            # 训练初期使用较高的drop率，逐渐降低到目标值
            progress = self.current_epoch / self.warmup_epochs
            current_drop = self.max_drop_rate + (base_drop - self.max_drop_rate) * progress
            return max(current_drop, self.min_drop_rate)
        else:
            return base_drop

    def _stabilized_sampling(self, weight, L_out):
        """稳定化的token采样"""
        if self.training:
            # 策略1: 添加小量噪声来稳定训练
            noise = torch.randn_like(weight) * self.noise_scale
            weight = weight + noise
            
            # 策略2: 使用温度缩放来平滑分布
            weight = weight / self.temperature
        
        # 执行topk采样
        sampled_inds = torch.topk(weight, L_out, dim=-1)[1]
        return sampled_inds

    def forward(self, Q, K, V, drop=0.5):
        B, T_q, D = Q.shape
        _, T_k, _ = K.shape

        # 应用渐进式采样策略
        progressive_drop = self._get_progressive_drop_rate(drop)
        L_out = int(T_q * progressive_drop)
        
        Q = self.q_proj(Q)
        K = self.k_proj(K)
        V = self.v_proj(V)

        dim_split = self.dim_LIN // self.num_heads
        # 分头并调整维度
        Q = Q.view(B, T_q, self.num_heads, dim_split).transpose(1, 2).contiguous().view(B * self.num_heads, T_q, dim_split)
        K = K.view(B, T_k, self.num_heads, dim_split).transpose(1, 2).contiguous().view(B * self.num_heads, T_k, dim_split)
        V = V.view(B, T_k, self.num_heads, dim_split).transpose(1, 2).contiguous().view(B * self.num_heads, T_k, dim_split)
        
        Q = Q / math.sqrt(dim_split)
        # 计算注意力权重
        A = torch.softmax(torch.bmm(Q, K.transpose(1, 2)), dim=2) # A: (B*h, T_q, T_k)
        A = self.dropout(A)

        if progressive_drop != 1 and T_q > 1:
            # 计算每个头中各位置的最大注意力得分，再将各头求和
            weight = A.view(B, self.num_heads, T_q, T_k).max(-1)[0] # max over T_k
            weight = weight.sum(1)  # shape: (B, T_q)
            
            # 使用稳定化采样
            if L_out <= 0: L_out = 1
            sampled_inds = self._stabilized_sampling(weight, L_out)
            
            # 将 A 从 (B*num_heads, T_q, T_k) 调整为 (B, num_heads, T_q, T_k)，再根据 sampled_inds 采样
            A = torch.gather(
                A.unflatten(0, (-1, self.num_heads)),
                2,
                sampled_inds[:, None, :, None].repeat(1, self.num_heads, 1, T_k)
            ).flatten(0, 1)
        else:
            weight = None
            sampled_inds = None
            
        # 注意力加权聚合
        O = torch.bmm(A, V) # O: (B*h, L_out, D_split)
        # 恢复多头结构
        O = O.view(B, self.num_heads, L_out, dim_split).transpose(1, 2).contiguous().view(B, L_out, self.dim_LIN)
        O = linear(O, self.out_proj.weight, self.out_proj.bias)
        return O, weight, sampled_inds

class SGSA(nn.Module):
    """
    Advanced Adaptive Multi-Head Self-Attention
    改进点: 
    1. 混合重要性评分 (Hybrid Scoring: Attention + L2 Norm)
    2. 注意力重归一化 (Re-normalization)
    3. 更加鲁棒的采样策略
    """
    
    def __init__(self, dim, num_heads, dropout=0.0):
        super(SGSA, self).__init__()
        self.dim_LIN = dim
        self.num_heads = num_heads
        
        # 基础投影层
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        
        # --- 渐进式采样参数 ---
        self.current_epoch = 0
        self.warmup_epochs = 10
        self.min_drop_rate = 0.3
        self.max_drop_rate = 0.8
        
        # --- 稳定化参数 ---
        self.noise_scale = 0.01
        self.temperature = 0.1
        
        # --- 新增: 评分平衡参数 ---
        # alpha 用于平衡 Attention分数 和 L2特征模长 的权重
        # 0.5 表示两者同等重要
        self.alpha = 0.5 

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.constant_(self.k_proj.bias, 0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.constant_(self.v_proj.bias, 0)
        nn.init.constant_(self.out_proj.bias, 0)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _get_progressive_drop_rate(self, base_drop):
        if not self.training:
            return base_drop
        if self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            current_drop = self.max_drop_rate + (base_drop - self.max_drop_rate) * progress
            return max(current_drop, self.min_drop_rate)
        else:
            return base_drop

    def _calculate_hybrid_score(self, A, Q_features):
        """
        计算混合重要性分数
        A: [B, Heads, T_q, T_k] - 注意力矩阵
        Q_features: [B, T_q, Dim] - Query的原始特征
        """
        B, H, T_q, T_k = A.shape
        
        # 1. Attention Saliency (注意力显著性)
        # 计算每个 Query 被关注的程度 (或者它关注别人的强烈程度)
        # 这里使用 max over keys (原本的逻辑)，并对 heads 取平均
        attn_score = A.max(dim=-1)[0].mean(dim=1) # [B, T_q]
        
        # 2. Feature Magnitude (特征模长显著性)
        # L2-Norm 往往代表特征的信息量
        feature_score = torch.norm(Q_features, p=2, dim=-1) # [B, T_q]
        
        # 3. 归一化并融合
        # 将两者都归一化到 [0, 1] 区间以便相加
        attn_score = (attn_score - attn_score.min(1, keepdim=True)[0]) / \
                     (attn_score.max(1, keepdim=True)[0] - attn_score.min(1, keepdim=True)[0] + 1e-6)
                     
        feature_score = (feature_score - feature_score.min(1, keepdim=True)[0]) / \
                        (feature_score.max(1, keepdim=True)[0] - feature_score.min(1, keepdim=True)[0] + 1e-6)
        
        # 混合分数
        combined_score = self.alpha * attn_score + (1 - self.alpha) * feature_score
        return combined_score

    def _stabilized_sampling(self, weight, L_out):
        """稳定化的采样"""
        if self.training:
            # 注入噪声防止过拟合于特定的 token
            noise = torch.randn_like(weight) * self.noise_scale
            weight = weight + noise
            # 温度平滑
            weight = weight / self.temperature
            
        # Top-K 采样
        sampled_inds = torch.topk(weight, L_out, dim=-1)[1] # [B, L_out]
        return sampled_inds

    def forward(self, Q_in, K_in, V_in, drop=0.5):
        """
        Q_in, K_in, V_in: 输入张量 [B, T, D]
        """
        B, T_q, D = Q_in.shape
        _, T_k, _ = K_in.shape

        # 1. 线性投影
        Q = self.q_proj(Q_in)
        K = self.k_proj(K_in)
        V = self.v_proj(V_in)

        # 2. 多头维度变换
        dim_split = self.dim_LIN // self.num_heads
        Q_split = Q.view(B, T_q, self.num_heads, dim_split).transpose(1, 2) # [B, H, T_q, D_s]
        K_split = K.view(B, T_k, self.num_heads, dim_split).transpose(1, 2)
        V_split = V.view(B, T_k, self.num_heads, dim_split).transpose(1, 2)

        # 3. 计算原始 Attention Map
        # Scale Dot-Product
        scores = torch.matmul(Q_split, K_split.transpose(-2, -1)) / math.sqrt(dim_split)
        A = torch.softmax(scores, dim=-1) # [B, H, T_q, T_k]
        A = self.dropout(A)

        # 4. 决定输出长度
        progressive_drop = self._get_progressive_drop_rate(drop)
        L_out = int(T_q * progressive_drop)

        # 5. 核心逻辑: 采样与剪枝
        if progressive_drop < 1.0 and T_q > 1 and L_out > 0:
            # --- 改进点 A: 使用混合评分 ---
            # 传入 Q_in (原始特征) 或 Q (投影后特征) 均可，这里用 Q
            weight = self._calculate_hybrid_score(A, Q) 
            
            # --- 采样索引 ---
            sampled_inds = self._stabilized_sampling(weight, L_out) # [B, L_out]
            
            # --- 准备 Gather ---
            # 扩展索引以匹配多头 Attention 的维度 [B, H, L_out, T_k]
            #我们需要在 dim=2 (T_q维度) 上进行 gather
            gather_inds = sampled_inds.view(B, 1, L_out, 1).expand(-1, self.num_heads, -1, T_k)
            
            # --- 提取保留的 Attention ---
            A_sampled = torch.gather(A, 2, gather_inds)
            
            # --- 改进点 B: 简单的信息补偿 (Renormalization) ---
            # 由于我们丢弃了一些行，剩下的 Attention 权重可能不够强。
            # 这里我们简单地对 key 维度重新归一化，或者将丢弃的能量加回来
            # 简单策略：Renormalize to sum to 1
            A_sampled_sum = A_sampled.sum(dim=-1, keepdim=True) + 1e-6
            A_sampled = A_sampled / A_sampled_sum
            
            # 更新 A
            A = A_sampled
            
            # 为了返回给外部可视化或其他用途，保留 weight
            final_weight = weight
        else:
            final_weight = None
            sampled_inds = None

        # 6. 聚合 Values
        # A: [B, H, L_out, T_k] * V: [B, H, T_k, D_s] -> [B, H, L_out, D_s]
        O = torch.matmul(A, V_split)
        
        # 7. 恢复维度
        O = O.transpose(1, 2).contiguous().view(B, -1, self.dim_LIN) # [B, L_out, D]
        O = self.out_proj(O)

        return O, final_weight, sampled_inds

class ToProxy2_SGSA(nn.Module):
    """Enhanced ToProxy2 with SGSA"""
    
    def __init__(self, dim=512, head_num=8, agent_q_num=8 ** 2, agent_kv_num=8 ** 2,
                 norm=nn.InstanceNorm2d, act=nn.SiLU):
        super().__init__()
        self.agent_q_num = agent_q_num
        self.head = head_num

        # --- PE 模块的初始化 ---
        # 1. 图像PE生成器 (动态，尺寸无关)
        self.image_pos_embed = build_position_embedding('sine', (1, 1), dim)  # dummy_size 无所谓

        # 2. 代理Q的PE (固定尺寸，在此直接生成并存储)
        agent_q_size = (int(agent_q_num ** 0.5), int(agent_q_num ** 0.5))
        q_agent_pos_embed_generator = build_position_embedding('sine', agent_q_size, dim)
        # 创建一个假的张量来调用生成器一次
        dummy_q_tensor = torch.zeros(1, dim, *agent_q_size)
        q_agent_pe = q_agent_pos_embed_generator(dummy_q_tensor)
        # 将生成的固定 PE 注册为 buffer，而不是重新计算
        self.register_buffer('q_agent_pe', q_agent_pe)

        # 3. 代理KV的PE (同理)
        agent_kv_size = (int(agent_kv_num ** 0.5), int(agent_kv_num ** 0.5))
        kv_agent_pos_embed_generator = build_position_embedding('sine', agent_kv_size, dim)
        dummy_kv_tensor = torch.zeros(1, dim, *agent_kv_size)
        kv_agent_pe = kv_agent_pos_embed_generator(dummy_kv_tensor)
        self.register_buffer('kv_agent_pe', kv_agent_pe)

        # --- 原始模块定义 ---
        self.learnable_q = nn.Parameter(torch.zeros(1, agent_q_num, dim))
        self.learnable_kv = nn.Parameter(torch.zeros(1, agent_kv_num, dim))
        self.q = nn.Linear(dim, dim)
        self.kv_x = nn.Linear(dim, dim * 2)
        self.kv_y = nn.Linear(dim, dim * 2)
        self.norm = norm(dim) if norm else nn.Identity()
        self.act = act()
        self.q_2 = nn.Linear(dim, dim)

        # 使用增强版的SGSA
        # self.attn1 = En_Adaptive_MHSA(dim=dim, num_heads=head_num)
        # self.attn2 = En_Adaptive_MHSA(dim=dim, num_heads=head_num)
        self.attn1 = SGSA(dim=dim, num_heads=head_num)
        self.attn2 = SGSA(dim=dim, num_heads=head_num)
        
    def set_epoch(self, epoch):
        """设置当前epoch，传递给内部的SGSA模块"""
        self.attn1.set_epoch(epoch)
        self.attn2.set_epoch(epoch)

    def forward(self, x):
        # 假设 x 的 shape 是 (b, h, w, c)
        b, h, w, c = x.shape

        # --- Attention 1: Agent-to-Image ---
        # 1. 动态生成图像PE，直接使用预计算好的代理PE
        image_pe = self.image_pos_embed(x.permute(0, 3, 1, 2))

        # 2. 为 learnable_q 添加 PE (此时 batch_size 还是 1)
        q_source_base = self.learnable_q + self.q_agent_pe  # Shape: [1, 64, 512]

        # 3. 【关键步骤】将带 PE 的代理查询沿 batch 维度复制 b 份
        q_source = q_source_base.repeat(b, 1, 1)  # Shape: [b, 64, 512]

        # 4. 为图像特征添加 PE
        kv_source = x.reshape(b, h * w, c) + image_pe

        # 5. 投射到 Q, K, V
        q = self.q(q_source)  # q 的 shape 现在是 [b, 64, 512]
        k, v = self.kv_x(kv_source).chunk(2, dim=-1)  # k, v 的 shape 是 [b, H*W, 512]
        
        # 现在 q, k, v 的 batch_size 都是 b，可以安全地传入 attn1
        atn_out, _, sampled_inds = self.attn1(q, k, v, drop=0.5)

        if sampled_inds is not None:
            full_len_out = torch.zeros_like(q)
            sampled_inds_expanded = sampled_inds.unsqueeze(-1).expand(-1, -1, c)
            full_len_out = full_len_out.scatter(1, sampled_inds_expanded, atn_out)
            atn_out = full_len_out

        h1 = w1 = int(self.agent_q_num ** 0.5)
        out = atn_out.reshape(b, h1, w1, -1).permute(0, 3, 1, 2)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        out = out + x
        mskip = out

        # --- Attention 2: Image-to-Agent (同理) ---
        q_source = out.reshape(b, h * w, c) + image_pe

        # 【关键步骤】同样需要对 learnable_kv 进行 repeat
        kv_source_base = self.learnable_kv + self.kv_agent_pe
        kv_source = kv_source_base.repeat(b, 1, 1)

        q = self.q_2(q_source)
        k, v = self.kv_y(kv_source).chunk(2, dim=-1)

        atn_out2, _, sampled_inds2 = self.attn2(q, k, v, drop=0.5)

        if sampled_inds2 is not None:
            full_len_out = torch.zeros_like(q)
            sampled_inds_expanded = sampled_inds2.unsqueeze(-1).expand(-1, -1, c)
            full_len_out = full_len_out.scatter(1, sampled_inds_expanded, atn_out2)
            atn_out2 = full_len_out

        out = atn_out2.reshape(b, h, w, -1) + mskip + x

        return out.contiguous()

class S3_Block(nn.Module):
    def __init__(self,hidden_dim: int = 0,feature_size=(8, 8)):
        
        super().__init__()
        self.toProxy = ToProxy2_SGSA(dim=hidden_dim)
        self.conv33 = DWConv(hidden_dim, hidden_dim, 3,padding=1)
        self.conv55 = DWConv(hidden_dim, hidden_dim, 5,padding=2)
        self.mlp = nn.Sequential(nn.Conv2d(hidden_dim*2, hidden_dim, 1),nn.InstanceNorm2d(hidden_dim),nn.SiLU())  
        self.l = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.InstanceNorm2d(hidden_dim)
        self.act = nn.SiLU()
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        conv_input = x.permute(0, 3, 1, 2)
        out = self.toProxy(x)
        y33 = self.conv33(conv_input).permute(0,2,3,1) # b h w c 
        y55 = self.conv55(conv_input).permute(0,2,3,1) # b h w c
        y33 = self.mlp(torch.cat([y33,y55],dim=-1).permute(0,3,1,2)).permute(0,2,3,1)
        
        out = self.act(self.norm(self.l(out+y33).permute(0,3,1,2))).permute(0,2,3,1)
        out = self.l2(out)
        out = out + x
        
        return out.contiguous()
    

class S3BlockWrapper(nn.Module):
    def __init__(self,hidden_dim: int = 0):
            super().__init__()
            self.blocks = nn.ModuleList([S3_Block(hidden_dim=hidden_dim) for _ in range(1)])
           
    def forward(self, x):
        # x -  B H W C
        out = x
        for block in self.blocks:
            out = block(out)
        return out
    


def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)
        return x

class DecoderStage(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([S3BlockWrapper(hidden_dim=dim) for i in range(depth)])

        if True:  
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self, dims_decoder=None, depths_decoder=[3, 9, 9, 7], drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        # 如果没有提供dims_decoder，使用默认值
        if dims_decoder is None:
            dims_decoder = [512, 256, 128, 64]
        
        # 确保dims_decoder和depths_decoder长度一致
        assert len(dims_decoder) == len(depths_decoder), \
            f"dims_decoder长度({len(dims_decoder)})必须与depths_decoder长度({len(depths_decoder)})一致"
        
        print(f"Decoder初始化 - 维度: {dims_decoder}, 深度: {depths_decoder}")
        
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]
        self.layers_up = nn.ModuleList()
        for i_layer in range(len(depths_decoder)):
            layer = DecoderStage(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                # upsample=PatchExpand2D_Fixed if (i_layer != 0) else None,
            )
            self.layers_up.append(layer)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x = rearrange(x,'b c h w -> b h w c')
        out_features = []
        for i, layer in enumerate(self.layers_up):
            x = layer(x)
            if i != 0:
                out_features.insert(0, rearrange(x,'b h w c -> b c h w'))
        return out_features

# 改进后的 FPN 模块，整合 Exchange 机制
class EFPN(nn.Module):
    def __init__(self, block, layers, input_dims=[128, 256, 512], output_dim=512,
                 width_per_group=64, norm_layer=None, bn_threshold=2e-2):
        super(EFPN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.base_width = width_per_group
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.bn_threshold = bn_threshold

        # 动态计算中间维度
        intermediate_dim1 = input_dims[0] * 2
        intermediate_dim2 = output_dim

        # ResNet层设置
        self.inplanes = intermediate_dim2
        self.dilation = 1
        planes = intermediate_dim2 // block.expansion
        self.bn_layer = self._make_layer(block, planes, layers, stride=2)

        # 基础卷积层
        self.conv1 = conv3x3(input_dims[0], intermediate_dim1, 2)
        self.bn1 = norm_layer(intermediate_dim1)
        self.conv2 = conv3x3(intermediate_dim1, intermediate_dim2, 2)
        self.bn2 = norm_layer(intermediate_dim2)

        # 适配层
        self.conv21 = nn.Conv2d(intermediate_dim1, intermediate_dim1, 1)
        self.bn21 = norm_layer(intermediate_dim1)
        self.adapt_conv1 = nn.Conv2d(input_dims[1], intermediate_dim1, 1)
        self.adapt_bn1 = norm_layer(intermediate_dim1)

        self.conv31 = nn.Conv2d(intermediate_dim2, intermediate_dim2, 1)
        self.bn31 = norm_layer(intermediate_dim2)
        self.adapt_conv2 = nn.Conv2d(input_dims[2], intermediate_dim2, 1)
        self.adapt_bn2 = norm_layer(intermediate_dim2)

        self.convf = nn.Conv2d(intermediate_dim2, intermediate_dim2, 1)
        self.bnf = norm_layer(intermediate_dim2)
        self.relu = nn.ReLU(inplace=True)
        # 添加 Exchange 模块用于智能特征融合
        self.exchange1 = Exchange()  # 用于第一阶段融合
        self.exchange2 = Exchange()  # 用于第二阶段融合

        # 为 Exchange 模块创建对应的 BN 层列表
        self.bn_exchange1 = [self.bn21, self.bn1]  # 用于 fpn0 和 x1_adapted 之间的交换
        self.bn_exchange2 = [self.bn31, self.bn2]  # 用于 fpn2 和 x2_adapted 之间的交换

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, base_width=self.base_width, 
                  dilation=previous_dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, base_width=self.base_width, 
                      dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 第一阶段处理
        fpn0 = self.relu(self.bn1(self.conv1(x[0])))

        # 适配第二个输入
        x1_adapted = self.relu(self.adapt_bn1(self.adapt_conv1(x[1])))

        # 使用 Exchange 进行智能融合而不是简单相加
        # 将 fpn0 和 x1_adapted 作为两个分支输入 Exchange 模块
        exchange_inputs1 = [fpn0, x1_adapted]
        exchanged_features1 = self.exchange1(exchange_inputs1, self.bn_exchange1, self.bn_threshold)
        fpn1 = self.relu(self.bn21(self.conv21(exchanged_features1[0]))) + exchanged_features1[1]

        # 第二阶段处理
        fpn2 = self.relu(self.bn2(self.conv2(fpn1)))

        # 适配第三个输入
        x2_adapted = self.relu(self.adapt_bn2(self.adapt_conv2(x[2])))

        # 再次使用 Exchange 进行智能融合
        exchange_inputs2 = [fpn2, x2_adapted]
        # print("exchange neck")
        exchanged_features2 = self.exchange2(exchange_inputs2, self.bn_exchange2, self.bn_threshold)
        sv_features = exchanged_features2[0] + self.relu(self.bn31(self.conv31(exchanged_features2[1])))

        # 最终处理
        sv_features = self.relu(self.bnf(self.convf(sv_features)))
        sv_features = self.bn_layer(sv_features)

        return sv_features.contiguous()
    
    
class S3AD_module(nn.Module):
    def __init__(self, model_t, model_s):
        super(S3AD_module, self).__init__()
        self.net_t = get_model(model_t)
        
        # 根据teacher网络类型动态设置维度
        teacher_name = model_t.name.lower()
        if 'clip' in teacher_name or 'rn50' in teacher_name or 'resnet50' in teacher_name:
            # CLIP ResNet50或标准ResNet50的特征维度 (实际测试得出: [256, 512, 1024])
            print("检测到ResNet50架构，使用相应的特征维度配置")
            teacher_dims = [256, 512, 1024]  # out_indices=[1,2,3]对应的维度
            neck_output_dim = 1024  # 使用1024，能被expansion=4整除
            decoder_dims = [1024, 512, 256, 128]  # Decoder的维度配置
            # decoder_dims = [1024, 1024, 512, 256]  # Decoder的维度配置
        elif 'resnet34' in teacher_name or 'rn34' in teacher_name or 'timm_resnet34' in teacher_name:
            # ResNet34的特征维度 (实际测试得出: [64, 128, 256])
            print("检测到ResNet34架构，使用相应的特征维度配置")
            teacher_dims = [64, 128, 256]
            neck_output_dim = 512  # 使用512确保能被expansion=4整除
            decoder_dims = [512, 256, 128, 64]  # 对应调整decoder维度
        else:
            # 默认配置（假设为ResNet34）
            print(f"未识别的teacher架构: {teacher_name}，使用默认ResNet34配置")
            teacher_dims = [64, 128, 256]
            neck_output_dim = 512  # 使用512确保能被expansion=4整除
            decoder_dims = [512, 256, 128, 64]  # 对应调整decoder维度
        
        # 保存配置以便调试
        self.teacher_dims = teacher_dims
        self.neck_output_dim = neck_output_dim
        self.decoder_dims = decoder_dims
        
        print(f"Teacher特征维度: {teacher_dims}")
        print(f"Neck输出维度: {neck_output_dim}")
        print(f"Decoder维度: {decoder_dims}")
        
        # 添加 channel adapter 用于维度匹配 (在 net_t 和 Neck 之间定义)
        # if 'clip' in teacher_name or 'rn50' in teacher_name or 'resnet50' in teacher_name:
        if 1==0:
            self.is_rn50 = True
            # decoder_dims = [1024,512,256,128] → feats_s 通道 = [128,256,512] (由于 insert(0, ...) 反向)
            # teacher_dims = [256, 512, 1024]
            # 将 student 特征升维到 teacher 维度 (与日志中的结构一致)
            decoder_output_dims = [decoder_dims[-i-1] for i in range(len(teacher_dims))]  # [128, 256, 512]
            self.channel_adapters = nn.ModuleList([
                nn.Conv2d(decoder_output_dims[i], teacher_dims[i], kernel_size=1)  # i=0: 128->256, i=1:256->512, i=2:512->1024
                for i in range(len(teacher_dims))
            ])
            print(f"初始化 channel adapters: decoder_output_dims{decoder_output_dims} -> teacher_dims{teacher_dims}")
        else:
            self.is_rn50 = False
            self.channel_adapters = None

        # 使用动态维度初始化Neck和Decoder
        # self.Neck = Neck(Bottleneck, 3, input_dims=teacher_dims, output_dim=neck_output_dim)
        self.Neck = EFPN(Bottleneck, 3, input_dims=teacher_dims, output_dim=neck_output_dim)
        
        # 更新model_s配置
        final_model_s = model_s.copy()
        final_model_s['dims_decoder'] = decoder_dims
        self.net_s = Decoder(depths_decoder=final_model_s['depths_decoder'], 
                           dims_decoder=final_model_s['dims_decoder'])

        self.frozen_layers = ['net_t']

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)

        return self

    def _compute_teacher_features(self, imgs):
        """计算教师网络特征"""
        with torch.no_grad():
            feats_t = self.net_t(imgs)
        return [f.detach() for f in feats_t]

    def forward(self, imgs):
        # Always use the current images; batch-index caches are unsafe with shuffled data.
        feats_t = self._compute_teacher_features(imgs)

        # 后续处理保持不变
        oce_out = self.Neck(feats_t)  # 16 512 8 8
        b, c, h, w = oce_out.shape
        scale = 20 if self.training else 0
        oce_out = add_jitter(oce_out.reshape(b, c, h * w).permute(0, 2, 1), scale=scale).reshape(b, h, w, c).permute(0,
                                                                                                                     3,
                                                                                                                     1,
                                                                                                                     2).contiguous()
        feats_s = self.net_s(oce_out)
        
        # 如果是ResNet50-based teacher，应用channel adapters进行维度匹配
        # if hasattr(self, 'is_rn50') and self.is_rn50 and self.channel_adapters is not None:
        if 1==0:
            # 将 student 特征升维到 teacher 维度 (与日志中的结构一致)
            feats_s_adapted = []
            for i, feat in enumerate(feats_s):
                adapted = self.channel_adapters[i](feat)  # 升维：128->256, 256->512, 512->1024
                feats_s_adapted.append(adapted)
            return feats_t, feats_s_adapted
        else:
            return feats_t, feats_s  # 非 RN50 时，返回原始特征

@MODEL.register_module
def S3AD(pretrained=False, **kwargs):
    model = S3AD_module(**kwargs)
    return model
