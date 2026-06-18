import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    二分类 / 多标签分类 版本的 Focal Loss

    参数:
        alpha: 正类(target=1)的权重
               负类(target=0)的权重会自动变成 (1 - alpha)

               例如:
               - alpha = 0.75: 更重视正类
               - alpha = 0.25: 更重视负类
               - alpha = 0.5 : 正负类一样重要

        gamma: 聚焦参数，用来降低“简单样本”的损失权重
               - gamma = 0: 退化为普通 BCE（二元交叉熵损失）
               - gamma 越大: 越关注难样本，越忽略简单样本

        reduction:
               - 'none' : 不聚合，直接返回每个位置的 loss
               - 'mean' : 对所有位置取平均
               - 'sum'  : 对所有位置求和
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        参数:
            logits:
                模型的原始输出，尚未经过 sigmoid
                shape 可以是:
                - [B]
                - [B, 1]
                - [B, C]        (多标签分类)
                - [B, 1, H, W]  (二值分割)
                总之：任意 shape 都行

            targets:
                真实标签，必须与 logits 同 shape
                每个位置的值应为 0 或 1

        返回:
            loss 标量，或逐元素 loss（取决于 reduction）
        """

        # BCEWithLogits 要求 targets 是浮点型
        targets = targets.float()

        # ------------------------------------------------------------
        # 第1步：把 logits 转成概率
        # logits 是任意实数
        # probs = sigmoid(logits) 后，范围变成 (0, 1)
        # 表示“预测为正类(target=1)的概率”
        # ------------------------------------------------------------
        probs = torch.sigmoid(logits)

        # ------------------------------------------------------------
        # 第2步：先计算普通的 BCE loss（reduction='none'，算出每一项结果）
        # 如果是reduction='none'，那输出就是和logits维度一样
        # 如果是reduction='mean'或者'sum'，那就只输出一个float了
        # ------------------------------------------------------------
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        # 也可以用F.binary_cross_entropy(probs, targets, reduction='none')，根据手动算出的probs来计算

        # ------------------------------------------------------------
        # 第3步：构造 p_t
        # p_t 的含义是：模型给“真实类别”分配的概率
        #
        # 分两种情况：
        # 1) 如果 target = 1，那么 p_t = probs
        # 2) 如果 target = 0，那么 p_t = 1 - probs
        #
        # 所以统一写成：
        # p_t = probs * targets + (1 - probs) * (1 - targets)
        # ------------------------------------------------------------
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # ------------------------------------------------------------
        # 第4步：构造 alpha_t（类别平衡权重）
        #
        # 如果 target = 1（正类），权重 = alpha
        # 如果 target = 0（负类），权重 = 1 - alpha
        #
        # 所以：
        # - alpha > 0.5  => 更重视正类
        # - alpha < 0.5  => 更重视负类
        #
        # 注意：
        # alpha 永远是给 target=1 这一类的，不是自动给“少数类”的
        # 如果你把标签编码反过来了，alpha 的含义也会跟着反过来
        # ------------------------------------------------------------
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # ------------------------------------------------------------
        # 第5步：构造 focal 调制因子
        #
        # (1 - p_t)^gamma
        #
        # 当一个样本很好分时，p_t 会很大，接近 1
        # 这时 (1 - p_t) 很小，整个权重就会很小
        # => 简单样本的损失被压低
        #
        # 当一个样本很难分时，p_t 会较小
        # 这时 (1 - p_t) 较大，整个权重不会太小
        # => 难样本保留较大的损失
        #
        # gamma 越大，这种“抑制简单样本”的效果越强
        # ------------------------------------------------------------
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)

        # ------------------------------------------------------------
        # 第6步：最终 loss = focal 权重 × 普通 BCE
        # ------------------------------------------------------------
        loss = focal_weight * bce_loss

        # ------------------------------------------------------------
        # 第7步：按 reduction 方式返回
        # ------------------------------------------------------------
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss