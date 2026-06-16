import torch
import torch.nn.functional as F

def clip_loss(image_emb, text_emb, temperature=0.07):
    # normalize
    image_emb = F.normalize(image_emb, dim=-1) # L2 normalization，即向量除以模长，变为单位向量
    text_emb = F.normalize(text_emb, dim=-1)

    # similarity logits
    logits = image_emb @ text_emb.T / temperature # 相似度矩阵，带温度(temperature<1)的缩放

    # diagonal matches are positives
    B = image_emb.size(0) # image_emb是[B, D]，B是Batch Size，D是Embedding Dimension
    labels = torch.arange(B, device=logits.device) # 一个列向量，为[0, 1, 2, ..., B-1]

    # symmetric contrastive loss
    loss_i = F.cross_entropy(logits, labels) # 图片对文本的损失
    loss_t = F.cross_entropy(logits.T, labels) # 文本对图片的损失
    return (loss_i + loss_t) / 2 # 平均损失