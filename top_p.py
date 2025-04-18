import numpy as np
def top_p_sampling(logits, p=0.9):
    """
    进行top-p采样
    通过对logits进行softmax计算得到概率分布，然后对概率进行排序并计算累积概率，
    找到累积概率大于等于p的索引，最后从这些索引中进行采样。
    """
    # 转化为概率
    logits = np.exp(logits) / np.sum(np.exp(logits))  
    # 对概率进行排序并得到索引
    # 注意argsort和sort的区别 同时这里是从大到小进行排序
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    # 计算累积概率
    cumulative_probs = np.cumsum(sorted_logits)
    # 得到截断概率的索引
    cutoff_index = np.where(cumulative_probs >= p)[0][0]
    # 得到top_p的索引
    top_p_indices = sorted_indices[:cutoff_index + 1]
    top_p_probs = sorted_logits[top_p_indices]
    # 归一化截断概率
    top_p_probs = top_p_probs / np.sum(top_p_probs)
    # top_p采样
    sampled_index = np.random.choice(top_p_indices, p=top_p_probs)

    return sampled_index

logits = [0.2,0.4,0.5,0.3]
print(top_p_sampling(logits, p=0.7))