import numpy as np
def top_k_sampling(logits, k=2):
    probs = np.exp(logits)/np.sum(np.exp(logits))
    indices = np.argsort(probs)[-k:][::-1]
    topk_probs = probs[indices]
    topk_probs = topk_probs / np.sum(topk_probs)

    sample_index = np.random.choice(indices, p=topk_probs)
    return sample_index
logits = [0.2,0.4,0.5,0.3]
print(top_k_sampling(logits, k=2))
