import torch
import torch.nn as nn
import torch.nn.functional as F
# from trl import DPOTrainer


def simplified_dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    ref_chosen_logps: torch.FloatTensor,
    ref_rejected_logps: torch.FloatTensor,
    beta: float = 0.1,
):
    """
    Simplified DPO loss function.
    
    Args:
        policy_chosen_logps: Policy model log probs for chosen responses (batch_size,)
        policy_rejected_logps: Policy model log probs for rejected responses (batch_size,)
        ref_chosen_logps: Reference model log probs for chosen responses (batch_size,)
        ref_rejected_logps: Reference model log probs for rejected responses (batch_size,)
        beta: Temperature parameter (typically 0.1-0.5)
        reference_free: Whether to ignore reference model (ref_logps=0)
    
    Returns:
        losses: DPO losses for each example (batch_size,)
        chosen_rewards: Rewards for chosen responses (batch_size,)
        rejected_rewards: Rewards for rejected responses (batch_size,)
    """
    # Calculate log ratios 
    # 已经是log概率了，所以直接相减即可
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps

    # Compute logits (difference between chosen and rejected log ratios)
    logits = chosen_logratios - rejected_logratios

    # Compute DPO loss (sigmoid version)
    losses = -F.logsigmoid(beta * logits)

    return losses

# 示例
if __name__ == "__main__":
    # 假设我们有以下logits
    policy_chosen_logps = torch.tensor([0.9, 0.6, 0.7])
    policy_rejected_logps = torch.tensor([0.6, 0.2, 0.6])
    ref_chosen_logps = torch.tensor([0.5, 0.4, 0.3])
    ref_rejected_logps = torch.tensor([0.4, 0.3, 0.2])

    # 调用简化的DPO损失函数
    losses = simplified_dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=0.4
    )

    print("DPO Losses:", losses)