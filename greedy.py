import torch
import torch.nn as nn
import torch.nn.functional as F

def greedy_search(model,input_ids,max_length,eos_token_id=None):
    model.eval()
    device = input_ids.device
    batch_size = input_ids.size(0)
    # batch_size seq_len
    sequences = input_ids.clone().to(device)

    for step in range(max_length):
        with torch.no_grad(): 
            # batch_size seq_len vocab_size
            outputs = model(sequences)
            # batch_size vocab_size
            next_token_logits = outputs.logits[:,-1,:]
        # log_probs = F.log_softmax(next_token_logits, dim=-1)
        # batch_size
        next_token = next_token_logits.argmax(dim=-1)
        sequences = torch.cat([sequences,next_token.unsqueeze(-1)],dim=-1)
        if eos_token_id is not None:
            if (sequences[:,-1]==eos_token_id).all():
                break
        return sequences
