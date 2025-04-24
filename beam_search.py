import torch
import torch.nn.functional as F

def beam_search(model,input_ids,max_length,beam_width,eos_token_id=None):
    model.eval()
    device = input_ids.device
    batch_size = input_ids.size(0)

    beam_scores = torch.zeros(batch_size,beam_width).to(device)
    # batch_size beam_width
    beam_sequences = input_ids.unsqueeze(1).repeat(1,beam_width,1).to(device)

    for step in range(max_length):
        flat_sequeeces = beam_sequences.view(-1,beam_sequences.size(-1))
        with torch.no_grad():
            outputs = model(flat_sequeeces)
            next_token_logits = outputs.logits[:,-1,:]
        # batch_size*beam_width,vocab_size
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        if step == 0:
            scores = log_probs[:batch_size]
            topk_scores, topk_indices = scores.topk(beam_width, dim=-1)
        else:
            scores = beam_scores.view(-1,1) + log_probs
            # batch_size,beam_width*vocab_size
            scores = scores.view(batch_size,-1)
            topk_scores, topk_indices = scores.topk(beam_width,dim=-1)
        vocab_size = log_probs.size(-1)
        beam_indices = topk_indices // vocab_size
        token_indices = topk_indices % vocab_size

        beam_sequences = torch.cat([beam_sequences[torch.arange(batch_size).unsqueeze(1),beam_indices],token_indices.unsqueeze(-1)],dim=-1)
        beam_scores = topk_scores

        if eos_token_id is not None:
            if (beam_sequences[:,:,-1]==eos_token_id).all():
                break
    # 选择分数最高的序列
    _, best_indices = beam_scores.max(dim=1)
    return beam_sequences[torch.arange(batch_size),best_indices]


batch_size = 2
beam_width = 3
seq_len = 4
beam_sequences = torch.randn(batch_size,beam_width,seq_len)
best_indices = [1,2]
print(beam_sequences)
print(beam_sequences[torch.arange(batch_size),best_indices])

