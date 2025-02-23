def get_model_size(model):
  total_params = sum(p.numel() for p in model.parameters())
  total_size = total_params * 4 / (1024**2)
  return total_size, total_params

def generate_text_simple(model, idx, max_new_tokens, context_size): 
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] 
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]  #Focuses only on the last time step, so that (batch, n_token, vocab_size) becomes (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1) 
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) 
        idx = torch.cat((idx, idx_next), dim=1)   # Appends sampled index to the running sequence, where idx has shape (batch, n_tokens+1)
    return idx