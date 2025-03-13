import torch
from model import args
from torch.utils.data import Dataset, DataLoader

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

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Add batch dimension
    return encoded_tensor.to(args.device)

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)   # Removes batch dimension
    return tokenizer.decode(flat.tolist())


class ModelDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text) 
        for i in range(0, len(token_ids) - max_length, stride): 
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1] #[-1] why not take only the last element from here?
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self): 
        return len(self.input_ids)
    def __getitem__(self, idx): 
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(text, tokenizer, batch_size=args.max_batch_size,
                         max_length=args.max_seq_len, stride=128, shuffle=True,
                         drop_last=True, num_workers=0): 
    dataset = ModelDataset(text, tokenizer, max_length, stride) 
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last, 
        num_workers=num_workers 
    )
    return dataloader
