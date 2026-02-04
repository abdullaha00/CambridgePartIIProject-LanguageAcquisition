
import torch


# ===== DKT
def collate_dkt(batch):
    # batch: list of (uid, q, a)
    
    uids, q_ids, correct_list = zip(*batch)

    T_max = max(len(q_seq) for q_seq in q_ids)
    B = len(q_ids)

    q_ids_padded = torch.zeros((B, T_max), dtype=torch.long)
    correct_list_padded = torch.zeros((B, T_max), dtype=torch.long)
    mask = torch.zeros((B, T_max), dtype=torch.bool)

    for i, (q, a) in enumerate(zip(q_ids, correct_list)):
        T = len(q)
        q_ids_padded[i, :T] = q
        correct_list_padded[i, :T] = a
        mask[i, :T] = 1
    
    return uids, q_ids_padded, correct_list_padded, mask

# ===== LMKT
def lmkt_collate(batch, pad_token_id: int):
    # Batch: list[Tensor(seq_len)]

    T_max = max(x.numel() for x in batch)
    B = len(batch)

    seqs_padded = torch.full((B, T_max), pad_token_id, dtype=torch.long)
    mask = torch.zeros((B, T_max), dtype=torch.long) 

    for i, seq in enumerate(batch):
        T = seq.numel()
        seqs_padded[i, :T] = seq
        mask[i, :T] = 1
    
    #===== -100 signifies ignore for LM loss =====
    labels = seqs_padded.clone()
    labels[mask == 0] = -100
    
    return {"input_ids": seqs_padded, "attention_mask": mask, "labels": labels}

