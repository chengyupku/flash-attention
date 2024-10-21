import torch
import torch.nn.functional as F

batch = 8
seqlen_q = 1024
seqlen_kv = 2048
nheads = 16
dim = 128
block_M = seqlen_q
block_N = 64

def attention(Q, K, V):
    dim = Q.size(-1)
    scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)
    return output


def flash(Q, K, V):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    acc_s = torch.empty((batch, nheads, block_M, block_N), device="cuda", dtype=torch.float)
    acc_s_cast = torch.empty((batch, nheads, block_M, block_N), device="cuda", dtype=torch.float16)
    acc_o = torch.empty((batch, block_M, nheads, dim), device="cuda", dtype=torch.float)
    scores_max = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    scores_max_prev = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    scores_scale = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    scores_sum = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    logsum = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    acc_o.fill_(0)
    logsum.fill_(0)
    scores_max.fill_(float('-inf'))
    Q *= scale

    for i in range(int(seqlen_kv / block_N)):
        acc_s.fill_(0)
        acc_s = torch.einsum('bqhd,bkhd->bhqk', Q, K[:, i * block_N : (i + 1) * block_N, :, :]) # [batch, seqlen, nheads, block_N]
        scores_max_prev = scores_max
        scores_max = acc_s.max(dim=-1, keepdim=False).values # [blockM]
        scores_scale = torch.exp2(scores_max_prev - scores_max)
        acc_o *= scores_scale[:, :, :, None].transpose(1, 2)
        acc_s = torch.exp2(acc_s - scores_max[:, :, :, None])
        # print("acc_s:", acc_s)
        acc_s_cast = acc_s.to(torch.float16)
        acc_o += torch.einsum('bhqk,bkhd->bqhd', acc_s_cast, V[:, i * block_N : (i + 1) * block_N, :, :])
        scores_sum = acc_s.sum(dim=-1, keepdim=False)
        logsum = logsum * scores_scale + scores_sum
        # print("acc_o:", acc_o.size())
        # print("logsum:", logsum.size())
    acc_o /= logsum[:, :, :, None].transpose(1, 2)
    return acc_o.to(torch.float16)


num_split = 4
def flash_split(Q, K, V):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    acc_s = torch.empty((batch, nheads, block_M, block_N), device="cuda", dtype=torch.float)
    acc_s_cast = torch.empty((batch, nheads, block_M, block_N), device="cuda", dtype=torch.float16)
    acc_o = torch.empty((batch, block_M, nheads, dim), device="cuda", dtype=torch.float)
    scores_max = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    scores_max_prev = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    scores_scale = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    scores_sum = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    logsum = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    gacc_o = torch.empty((num_split, batch, block_M, nheads, dim), device="cuda", dtype=torch.float)
    glogsum = torch.empty((num_split, batch, nheads, block_M), device="cuda", dtype=torch.float)
    gscore_max = torch.empty((num_split, batch, nheads, block_M), device="cuda", dtype=torch.float)
    
    Q *= scale

    for ks in range(num_split):
        acc_o.fill_(0)
        logsum.fill_(0)
        scores_max.fill_(float('-inf'))
        scores_max_prev.fill_(float('-inf'))
        for i in range(int((seqlen_kv // num_split) / block_N)):
            acc_s.fill_(0)
            acc_s = torch.einsum('bqhd,bkhd->bhqk', Q, K[:, (seqlen_kv // num_split) * ks + i * block_N : (seqlen_kv // num_split) * ks + (i + 1) * block_N, :, :]) # [batch, seqlen, nheads, block_N]
            scores_max_prev = scores_max
            scores_max = acc_s.max(dim=-1, keepdim=False).values # [blockM]
            scores_scale = torch.exp2(scores_max_prev - scores_max)
            acc_o *= scores_scale[:, :, :, None].transpose(1, 2)
            acc_s = torch.exp2(acc_s - scores_max[:, :, :, None])
            acc_s_cast = acc_s.to(torch.float16)
            acc_o += torch.einsum('bhqk,bkhd->bqhd', acc_s_cast, V[:, (seqlen_kv // num_split) * ks + i * block_N : (seqlen_kv // num_split) * ks + (i + 1) * block_N, :, :])
            scores_sum = acc_s.sum(dim=-1, keepdim=False)
            logsum = logsum * scores_scale + scores_sum
        acc_o /= logsum[:, :, :, None].transpose(1, 2)
        logsum = torch.log2(logsum) + scores_max
        gacc_o[ks, :, :, :, :] = acc_o
        glogsum[ks, :, :, :] = logsum
        # gscore_max[ks, :, :, :] = scores_max


    # Reduce
    # fz = torch.empty((batch, block_M, nheads, dim), device="cuda", dtype=torch.float).fill_(0)
    # fm = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float).fill_(0)
    o = torch.empty((batch, block_M, nheads, dim), device="cuda", dtype=torch.float).fill_(0)
    lse_logsum = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float).fill_(0)

    lse_max = glogsum.max(dim=0, keepdim=False).values
    for ks in range(num_split):
        lse = glogsum[ks, :, :, :]
        lse_logsum += torch.exp2(lse - lse_max)
    lse_logsum = torch.log2(lse_logsum) + lse_max
    for ks in range(num_split):
        lse = glogsum[ks, :, :, :]
        scale = torch.exp2(lse - lse_logsum)
        o += gacc_o[ks, :, :, :, :] * scale[:, :, :, None].transpose(1, 2)
    return o.to(torch.float16)

if __name__ == "__main__":
    Q = torch.randn((batch, seqlen_q, nheads, dim), dtype=torch.float16, device="cuda")
    K = torch.randn((batch, seqlen_kv, nheads, dim), dtype=torch.float16, device="cuda")
    V = torch.randn((batch, seqlen_kv, nheads, dim), dtype=torch.float16, device="cuda")
    # Q = torch.ones((batch, seqlen_q, nheads, dim), dtype=torch.float16, device="cuda")
    # K = torch.ones((batch, seqlen_kv, nheads, dim), dtype=torch.float16, device="cuda")
    # V = torch.ones((batch, seqlen_kv, nheads, dim), dtype=torch.float16, device="cuda")

    ref = attention(Q, K, V)
    out = flash_split(Q, K, V)
    
    # print("ref:", ref)
    # print("out:", out)
    print(torch.allclose(ref, out, rtol=1e-2, atol=1e-2))