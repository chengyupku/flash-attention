import torch
from flash_attn_interface import flash_attn_func
from flash_attn_interface import flash_attn_with_kvcache

dtype = torch.float16
device = torch.device("cuda")

def run(mode, batch, seqlen_q, seqlen_kv, nheads, headdim, casual):
    if mode == "prefill":
        func = flash_attn_func
    elif mode == "decode":
        func = flash_attn_with_kvcache
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    Q = torch.randn((batch, seqlen_q, nheads, headdim), dtype=dtype, device=device)
    K = torch.randn((batch, seqlen_kv, nheads, headdim), dtype=dtype, device=device)
    V = torch.randn((batch, seqlen_kv, nheads, headdim), dtype=dtype, device=device)

    warmups = 10
    iterations = 10
    for _ in range(warmups):
        out = func(Q, K, V, causal=casual)
        torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    for _ in range(iterations):
        start_event.record()
        out = func(Q, K, V, causal=casual)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        total_time += elapsed_time

    avg_time = total_time / iterations

    flops_per_matmul = 2.0 * batch * nheads * seqlen_q * seqlen_kv * headdim
    total_flops = 2 * flops_per_matmul
    if casual:
        total_flops *= 0.5
    tflops = total_flops / (avg_time * 1e-3) / 1e12
    return tflops, avg_time

def run_once(mode, batch, seqlen_q, seqlen_kv, nheads, headdim, casual):
    if mode == "prefill":
        func = flash_attn_func
    elif mode == "decode":
        func = flash_attn_with_kvcache
    else:
        raise ValueError(f"Unknown mode: {mode}")
    Q = torch.ones((batch, seqlen_q, nheads, headdim), dtype=dtype, device=device)
    K = torch.ones((batch, seqlen_kv, nheads, headdim), dtype=dtype, device=device)
    V = torch.ones((batch, seqlen_kv, nheads, headdim), dtype=dtype, device=device)
    out = func(Q, K, V, causal=casual)
    print("out:", out)


if __name__ == "__main__":

    configs = [
        {'model': 'GPT2', 'h': 12, 'n_ctx': 1024, 'd_head': 64},
        {'model': 'BERT-small', 'h': 8, 'n_ctx': 512, 'd_head': 64},
        {'model': 'BERT-base', 'h': 12, 'n_ctx': 512, 'd_head': 64},
        {'model': 'BERT-large', 'h': 16, 'n_ctx': 512, 'd_head': 64},
        {'model': 'DiT-S-2', 'h': 6, 'n_ctx': 1024, 'd_head': 64},
        {'model': 'DiT-B-2', 'h': 12, 'n_ctx': 1024, 'd_head': 64},
        {'model': 'OPT-350M', 'h': 16, 'n_ctx': 2048, 'd_head': 64},
        {'model': 'OPT-13B', 'h': 40, 'n_ctx': 2048, 'd_head': 128},
        {'model': 'OPT-175B', 'h': 96, 'n_ctx': 2048, 'd_head': 128},
        {'model': 'Llama3-8b', 'h': 32, 'n_ctx': 4096, 'd_head': 128},
        {'model': 'Llama3-70b', 'h': 64, 'n_ctx': 4096, 'd_head': 128},
        {'model': 'Llama3-8b', 'h': 32, 'n_ctx': 8192, 'd_head': 128},
        {'model': 'Llama3-70b', 'h': 64, 'n_ctx': 8192, 'd_head': 128},
    ]

    batchs = [1, 64]
    seqlen_qs = [1, 128]

    for casual in [True, False]:
        for batch in batchs:
            for config in configs:
                # print(f"Running {config['model']} with batch={batch}, seqlen_q={config['n_ctx']}")
                tflops, avg_time = run("prefill", batch, config['n_ctx'], config['n_ctx'], config['h'], config['d_head'], casual=casual)
                print(f"TFLOPS: {tflops:.2f}, avg_time: {avg_time:.4f} ms")

    # batch, seqlen_q, seqlen_kv, nheads, head_dim = 1, 4096, 4096, 32, 64
    # run_once(1, 4096, 4096, 1, 128, False)
    # tflops, avg_time = run(batch, seqlen_q, seqlen_kv, nheads, head_dim, casual=False)
    # print(f"TFLOPS: {tflops:.2f}, avg_time: {avg_time:.2f} ms")