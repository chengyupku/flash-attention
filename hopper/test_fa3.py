import torch
from flash_attn_interface import flash_attn_func
from flash_attn_interface import flash_attn_with_kvcache

dtype = torch.float16
device = torch.device("cuda")

def run(batch, seqlen_q, seqlen_kv, nheads, headdim, casual):
    Q = torch.randn((batch, seqlen_q, nheads, headdim), dtype=dtype, device=device)
    K = torch.randn((batch, seqlen_kv, nheads, headdim), dtype=dtype, device=device)
    V = torch.randn((batch, seqlen_kv, nheads, headdim), dtype=dtype, device=device)

    warmups = 10
    iterations = 10
    for _ in range(warmups):
        out = flash_attn_with_kvcache(Q, K, V, causal=casual)
        torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    for _ in range(iterations):
        start_event.record()
        out = flash_attn_with_kvcache(Q, K, V, causal=casual)
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

def run_once(batch, seqlen_q, seqlen_kv, nheads, headdim, casual):
    Q = torch.ones((batch, seqlen_q, nheads, headdim), dtype=dtype, device=device)
    K = torch.ones((batch, seqlen_kv, nheads, headdim), dtype=dtype, device=device)
    V = torch.ones((batch, seqlen_kv, nheads, headdim), dtype=dtype, device=device)
    out = flash_attn_with_kvcache(Q, K, V, causal=casual)
    print("out:", out)


if __name__ == "__main__":

    configs = [
        {'model': 'Llama3-8b', 'h': 32, 'n_ctx': 8192, 'd_head': 128},
        # {'model': 'Llama3-70b', 'h': 64, 'n_ctx': 8192, 'd_head': 128},
    ]

    batchs = [1]
    seqlen_qs = [1, 128]

    for batch in batchs:
        for seqlen_q in seqlen_qs:
            for config in configs:
                print(f"Running {config['model']} with batch={batch}, seqlen_q={seqlen_q}")
                tflops, avg_time = run(batch, seqlen_q, config['n_ctx'], config['h'], config['d_head'], casual=False)
                print(f"TFLOPS: {tflops:.2f}, avg_time: {avg_time:.4f} ms")

    # batch, seqlen_q, seqlen_kv, nheads, head_dim = 1, 4096, 4096, 32, 64
    # run_once(1, 4096, 4096, 1, 128, False)
    # tflops, avg_time = run(batch, seqlen_q, seqlen_kv, nheads, head_dim, casual=False)
    # print(f"TFLOPS: {tflops:.2f}, avg_time: {avg_time:.2f} ms")