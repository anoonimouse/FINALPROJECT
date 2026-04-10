import os
import time
import torch
from thop import profile


def analyze_model_resources(model, input_size=(1, 3, 32, 32), device='cpu'):
    """Profile a model's params, MACs, disk size, memory, and latency."""
    results = {}

    model = model.to(device)
    dummy_input = torch.randn(*input_size).to(device)

    # param counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results['Total Params (K)'] = total_params / 1000
    results['Trainable Params (K)'] = trainable_params / 1000

    # MAC count via thop
    macs, _ = profile(model, inputs=(dummy_input, ), verbose=False)
    results['MACs (M)'] = macs / 1e6

    # serialized size on disk
    temp_file = "temp_model_size.pt"
    torch.save(model.state_dict(), temp_file)
    size_kb = os.path.getsize(temp_file) / 1024
    if os.path.exists(temp_file):
        os.remove(temp_file)
    results['Size (KB)'] = size_kb

    # peak memory estimate
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        _ = model(dummy_input)
        peak_mem_kb = torch.cuda.max_memory_allocated() / 1024
    else:
        # rough cpu estimate: params + activations, not exact but ballpark
        peak_mem_kb = (dummy_input.element_size() * dummy_input.nelement() + total_params * 4) / 1024 * 2
    results['Peak Mem (KB)'] = peak_mem_kb

    # latency (avg over 100 runs after warmup)
    for _ in range(10):
        _ = model(dummy_input)

    if device == 'cuda':
        torch.cuda.synchronize()

    start_time = time.time()
    num_runs = 100
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    if device == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()
    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000
    results['Latency (ms)'] = avg_latency_ms

    return results


def print_analysis_table(results, model_name="TinyMCUNet"):
    print("-" * 50)
    print(f"Resource Analysis: {model_name}")
    print("-" * 50)
    for k, v in results.items():
        print(f"{k:25s} : {v:.2f}")
    print("-" * 50)


if __name__ == '__main__':
    from model import TinyMCUNet

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}...")

    net = TinyMCUNet(width_mult=1.0)
    net.eval()

    metrics = analyze_model_resources(net, device=device)
    print_analysis_table(metrics)
