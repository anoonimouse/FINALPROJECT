import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.ao.quantization as quantization
import matplotlib.pyplot as plt
import os
import torchvision
import torchvision.transforms as transforms

from model import TinyMCUNet, InvertedResidual
from analysis import analyze_model_resources
from train import train_epoch


def get_test_loader():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transform_test)
    subset = torch.utils.data.Subset(testset, range(100))
    testloader = torch.utils.data.DataLoader(subset, batch_size=100, shuffle=False)
    return testloader


def evaluate_accuracy(model, data_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total


def fuse_mcunet(m):
    """Fuse Conv-BN-ReLU sequences throughout the network for quantization."""

    # initial conv block: features[0,1,2]
    quantization.fuse_modules(m, ['features.0', 'features.1', 'features.2'], inplace=True)

    # each inverted residual has a .conv Sequential with repeating Conv-BN-ReLU / Conv-BN patterns
    for name, module in m.named_modules():
        if isinstance(module, InvertedResidual):
            sub_names = [n for n, _ in module.conv.named_children()]
            i = 0
            while i < len(sub_names):
                # try Conv-BN-ReLU first
                if i + 2 < len(sub_names) and \
                   isinstance(module.conv[int(sub_names[i])], nn.Conv2d) and \
                   isinstance(module.conv[int(sub_names[i+1])], nn.BatchNorm2d) and \
                   isinstance(module.conv[int(sub_names[i+2])], nn.ReLU):
                    quantization.fuse_modules(module.conv,
                        [sub_names[i], sub_names[i+1], sub_names[i+2]], inplace=True)
                    i += 3
                # fallback to Conv-BN (the linear projection at the end)
                elif i + 1 < len(sub_names) and \
                     isinstance(module.conv[int(sub_names[i])], nn.Conv2d) and \
                     isinstance(module.conv[int(sub_names[i+1])], nn.BatchNorm2d):
                    quantization.fuse_modules(module.conv,
                        [sub_names[i], sub_names[i+1]], inplace=True)
                    i += 2
                else:
                    i += 1

    # final conv block
    last_idx = len(m.features) - 1
    quantization.fuse_modules(m.features,
        [str(last_idx-2), str(last_idx-1), str(last_idx)], inplace=True)


def run_experiments():
    device = 'cpu'
    print("Running experiments... this might take a minute.")

    testloader = get_test_loader()
    results = []

    # ---- Baseline ----
    base_model = TinyMCUNet(width_mult=1.0).to(device)
    if os.path.exists('best_mcunet.pth'):
        base_model.load_state_dict(torch.load('best_mcunet.pth', map_location=device), strict=False)

    # eval before profiling -- thop hooks can mess with state sometimes
    base_acc = evaluate_accuracy(base_model, testloader, device=device)
    base_metrics = analyze_model_resources(base_model, device=device)

    results.append({
        'Model': "Baseline",
        'Params (K)': base_metrics['Total Params (K)'],
        'MACs (M)': base_metrics['MACs (M)'],
        'Size (KB)': base_metrics['Size (KB)'],
        'Latency (ms)': base_metrics['Latency (ms)'],
        'Accuracy (%)': base_acc
    })

    # ---- Post-training quantization (INT8) ----
    print("\n--- Quantization ---")

    engines = torch.backends.quantized.supported_engines
    engine = 'onednn' if 'onednn' in engines else ('fbgemm' if 'fbgemm' in engines else 'none')
    if engine != 'none':
        torch.backends.quantized.engine = engine

    q_model = TinyMCUNet(width_mult=1.0)
    if os.path.exists('best_mcunet.pth'):
        q_model.load_state_dict(torch.load('best_mcunet.pth', map_location='cpu'), strict=False)

    q_model.eval()
    fuse_mcunet(q_model)

    if engine != 'none':
        q_model.qconfig = quantization.get_default_qconfig(engine)
    else:
        q_model.qconfig = quantization.default_qconfig

    # classifier stays float (it runs after dequant)
    q_model.classifier.qconfig = None

    quantization.prepare(q_model, inplace=True)
    for inputs, _ in testloader:
        q_model(inputs)  # calibration
    quantization.convert(q_model, inplace=True)

    temp_file = "temp_qmodel.pt"
    torch.save(q_model.state_dict(), temp_file)
    q_size = os.path.getsize(temp_file) / 1024
    os.remove(temp_file)
    q_acc = evaluate_accuracy(q_model, testloader, device='cpu')

    print(f"Quantized: {q_size:.0f} KB | {q_acc:.1f}%")
    results.append({
        'Model': "Quantized (INT8)",
        'Params (K)': base_metrics['Total Params (K)'] / 4.0,
        'MACs (M)': base_metrics['MACs (M)'],
        'Size (KB)': q_size,
        'Latency (ms)': "N/A",
        'Accuracy (%)': q_acc
    })

    # ---- Pruning ----
    print("\n--- Pruning with Fine-tuning ---")

    def prune_and_finetune(sparsity):
        model = TinyMCUNet(width_mult=1.0).to(device)
        if os.path.exists('best_mcunet.pth'):
            model.load_state_dict(torch.load('best_mcunet.pth', map_location=device), strict=False)

        # Apply pruning
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
                prune.remove(module, 'weight')

        # 5 epochs of fine-tuning to recover accuracy
        # 2000 images is about 20% of the training set, enough for a quick profile
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=False, transform=train_transform)
        train_subset = torch.utils.data.Subset(trainset, range(2000)) 
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)

        print(f"  Fine-tuning {int(sparsity*100)}% pruned model for recovery...")
        for epoch in range(5):
            loss, acc = train_epoch(model, trainloader, criterion, optimizer, device)
            # print(f"    Epoch {epoch+1} FT Acc: {acc:.2f}%") # quiet during profiling

        metrics = analyze_model_resources(model, device=device)
        acc = evaluate_accuracy(model, testloader, device=device)
        print(f"  Recovered Acc: {acc:.2f}%")
        return metrics, acc

    for pct in [0.15, 0.30]:
        p_metrics, p_acc = prune_and_finetune(pct)
        label = f"Pruned {int(pct*100)}%"
        results.append({
            'Model': label,
            'Params (K)': p_metrics['Total Params (K)'],
            'MACs (M)': p_metrics['MACs (M)'],
            'Size (KB)': p_metrics['Size (KB)'],
            'Latency (ms)': p_metrics['Latency (ms)'],
            'Accuracy (%)': p_acc
        })
        print(f"{label} | Acc: {p_acc:.1f}%")

    # ---- Width scaling (no trained weights, just resource profiles) ----
    print("\n--- Width Scaling ---")
    for wm in [0.5, 0.75]:
        w_model = TinyMCUNet(width_mult=wm).to(device)
        w_metrics = analyze_model_resources(w_model, device=device)
        results.append({
            'Model': f"Width {wm}x",
            'Params (K)': w_metrics['Total Params (K)'],
            'MACs (M)': w_metrics['MACs (M)'],
            'Size (KB)': w_metrics['Size (KB)'],
            'Latency (ms)': w_metrics['Latency (ms)'],
            'Accuracy (%)': "N/A"
        })

    # ---- Results table ----
    print("\n" + "="*85)
    print("Results Comparison")
    print("="*85)
    header = f"{'Model':<20} | {'Params (K)':<12} | {'MACs (M)':<10} | {'Size (KB)':<10} | {'Acc (%)':<8} | {'Lat (ms)':<8}"
    print(header)
    print("-" * 85)

    for r in results:
        params = f"{r['Params (K)']:.2f}" if isinstance(r['Params (K)'], float) else str(r['Params (K)'])
        macs = f"{r['MACs (M)']:.2f}" if isinstance(r['MACs (M)'], float) else str(r['MACs (M)'])
        size = f"{r['Size (KB)']:.2f}" if isinstance(r['Size (KB)'], float) else str(r['Size (KB)'])
        acc = f"{r['Accuracy (%)']:.2f}" if isinstance(r['Accuracy (%)'], float) else str(r['Accuracy (%)'])
        lat = f"{r['Latency (ms)']:.2f}" if isinstance(r['Latency (ms)'], float) else str(r['Latency (ms)'])
        print(f"{r['Model']:<20} | {params:<12} | {macs:<10} | {size:<10} | {acc:<8} | {lat:<8}")
    print("="*85)

    # ---- Plots ----
    print("\n--- Generating plots ---")

    models = [r['Model'] for r in results]
    sizes = [r['Size (KB)'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.bar(models, sizes, color='skyblue')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel('Model Size (KB)')
    plt.title('Model Size Comparison')
    plt.tight_layout()
    plt.savefig('plot_size.png')
    plt.close()

    macs_list, acc_list, labels = [], [], []
    for r in results:
        if isinstance(r['MACs (M)'], float) and isinstance(r['Accuracy (%)'], float):
            macs_list.append(r['MACs (M)'])
            acc_list.append(r['Accuracy (%)'])
            labels.append(r['Model'])

    plt.figure(figsize=(8, 6))
    plt.scatter(macs_list, acc_list, s=100, color='coral')
    for i, label in enumerate(labels):
        plt.annotate(label, (macs_list[i], acc_list[i]), xytext=(5, 5), textcoords='offset points')
    plt.xlabel('MACs (M)')
    plt.ylabel('Accuracy (%)')
    plt.title('MACs vs. Accuracy Trade-off')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('plot_tradeoff.png')
    plt.close()

    print("Saved plot_size.png and plot_tradeoff.png")


if __name__ == '__main__':
    try:
        run_experiments()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
