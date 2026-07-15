import torch
import sys
from tide.config import TiDEConfig
from tide.datasets.electricity_dataset import build_dataloaders
from tide.evaluation.evaluate import evaluate
from tide.inference.predict import load_model
from pathlib import Path

def main():
    cfg = TiDEConfig()
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"
    device = torch.device(device_str)

    print("Building dataloaders ...")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    print("Loading best model ...")
    model = load_model(cfg, device=device_str)

    # Redirect stdout to a file to capture evaluation output
    log_path = cfg.output_dir / "metrics.txt"
    print(f"Evaluating and saving print output to {log_path} ...")
    
    import io
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    metrics = evaluate(
        model=model,
        test_loader=test_loader,
        cfg=cfg,
        device=device,
        train_targets=train_loader.dataset.data[:, 0].cpu().numpy(),
    )
    
    sys.stdout = old_stdout
    output_text = new_stdout.getvalue()
    print(output_text)
    
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(output_text)

if __name__ == "__main__":
    main()
