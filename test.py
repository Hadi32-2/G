import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from helpers import get_device
from resnet import ResNet18
from data import dataloaders  # uses CIFAR-10

def test_model_on_cifar10():
    device = get_device()
    model = ResNet18(num_classes=10, dropout=False).to(device)

    # بارگذاری checkpoint فقط یکبار کافی است و بهتر است با map_location=device بارگذاری کنیم
    checkpoint_path = "./results/model_cifar10_best.pth"
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)

    
    model.eval()  # حالت ارزیابی مدل

    val_loader = dataloaders["val"]

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"✅ CIFAR-10 Test F1-score (Raw model): {f1:.4f}")

    # نمایش چند پیش‌بینی نمونه
    num_samples = 8
    sample_inputs, sample_labels = next(iter(val_loader))
    sample_inputs, sample_labels = sample_inputs.to(device), sample_labels.to(device)

    with torch.no_grad():
        sample_outputs = model(sample_inputs)
        sample_preds = torch.argmax(sample_outputs, dim=1)

    data = {
        "True Label": sample_labels[:num_samples].cpu().numpy(),
        "Predicted Label": sample_preds[:num_samples].cpu().numpy(),
    }
    df = pd.DataFrame(data)
    print("\n📊 Sample Prediction Table (Raw):")
    print(df)

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis("off")
    ax.table(cellText=df.values,
             colLabels=df.columns,
             cellLoc='center',
             loc='center',
             colColours=["#cccccc"] * df.shape[1])
    plt.title("Prediction Table", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig("raw_cifar10_predictions.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    test_model_on_cifar10()
