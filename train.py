import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.models import resnet18
import os
import copy
import time
from sklearn.metrics import f1_score

from helpers import get_device
from losses import relu_evidence, edl_mse_loss_with_prospect

os.makedirs('checkpoints', exist_ok=True)


def is_custom_loss(fn):
    return callable(fn) and fn.__name__ == "edl_mse_loss_with_prospect"


def train_model(
    model,
    dataloaders,
    num_classes,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=50,
    device=None,
    uncertainty=True,
    source_distribution=None,
):
    print(">>> train_model started <<<")
    since = time.time()

    if not device:
        device = get_device()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = float('-inf')
    model_saved = False

    losses = {"loss": [], "phase": [], "epoch": []}
    accuracy = {"accuracy": [], "phase": [], "epoch": []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            all_preds = []
            all_labels = []

            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    if uncertainty and is_custom_loss(criterion):
                        loss, normalized_edl_loss, normalized_certainty_penalty = criterion(
                            outputs, labels, epoch, num_classes, 10, device,
                            source_distribution, lambda_certainty=0.5
                        )
                    else:
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    all_preds.extend(preds.detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())
                    running_loss += loss.item() * inputs.size(0)

            if scheduler is not None and phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = f1_score(all_labels, all_preds, average='macro')

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | F1-score: {epoch_acc:.4f}")

            losses["loss"].append(epoch_loss)
            losses["phase"].append(phase)
            losses["epoch"].append(epoch)
            accuracy["accuracy"].append(epoch_acc)
            accuracy["epoch"].append(epoch)
            accuracy["phase"].append(phase)

            if phase == "val" and epoch_acc > best_acc:
                print(f"✅ New best model found at epoch {epoch+1}")
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                model_saved = True

                try:
                    save_path = os.path.abspath("checkpoints/model_cifar10_best.pth")
                    torch.save(best_model_wts, save_path)
                    print(f"✅ Best model saved to {save_path}")
                except Exception as e:
                    print(f"⛔ Failed to save best model: {e}")

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val F1-score: {best_acc:.4f}")

    if not model_saved:
        print("⚠️ No improvement during training — best model was not saved.")

    try:
        final_save_path = os.path.abspath("checkpoints/model_cifar10_final.pth")
        torch.save(model.state_dict(), final_save_path)
        print(f"✅ Final model saved to {final_save_path}")
    except Exception as e:
        print(f"⛔ Failed to save final model: {e}")

    model.load_state_dict(best_model_wts)
    metrics = (losses, accuracy)
    return model, metrics


def main():
    print(">>> main started <<<")

    device = get_device()
    print(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261))
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    dataloaders = {"train": train_loader, "val": val_loader}

    num_classes = 10

    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = edl_mse_loss_with_prospect  # یا torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    trained_model, metrics = train_model(
        model=model,
        dataloaders=dataloaders,
        num_classes=num_classes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=100,
        device=device,
        uncertainty=True,
        source_distribution=None,
    )

    print("Training finished!")


if __name__ == "__main__":
    main()
