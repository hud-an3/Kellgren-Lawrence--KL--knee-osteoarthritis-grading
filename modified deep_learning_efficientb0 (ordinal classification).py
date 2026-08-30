import torch
import torch.nn as nn
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import os
from collections import Counter

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from eda import load_dataset, Train_data, Val_data, Test_data, preprocess_cnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, class_names = load_dataset(Train_data)
num_classes = len(class_names)


train_transform = transforms.Compose([
    preprocess_cnn,
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_test_transform = transforms.Compose([
    preprocess_cnn,
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


batch_size = 32

train_ds = datasets.ImageFolder(Train_data, transform=train_transform)
val_ds = datasets.ImageFolder(Val_data, transform=val_test_transform)
test_ds = datasets.ImageFolder(Test_data, transform=val_test_transform)

# Weighted sampler for class imbalance
class_counts = Counter(train_ds.targets)
class_weights = {c: 1.0 / class_counts[c] for c in class_counts}
sample_weights = [class_weights[t] for t in train_ds.targets]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# Freeze all feature extractor layers
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze last MBConv block
for param in model.features[-3].parameters():
    param.requires_grad = True

# Train classifier head
for param in model.classifier.parameters():
    param.requires_grad = True

in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, num_classes - 1)
)

model = model.to(device)


optimizer = optim.AdamW([
    {"params": model.features[-3:].parameters(), "lr": 1e-4},
    {"params": model.classifier.parameters(), "lr": 3e-4}
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5
)


# Corn loss

def corn_loss(logits, labels):
    """
    logits: (B, C-1)
    labels: (B,)
    """
    num_classes = logits.shape[1] + 1
    loss = 0.0
    for i in range(num_classes - 1):
        binary_labels = (labels > i).float()
        loss += F.binary_cross_entropy_with_logits(logits[:, i], binary_labels)
    return loss

best_val_acc = 0.0
patience_counter = 0
early_stop_patience = 6

for epoch in range(num_epochs):
    # Train
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = corn_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    y_val_true, y_val_pred = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = corn_loss(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = torch.sum(probs > 0.5, dim=1)

            y_val_true.extend(labels.cpu().numpy())
            y_val_pred.extend(preds.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(y_val_true, y_val_pred)
    scheduler.step(avg_val_loss)

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print("Early stopping triggered")
            break

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f}")


# Test

model.load_state_dict(torch.load("best_model.pth"))
model.eval()
y_true, y_pred = [], []
all_probs = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        probs = torch.sigmoid(outputs)
        preds = torch.sum(probs > 0.5, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        # Save full probability distribution
        batch_probs = torch.zeros((imgs.size(0), num_classes)).to(device)
        for k in range(num_classes):
            if k == 0:
                batch_probs[:, k] = 1 - probs[:, 0]
            elif k == num_classes - 1:
                batch_probs[:, k] = probs[:, -1]
            else:
                batch_probs[:, k] = probs[:, k-1] - probs[:, k]

        all_probs.append(batch_probs.cpu().numpy())

cnn_probs = np.vstack(all_probs)
np.save("cnn_probs.npy", cnn_probs)

print("TEST METRICS")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
print("Cohen Kappa:", cohen_kappa_score(y_true, y_pred))
