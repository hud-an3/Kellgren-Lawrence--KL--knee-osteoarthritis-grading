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
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    preprocess_cnn,
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


batch_size = 32

train_ds = datasets.ImageFolder(Train_data, transform=train_transform)
val_ds   = datasets.ImageFolder(Val_data, transform=val_test_transform)
test_ds  = datasets.ImageFolder(Test_data, transform=val_test_transform)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


train_labels = train_ds.targets
counts = Counter(train_labels)


criterion = nn.CrossEntropyLoss()

class_counts = Counter(train_ds.targets)
class_weights = {c: 1.0 / class_counts[c] for c in class_counts}
sample_weights = [class_weights[t] for t in train_ds.targets]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    sampler=sampler
)


model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
)



for param in model.features.parameters():
    param.requires_grad = False


for param in model.features[-3].parameters():
    param.requires_grad = True


for param in model.classifier.parameters():
    param.requires_grad = True




in_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, num_classes)
)


model = model.to(device)




optimizer = optim.AdamW([
    {"params": model.features[-3:].parameters(), "lr": 1e-4},
    {"params": model.classifier.parameters(), "lr": 3e-4}
], weight_decay=1e-4)


scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5
)



num_epochs = 10
best_val_acc = 0.0 
patience_counter = 0
early_stop_patience = 6

for epoch in range(num_epochs):
    
  
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

   
    model.eval()
    val_loss = 0.0
    y_val_true, y_val_pred = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
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



model.load_state_dict(torch.load("best_model.pth"))
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("TEST METRICS")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
print("Cohen Kappa:", cohen_kappa_score(y_true, y_pred))


def generate_gradcam(model, input_tensor, target_class):
    model.eval()
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.features[-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    output[0, target_class].backward()

    # Global Average Pooling on gradients
    grads = gradients[0].mean(dim=(2, 3), keepdim=True)

    cam = (grads * activations[0]).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    # Resize CAM to input size (224×224)
    cam = F.interpolate(
        cam,
        size=input_tensor.shape[2:],
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze().detach().cpu().numpy()
    cam = cam / (cam.max() + 1e-8)

    fh.remove()
    bh.remove()

    return cam
