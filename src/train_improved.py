"""
Improved TinyCNN training for Kaggle submission.
Run this in a NEW Colab cell BEFORE the Kaggle submission cell (Section 3.2).
Requires GPU runtime. Training takes ~15-20 minutes total.
"""
import os, copy, time, json
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── 1. Dataset helpers (same as notebook) ──────────────────────────
class TinyImageNet30Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class TinyImageNet30TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        if os.path.isdir(os.path.join(test_dir, 'test_set')):
            self.test_dir = os.path.join(test_dir, 'test_set')
        self.files = sorted([f for f in os.listdir(self.test_dir) if f.lower().endswith('.jpeg')])
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.test_dir, fname)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, fname

def load_dataset_from_folder(data_dir):
    paths, labels = [], []
    class_to_idx = {}
    for i, cls in enumerate(sorted(os.listdir(data_dir))):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        class_to_idx[cls] = i
        for f in sorted(os.listdir(cls_dir)):
            if f.lower().endswith(('.jpeg', '.jpg', '.png')):
                paths.append(os.path.join(cls_dir, f))
                labels.append(i)
    return paths, labels, class_to_idx

# ── 2. Improved TinyCNN (wider, same 5-layer structure) ───────────
class TinyCNN(nn.Module):
    """5-conv CNN for TinyImageNet30 — wider channels for better capacity."""
    def __init__(self, num_classes=30, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),               # 64 -> 32

            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),               # 32 -> 16

            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Layer 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),               # 16 -> 8

            # Layer 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),               # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── 3. CutMix helper ──────────────────────────────────────────────
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    # Random bounding box
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, x1:x2, y1:y2] = x[index, :, x1:x2, y1:y2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, y, y[index], lam


# ── 4. Training function with scheduler + CutMix ──────────────────
def train_model_improved(model, train_loader, val_loader, epochs=80, lr=1e-3,
                         weight_decay=5e-4, label_smooth=0.1, cutmix_prob=0.5,
                         save_path='best_improved.pth'):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_w = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # Train
        model.train()
        rl, rc, rt = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # CutMix with probability
            if np.random.rand() < cutmix_prob:
                images, targets_a, targets_b, lam = cutmix_data(images, labels)
                outputs = model(images)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            rl += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            rc += (preds == labels).sum().item()
            rt += labels.size(0)

        scheduler.step()
        train_loss = rl / rt
        train_acc = 100.0 * rc / rt

        # Val
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        val_criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = val_criterion(outputs, labels)
                vl += loss.item() * images.size(0)
                vc += (outputs.argmax(1) == labels).sum().item()
                vt += labels.size(0)
        val_loss = vl / vt
        val_acc = 100.0 * vc / vt

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_w = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch+1:3d}/{epochs} | '
                  f'train {train_acc:.1f}% | val {val_acc:.1f}% | '
                  f'best {best_val_acc:.1f}% | lr={lr_now:.6f}')

    model.load_state_dict(best_w)
    torch.save(best_w, save_path)
    print(f'\nBest val acc: {best_val_acc:.2f}% -> saved to {save_path}')
    return history


# ── 5. Data loading ────────────────────────────────────────────────
DATA_DIR = './train_set'
all_paths, all_labels, class_to_idx = load_dataset_from_folder(DATA_DIR)
NUM_CLASSES = len(class_to_idx)
print(f'Found {len(all_paths)} images, {NUM_CLASSES} classes')

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths, all_labels, test_size=0.2, random_state=0, stratify=all_labels
)

# Strong augmentation for training
train_transform_strong = transforms.Compose([
    transforms.Resize((72, 72)),                        # slight up-size
    transforms.RandomCrop(64),                          # then crop back to 64
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
    transforms.RandomGrayscale(0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
])

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

BATCH_SIZE = 128
NUM_WORKERS = 2

train_dataset = TinyImageNet30Dataset(train_paths, train_labels, train_transform_strong)
val_dataset = TinyImageNet30Dataset(val_paths, val_labels, val_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')

# ── 6. Train 3 models with different seeds ─────────────────────────
EPOCHS = 80
SEEDS = [0, 42, 123]
model_paths = []

for i, seed in enumerate(SEEDS):
    print(f'\n{"="*60}')
    print(f'Training model {i+1}/{len(SEEDS)} (seed={seed})')
    print(f'{"="*60}')
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TinyCNN(num_classes=NUM_CLASSES, dropout=0.3).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Params: {params:,}')

    save_path = f'best_improved_seed{seed}.pth'
    history = train_model_improved(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=1e-3, weight_decay=5e-4,
        label_smooth=0.1, cutmix_prob=0.5,
        save_path=save_path
    )
    model_paths.append(save_path)

    # Save history
    with open(f'history_improved_seed{seed}.json', 'w') as f:
        json.dump(history, f)

# ── 7. Ensemble + TTA prediction ──────────────────────────────────
print(f'\n{"="*60}')
print('Generating Kaggle submission with ensemble + TTA')
print(f'{"="*60}')

# Load all trained models
models = []
for path in model_paths:
    m = TinyCNN(num_classes=NUM_CLASSES, dropout=0.0).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    models.append(m)
    print(f'Loaded {path}')

# TTA transforms: original + horizontal flip + 5-crop
tta_transforms = [
    # Original
    transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # Horizontal flip
    transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # Center crop from larger
    transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # Center crop + flip
    transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.CenterCrop(64),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]

# Collect logits: N_models x N_tta views
all_logits = []
fnames_list = None
for m in models:
    for tta_t in tta_transforms:
        test_ds = TinyImageNet30TestDataset('./test_set', tta_t)
        test_dl = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
        logits_list, names = [], []
        with torch.no_grad():
            for images, fnames in test_dl:
                images = images.to(device)
                logits_list.append(m(images))
                names.extend(fnames)
        all_logits.append(torch.cat(logits_list))
        if fnames_list is None:
            fnames_list = names

print(f'Total forward passes: {len(all_logits)} ({len(models)} models x {len(tta_transforms)} TTA)')

# Average logits and predict
avg_logits = sum(all_logits) / len(all_logits)
preds = avg_logits.argmax(1).cpu().numpy().tolist()

# Validation accuracy of ensemble
val_logits_all = []
for m in models:
    vl = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            vl.append(m(images))
    val_logits_all.append(torch.cat(vl))
val_avg = sum(val_logits_all) / len(val_logits_all)
val_labels_all = []
for images, labels in val_loader:
    val_labels_all.append(labels)
val_labels_t = torch.cat(val_labels_all).to(device)
val_acc = (val_avg.argmax(1) == val_labels_t).float().mean().item() * 100
print(f'Ensemble validation accuracy: {val_acc:.2f}%')

# Save submission
submission = pd.DataFrame({'Id': fnames_list, 'Category': preds})
submission.to_csv('sc21xl2.csv', index=False)
print(f'Saved sc21xl2.csv ({len(submission)} rows)')
print('Done! Upload sc21xl2.csv to Kaggle.')
