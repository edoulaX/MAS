import os
import torch
import torchvision.transforms as T
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification, ViTFeatureExtractor
from diffusers import StableDiffusionImg2ImgPipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import albumentations as A
from glob import glob
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MaskGenerator:
    @staticmethod
    def apply_donut_mask(image, inner_ratio=0.25, outer_ratio=0.75):
        """Apply a donut-shaped mask to the image"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w//2, h//2), int(w * outer_ratio/2), 255, -1)
        cv2.circle(mask, (w//2, h//2), int(w * inner_ratio/2), 0, -1)
        return cv2.bitwise_and(image, image, mask=mask)
    
    @staticmethod
    def apply_center_mask(image, ratio=0.5):
        """Apply a center rectangular mask"""
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        center_h, center_w = int(h * ratio), int(w * ratio)
        start_h, start_w = (h - center_h)//2, (w - center_w)//2
        mask[start_h:start_h+center_h, start_w:start_w+center_w] = 0
        return cv2.bitwise_and(image, image, mask=mask)
    
    @staticmethod
    def apply_random_mask(image, num_patches=5):
        """Apply random rectangular masks"""
        h, w = image.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        for _ in range(num_patches):
            x1, y1 = np.random.randint(0, w-50), np.random.randint(0, h-50)
            patch_w, patch_h = np.random.randint(20, 50), np.random.randint(20, 50)
            mask[y1:y1+patch_h, x1:x1+patch_w] = 0
        return cv2.bitwise_and(image, image, mask=mask)

class AugmentedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[cls]))
        
        # Define augmentation pipeline
        self.aug_pipeline = A.Compose([
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.ISONoise(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.MotionBlur(p=0.5),
                A.GaussianBlur(p=0.5),
            ], p=0.5),
            A.ColorJitter(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ])
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply random masking
        img_np = np.array(image)
        mask_type = np.random.choice(['donut', 'center', 'random'])
        if mask_type == 'donut':
            img_np = MaskGenerator.apply_donut_mask(img_np)
        elif mask_type == 'center':
            img_np = MaskGenerator.apply_center_mask(img_np)
        else:
            img_np = MaskGenerator.apply_random_mask(img_np)
            
        # Apply additional augmentations
        augmented = self.aug_pipeline(image=img_np)
        img_np = augmented['image']
        
        if self.transform:
            image = self.transform(Image.fromarray(img_np))
            
        return image, label
    
    def __len__(self):
        return len(self.samples)

class NoveltyViT(torch.nn.Module):
    def __init__(self, num_classes, pretrained_model="google/vit-base-patch16-224-in21k"):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes
        )
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(pretrained_model)
        
        # Energy-based novelty detection head
        self.energy_threshold = None
        self.embedding_mean = None
        self.embedding_cov = None
        
    def forward(self, x):
        outputs = self.vit(x, output_hidden_states=True)
        logits = outputs.logits
        embeddings = outputs.hidden_states[-1][:, 0]  # CLS token
        return logits, embeddings
    
    def compute_energy(self, logits):
        return -torch.logsumexp(logits, dim=1)
    
    def mahalanobis_distance(self, embeddings):
        if self.embedding_mean is None or self.embedding_cov is None:
            return None
        diff = embeddings - self.embedding_mean
        inv_covmat = torch.linalg.inv(self.embedding_cov)
        return torch.sqrt(torch.sum(torch.matmul(diff, inv_covmat) * diff, dim=1))
    
    def calibrate_novelty_detection(self, dataloader, percentile=95):
        """Calibrate novelty detection thresholds using validation data"""
        energies = []
        embeddings_list = []
        
        self.eval()
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.cuda()
                logits, embeddings = self(x)
                energies.append(self.compute_energy(logits).cpu())
                embeddings_list.append(embeddings.cpu())
        
        energies = torch.cat(energies)
        embeddings = torch.cat(embeddings_list)
        
        # Set energy threshold
        self.energy_threshold = torch.quantile(energies, percentile/100)
        
        # Compute embedding statistics
        self.embedding_mean = embeddings.mean(0)
        self.embedding_cov = torch.cov(embeddings.T)
    
    def is_novel(self, x):
        """Determine if input is novel using multiple criteria"""
        self.eval()
        with torch.no_grad():
            logits, embeddings = self(x)
            energy = self.compute_energy(logits)
            mahalanobis_dist = self.mahalanobis_distance(embeddings)
            
            # Combine multiple novelty signals
            energy_novel = energy > self.energy_threshold
            conf_novel = torch.max(torch.softmax(logits, dim=1), dim=1)[0] < 0.5
            
            return energy_novel | conf_novel

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar.set_postfix(
            loss=total_loss/(batch_idx+1),
            acc=100.*correct/total
        )
    
    return total_loss/len(dataloader), 100.*correct/total

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss/len(dataloader), 100.*correct/total

def plot_embeddings(model, dataloader, method='tsne'):
    """Visualize embeddings using t-SNE or UMAP"""
    embeddings = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.cuda()
            _, emb = model(x)
            embeddings.append(emb.cpu().numpy())
            labels.extend(y.numpy())
    
    embeddings = np.concatenate(embeddings)
    
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    else:
        import umap
        reducer = umap.UMAP(random_state=42)
    
    embedding_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(f'Embeddings visualization using {method.upper()}')
    plt.show()

def plot_novelty_scores(model, known_loader, novel_loader):
    """Plot distribution of novelty scores"""
    known_scores = []
    novel_scores = []
    
    model.eval()
    with torch.no_grad():
        for x, _ in known_loader:
            x = x.cuda()
            logits, _ = model(x)
            energy = model.compute_energy(logits)
            known_scores.extend(energy.cpu().numpy())
        
        for x, _ in novel_loader:
            x = x.cuda()
            logits, _ = model(x)
            energy = model.compute_energy(logits)
            novel_scores.extend(energy.cpu().numpy())
    
    plt.figure(figsize=(10, 6))
    plt.hist(known_scores, bins=50, alpha=0.5, label='Known')
    plt.hist(novel_scores, bins=50, alpha=0.5, label='Novel')
    plt.axvline(model.energy_threshold.cpu().numpy(), color='r', linestyle='--', label='Threshold')
    plt.xlabel('Energy Score')
    plt.ylabel('Count')
    plt.legend()
    plt.title('Distribution of Novelty Scores')
    plt.show()

if __name__ == "__main__":
    # Configuration
    config = {
        'num_epochs': 30,
        'batch_size': 32,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'num_classes': 3,  # Update based on your dataset
    }

    # Data transforms
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize datasets and dataloaders
    train_dataset = AugmentedImageDataset('path/to/train', transform=transform)
    val_dataset = AugmentedImageDataset('path/to/val', transform=transform)
    novel_dataset = AugmentedImageDataset('path/to/novel', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    novel_loader = DataLoader(novel_dataset, batch_size=config['batch_size'])
    
    # Initialize model and training components
    model = NoveltyViT(num_classes=config['num_classes']).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs']
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0
    for epoch in range(config['num_epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{config["num_epochs"]}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Calibrate novelty detection
    model.calibrate_novelty_detection(val_loader)
    
    # Visualize results
    plot_embeddings(model, val_loader, method='tsne')
    plot_novelty_scores(model, val_loader, novel_loader) 