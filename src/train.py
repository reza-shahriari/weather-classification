import torch
from torch.utils.data import DataLoader
from dataloader import WeatherDataset
from model import WeatherClassifier
import torch.optim as optim
import torchvision
import torch.nn as nn
import cv2
import sys
import os
from pathlib import Path
ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)
from utils.Augment import WeatherAug

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datetime import datetime
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import math

from multiprocessing import Pool
from functools import partial

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def calculate_metrics(outputs, labels):
    _, predicted = outputs.max(1)
    accuracy = predicted.eq(labels).sum().item() / labels.size(0)
    f1 = f1_score(labels.cpu(), predicted.cpu(), average='weighted')
    cm = confusion_matrix(labels.cpu(), predicted.cpu())
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def visualize_augmentations(dataset, num_examples=5):
    fig, axes = plt.subplots(num_examples, 2, figsize=(10, 4*num_examples))
    for i in range(num_examples):
        original_img = dataset.get_original_image(i)
        augmented_img = dataset[i][0].permute(1,2,0).numpy()
        axes[i,0].imshow(original_img)
        axes[i,0].set_title('Original')
        axes[i,1].imshow(augmented_img)
        axes[i,1].set_title('Augmented')
    plt.tight_layout()
    return fig   

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)

def find_lr(model, train_loader, optimizer, criterion, device,
            init_lr=1e-7, final_lr=1.0, beta=0.98):
    num = len(train_loader) - 1
    mult = (final_lr / init_lr) ** (1/num)
    lr = init_lr
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    losses = []
    log_lrs = []
    
    model.train()
    for batch_num, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"NaN detected at lr={lr}")
            break
            
        # Update smoothed loss
        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**(batch_num + 1))
        
        # Store values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        
        # Stop if loss is exploding
        if batch_num > 0 and smoothed_loss > 4 * best_loss:
            break
            
        if smoothed_loss < best_loss or batch_num == 0:
            best_loss = smoothed_loss
            
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
        
        if lr > final_lr:
            break
    
    return log_lrs[:-1], losses[:-1]  # Remove the last point where loss exploded

prep_aug = WeatherAug(hyp='cfg/augment_params.yaml', realtime=False)

def process_single_image(img_path):
    img = cv2.imread(str(img_path))
    img, transforms = prep_aug(img)
    # Ensure image is in valid range before saving
    img = np.clip(img, 0, 255).astype(np.uint8)  # Add clipping and convert to uint8
    cv2.imwrite(str(img_path), img)
    return transforms

def preprocess_dataset(data_path, hyp, num_workers=4):
    img_files = []
    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        img_files.extend([os.path.join(class_path, f) for f in os.listdir(class_path)])
    
    with Pool(num_workers) as pool:
        transforms = pool.map(process_single_image, img_files)
        pool.close()
        pool.join()
def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig
def visualize_batch(images, predictions, labels, class_names, writer, epoch, prefix="Training"):
    # Select up to 8 random images from the batch
    n = min(8, images.shape[0])
    idx = torch.randperm(images.shape[0])[:n]
    images = images[idx]
    predictions = predictions[idx]
    labels = labels[idx]
    
    # Create a grid of images
    grid = torchvision.utils.make_grid(images, nrow=4, normalize=True)
    
    # Add text annotations for predictions and ground truth
    fig = plt.figure(figsize=(15, 8))
    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.axis('off')
    
    for i in range(n):
        plt.text(i % 4 * images.shape[3] + 5, i // 4 * images.shape[2] + 15, 
                f'Pred: {class_names[predictions[i]]}\nTrue: {class_names[labels[i]]}',
                color='white', backgroundcolor='black')
    
    writer.add_figure(f'{prefix}/Predictions', fig, epoch)

def log_sample_predictions(model, val_loader, writer, epoch, device, class_names):
    model.eval()
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predictions = outputs.max(1)
    
    visualize_batch(images, predictions, labels, class_names, writer, epoch, "Validation")

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    
    val_pbar = tqdm(val_loader, desc='Validation')
    with torch.no_grad():
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            
            # Update progress bar with current loss
            val_pbar.set_postfix({'loss': val_loss/len(val_loader)})
    
    val_loss = val_loss / len(val_loader)
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    val_cm = confusion_matrix(val_labels, val_preds)
    
    return val_loss, val_f1, val_cm

def plot_lr_finder(log_lrs, losses):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(log_lrs[10:-5], losses[10:-5])
    ax.set_xlabel('Learning Rate (log scale)')
    ax.set_ylabel('Loss')
    ax.set_xscale('log')
    ax.set_title('Learning Rate Finder')
    ax.grid(True)
    return fig

def train_model(data_dir, hyp='cfg/augment_params.yaml', num_workers=8, 
                num_epochs=10, batch_size=32, learning_rate=0.0001,test_frequency=5):
    run_name = f"weather_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f'runs/{run_name}')
    
    train_path = os.path.join(data_dir, 'train')
    preprocess_dataset(train_path, hyp, num_workers)  # Ensure this doesn't normalize
    
    train_dataset = WeatherDataset(os.path.join(data_dir, 'train'), hyp=hyp, augment=True)
    val_dataset = WeatherDataset(os.path.join(data_dir, 'val'), hyp='', augment=False)
    test_dataset = WeatherDataset(os.path.join(data_dir, 'test'), hyp='', augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader =DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    early_stopping = EarlyStopping(patience=7)
    scaler = GradScaler()
    
    model = WeatherClassifier(num_classes=len(train_dataset.classes))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Save initial state before LR finder
    initial_state = model.state_dict().copy()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    


    # Restore initial model state and optimizer
    model.load_state_dict(initial_state)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    best_val_loss = float('inf')
    model_save_dir = f'models/{run_name}'
    os.makedirs(model_save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for step, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            train_pbar.set_postfix({'loss': train_loss/(step+1)})
        
         # Log training batch
        
        
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, predictions = outputs.max(1)
        visualize_batch(images, predictions, labels, train_dataset.classes, writer, epoch)
        
        # Log validation batch
        log_sample_predictions(model, val_loader, writer, epoch, device, train_dataset.classes)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        train_cm = confusion_matrix(train_labels, train_preds)
        writer.add_scalar('Training/Loss', train_loss/len(train_loader), epoch)
        writer.add_scalar('Training/F1_Score', train_f1, epoch)
        writer.add_figure('Training/Confusion_Matrix', plot_confusion_matrix(train_cm, train_dataset.classes), epoch)
        
        val_loss, val_f1, val_cm = validate(model, val_loader, criterion, device)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/F1_Score', val_f1, epoch)
        writer.add_figure('Validation/Confusion_Matrix', plot_confusion_matrix(val_cm, train_dataset.classes), epoch)
        
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print("Early stopping triggered")
            break
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{model_save_dir}/best_model.pth')
    
        if (epoch + 1) % test_frequency == 0:
            test_loss, test_f1, test_cm = validate(model, test_loader, criterion, device)
            writer.add_scalar('Test/Loss', test_loss, epoch)
            writer.add_scalar('Test/F1_Score', test_f1, epoch)
            writer.add_figure('Test/Confusion_Matrix', 
                            plot_confusion_matrix(test_cm, train_dataset.classes), epoch)
            print(f'Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}')
    writer.close()
    return model

def cleanup_augmented_images(train_dir):
    for root, _, files in os.walk(train_dir):
        for file in files:
            if '_augmented_' in file:
                os.remove(os.path.join(root, file))

if __name__ == "__main__":
    cleanup_augmented_images('dataset/train')
    model = train_model('dataset',num_epochs=100)