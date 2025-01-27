import torch
from torch.utils.data import DataLoader
from dataloader import WeatherDataset
from model import WeatherClassifier
import torch.optim as optim
import torch.nn as nn
import cv2
from image_utils.Augment import WeatherAug 
def preprocess_dataset(data_path, hyp, num_workers=8):
    """
    Preprocess dataset with weather augmentations
    """
    import shutil
    from pathlib import Path
    from multiprocessing import Pool
    from functools import partial
    
    # Create augmented directory if not exists
    aug_path = Path(data_path) / 'augmented'
    if aug_path.exists():
        shutil.rmtree(aug_path)
    aug_path.mkdir(exist_ok=True)
    
    # Get all images
    img_files = list(Path(data_path).rglob('*.jpg'))
    
    # Initialize augmenter
    augmenter = WeatherAug(hyp=hyp, realtime=False)
    
    def process_single_image(img_path, aug_path):
        img = cv2.imread(str(img_path))
        aug_img, transforms = augmenter(img)
        save_path = aug_path / f"{img_path.stem}_aug_{transforms}.jpg"
        cv2.imwrite(str(save_path), aug_img)
        
    # Process images in parallel
    with Pool(num_workers) as pool:
        pool.map(partial(process_single_image, aug_path=aug_path), img_files)

def train_model(data_dir, hyp='image_utils/augment_config.py',num_workers=8,num_epochs=10, batch_size=32):
    # Create dataset and dataloader
    preprocess_dataset('dataset',hyp,num_workers)
    dataset = WeatherDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = WeatherClassifier(num_classes=len(dataset.classes))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}')
    
    return model
