from torch.utils.data import Dataset, DataLoader
import os
import cv2
from PIL import ImageFile
import sys
import os
from pathlib import Path
import numpy as np
ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)
from utils.Augment import WeatherAug
ImageFile.LOAD_TRUNCATED_IMAGES = True

class WeatherDataset(Dataset):
    def __init__(self, data_dir, hyp='cfg/augment_params.yaml', augment=True, img_size=224):
        self.data_dir = data_dir
        self.classes = os.listdir(data_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.imgsz = img_size
        
        self.image_paths = []
        self.labels = []
        self.augment = augment
        if augment:
            self.augmenter = WeatherAug(hyp=hyp, realtime=True)

        for class_name in self.classes:
            class_path = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img_name))
                self.labels.append(self.class_to_idx[class_name])
    
    def resize_with_aspect_ratio(self, img):
        h, w, _ = img.shape
        if h > w:
            new_h = self.imgsz
            new_w = int(w * self.imgsz / h)
        else:
            new_w = self.imgsz
            new_h = int(h * self.imgsz / w)
            
        resized = cv2.resize(img, (new_w, new_h))
        
        # Create a black canvas of target size
        final_img = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_x = (self.imgsz - new_w) // 2
        pad_y = (self.imgsz - new_h) // 2
        
        # Place the resized image in the center
        final_img[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        return final_img

    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        
        # Resize if larger than max_size while keeping aspect ratio
        img = self.resize_with_aspect_ratio(img)
        
        if self.augment:
            img, transforms = self.augmenter(img)
        img = self.resize_with_aspect_ratio(img)
        # Convert to torch tensor 
        img = img.astype('float32') / 255.0
        img = img.transpose(2, 0, 1)  # Convert to CHW format
            
        return img, self.labels[idx]
