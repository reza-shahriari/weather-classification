from torch.utils.data import Dataset, DataLoader
import os
import cv2
from PIL import ImageFile
from image_utils.Augment import WeatherAug
ImageFile.LOAD_TRUNCATED_IMAGES = True

class WeatherDataset(Dataset):
    def __init__(self, data_dir,  hyp, augment=True):
        self.data_dir = data_dir
        self.classes = os.listdir(data_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        if augment:
            self.augmenter = WeatherAug(hyp=hyp, realtime=True)
    
        for class_name in self.classes:
            class_path = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img_name))
                self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        
        if self.augment:
            img, transforms = self.augmenter(img)
            
        return img,self.labels[idx]
