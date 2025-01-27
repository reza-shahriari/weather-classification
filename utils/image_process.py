from PIL import Image
import imghdr
import os
import shutil
from glob import glob
import zipfile
import random
from sklearn.model_selection import train_test_split
class ImageProcessor:
    def __init__(self,data_dir='dataset'):
        self.data_dir = data_dir
        
    
    def validate_image(self,image_path):
        try:
            img = Image.open(image_path)
            img.verify()
            
            # Check if it's a valid image format
            if imghdr.what(image_path) not in ['jpeg', 'png', 'jpg']:
                return False
                
            # Check minimum dimensions
            if img.size[0] < 100 or img.size[1] < 100:
                return False
                
            return True
        except:
            return False

    def clean_dataset(self):
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if not self.validate_image(file_path):
                    os.remove(file_path)
                    print(f"Removed invalid image: {file_path}")
    
    def move_images(self,source_dir,destination_dir,folder_based=True,names=[]):
        if folder_based:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if self.validate_image(file_path):
                        os.rename(file_path, os.path.join(destination_dir, file))
                        print(f"Moved valid image: {file_path}")
        else:
            imgs = glob(os.path.join(source_dir,'*.jpg'))
            for img in imgs:
                for name in names:
                    if name in os.path.basename(img).lower():
                        os.rename(img, os.path.join(destination_dir,name, os.path.basename(img)))
                        print(f"Moved valid image: {img}")
    
    def create_dataset(self):
        dataset_main_path = 'dataset'
        dataset_catagories = ['cloudy','rainy','shine','sunrise','clear',]
        os.makedirs(dataset_main_path, exist_ok=True)
        for catagory in dataset_catagories:
            os.makedirs(os.path.join(dataset_main_path,catagory),exist_ok=True)
        folders = []
        for file in glob(dataset_main_path+'/*'):
            if file[:-4] in['.zip','.rar']:
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(file[:-4])
                os.remove(file)
        folders = [
            'dataset/Weather-Classification-main/Multi-class Weather Dataset',
            'dataset/dataset2',
            'dataset/data_sets/data_sets/artificial',
            'dataset/data_sets/data_sets/real'
        ]
        self.move_images(os.path.join(folders[0],'Cloudy'),'dataset/cloudy')
        self.move_images(os.path.join(folders[0],'Rain'),'dataset/rainy')
        self.move_images(os.path.join(folders[0],'Shine'),'dataset/shine')
        self.move_images(os.path.join(folders[0],'Sunrise'),'dataset/sunrise')
        self.move_images(folders[1],'dataset',False,['cloudy','rainy','shine','sunrise'])
        self.move_images(os.path.join(folders[2],'clear'),'dataset/clear')
        self.move_images(os.path.join(folders[3],'clear'),'dataset/clear')
        self.move_images(os.path.join(folders[2],'synthetic_rain'),'dataset/rainy')
        self.move_images(os.path.join(folders[3],'rain_combine'),'dataset/rainy')
        shutil.rmtree('dataset/Weather-Classification-main/')
        shutil.rmtree('dataset/dataset2')
        shutil.rmtree('dataset/data_sets')
    
    def train_val_test_split(self):

        # Create train, val, test directories
        splits = ['train', 'val', 'test']
        categories = [os.path.basename(folder) for folder in glob(os.path.join(self.data_dir, '*')) if os.path.isdir(folder)]
        for s in splits:
            if s in categories:
                categories.remove(s)
        for category in categories:
            category_path = os.path.join(self.data_dir, category)
            if os.path.exists(category_path):
                num_images = len([f for f in os.listdir(category_path) if self.validate_image(os.path.join(category_path, f))])
                print(f"{category}: {num_images} valid images")

        
        
        for split in splits:
            split_path = os.path.join(self.data_dir, split)
            os.makedirs(split_path, exist_ok=True)
            for category in categories:
                os.makedirs(os.path.join(split_path, category), exist_ok=True)
        
        # Split data for each category
        for category in categories:
            category_path = os.path.join(self.data_dir, category)
            if not os.path.exists(category_path):
                continue
                
            # Get all images in category
            images = [f for f in os.listdir(category_path) if self.validate_image(os.path.join(category_path, f))]
            
            # Split into train (70%), validation (15%), test (15%)
            train_imgs, test_val_imgs = train_test_split(images, train_size=0.7, random_state=42)
            val_imgs, test_imgs = train_test_split(test_val_imgs, train_size=0.5, random_state=42)
            
            # Move images to respective folders
            for img in train_imgs:
                src = os.path.join(category_path, img)
                dst = os.path.join(self.data_dir, 'train', category, img)
                shutil.move(src, dst)
                
            for img in val_imgs:
                src = os.path.join(category_path, img)
                dst = os.path.join(self.data_dir, 'val', category, img)
                shutil.move(src, dst)
                
            for img in test_imgs:
                src = os.path.join(category_path, img)
                dst = os.path.join(self.data_dir, 'test', category, img)
                shutil.move(src, dst)
            
            # Remove empty category folder
            if len(os.listdir(category_path)) == 0:
                os.rmdir(category_path)
    def reverse_split(self):
        splits = ['train', 'val', 'test']
        categories = [os.path.basename(folder) for folder in glob(os.path.join(self.data_dir, '*')) if os.path.isdir(folder)]
        for s in splits:
            if s in categories:
                categories.remove(s)
        # Create category folders if they don't exist
        for category in categories:
            os.makedirs(os.path.join(self.data_dir, category), exist_ok=True)
        
        # Move all images back to their category folders
        for split in splits:
            split_path = os.path.join(self.data_dir, split)
            if not os.path.exists(split_path):
                continue
                
            for category in categories:
                category_path = os.path.join(split_path, category)
                if not os.path.exists(category_path):
                    continue
                    
                # Move all images from split/category to category
                for img in os.listdir(category_path):
                    src = os.path.join(category_path, img)
                    dst = os.path.join(self.data_dir, category, img)
                    shutil.move(src, dst)
                
                # Remove empty category folder
                os.rmdir(category_path)
            
            # Remove empty split folder
            os.rmdir(split_path)


processor = ImageProcessor()
processor.train_val_test_split()
            