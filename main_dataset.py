import torch
import cv2
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class Cifar10Dataset(Dataset):
    def __init__(self, root, is_transform=False):
        self.root = root if os.path.exists(root) else None
        self.is_transform = is_transform
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.normarlize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.pic = []
        self.label = []
        self.label_name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        
        if self.load_path_list():
            print(f"dataset loaded success!")
        else:
            print(f"dataset loaded failed!")

    def load_path_list(self):
        if self.root is None:
            return False
        
        for index ,label in enumerate (self.label_name):
            # 迭代類別資料夾
            label_path = os.path.join(self.root, label)
            for img in os.listdir(label_path):
                # 迭代圖片
                img_path = os.path.join(label_path, img)
                self.pic.append(img_path)
                self.label.append(index)
        
        return True
        
    def __getitem__(self, idx):
        image_path = self.pic[idx]
        label_path = self.label[idx]
        
        image = cv2.imread(image_path)
        
        image_tensor = self.data_argumentation(image)
        label_tensor = torch.tensor(label_path, dtype=torch.long)
        
        return image_tensor, label_tensor
        
    def data_argumentation(self, image):
        image = cv2.resize(image, (32, 32))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = Image.fromarray(image)
        
        if self.is_transform:
            image = self.transform(image)
        else:
            image = self.normarlize(image)
            
        return image
        
    def __len__(self):
        return len(self.label)
    
if __name__ == "__main__":
    dataset = Cifar10Dataset("train", is_transform=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1)
    for i, (image, label) in enumerate(dataloader):
        print(f"image shape: {image.shape}, label shape: {label.shape}")
        print(f"image dtype: {type(image)}, label dtype: {type(label)}")
        break