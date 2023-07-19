import numpy as np
from torchvision import datasets, transforms
import pandas as pd
from PIL import Image
from typing import Any, Tuple
from numpy import asarray


class XrayDataset():
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform
        train: bool = True
        #self.img_dir = img_dir#'./final_data'
        self.data: Any = []
        self.L3 = []
        self.L2 = []
        self.L1=[]

        for idx in range(len(self.img_labels)):
            img_path = self.img_labels.iloc[idx,0]
            L3 = self.img_labels.iloc[idx, 1]
            L2 = self.img_labels.iloc[idx, 2]
            L1 = self.img_labels.iloc[idx, 3]
     
            img = Image.open(f'{img_path}')
            img = img.resize((224, 224))
            img = img.convert("RGB")
            img = asarray(img)
        
            self.data.append(img)
            self.L3.append(L3)
            self.L2.append(L2)
            self.L1.append(L1)
 
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index:int) -> Tuple[Any, Any]:
        
        image, target = self.data[index], self.L3[index] , self.L2[index], self.L1[index]
     
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            L3 = self.target_transform(L3)
        if self.target_transform:
            L2 = self.target_transform(L2)
        if self.target_transform:
            L1 = self.target_transform(L1)

        return image, L3, L2, L1

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

class iOA(iData):
    use_path = False 
    
    train_trsf = [
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1,contrast=0.1),    
    ]
    test_trsf = [
        transforms.Resize(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(16).tolist()

    def download_data(self): 
        train_dataset = XrayDataset(annotations_file='~/data/sample_train.csv')
        test_dataset = XrayDataset(annotations_file='~/data/sample_test.csv')
        self.train_data, self.train_L3,self.train_L2,self.train_L1 = np.array(train_dataset.data), np.array(
            train_dataset.L3) , np.array(train_dataset.L2) , np.array(train_dataset.L1)
    
        self.test_data, self.test_L3,self.test_L2,self.test_L1 =np.array(test_dataset.data), np.array(test_dataset.L3
            ) , np.array(test_dataset.L2), np.array(test_dataset.L1
        )
