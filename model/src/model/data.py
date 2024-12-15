import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ray.data import Dataset, from_items
from PIL import Image
import numpy as np

class Data:

    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __init__(self, input_path, output_path=None, train=True):

        self.input_path = input_path
        self.output_path = output_path
        self.train = train


    def get_transforms(self):
        if self.train:
            return Data.train_transform
        return Data.test_transform

    def _preprocess(self, image_data):

        image_path = image_data["image_path"]
        label = image_data["label"]

        transform = Data.get_transforms()

        image = Image.open(image_path).convert('RGB')

        tensor = transform(image)

        return {
            "images": tensor.numpy(),
            "labels": label
        }
    
    def create_dataset(self):
        image_data = []
        class_to_idx = {}

        for idx, class_name in enumerate(os.listdir(self.input_path)):

            class_path = os.path.join(self.input_path, class_name)
            class_to_idx[class_name] = idx 

            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                image_data.append({
                    "image_path": image_path,
                    "label": idx
                })

        ds = from_items(image_data)
        ds = ds.map(self._preprocess)


        return ds, class_to_idx
    

if __name__ == "__main__":

    data = Data(input_path="/home/fahad/study/kserving/data/train")
    ds, class_to_idx = data.create_dataset()
    import pdb; pdb.set_trace()