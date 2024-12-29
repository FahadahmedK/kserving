import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ray.data import Dataset, from_items
from PIL import Image
import numpy as np


# def apply_and_save_transform(image, transform, transform_name):
#     transformed_image = transform(image)
#     if isinstance(transformed_image, torch.Tensor):
#         transformed_image = transforms.ToPILImage()(transformed_image)
#     transformed_image.save(os.path.join(f"{transform_name}.jpg"))

def denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


class Data:

    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    
    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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

        transform = self.get_transforms()

        image = Image.open(image_path).convert('RGB')
        # Apply and save each transformation
        tensor = transform(image)
        # print(image_path)
        # print(label)
        # torchvision.utils.save_image(torch.clamp(tensor, 0, 1), 'transformed_image.png')

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