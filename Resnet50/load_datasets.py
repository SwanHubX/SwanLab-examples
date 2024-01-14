import csv
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class DatasetLoader(Dataset):
    def __init__(self, csv_path, image_size=(128, 128), mode='train'):
        """
        Initialize the DatasetLoader with the location of the csv file, the desired image size, and mode.
        It loads the data from the CSV and sets up the appropriate image transformation pipeline based on the mode.
        """
        self.csv_file = csv_path
        self.image_size = image_size
        self.mode = mode  # 'train' or 'test'
        self.data = self.load_data()
        if mode == 'train':
            self.transform = self.set_train_transforms()
        elif mode == 'test':
            self.transform = self.set_test_transforms()
        else:
            raise ValueError("`mode` can only be train or test")

        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def load_data(self):
        """
        Load data from the CSV file and return it as a list.
        """
        with open(self.csv_file, 'r') as file:
            return list(csv.reader(file))

    def set_train_transforms(self):
        """
        Define the image transformation pipeline for training using torchvision transforms.
        Includes common augmentations to introduce variability and robustness.
        """
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def set_test_transforms(self):
        """
        Define the image transformation pipeline for testing using torchvision transforms.
        Generally simpler as no augmentation is applied, only resizing and normalization.
        """
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path):
        """
        Preprocess the image: Read the image, apply transformations, and return the transformed image.
        """
        full_path = os.path.join(self.current_dir, 'datasets', image_path)
        image = Image.open(full_path)
        return self.transform(image)

    def __len__(self):
        """
        Return the number of items in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Return the preprocessed image and its label at the specified index from the dataset.
        """
        image_path, label = self.data[index]
        image = self.preprocess_image(image_path)
        return image, int(label)

