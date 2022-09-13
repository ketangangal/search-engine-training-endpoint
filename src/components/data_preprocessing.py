from src.components import TRAIN_DATA_PATH, TEST_DATA_, VALID_DATA_PATH
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class DataPreprocessing:
    def __init__(self): 
        self.BATCH_SIZE = 32
        self.TRAIN_DATA_PATH = TRAIN_DATA_PATH
        self.TEST_DATA_PATH = TEST_DATA_
        self.VALID_DATA_PATH = VALID_DATA_PATH

    def transformations(self):
        TRANSFORM_IMG = transforms.Compose(
                            [transforms.Resize(512),
                            transforms.CenterCrop(512),
                            transforms.RandomHorizontalFlip(p=0.9),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225] )]
                            )

        return TRANSFORM_IMG

    def create_loaders(self):
        TRANSFORM_IMG = self.transformations()

        train_data = ImageFolder(root=self.TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
        test_data = ImageFolder(root=self.TEST_DATA_PATH, transform=TRANSFORM_IMG)
        valid_data = ImageFolder(root=self.TEST_DATA_PATH, transform=TRANSFORM_IMG)

        train_data_loader = DataLoader(train_data, batch_size=self.BATCH_SIZE, shuffle=False,  num_workers=4)
        test_data_loader  = DataLoader(test_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4) 
        valid_data_loader = DataLoader(valid_data, batch_size=self.BATCH_SIZE, shuffle=True,  num_workers=4)

        return train_data_loader, test_data_loader, valid_data_loader



    def run_step(self):
        train_data_loader, test_data_loader, valid_data_loader = self.create_loaders()
        num_epochs = 1
        for epoch in range(num_epochs):
            print()
            for batch_idx, (x, y) in enumerate(train_data_loader):
                if batch_idx >= 3:
                    break
                print(" Batch index:", batch_idx, end="")
                print(" | Batch size:", y.shape[0], end="")
                print(" | x shape:", x.shape, end="")
                print(" | y shape:", y.shape)            

        print("Labels from current batch:", y)


if __name__ == "__main__":
    dp = DataPreprocessing()
    dp.run_step()