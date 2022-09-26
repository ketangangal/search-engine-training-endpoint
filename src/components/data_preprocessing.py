from src.entity.config_entity import DataPreprocessingConfig
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


class DataPreprocessing:
    def __init__(self):
        self.config = DataPreprocessingConfig()

    def transformations(self):
        try:
            """
            Transformation Method Provides TRANSFORM_IMG object. Its pytorch's transformation class to apply on images.
            :return: TRANSFORM_IMG
            """
            TRANSFORM_IMG = transforms.Compose(
                [transforms.Resize(self.config.IMAGE_SIZE),
                 transforms.CenterCrop(256),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])]
            )

            return TRANSFORM_IMG
        except Exception as e:
            raise e

    def create_loaders(self, TRANSFORM_IMG):
        """
        The create_loaders method takes Transformations and create dataloaders.
        :param TRANSFORM_IMG:
        :return: Dict of train, test, valid Loaders
        """
        try:
            print("Generating DataLoaders : ")
            result = {}
            for _ in tqdm(range(1)):
                train_data = ImageFolder(root=self.config.TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
                test_data = ImageFolder(root=self.config.TEST_DATA_PATH, transform=TRANSFORM_IMG)
                valid_data = ImageFolder(root=self.config.TEST_DATA_PATH, transform=TRANSFORM_IMG)

                train_data_loader = DataLoader(train_data, batch_size=self.config.BATCH_SIZE,
                                               shuffle=True, num_workers=1)
                test_data_loader = DataLoader(test_data, batch_size=self.config.BATCH_SIZE,
                                              shuffle=False, num_workers=1)
                valid_data_loader = DataLoader(valid_data, batch_size=self.config.BATCH_SIZE,
                                               shuffle=False, num_workers=1)

                result = {
                    "train_data_loader": (train_data_loader, train_data),
                    "test_data_loader": (test_data_loader, test_data),
                    "valid_data_loader": (valid_data_loader, valid_data)
                }
            return result
        except Exception as e:
            raise e

    def run_step(self):
        try:
            """
            This methods calls all the private methods.
            :return: Response of Process
            """
            TRANSFORM_IMG = self.transformations()
            result = self.create_loaders(TRANSFORM_IMG)
            return result
        except Exception as e:
            raise e


if __name__ == "__main__":
    # Data Ingestion Can be replaced like this
    # https://aws.amazon.com/blogs/machine-learning/announcing-the-amazon-s3-plugin-for-pytorch/
    dp = DataPreprocessing()
    loaders = dp.run_step()
    for i in loaders["train_data_loader"][0]:
        break
