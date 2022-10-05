from src.entity.config_entity import DataIngestionConfig
from src.utils.storage_handler import S3Connector
from from_root import from_root
import splitfolders
import os


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        # self.connection = S3Connector()
        # self.client = self.connection.client

    # def download_dir(self):
    #     """
    #     params:
    #     - prefix: pattern to match in s3
    #     - local: local path to folder in which to place files
    #     - bucket: s3 bucket with target contents
    #     - client: initialized s3 client object
    #
    #     """
    #     try:
    #         # keys = []
    #         # dirs = []
    #         # next_token = ''
    #         # base_kwargs = {'Bucket': self.config.BUCKET, 'Prefix': self.config.PREFIX}
    #         # while next_token is not None:
    #         #     kwargs = base_kwargs.copy()
    #         #     if next_token != '':
    #         #         kwargs.update({'ContinuationToken': next_token})
    #         #
    #         #     results = self.client.list_objects_v2(**kwargs)
    #         #     contents = results.get('Contents')
    #         #     for i in contents:
    #         #         k = i.get('Key')
    #         #         if k[-1] != '/':
    #         #             keys.append(k)
    #         #         else:
    #         #             dirs.append(k)
    #         #     next_token = results.get('NextContinuationToken')
    #
    #         print("\n====================== Fetching Data ==============================\n")
    #
    #         data_path = os.path.join(from_root(), self.config.RAW, self.config.PREFIX)
    #         os.system(f"aws s3 sync s3://image-database-system/images/ {data_path}")
    #
    #         # for d in tqdm(dirs, desc="Creating Directories : "):
    #         #     destination_path = os.path.join(self.config.RAW, d)
    #         #     if not os.path.exists(os.path.dirname(destination_path)):
    #         #         os.makedirs(os.path.dirname(destination_path))
    #         #
    #         # for k in tqdm(keys, desc="Downloading Images : "):
    #         #     destination_path = os.path.join(self.config.RAW, k)
    #         #     if not os.path.exists(os.path.dirname(destination_path)):
    #         #         os.makedirs(os.path.dirname(destination_path))
    #         #     if not os.path.exists(destination_path):
    #         #         self.client.download_file(self.config.BUCKET, k, destination_path)
    #
    #         print("\n====================== Fetching Completed ==========================\n")
    #
    #     except Exception as e:
    #         raise e

    def split_data(self):
        """
        This Method is Responsible for splitting.
        :return:
        """
        try:
            splitfolders.ratio(
                input=os.path.join(self.config.RAW, self.config.PREFIX),
                output=self.config.SPLIT,
                seed=self.config.SEED,
                ratio=self.config.RATIO,
                group_prefix=None, move=False
            )
        except Exception as e:
            raise e

    def run_step(self):
        # self.download_dir()
        self.split_data()
        return {"Response": "Completed Data Ingestion"}


if __name__ == "__main__":
    paths = ["data", r"data\raw", r"data\splitted", r"data\embeddings",
             "model", r"model\benchmark", r"model\finetuned"]

    for folder in paths:
        path = os.path.join(from_root(), folder)
        print(path)
        if not os.path.exists(path):
            os.mkdir(folder)

    dc = DataIngestion()
    print(dc.run_step())
