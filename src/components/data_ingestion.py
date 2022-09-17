import splitfolders
from tqdm import tqdm
import shutil
import boto3
import os


class DataIngestion:
    def __init__(self, prefix, raw, split, bucket):
        self.prefix = prefix
        self.raw = raw
        self.split = split
        self.bucket = bucket
        self.client = boto3.client('s3')

    def download_dir(self):
        """
        params:
        - prefix: pattern to match in s3
        - local: local path to folder in which to place files
        - bucket: s3 bucket with target contents
        - client: initialized s3 client object

        """
        keys = []
        dirs = []
        next_token = ''
        base_kwargs = {'Bucket': self.bucket, 'Prefix': self.prefix}
        while next_token is not None:
            kwargs = base_kwargs.copy()
            if next_token != '':
                kwargs.update({'ContinuationToken': next_token})

            results = self.client.list_objects_v2(**kwargs)
            contents = results.get('Contents')
            for i in contents:
                k = i.get('Key')
                if k[-1] != '/':
                    keys.append(k)
                else:
                    dirs.append(k)
            next_token = results.get('NextContinuationToken')
        print("====================== Fetching Data ==============================\n")

        for d in tqdm(dirs, desc="Creating Directories : "):
            dest_pathname = os.path.join(self.raw, d)
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))

        for k in tqdm(keys, desc="Downloading Images : "):
            dest_pathname = os.path.join(self.raw, k)
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            self.client.download_file(self.bucket, k, dest_pathname)

        print("\n====================== Fetching Completed ==========================")

    def split_data(self):
        try:
            splitfolders.ratio(input=self.raw + "/images/", output=self.split, seed=1337, ratio=(.8, .1, .1),
                               group_prefix=None, move=False)
            shutil.rmtree(self.raw + "/images/")
        except Exception as e:
            raise e

    def run_step(self):
        self.download_dir()
        self.split_data()
        return {"Response": "Completed"}


if __name__ == "__main__":
    dc = DataIngestion("images/", "data/raw", "data/splitted", "image-database-system")
    dc.run_step()
