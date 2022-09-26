from src.utils.database_handler import MongoDBClient
from src.entity.config_entity import AnnoyConfig
from annoy import AnnoyIndex
from typing_extensions import Literal
from tqdm import tqdm
import json


class CustomAnnoy(AnnoyIndex):
    def __init__(self, f: int, metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"]):
        super().__init__(f, metric)
        self.label = []

    # noinspection PyMethodOverriding
    def add_item(self, i: int, vector, label: str) -> None:
        super().add_item(i, vector)
        self.label.append(label)

    def get_nns_by_vector(
            self, vector, n: int, search_k: int = ..., include_distances: Literal[False] = ...):
        indexes = super().get_nns_by_vector(vector, n, search_k, include_distances)
        labels = [self.label[link] for link in indexes]
        return labels

    def load(self, fn: str, prefault: bool = ...):
        super().load(fn)
        path = fn.replace(".ann", ".json")
        self.label = json.load(open(path, "r"))

    def save(self, fn: str, prefault: bool = ...):
        super().save(fn)
        path = fn.replace(".ann", ".json")
        json.dump(self.label, open(path, "w"))


class Annoy(object):
    def __init__(self):
        self.config = AnnoyConfig()
        self.mongo = MongoDBClient()
        self.result = self.mongo.get_collection_documents()["Info"]

    def build_annoy_format(self):
        Ann = CustomAnnoy(256, 'euclidean')
        print("Creating Ann for predictions : ")
        for i, record in tqdm(enumerate(self.result), total=8677):
            Ann.add_item(i, record["images"], record["s3_link"])

        Ann.build(100)
        Ann.save(self.config.EMBEDDING_STORE_PATH)
        return True

    def run_step(self):
        self.build_annoy_format()


if __name__ == "__main__":
    ann = Annoy()
    ann.run_step()
