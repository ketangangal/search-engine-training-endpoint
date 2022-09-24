from annoy import AnnoyIndex
from from_root import from_root
from typing import Literal
import os
import json


# noinspection PyTypeChecker
class CustomAnnoy(AnnoyIndex):
    def __init__(self, f: int, metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"]):
        super().__init__(f, metric)
        self.label = []

    # noinspection PyMethodOverriding
    def add_item(self, i: int, vector, label: str) -> None:
        super().add_item(i, vector)
        self.label.append(label)

    def get_nns_by_vector(
            self, vector, n: int, search_k: int = ..., include_distances: Literal[False] = ...
    ) -> list[str]:
        indexes = super().get_nns_by_vector(vector, n)
        labels = [self.label[link] for link in indexes]
        return labels

    def load(self, fn: str, prefault: bool = ...) -> Literal[True]:
        super().load(fn)
        path = fn.replace(".ann", ".json")
        self.label = json.load(open(path, "r"))

    def save(self, fn: str, prefault: bool = ...) -> Literal[True]:
        super().save(fn)
        path = fn.replace(".ann", ".json")
        json.dump(self.label, open(path, "w"))


class Inferncing(object):
    def __init__(self):
        self.ann = CustomAnnoy(256, 'euclidean')

    def test(self, embedding):
        path = os.path.join(from_root(), "data", "embeddings", "embeddings.ann")
        self.ann.load(path)
        print(self.ann.get_nns_by_vector(embedding, 5))

