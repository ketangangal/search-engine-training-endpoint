from src.entity.config_entity import DatabaseConfig
from pymongo import MongoClient
from typing import List, Dict, Any


class MongoDBClient(object):
    def __init__(self):
        self.config = DatabaseConfig()
        url = self.config.URL.replace("<username>", self.config.USERNAME).replace("<password>", self.config.PASSWORD)
        self.client = MongoClient(url)

    def insert_bulk_record(self, documents: List[Dict[str, Any]]):
        try:
            db = self.client[self.config.DBNAME]
            collection = self.config.COLLECTION
            if collection not in db.list_collection_names():
                db.create_collection(collection)
            result = db[collection].insert_many(documents)
            return {"Response": "Success", "Inserted Documents": len(result.inserted_ids)}
        except Exception as e:
            raise e

    def get_collection_documents(self):
        try:
            db = self.client[self.config.DBNAME]
            collection = self.config.COLLECTION
            result = db[collection].find()
            return {"Response": "Success", "Info": result}
        except Exception as e:
            raise e

    def drop_collection(self):
        try:
            db = self.client[self.config.DBNAME]
            collection = self.config.COLLECTION
            db[collection].drop()
            return {"Response": "Success"}
        except Exception as e:
            raise e


if __name__ == "__main__":
    data = [
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"},
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"},
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"},
        {"embedding": [1, 2, 3, 4, 5, 6], "label": 1, "link": "https://test.com/"}
    ]

    mongo = MongoDBClient()
    print(mongo.insert_bulk_record(data))
    # print(mongo.drop_collection())
    # result = mongo.get_collection_documents()
    # print(result["Info"])
