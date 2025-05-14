import os
os.environ['HF_HOME'] = "/aiau010_scratch/azm0269/hub"

import pandas as pd
from datasets import load_dataset
from config_folder import client_config_file

class TaskDataset:
    def __init__(self):
        """dataset class for all datasets"""

    def get_dataset(self, name=None, file_name=None, split="train"):
        # import os
        # print(os.listdir())
        if file_name!=None:
            try:
                if file_name in list(client_config_file.DATASETS.keys()):
                    df = pd.read_csv(f"./data/raw_data/{file_name}.csv")
                    return df
            except FileNotFoundError:
                print("Local file does not exist. Loading data from Hub.")
        if file_name==None:
            if name=="qiaojin/PubMedQA":
                print(f"Loading Dataset: {name} | Split Type: {split}")
                dataset = load_dataset(
                    name, "pqa_labeled", split=split
                )
                return dataset.to_pandas()
            elif name=="Salesforce/cos_e":
                print(f"Loading Dataset: {name} | Split Type: {split}")
                dataset = load_dataset(
                    name, "v1.11", split=split
                )
                return dataset.to_pandas()
            else:
                print(f"Loading Dataset: {name} | Split Type: {split}")
                dataset = load_dataset(
                    name, split=split
                )
                return dataset.to_pandas()
