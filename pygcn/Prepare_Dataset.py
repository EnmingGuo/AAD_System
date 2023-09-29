from StaticDataset import StaticDataset
from DynamicDataset import DynamicDataset
# def create_static_dataset(storePath,rawPath,progress):
#     dataset = StaticDataset(storePath,rawPath)
#     print("The data is ready!")
#     def get_Progress():
#         return progress[0]
#     return dataset
class DatasetProgress:
    def __init__(self):
        self.progress = [0]

    def create_static_dataset(self, storePath, rawPath):
        if storePath == "C:/Users/lijl7/Desktop/AAD_System/pygcn/Static":
            dataset = StaticDataset(root  = storePath,raw_path= rawPath)
        else :
            dataset = DynamicDataset(root  = storePath, raw_path= rawPath)
        print(dataset)
        print("The data is ready!")
        self.progress[0] = 100  # 更新进度为100
        return dataset.data_list

    def get_progress(self):
        return self.progress[0]
