import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from feature_capture import *
import os
import scipy.io as sio
# 这里给出大家注释方便理解
# 程序只要第一次运行后，processed文件生成后就不会执行proces函数，而且只要不重写download()和process()方法，也会直接跳过下载和处理。
class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print(self.data) # 输出torch.load加载的数据集data
        # print(root) # MYdata
        # print(self.data) # Data(x=[3, 1], edge_index=[2, 4], y=[3])
        # print(self.slices) # defaultdict(<class 'dict'>, {'x': tensor([0, 3, 6]), 'edge_index': tensor([ 0,  4, 10]), 'y': tensor([0, 3, 6])})
        # print(self.processed_paths[0]) # MYdata\processed\datas.pt
    # 返回数据集源文件名，告诉原始的数据集存放在哪个文件夹下面，如果数据集已经存放进去了，那么就会直接从raw文件夹中读取。
    @property
    def raw_file_names(self):
        # pass # 不能使用pass，会报join() argument must be str or bytes, not 'NoneType'错误
        return []
    # 首先寻找processed_paths[0]路径下的文件名也就是之前process方法保存的文件名
    @property
    def processed_file_names(self):
        return ['datas.pt']
    # 用于从网上下载数据集，下载原始数据到指定的文件夹下，自己的数据集可以跳过
    def download(self):
        pass
    # 生成数据集所用的方法，程序第一次运行才执行并生成processed文件夹的处理过后数据的文件，否则必须删除已经生成的processed文件夹中的所有文件才会重新执行此函数
    # def process(self):
    #     # Read data into huge `Data` list.
    #     # Read data into huge `Data` list.
    #     # 这里用于构建data
    #     edge_index1 = torch.tensor([[0, 1, 1, 2],
    #                                [1, 0, 2, 1]], dtype=torch.int64)
    #     edge_index2 = torch.tensor([[0, 1, 1, 2 ,0 ,1],
    #                                 [1, 0, 2, 1 ,0 ,1]], dtype=torch.int64)
    #     # 节点及每个节点的特征：从0号节点开始
    #     X = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    #     # 每个节点的标签：从0号节点开始-两类0，1
    #     y2 = torch.tensor([0], dtype=torch.long)
    #     y1 = torch.tensor([1],dtype=torch.long)
    #     # 创建data数据
    #     data1 = Data(x=X, edge_index=edge_index1, y=y2)
    #     data2 = Data(x=X, edge_index=edge_index2, y=y1)
    #     data3 = Data(x=X, edge_index=edge_index2, y=y1)
    #     # 将data放入datalist
    #     data_list = [data1,data2,data3]
    #     # data_list = data_list.append(data)
    #     if self.pre_filter is not None: # pre_filter函数可以在保存之前手动过滤掉数据对象。用例可能涉及数据对象属于特定类的限制。默认None
    #         data_list = [data for data in data_list if self.pre_filter(data)]
    #     if self.pre_transform is not None: # pre_transform函数在将数据对象保存到磁盘之前应用转换(因此它最好用于只需执行一次的大量预计算)，默认None
    #         data_list = [self.pre_transform(data) for data in data_list]
    #     data, slices = self.collate(data_list) # 直接保存list可能很慢，所以使用collate函数转换成大的torch_geometric.data.Data对象
    #     # print(data)
    #     torch.save((data, slices), self.processed_paths[0])
    def process(self):
        data_list = []
        filePath = 'C:\\Users\lijl7\Desktop\AAD_System\Mat'
        y2 = torch.tensor([0], dtype=torch.long)
        y1 = torch.tensor([1], dtype=torch.long)
        x_list = []
        for x_num in range(0,90):
            x_temp = [x_num]
            x_list.append(x_temp)
        #x_list = [x_list,x_list]
        X = torch.tensor(x_list, dtype=torch.float)
        i = 1
        for root, dirs, files in os.walk(filePath):

            # root 表示当前正在访问的文件夹路径
            # dirs 表示该文件夹下的子目录名list
            # files 表示该文件夹下的文件list

            # 遍历文件
            for f in files:
                IndexF = []
                IndexT = []
                Path = os.path.join(root, f)
                load_data = sio.loadmat(Path)
                matrix = load_data['features']
                get_degree(matrix)
                for row in range(0,90):
                    for col in range(0,90):
                        if(matrix[row][col]==1):
                            IndexF.append(row)
                            IndexT.append(col)
                #Edge_List = [[IndexF,IndexT],[IndexF,IndexT]]
                Edge_List = [IndexF,IndexT]
                edge_index = torch.tensor(Edge_List, dtype=torch.int64)
                ytemp=torch.tensor([(i-1)/5], dtype=torch.long)
                print(ytemp)
                data=Data(x=X,edge_index=edge_index,y=ytemp)
                data_list.append(data)
                # if (i<=445):
                #     data = Data(x=X, edge_index=edge_index, y=y2)
                #     data_list . append(data)
                # else:
                #     data = Data(x=X, edge_index=edge_index, y=y1)
                #     data_list .append(data)
                i = i+1
        data, slices = self.collate(data_list) # 直接保存list可能很慢，所以使用collate函数转换成大的torch_geometric.data.Data对象
        # print(data)
        torch.save((data, slices), self.processed_paths[0])
