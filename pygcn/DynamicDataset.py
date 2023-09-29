import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from feature_capture import *
import os
import scipy.io as sio
# 这里给出大家注释方便理解
# 程序只要第一次运行后，processed文件生成后就不会执行proces函数，而且只要不重写download()和process()方法，也会直接跳过下载和处理。
class DynamicDataset(InMemoryDataset):
    def __init__(self, root,raw_path,transform=None, pre_transform=None):
        self.raw_path = raw_path
        super().__init__(root, transform, pre_transform)
        self.data_list = torch.load(self.processed_paths[0])

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
    def process(self):
        data_list = []
        filePath = self.raw_path
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

                Edge_List = [IndexF,IndexT]
                edge_index = torch.tensor(Edge_List, dtype=torch.int64)

                # 获取对应y 特征的类别
                ytemp=torch.tensor([(i-1)/5], dtype=torch.long)

                # 处理获得对应的特征
                degree = get_degree(matrix)
                degree_centrality = get_degree_centrality(matrix)
                betweeness_centrality = get_betweeness_centrality(matrix)
                pagerank = get_pagerank(matrix)
                closeness_centrality = get_closeness_centrality(matrix)
                flow_efficencies = get_flow_efficencies(matrix)
                KS = get_KS(matrix)[0]
                local_efficiency = get_local_efficiency(matrix)

                # 将特征处理成为相应的数据
                x_list = []
                for x_num in range(0, 90):
                    x_temp = []
                    x_temp.append(degree[x_num])
                    x_temp.append(degree_centrality[x_num])
                    x_temp.append(betweeness_centrality[x_num])
                    x_temp.append(pagerank[x_num])
                    x_temp.append(closeness_centrality[x_num])
                    x_temp.append(flow_efficencies[x_num])
                    x_temp.append(KS[x_num])
                    x_temp.append(local_efficiency[x_num])
                    x_list.append(x_temp)
                X = torch.tensor(x_list, dtype=torch.float)
                data = Data(x=X,edge_index=edge_index,y=ytemp)
                data_list.append(data)
                i = i+1
        # data, slices = self.collate(data_list) # 直接保存list可能很慢，所以使用collate函数转换成大的torch_geometric.data.Data对象
        # print(data)
        torch.save((data_list), self.processed_paths[0])
