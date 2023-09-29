import torch
import copy
from torch_geometric.data import DataLoader
import warnings
from torch_geometric.data import InMemoryDataset
from StaticDataset import StaticDataset
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv

class MyDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.data, self.slices = self.collate(data_list)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,dataset):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels,dataset):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

def test_case(mydata):
    rawdataset = StaticDataset("C:\\Users\lijl7\Desktop\AAD_System\pygcn\Static",
                            "C:\\Users\lijl7\Desktop\AAD_System\Mat")  # 创建数据集对象
    dataset = MyDataset(rawdataset.data_list)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = GCN(hidden_channels=64, dataset=dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = torch.nn.CrossEntropyLoss()
    Train_acc = 0;
    Test_result= None;
    choosed_device = torch.device('cpu')
    for epoch in range(1, 10):
        train(optimizer, crit, model, data_loader,choosed_device)
        train_acc = test(model, data_loader,choosed_device)
        test_result = get_result(model, mydata)
        print(f"The {epoch:03d}  round of training is completed!")
        if train_acc > Train_acc:
            Test_result = test_result

    return Test_result


def modify_data(data,features):
    variable_list = ['Degree',  'Degree Centrality',  'Betweeness Centrality',  'Pagerank', 'Closeness Centrality', 'Flow Coefficiency', 'Kernel Shell', 'Local Efficiency']
    # Create an empty dictionary
    variable_dict = {}
    # Assign IDs to the variables
    for i, variable in enumerate(variable_list):
        variable_dict[i] = variable
    print(data)
    columns_to_keep = []
    for col_idx, variable_name in variable_dict.items():
        if variable_name in features:
            columns_to_keep.append(col_idx)
    new_datalist = []
    for data_item in data:
        # 获取原始x列表
        original_x = data_item.x
        # 生成新的x列表，仅保留指定的列
        new_x = [original_x[:, col] for col in columns_to_keep]
        # 将新的x列表赋值回data对象
        data_item.x = torch.stack(new_x, dim=1)
        new_datalist.append(data_item)
    print("停顿一下")
    print(new_datalist)
    mydataset = MyDataset(new_datalist)
    return mydataset

def static_gcn(data):
    feature = data.get('feature')
    print(feature)
    original_dataset = data.get('train_data')
    print("Mark一下")
    print(original_dataset)
    new_dataset = copy.deepcopy(original_dataset)
    dataset = modify_data(new_dataset, feature)

    max_epoch = data.get('Epoch');
    is_shuffle = data.get('is_shuffle')
    batch_size = data.get('Batch_size')
    test_ratio = data.get('Test_ratio')
    device = data.get('device')
    selected_optimizer = data.get('optimizer')

    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    choosed_device = torch.device('cpu')
    # device
    if(device == 'CPU'):
        choosed_device = torch .device('cpu')
    elif(device == 'GPU'):
        choosed_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Device', device)
    torch.manual_seed(12345)
    if(is_shuffle==1):
        dataset = dataset.shuffle()

    #  dataset_lower
    dataset_lower = int(890*(1-test_ratio))
    train_dataset = dataset[:dataset_lower]
    test_dataset = dataset[dataset_lower:]
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    if(is_shuffle==1):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()

    model = GCN(hidden_channels=64,dataset = dataset).to(choosed_device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    train_acc_list = []
    test_acc_list = []
    for epoch in range(1, max_epoch):
        train(optimizer,criterion,model, train_loader,choosed_device)
        train_acc = test(model, train_loader,choosed_device)
        test_acc = test(model, test_loader,choosed_device)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        train_acc_list . append(train_acc)
        test_acc_list . append(test_acc)
    if(selected_optimizer == 'Average Epoch'):
        mid_train_acc_list = train_acc_list[len(train_acc_list) // 2]
        mid_test_acc_list = test_acc_list[len(test_acc_list)//2]
        return [ mid_train_acc_list,mid_test_acc_list]

    elif(selected_optimizer == 'Optimal Epoch'):
        # 创建一个空字典来存储test_acc_list中的数值和对应的train_acc_list的值
        acc_dict = {}

        # 遍历test_acc_list
        for i, acc in enumerate(test_acc_list):
            if acc not in acc_dict:
                # 如果数值不在字典中，添加键值对
                acc_dict[acc] = train_acc_list[i]
            else:
                # 如果数值已经存在于字典中，比较train_acc_list的值，更新为更大的值
                acc_dict[acc] = max(acc_dict[acc], train_acc_list[i])

        # 获取最大的test_acc_list数值
        max_test_acc = max(test_acc_list)

        # 获取对应的train_acc_list的值
        corresponding_train_acc = acc_dict[max_test_acc]
        return [corresponding_train_acc, max_test_acc]

    else:
        last_train_acc_list = train_acc_list[-1]
        last_test_acc_list = test_acc_list[-1]
        return [last_train_acc_list, last_test_acc_list]


def train(optimizer,criterion,model,data_loader,device):
    model.train()
    for data in data_loader:  # Iterate in batches over the training dataset.
        optimizer.zero_grad()  # Clear gradients.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out,data.y)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.


def test(model,data_loader,device):
    model.eval()
    correct = 0
    for data in data_loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(data_loader.dataset)  # Derive ratio of correct predictions.

def get_result(model,mydata):
    model.eval()
    out= model(mydata.x,mydata.edge_index,mydata.batch)
    probabilities = F.softmax(out, dim=1)
    return probabilities