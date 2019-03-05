import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
#from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints
import torch_geometric
from torch_geometric.utils import to_undirected



class MyTransform(object):
    def __call__(self, data):
        #data.face, data.x = None, torch.ones(data.num_nodes, 1)
        num = 1024
        pos,face = data.pos,data.face
        assert not pos.is_cuda and not face.is_cuda
        assert pos.size(1) == 3 and face.size(0) == 3

        area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
        area = torch.sqrt((area**2).sum(dim=-1)) / 2
        
        prob = area / area.sum()
        sample = torch.multinomial(prob, num, replacement=True)
        face = face[:, sample]

        frac = torch.rand(num, 2)
        mask = frac.sum(dim=-1) > 1
        frac[mask] = 1 - frac[mask]

        pos_sampled = pos[face[0]]
        pos_sampled += frac[:, :1] * (pos[face[1]] - pos[face[0]])
        pos_sampled += frac[:, 1:] * (pos[face[2]] - pos[face[0]])

        data.pos = pos_sampled

        edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
        edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
        data.edge_index = edge_index
        data.face = None
        return data

#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MUTAG')
#dataset = TUDataset(path, name='MUTAG').shuffle()
pre_transform = T.Compose([T.NNGraph(k=6), MyTransform()])
train_dataset = ModelNet(root='/home/code/geo/point_gcn/ModelNet',name='10',train=True, 
                    pre_transform=pre_transform)
test_dataset = ModelNet(root='/home/code/geo/point_gcn/ModelNet',name='10',train=False, 
                    pre_transform=pre_transform)
#test_dataset = dataset[:len(dataset) // 10]
#train_dataset = dataset[len(dataset) // 10:]
test_loader = DataLoader(test_dataset, batch_size=1)
train_loader = DataLoader(train_dataset, batch_size=1)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(train_dataset.num_features, 16)
        self.conv2 = GCNConv(16, train_dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        print(type(x))
        print(x.shape)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        print(type(data))
        print(data)
        #output = model(data.x, data.edge_index, data.batch)
        output = model(data)
        print(data.y)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1, 101):
    train_loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                       train_acc, test_acc))

