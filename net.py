import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv

class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()
    
class TemporalConvNet(nn.Module):
    """
    time dilation convolution
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs : channel's number of input data's feature
            num_channels : numbers of data feature tranform channels, the last is the output channel
            kernel_size : using 1d convolution, so the real kernel is (1, kernel_size) 
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i 
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            padding = (kernel_size - 1) * dilation_size 
            
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation_size), padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            
            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
            # layers += [nn.Sequential(self.conv, self.relu, self.dropout)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01) 

    def forward(self, x):
        """ 
        like ResNet
        Args:
            X : input data of shape (B, N, T, F) --->(BatchSize,Nodes,TimeSeries,Feature)
        """
        y = F.relu(self.network(x) + self.downsample(x) if self.downsample else x) 
        
        return y

class GraphConstructor(nn.Module):
    def __init__(self, num_nodes, k, dim, alpha=3):
        super(GraphConstructor, self).__init__()
        self.num_nodes = torch.tensor(num_nodes)
        self.k = torch.tensor(k)
        self.dim = torch.tensor(dim)
        self.alpha = torch.tensor(alpha)

        self.emb1 = nn.Embedding(num_nodes, dim)
        self.emb2 = nn.Embedding(num_nodes, dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, idx):
        node_vec1 = self.emb1(idx)
        node_vec2 = self.emb2(idx)
        node_vec1 = torch.tanh(self.alpha * self.lin1(node_vec1))
        node_vec2 = torch.tanh(self.alpha * self.lin1(node_vec2))

        a = torch.mm(node_vec1, node_vec2.transpose(1,0))-torch.mm(node_vec2, node_vec1.transpose(1,0))

        adj = F.relu(torch.tanh(self.alpha * a))

        top_k_values, top_k_indices = torch.topk(adj, self.k, dim=1)
        mask = torch.zeros_like(adj)
        mask.scatter_(1, top_k_indices, top_k_values.fill_(1))

        adj = adj * mask
        tmp_index = torch.nonzero(adj).T
        tmp_data= adj[tmp_index[0], tmp_index[1]]
        index = tmp_index
        data = tmp_data
        return index,data

class GCNWithEdgeWeights(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(GCNWithEdgeWeights, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, num_classes)

    
    def forward(self, x, edge_index,edge_weight):
        x = self.conv1(x, edge_index,edge_weight)
        x = torch.relu(x)
        x = F.dropout(x, p=0.5,training=self.training)
        x = self.conv2(x, edge_index,edge_weight)
        return x

#LSTM
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        return out


# NET
class NET(nn.Module):
    def __init__(self,in_channels, out_channels,num_nodes,ktop,node_dim,tanhalpha,num_features,hidden_channels,num_classes):
        super(NET, self).__init__()
        self.TC = TemporalConvNet(num_inputs=in_channels,num_channels=out_channels)
        
        # self.idx = torch.arange(num_nodes).to('cuda:0')
        # self.GConstructor = GraphConstructor(num_nodes, k=ktop, dim=node_dim, alpha=tanhalpha)
        # self.GCN = GCNWithEdgeWeights(in_features = num_features, hidden_features = hidden_channels, num_classes = num_classes)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(),
            #nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=5, stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(),
            #nn.Dropout(0.5)
        )
        # 全连接层修改了
        self.fc1 = nn.Sequential(
            nn.Linear(12 * 62 * 25, 62*2, bias=True),
            nn.Dropout(0.1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(62*2, 4, bias=True),
            #nn.Dropout(0.5)
        )
             
    def forward(self, x):
#        auto_edge_index,auto_edge_weight = self.GConstructor(self.idx)
        # x = self.TC(x)
#        x = self.GCN(x,auto_edge_index,auto_edge_weight)+x
        x = torch.unsqueeze(x, dim=1) 
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.reshape(x,(x.size(0),-1))
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# NET
class TCN_CNN(nn.Module):
    def __init__(self,in_channels, out_channels,num_nodes,ktop,node_dim,tanhalpha,num_features,hidden_channels,num_classes):
        super(TCN_CNN, self).__init__()
        self.TC = TemporalConvNet(num_inputs=in_channels,num_channels=out_channels)
        
        # self.idx = torch.arange(num_nodes).to('cuda:0')
        # self.GConstructor = GraphConstructor(num_nodes, k=ktop, dim=node_dim, alpha=tanhalpha)
        # self.GCN = GCNWithEdgeWeights(in_features = num_features, hidden_features = hidden_channels, num_classes = num_classes)
        
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 6, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(),
            # nn.Dropout(0.1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 8, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            # nn.Dropout(0.1)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 10, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            # nn.Dropout(0.1)
        )
        
        
        self.fc1 = nn.Sequential(
            nn.Linear(10 * 62 * 25, 62, bias=True),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(62, 4, bias=True),
            # nn.Dropout(0.2)
        )
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2)
        
        
    def forward(self, x):
        # auto_edge_index,auto_edge_weight = self.GConstructor(self.idx)
        # x = self.GCN(x,auto_edge_index,auto_edge_weight)+x
        
        x = self.TC(x)
        # x = torch.unsqueeze(x, dim=1) 
        
        x = self.layer1(x)
        # x = self.Maxpool(x)
        x = self.layer2(x)
        # x = self.Maxpool(x)
        x = self.layer3(x)
        # x = self.Maxpool(x)
        
        x = torch.reshape(x,(x.size(0),-1))
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        # x = torch.relu(x)
        # x = self.fc3(x)
        return x


# NET
class TCN3DNet(nn.Module):
    def __init__(self,in_channels, out_channels,num_nodes,ktop,node_dim,tanhalpha,num_features,hidden_channels,num_classes):
        super(TCN3DNet, self).__init__()
        self.TC = TemporalConvNet(num_inputs=in_channels,num_channels=out_channels)
                 
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 2, (2, 5, 5), stride=1, padding=(0, 2, 2),bias=True),
            nn.BatchNorm3d(2),
            nn.LeakyReLU(),
            # nn.Dropout(0.1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(2, 4, (2, 5, 5), stride=1, padding=(0, 2, 2),bias=True),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(),
            # nn.Dropout(0.1)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv3d(4, 6, (2, 5, 5), stride=1, padding=(0, 2, 2),bias=True),
            nn.BatchNorm3d(6),
            nn.LeakyReLU(),
            # nn.Dropout(0.1)
        )
        
        
        self.fc1 = nn.Sequential(
            nn.Linear(6 * 62 * 25, 71, bias=True),
            # nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(71, 4, bias=True),
            # nn.Dropout(0.2)
        )
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2)
        
        
    def forward(self, x):
        x = self.TC(x)
        x = torch.unsqueeze(x, dim=1) 
        x = self.layer1(x)
        # x = self.Maxpool(x)
        x = self.layer2(x)
        # x = self.Maxpool(x)
        x = self.layer3(x)
        # x = self.Maxpool(x)
        
        x = torch.reshape(x,(x.size(0),-1))
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    
# NET
class LOSO_NET(nn.Module):
    def __init__(self,in_channels, out_channels,num_nodes,ktop,node_dim,tanhalpha,num_features,hidden_channels,num_classes):
        super(LOSO_NET, self).__init__()
        self.TC = TemporalConvNet(num_inputs=in_channels,num_channels=out_channels)
        
        # self.idx = torch.arange(num_nodes).to('cuda:0')
        # self.GConstructor = GraphConstructor(num_nodes, k=ktop, dim=node_dim, alpha=tanhalpha)
        # self.GCN = GCNWithEdgeWeights(in_features = num_features, hidden_features = hidden_channels, num_classes = num_classes)

        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 12, kernel_size=5, stride=(1, 1), padding=(2, 2), bias=True),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(24 * 62 * 25, 62*8, bias=True),
        #     # nn.Dropout(0.1)
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(62*8, 24, bias=True),
        #     #nn.Dropout(0.5)
        # )
        # self.fc3 = nn.Sequential(
        #     nn.Linear(24, 4, bias=True),
        #     #nn.Dropout(0.5)
        # )
        
        self.fc1 = nn.Sequential(
            nn.Linear(12 * 62 * 25, 62*3, bias=True),
            nn.Dropout(0.6)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(62*3, 4, bias=True),
            nn.Dropout(0.6)
        )
        
    def forward(self, x):
        # auto_edge_index,auto_edge_weight = self.GConstructor(self.idx)
        # x = self.GCN(x,auto_edge_index,auto_edge_weight)+x
        
        x = self.TC(x)
        # x = torch.unsqueeze(x, dim=1) 
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = torch.reshape(x,(x.size(0),-1))
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        # x = torch.relu(x)
        # x = self.fc3(x)
        return x
    
class DepthWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthWiseConv, self).__init__()
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=in_channels
        )
        # 逐点卷积
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


# NET
class TCN_DSConvCNN(nn.Module):
    def __init__(self,in_channels, out_channels,num_nodes,ktop,node_dim,tanhalpha,num_features,hidden_channels,num_classes):
        super(TCN_DSConvCNN, self).__init__()
        self.TC = TemporalConvNet(num_inputs=in_channels,num_channels=out_channels)

        self.layer1 = nn.Sequential(
            DepthWiseConv(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            DepthWiseConv(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        # 全连接层修改了
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 62 * 25, 62*8, bias=True),
            #nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(62*8, 24, bias=True),
            #nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(24, 4, bias=True),
            #nn.Dropout(0.5)
        )
             
    def forward(self, x):
        
        x = self.TC(x)
        x = torch.unsqueeze(x, dim=1) 
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.reshape(x,(x.size(0),-1))
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


# 1DCNNsort
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(5, 30, kernel_size=5)
        self.conv2 = nn.Conv1d(30, 5, kernel_size=5)
        self.fc1 = nn.Linear(270, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 4)

             
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)
    
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, (1, 4), stride=1, padding=(0, 4 // 2), bias=False),
            nn.BatchNorm2d(8, momentum=0.01, affine=True, eps=1e-3),
            # DepthwiseConv2D
            Conv2dWithConstraint(8, 8 * 2, (21, 1), max_norm=1, stride=1, padding=(0, 0),groups=8, bias=False),
            nn.BatchNorm2d(8 * 2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=0.5))

        self.block2 = nn.Sequential(
            # SeparableConv2D =======================
            nn.Conv2d(8 * 2, 8 * 2, (1, 4), stride=1,padding=(0, 4 // 2), bias=False, groups=8 * 2),
            nn.Conv2d(8 * 2, 16, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(16, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=0.5))
        
        self.fc1 = nn.Linear(16*3, 10, bias=True)
        self.fc2 = nn.Linear(10, 4, bias=True)
        
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1) 
        x = self.block1(x)
        x = self.block2(x)       
        x = torch.reshape(x,(x.size(0),-1))
        x = self.fc1(x)
        x = self.fc2(x)
        return x



class ShallowNet(nn.Module):
    def __init__(self):
        super(ShallowNet, self).__init__()
        self.conv1 = nn.Conv1d(62, 124, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(124, 124, kernel_size=3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm1d(124)
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(124 * 12, 12*4)
        self.fc2 = nn.Linear(12*4, 4)
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = torch.relu(x)
        x = self.pooling(x)
        x = torch.reshape(x,(x.size(0),-1))
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x



class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.conv1 = nn.Conv1d(62, 93, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(93, 124, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(124, 248, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(93)
        self.batchnorm2 = nn.BatchNorm1d(124)
        self.batchnorm3 = nn.BatchNorm1d(248)
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(248 * 3, 60)
        self.fc2 = nn.Linear(60, 12)
        self.fc3 = nn.Linear(12, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.batchnorm1(x)
        x = self.pooling(x)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.batchnorm2(x)
        x = self.pooling(x)
        
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.batchnorm3(x)
        x = self.pooling(x)
        
        x = torch.reshape(x,(x.size(0),-1))
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x
    
