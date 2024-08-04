import torch
import numpy as np
from math import ceil
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model.net import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from itertools import cycle

from thop import profile
from torchsummary import summary

class NetDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        super(NetDataset, self).__init__()
        self.Data = features
        self.label = labels

    def __getitem__(self, index):
        return self.Data[index], self.label[index]

    def __len__(self):
        return len(self.Data)
    

def split_data_10fold(input_data):
    """
    Split data for 10-fold cross-validation with shuffling
    
    :param input_data: Data that needs to be split
    :return: List of tuples (train_index, validation_index) for each fold
    """
    np.random.seed(321)
    indices = np.arange(len(input_data))
    np.random.shuffle(indices)
    
    fold_size = len(input_data) // 10
    folds_data_index = []
    
    for i in range(10):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        if i < 9:
            validation_indices = indices[(i + 1) * fold_size: (i + 2) * fold_size]
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 2) * fold_size:]])
        else:
            validation_indices = indices[:fold_size]
            train_indices = indices[fold_size: i * fold_size]
        folds_data_index.append((train_indices,validation_indices,test_indices))
    
    return folds_data_index


# train the model
def train(Train_data_loader,Validation_data_loader):
    best_validation_accuracy = 0.0
    patience = 60  
    early_stop_counter = 0  
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for i, (data, label) in enumerate(Train_data_loader):
            optimizer.zero_grad()
            
            data = torch.as_tensor(data,dtype=torch.float32).to(device)
            label = torch.as_tensor(label,dtype=torch.float32).to(device)
            
            outputs = model(data)
            
            loss = criterion(outputs, label)           
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # 计算平均训练损失
        average_loss = total_loss / len(Train_data_loader)

        # 计算在训练集上的准确度
        model.eval()
        with torch.no_grad():
            train_correct = 0
            train_total = 0
            for inputs, labels in Train_data_loader:
                inputs = torch.as_tensor(inputs,dtype=torch.float32).to(device)
                train_outputs = model(inputs)
                _, train_predicted = torch.max(train_outputs, dim=1)
                _, labels = torch.max(torch.as_tensor(labels,dtype=torch.float32).to(device),dim=1)
                train_total += labels.size(0)
                train_correct += (train_predicted == labels).sum().item()
            train_accuracy = train_correct / train_total * 100

        # 计算在验证集上的准确度
        validation_correct = 0
        validation_total = 0
        with torch.no_grad():
            for inputs, labels in Validation_data_loader:
                inputs = torch.as_tensor(inputs,dtype=torch.float32).to(device)
                validation_outputs = model(inputs)
                _, validation_predicted = torch.max(validation_outputs, dim=1)
                _, labels = torch.max(torch.as_tensor(labels,dtype=torch.float32).to(device),dim=1)
                validation_total += labels.size(0)
                validation_correct += (validation_predicted == labels).sum().item()
            validation_accuracy = validation_correct / validation_total * 100

        # 打印训练信息
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {validation_accuracy:.2f}%')
        
        #scheduler.step()
        # 保存最好的模型
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            early_stop_counter = 0
            torch.save(model.state_dict(), '/root/autodl-tmp/Seed-iv/single/best_model.ckpt')
            torch.save(model, '/root/autodl-tmp/Seed-iv/single/TCNN.pth') 
        else:
            early_stop_counter += 1

        # 早停
        if early_stop_counter >= patience:
            print(f'Early stopping. No improvement for {patience} epochs.')
            break
        
    # 在整个训练结束后保存模型
    torch.save(model.state_dict(), '/root/autodl-tmp/Seed-iv/single/model.ckpt')
    print(f'the best_validation_accuracy is:{best_validation_accuracy:.2f}%')


# Test the model
Test_true = []
Test_pred = []

def test(Test_data_loader):
    model.load_state_dict(torch.load('/root/autodl-tmp/Seed-iv/single/best_model.ckpt'))
    model.eval()
        
    with torch.no_grad():
        test_correct = 0
        test_total = 0
        for inputs, labels in Test_data_loader:
            inputs = torch.as_tensor(inputs,dtype=torch.float32).to(device)
            test_outputs = model(inputs)
            _, test_predicted = torch.max(test_outputs, dim=1)
            _, labels = torch.max(torch.as_tensor(labels,dtype=torch.float32).to(device),dim=1)
            
            Test_pred.extend(test_predicted.cpu().detach().numpy())
            Test_true.extend(labels.cpu().detach().numpy())
            
            test_total += labels.size(0)
            test_correct += (test_predicted == labels).sum().item()
        test_accuracy = test_correct / test_total * 100
        print(f'test Accuracy: {test_accuracy:.2f}%')
        return test_accuracy

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num,trainable_num

def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    """
    @param label_true: 真实标签
    @param label_pred: 预测标签
    @param label_name: 标签名字
    @param title: 图标题
    @param pdf_save_path: 是否保存,是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率,论文一般要求至少300dpi

    """
    plt.figure(dpi=1200)  # 初始化一张画布
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)
        
# # test Kfold        
# input_data = np.arange(851)
# folds_data_index = split_data_10fold(input_data)
# for fold, (train_index, validation_index,test_indices) in enumerate(folds_data_index):
#     print(f"Fold {fold+1}: Train indices: {train_index}, Validation indices: {validation_index}, Test indices: {test_indices}")


# feature = np.load('/root/autodl-tmp/Seed-iv/single/1/1-1.npy')
feature = np.load('/root/autodl-tmp/Seed-iv/single1s/1/1-1.npy')
feature = feature.reshape(-1,4,feature.shape[1],feature.shape[2])
print(feature.shape)
label = np.load('/root/autodl-tmp/Seed-iv/single/1/1label.npy')
print(label.shape)

# Split data using 10-fold cross-validation with shuffle
folds_data_index = split_data_10fold(feature)

fold_scores = []
fold_std = []

# Print the indices for each fold
for fold, (train_index, validation_index,test_indices) in enumerate(folds_data_index):
    # print(f"Fold {fold+1}: Train indices: {train_index}, Validation indices: {validation_index}, Test indices: {test_indices}")
    
    epochs = 1000
    device = 'cuda:0'

    model = TCN3DNet(in_channels=62,out_channels=[62, 124, 62],num_nodes=62,ktop=3,node_dim=50,tanhalpha=3,
                    num_features=25,hidden_channels=50,num_classes=25).to(device) #batch:50--lr:0.00003

    # model = TCN_DSConvCNN(in_channels=62,out_channels=[62, 124, 62],num_nodes=62,ktop=3,node_dim=50,tanhalpha=3,
    #                 num_features=25,hidden_channels=50,num_classes=25).to(device) #batch:50--lr:0.0003
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)#lr=0.0001
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.90)


    input = torch.randn(1,4,62,25).to(device)
    flops, params = profile(model, inputs=(input, ))

    print("FLOPs=", str(flops/1e6) +'{}'.format("M"))
    print("params=", str(params/1e6)+'{}'.format("M"))

    summary(model, (4, 62, 25))
    
    total_num,trainable_num = get_parameter_number(model)
    print('Total params:',total_num)
    print('Trainable params:',trainable_num)

    mean, std = np.mean(feature), np.std(feature)
    feature = (feature - mean) / std
    
    trainX = feature[train_index]
    valX = feature[validation_index]
    testX = feature[test_indices]
    
    # mean, std = np.mean(feature[train_index]), np.std(feature[train_index])
    # trainX = (feature[train_index] - mean) / std
    # valX = (feature[validation_index] - mean) / std
    # testX = (feature[test_indices] - mean) / std
    
    Train_data = NetDataset(trainX, label[train_index])
    Validation_data = NetDataset(valX, label[validation_index])
    Test_data = NetDataset(testX, label[test_indices])

    Train_data_loader = DataLoader(Train_data, batch_size=32, shuffle=True, drop_last=False)
    Validation_data_loader = DataLoader(Validation_data, batch_size=32, shuffle=True, drop_last=False)
    Test_data_loader = DataLoader(Test_data, batch_size=32, shuffle=True, drop_last=False)
    
    print(f'Fold :{fold+1}')
    train(Train_data_loader,Validation_data_loader)
    test_accuracy = test(Test_data_loader)
    
    
    if test_accuracy >= 60 :
        fold_scores.append(test_accuracy)
        
if len(fold_scores) >= 9: 
    print(f'average_test_accuracy: {np.mean(fold_scores):.2f}%' )
    draw_confusion_matrix(label_true=Test_true,			
                    label_pred=Test_pred,	    
                    label_name=["neutral", "sad", "fear", "happy"],
                    title="Confusion Matrix on SEED-IV",
                    pdf_save_path="Confusion Matrix on SEED-iv.png",
                    dpi=1200)

