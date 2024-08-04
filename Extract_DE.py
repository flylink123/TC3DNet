import os
import math
import numpy as np
import scipy.io
from scipy.signal import butter, lfilter,cheby2, filtfilt

def butter_bandpass_filter(data, lowcut, highcut, samplingRate, order=4):
	nyq = 0.5 * samplingRate
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	y = lfilter(b, a, data)
	return y

def cheby2_bandpass_filter(data, lowcut, highcut, samplingRate, order=4, rp=1, rs=40):
    # 计算归一化截止频率
    nyq = 0.5 * samplingRate
    low = lowcut / nyq
    high = highcut / nyq
    # 设计Chebyshev Type II滤波器
    b, a = cheby2(order, rs, [low, high], btype='bandpass')
    # 应用滤波器
    filtered_data = filtfilt(b, a, data)
    return filtered_data


# def var(data):
#     return np.mean([math.pow(x - np.mean(data), 2) for x in data])

# 微分熵计算
def compute_DE(data):
    variance = np.var(data,ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


def Extract_DE(data, num_bands=25, N=4):
    frequency = 200
    
    n = data.shape[1]
    # 计算裁剪后的数量 k
    k = n // (frequency*N)
    cropped_n = k * (frequency*N)  # 确保能够整除800

    # 裁剪数据
    cropped_data = data[:, :cropped_n, :]
    cropped_data = cropped_data.transpose((2,0,1))
    
    
    trials = cropped_data.shape[0]
    channels = cropped_data.shape[1]
    samples = cropped_data.shape[2]
    
    
    # 采样点计算一个微分熵
    num_sample = int(samples/(frequency*N))
    
    # 微分熵特征
    DE_Characteristics = np.empty([0, num_bands, num_sample])
        
    for trial in range(trials):
        trail_single = cropped_data[trial, :]
        
        trail_de = np.empty([0, num_sample])
        
        for channel in range(channels):
            channel_single = trail_single[channel, :]
            
            channel_de = np.empty([0, num_sample])

            for band_index in range(num_bands):
                low_freq = band_index * 2
                high_freq = (band_index + 1) * 2
                
                # Apply bandpass filter
                channel_band = butter_bandpass_filter(channel_single, low_freq+0.01, high_freq+0.01, frequency)
                
                # Calculate DE for this band
                DE_band = np.zeros(shape=[0], dtype=float)
                for index in range(num_sample):
                    DE_band = np.append(DE_band, compute_DE(channel_band[index *frequency*N: (index + 1) *frequency*N])) 
                channel_de = np.concatenate((channel_de, DE_band.reshape(1, -1)), axis=0) 
            
            trail_de = np.vstack((trail_de, channel_de))
        
        trail_de = trail_de.reshape(channels, num_bands, num_sample)
        DE_Characteristics = np.vstack((DE_Characteristics, trail_de))
    
    DE_Characteristics = DE_Characteristics.reshape(trials, channels, num_bands, num_sample)
    
    return DE_Characteristics.transpose(0, 3, 1, 2)


def label_to_one_hot(label_list):
    num_classes = 4 
    num_samples = len(label_list)  # 获取样本数

    one_hot = np.zeros((num_samples, num_classes))  # 创建全零矩阵，用于存储独热编码
    
    for i, label in enumerate(label_list):
        one_hot[i, label] = 1  # 将对应标签位置置为1
    
    return one_hot

def GetFeatureLabel(folder_path,session_label):
    # 创建字典来存储后缀相同的变量
    merged_variables = {f"eeg{i}": [] for i in range(1, 25)}
    
    DE_feature = None
    label_arrays = []
    
    file_cnt = 0 
    
    all_mat_file = os.walk(folder_path)
    for path, dir_list, file_list in all_mat_file:
        for file_name in file_list:
            file_cnt += 1
            print('当前已处理到{}，总进度{}/{}'.format(file_name, file_cnt, len(file_list)))
            raw_mat = scipy.io.loadmat(os.path.join(folder_path, file_name))
            # 获取除了默认变量外的其他变量名列表
            var_names = [var for var in raw_mat if not var.startswith('__') and not var.endswith('__')]
            for variable in var_names:
                # 遍历每个变量名
                parts = variable.split('_')
                if len(parts) == 2 and parts[1].startswith('eeg'):
                    try:
                        eeg_num = int(parts[1][3:])  # 获取数字部分
                        if 1 <= eeg_num <= 24:
                            merged_variables[parts[1]].append(raw_mat[variable])
                    except ValueError:
                        pass
                    
    #merge session1RawData and label                 
    label_index = 0
    num_bands = 25
    channels = 62
    N = 4
    
    for var_suffix, var_values in merged_variables.items():
        feature = Extract_DE(np.dstack(var_values))
        #merged_variables[var_suffix] = feature
        feature = feature.reshape(-1,channels,num_bands)
        print(var_suffix,'has processed done')
        
        label_arrays.append(np.full(shape=[1,feature.shape[0]],fill_value=session_label[label_index]))
        label_index = label_index + 1
        
        if DE_feature is None:
            DE_feature = feature
        else:
            DE_feature = np.concatenate((DE_feature, feature), axis=0)
          
            
    print('future.shape',DE_feature.shape) 
    
    label_arrays = np.concatenate(label_arrays, axis=1)
    label_arrays = label_to_one_hot(np.squeeze(label_arrays))
    print('label_arrays.shape',label_arrays.shape)
    
    return DE_feature,label_arrays



#feature_folder_path
folder_path1 = 'E:/SEED_IV/SEED_IV/eeg_raw_data/1'
folder_path2 = 'E:/SEED_IV/SEED_IV/eeg_raw_data/2'
folder_path3 = 'E:/SEED_IV/SEED_IV/eeg_raw_data/3'

#label 
session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]

sessionfeature1,label_arrays1 = GetFeatureLabel(folder_path1,session1_label)
sessionfeature2,label_arrays2 = GetFeatureLabel(folder_path2,session2_label)
sessionfeature3,label_arrays3 = GetFeatureLabel(folder_path3,session3_label)

feature = np.concatenate((sessionfeature1, sessionfeature2,sessionfeature3), axis=0)
label = np.concatenate((label_arrays1, label_arrays2, label_arrays3), axis=0)

print('feature.shape:',feature.shape)
print('label.shape:',label.shape)

np.save('seed_iv_feature_25hzc4.npy',feature)
np.save('seed_iv_label_25hzc4.npy',label)