import torch
import torch.nn as nn
import math

from layers.WindowListSplit import splitWindowList, countResultWindows


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)


'''一维卷积块'''
class Conv1dBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super(Conv1dBlock, self).__init__()
        self.conv1d0 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1d0(x)
        return x
class RNNBlock(nn.Module):
    '''输入为Conv1dBlock卷积后的结果'''
    def __init__(self,sequenceDim,hidR):
        super(RNNBlock, self).__init__()
        self.gru = nn.GRU(sequenceDim,hidR,batch_first=True);# 默认gru第一个维度是时间步 batch_first设置第一个是batch
    def forward(self, x):
        x = x.permute(0, 2, 1)  # 变为 批量*通道维度，时间长，子序列窗口(conv的输出通道数目) 方便后续
        _,hn = self.gru(x)
        return hn
class ConvRNNBlock(nn.Module):
    '''输入为Conv1dBlock卷积后的结果'''
    def __init__(self,in_channels,out_channels,kernel_size,stride,hidR):
        super(ConvRNNBlock, self).__init__()
        self.conv1d = Conv1dBlock(in_channels, out_channels, kernel_size,stride)
        self.gru = RNNBlock(out_channels,hidR);
        self.in_channels = in_channels
        self.out_channels = out_channels
    def forward_conv(self,x):
        conv_output = self.conv1d(x)
        return conv_output
    def forward_rnn(self,x):
        rnn_output = self.gru(x)
        return rnn_output

    def forward(self, x):
        conv_output = self.conv1d(x)
        rnn_output = self.gru(conv_output)
        return conv_output,rnn_output

    def getOutChannels(self):
        return self.out_channels
    def getInChannels(self):
        return self.in_channels


import torch.nn.functional as F
'''卷积链之间不共享卷积层,集成由粗至细的'''
class PyramidalRNNEmbedding(nn.Module):
    def __init__(self, windows, d_model,rnnMixTemperature, dropout=0.1):
        super(PyramidalRNNEmbedding, self).__init__()
        # 构建多个卷积链
        split_windows_lists = splitWindowList(windows) #返回划分的倍数列表和最大公约数，每一个倍数有序链表构造一条卷积 # 24 48 72 96 144
        self.split_windows_lists = split_windows_lists # 储存供后边使用
        window_count = countResultWindows(split_windows_lists)
        self.rateParameter = nn.Parameter(torch.ones((window_count,1))/window_count)
        self.temperature = rnnMixTemperature # 通过命令行统一设置
        # 子序列编码
        self.hidRDict = {}
        for window in windows:
            self.hidRDict[window] = 0
        # 对每个窗口的卷积块进行计数
        for sub_windows in split_windows_lists:
            for window in sub_windows:
                self.hidRDict[window]+=1
        # 分配每个卷积窗口的隐层个数
        for window in windows:
            tmpHidR = d_model//len(windows)
            self.hidRDict[window] = int(tmpHidR/self.hidRDict[window]) # 同样的窗口 多个评分hidR 让不同窗口的hidR平均


        moduleLists  = []
        conv_channels = 128 #统一的卷积channel
        for window_list in split_windows_lists:
            in_channels = 1
            convRNNBlocks = []
            for i in range(len(window_list)):
                out_channels = window_list[i] # window 每一层卷积的输出通道数等于窗口大小
                # out_channels = 24 # 设置统一的通道数 方便反卷积
                kernel = int(out_channels/in_channels) # kernel等于与前一个窗口的倍数
                stride = kernel
                hidR = self.hidRDict[out_channels] # window
                if in_channels==1:
                    tmpConvRNNBlock = ConvRNNBlock(1, conv_channels, kernel, stride,hidR)
                else:
                    tmpConvRNNBlock = ConvRNNBlock(conv_channels, conv_channels, kernel, stride,hidR)
                convRNNBlocks.append(tmpConvRNNBlock)
                in_channels = out_channels
            convRnns = nn.ModuleList(convRNNBlocks)
            moduleLists.append(convRnns)
        self.nnModuleLists = nn.ModuleList(moduleLists)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.outLinear = nn.Linear(d_model,d_model)
    def forward(self, x, x_mark):
        # x: [Batch Variate Time]
        batchSize = x.shape[0]
        timeLength = x.shape[1]
        channel = x.shape[2]
        if x_mark is None:
            x = x.permute(0, 2, 1).reshape(batchSize * channel, 1, timeLength)
        else:
            x = torch.cat([x.permute(0, 2, 1), x_mark.permute(0, 2, 1)],dim=1)
            channel+=x_mark.shape[2]
            x = x.reshape(batchSize * channel, 1, timeLength)

        hns = [] # 保存所有卷积rnn链的结果
        #对每个卷积链进行操作，得到rnn表示
        for i,moduleList in enumerate(self.nnModuleLists):
            convMap = {} #临时存储卷积结果
            tmpX = x
            window_list = self.split_windows_lists[i]
            for  j in range(len(moduleList)):
                tmpConvRNNBlock = moduleList.__getitem__(j)
                tmpX = tmpConvRNNBlock.forward_conv(tmpX)
                # window = tmpConvRNNBlock.getOutChannels() #这样获取不到窗口大小
                window = window_list[j]
                convMap[window] = tmpX
            # 上采样融合
            lastWindow = None
            upScaleConv = tmpX
            for j in range(len(window_list)-1,-1,-1):
                tmpConvRNNBlock = moduleList.__getitem__(j)
                window = window_list[j]
                if lastWindow is not None: #从第二个开始采样
                    scale_factor = lastWindow/window
                    upScaleConv = F.upsample(upScaleConv,scale_factor=scale_factor)# ,mode='bilinear'不能直接用于三维
                    cha = upScaleConv.shape[-1] - convMap[window].shape[-1]
                    if cha !=0: #对于奇数等情况 补上上采样确实的维度
                        upScaleConv = torch.cat([upScaleConv,upScaleConv[:, :, -1:]],dim=-1)
                    rnn_input = upScaleConv + convMap[window]
                    del convMap[window] #释放内存
                else:
                    rnn_input = upScaleConv
                tmpHn = tmpConvRNNBlock.forward_rnn(rnn_input)
                hidR = self.hidRDict[window]  # window 获取window对应的隐层数量
                tmpHn = tmpHn.reshape(batchSize, channel, hidR)
                hns.append(tmpHn)

                lastWindow = window
        rate = torch.softmax(self.rateParameter/self.temperature,-1)  #
        # hns = [(hni*rate[i]) for i, hni in enumerate(list(hns.values()))]
        hns = [(hni*rate[i]) for i, hni in enumerate(hns)]
        hns = torch.cat(hns,dim=-1)
        hns = self.outLinear(hns) #类似多头注意力的线性层
        return hns

