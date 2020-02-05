import torch
from torch import nn
def conv3x3(in_channels):
    return nn.Conv1d(in_channels=in_channels,out_channels=in_channels*2,
                     padding=1,kernel_size=3,stride=1)
def conv1x1(in_channels,out_channels):
    return nn.Conv1d(in_channels=in_channels,out_channels=out_channels,
                     padding=0,kernel_size=1,stride=1)

def ception1(in_channels):
    return nn.Sequential(conv3x3(in_channels=in_channels),
                         nn.BatchNorm1d(in_channels*2),
                         nn.ReLU()
                         )
def conv5(in_channels):
    return nn.Conv1d(in_channels=in_channels,out_channels=in_channels*2,kernel_size=5,padding=0,stride=1)

def ception2(in_channels):
    return nn.Sequential(
        conv3x3(in_channels=in_channels),
        nn.BatchNorm1d(in_channels*2),
        nn.ReLU(),
        conv3x3(in_channels=in_channels*2),
        nn.BatchNorm1d(in_channels * 4),
        nn.ReLU()
    )
def ception3(in_channels):
    return nn.Sequential(
        conv3x3(in_channels=in_channels),
        nn.BatchNorm1d(in_channels * 2),
        nn.ReLU(),
        conv3x3(in_channels=in_channels * 2),
        nn.BatchNorm1d(in_channels * 4),
        nn.ReLU(),
        conv3x3(in_channels=in_channels * 4),
        nn.BatchNorm1d(in_channels * 8),
        nn.ReLU()
    )
def adoptblock(in_channels,out_channels):
    return nn.Sequential(
        conv1x1(in_channels=in_channels,out_channels=out_channels),
    )
def dilation(in_channels,expand):
    return nn.Conv1d(in_channels=in_channels, kernel_size=3,
                     out_channels=in_channels * 2, dilation=expand,
                     stride=1)
def dilationblock1(in_channels,expand=1):
    return nn.Sequential(
        dilation(in_channels=in_channels,expand=expand),
        dilation(in_channels=in_channels*2,expand=expand),
        dilation(in_channels=in_channels*4,expand=expand),
        nn.BatchNorm1d(in_channels * 8),
        nn.ReLU()
    )
def dilationblock2(in_channels,expand=2):
    return nn.Sequential(
        dilation(in_channels=in_channels, expand=expand),
        dilation(in_channels=in_channels*2, expand=expand),
        dilation(in_channels=in_channels*4, expand=expand),
        nn.BatchNorm1d(in_channels * 8),
        nn.ReLU()
    )
def dilationblock3(in_channels,expand=3):
    return nn.Sequential(
        dilation(in_channels=in_channels, expand=expand),
        dilation(in_channels=in_channels*2, expand=expand),
        dilation(in_channels=in_channels*4, expand=expand),
        nn.BatchNorm1d(in_channels * 8),
        nn.ReLU()
    )

class inceptionblock(nn.Module):
    def __init__(self,in_channels,output_size,num_classes,bidirectional,num_layers):
        super(inceptionblock,self).__init__()
        self.conv1=conv5(in_channels=in_channels)
        self.conv2=conv5(in_channels=in_channels*2)
        self.channel11=ception1(in_channels=in_channels*4)
        self.channel12 = ception2(in_channels=in_channels*4)
        self.channel13 = ception3(in_channels=in_channels*4)
        self.adjust1=conv1x1(in_channels=in_channels*4,out_channels=in_channels*4*14)
        self.adopt1=adoptblock(in_channels=in_channels*14*4,out_channels=in_channels*8*4)
        self.channel21=ception1(in_channels=in_channels*8*4)
        self.channel22 = ception2(in_channels=in_channels*8*4)
        self.channel23 = ception3(in_channels=in_channels*8*4)
        self.adjust2=conv1x1(in_channels=in_channels*8*4,out_channels=in_channels*8*14*4)
        self.adopt2=adoptblock(in_channels=in_channels*8*14*4,out_channels=in_channels*8*8)
        self.dila11 = dilationblock1(in_channels=in_channels)
        self.dila12 = dilationblock2(in_channels=in_channels)
        self.dila13 = dilationblock3(in_channels=in_channels)
        self.dila21 = dilationblock1(in_channels=in_channels*8)
        self.dila22 = dilationblock2(in_channels=in_channels*8)
        self.dila23 = dilationblock3(in_channels=in_channels*8)
        self.RoIpooling1=torch.nn.AdaptiveMaxPool1d(output_size=output_size)
        self.RoIpooling21 = torch.nn.AdaptiveMaxPool1d(output_size=output_size)
        self.RoIpooling22 = torch.nn.AdaptiveMaxPool1d(output_size=output_size)
        self.RoIpooling23 = torch.nn.AdaptiveMaxPool1d(output_size=output_size)
        self.RNNLayer = torch.nn.LSTM(input_size=1536, hidden_size=output_size,
                                      batch_first=True, num_layers=num_layers,
                                      dropout=0.3, bidirectional=bidirectional)
        if bidirectional:
            self.FC1 = torch.nn.Linear(num_layers * 2 * output_size, 256)
        else:
            self.FC1 = torch.nn.Linear(num_layers * output_size, 256)
        self.Drop1 = torch.nn.Dropout(p=0.3)
        self.Output = torch.nn.Linear(256, num_classes)
        self.action = torch.nn.Softmax(dim=1)

    def forward(self,x):
        dilation1=self.dila21(self.dila11(x))
        dilation2 = self.dila22(self.dila12(x))
        dilation3 = self.dila23 (self.dila13(x))
        dilationout = torch.cat((self.RoIpooling21(dilation1),
                                 self.RoIpooling22(dilation2),
                                 self.RoIpooling23(dilation3)), dim=1)
        x=self.conv2(self.conv1(x))
        residule=self.adjust1(x)
        output=torch.cat((self.channel11(x),self.channel12(x),self.channel13(x)),dim=1)
        output=output+residule
        output=self.adopt1(output)
        residule=self.adjust2(output)
        output = torch.cat((self.channel21(output), self.channel22(output), self.channel23(output)), dim=1)
        output = output + residule
        output = self.adopt2(output)
        output=self.RoIpooling1(output)
        RNNinput=torch.cat((output,dilationout),dim=1).permute(0, 2, 1)
        _, (h0, c0) = self.RNNLayer(RNNinput)
        out = h0.permute(1, 0, 2).contiguous()
        batch_size=out.size(0)
        out=out.view(batch_size, -1)
        output = self.action(self.Output(self.Drop1(self.FC1(out))))
        return output