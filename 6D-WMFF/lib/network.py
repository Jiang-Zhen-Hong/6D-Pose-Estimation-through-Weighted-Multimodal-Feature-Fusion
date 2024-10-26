
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet
from lib.extractors import resnet34
    
class InterMA1DChannelAttention(nn.Module):
    def __init__(self,input_channels, inernal_neurons,num_points):
        super(InterMA1DChannelAttention,self).__init__()
        self.fc1 = nn.Conv1d(input_channels, inernal_neurons, 1)
        self.fc2 = nn.Conv1d(inernal_neurons, input_channels, 1)
        self.ap = torch.nn.AvgPool1d(num_points)
        self.mp = torch.nn.MaxPool1d(num_points)
        self.input_channels = input_channels
    def forward(self,inputs):
        x1 = self.ap(inputs)
        x1 = self.fc1(x1)
        x1 = F.relu(x1,inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)

        x2 = self.mp(inputs)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)

        x = x1 + x2
        x.view(-1,self.input_channels,1)
        return x

class InterMA1Dblock(nn.Module):
    def __init__(self, in_channels,out_channels,num_points,Inter_Modality_ChannelAttention_reduce = 4):
        super(InterMA1Dblock,self).__init__()
        self.conv = nn.Conv1d(in_channels,in_channels,1)
        self.intermca = InterMA1DChannelAttention(in_channels,in_channels//Inter_Modality_ChannelAttention_reduce,num_points)
        self.out_channels = out_channels
    def forward(self, inputs):
        inputs = self.conv(inputs)
        inputs = F.relu(inputs)
        channel_att_vec = self.intermca(inputs)
        outputs = channel_att_vec*inputs
        return outputs

class InterMA2DChannelAttention(nn.Module):
    def __init__(self, input_channels, inernal_neurons):
        super(InterMA2DChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=inernal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=inernal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)

        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x

class InterMA2Dblock(nn.Module):
    def __init__(self,in_channels,out_channels,channelAttention_reduce=4):
        super(InterMA2Dblock,self).__init__()
        self.ca = InterMA2DChannelAttention(input_channels=in_channels//2,
                                          inernal_neurons=in_channels //2 // channelAttention_reduce)
        self.conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=(1, 1), padding=0)
        self.out_channels = out_channels
    def forward(self, inputs):
        x = self.conv(inputs)
        x = F.relu(x)
        channel_att_vec = self.ca(x)
        channel_att_vec = torch.cat([channel_att_vec,1-channel_att_vec],1)
        outputs = channel_att_vec * inputs
        return outputs


class IntraMAChannelAttention(nn.Module):
    def __init__(self, input_channels, inernal_neurons):
        super(IntraMAChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=inernal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=inernal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)

        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x

class IntraMAblock(nn.Module):
    def __init__(self,in_channels,out_channels,channelAttention_reduce=4):
        super(IntraMAblock,self).__init__()
        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        self.ca = IntraMAChannelAttention(input_channels=in_channels,
                                       inernal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels,in_channels,kernel_size=(1,1),padding=0)
    
    def forward(self, inputs):
        inputs = self.conv(inputs)
        inputs = F.relu(inputs)

        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatital_att = self.conv(x)
        out = spatital_att * inputs
        out = self.conv(out)
        return out

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):

        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1)

class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj, dropout_prob=0.5):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.maskcnn = ModifiedResnet()
        self.depthcnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points, dropout_prob) 
        
        self.conv1_r = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1408, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj*4, 1)
        self.conv4_t = torch.nn.Conv1d(128, num_obj*3, 1)
        self.conv4_c = torch.nn.Conv1d(128, num_obj*1, 1)
        
        self.test = torch.nn.Conv1d(1536, 1408, 1)

        self.intramargb = IntraMAblock(32,32)
        self.intramamask = IntraMAblock(32,32)
        self.intramadepth = IntraMAblock(32, 32)
        self.interma2d1 = InterMA2Dblock(64, 64)
        self.interma2d2 = InterMA2Dblock(64, 64)
        self.interma1d1 = InterMA1Dblock(1536, 1536, num_points)
        self.num_obj = num_obj

    def forward(self, img, x, choose, obj, mask,nm_in):
        out_img = self.cnn(img)
        out_mask = self.maskcnn(mask)
        out_nm = self.depthcnn(nm_in)

        out_img = self.intramargb(out_img)
        out_mask = self.intramamask(out_mask)
        out_nm = self.intramadepth(out_nm)
        img_f1 = torch.cat([out_img,out_mask],1)
        img_f2 = torch.cat([out_img,out_nm],1)

        img_f1 = self.interma2d1(img_f1)
        img_f2 = self.interma2d2(img_f2)
        img_f = torch.cat([img_f1,img_f2],1)

        bs, di, _, _ = out_img.size()
        bs2, di2, _, _ = img_f.size()

        emb = out_img.view(bs, di, -1)
        img_f = img_f.view(bs2, di2, -1)

        choose1 = choose.repeat(1, di, 1)
        choose2 = choose.repeat(1,di2,1)
        emb = torch.gather(emb, 2, choose1).contiguous()
        img_f = torch.gather(img_f,2,choose2).contiguous()

        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        ap_x = torch.cat([ap_x,img_f],1)
        ap_x = self.interma1d1(ap_x)

        ap_x = F.relu(self.test(ap_x))
        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))      

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])
        out_cx = torch.index_select(cx[b], 0, obj[b])
        
        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()
        
        return out_rx, out_tx, out_cx, emb.detach(), img_f.detach()
 


class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024)
        return ap_x

class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)
        
        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj*4) 
        self.conv3_t = torch.nn.Linear(128, num_obj*3) 

        self.convimg_f = torch.nn.Conv1d(128, 128, 1)
        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.conv = torch.nn.Linear(1152,1024)
        self.num_obj = num_obj

    def forward(self, x, emb, obj, img_f):
        bs = x.size()[0]
        
        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)
        img_f = F.relu(self.convimg_f(img_f))
        img_f = self.ap1(img_f)
        img_f = img_f.view(-1,128)
        ap_x = torch.cat([ap_x,img_f],1)
        ap_x = self.conv(ap_x)
        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))   

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])

        return out_rx, out_tx
