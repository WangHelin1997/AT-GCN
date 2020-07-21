import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import pickle
import numpy as np
import math
from torch.nn import Parameter
from torchlibrosa.augmentation import SpecAugmentation
from mul_att import *


class S_Attention(nn.Module):
    def __init__(self, in_channels):
        super(S_Attention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        y = self.pool(input)
        y = y.view(y.size(0), y.size(1))
        y = self.fc(y)
        y = y.view(y.size(0), y.size(1), 1, 1)
        y = y * input
        return y


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    # p is set to 0.2
    _adj = _adj * 0.2 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj


def gen_A_on(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    for i in range(0, 72):
        for j in range(0, 72):
            _adj[i, j] = 1.
    for i in range(72, 137):
        for j in range(72, 137):
            _adj[i, j] = 1.
    for i in range(137, 285):
        for j in range(137, 285):
            _adj[i, j] = 1.
    for i in range(285, 300):
        for j in range(285, 300):
            _adj[i, j] = 1.
    for i in range(300, 466):
        for j in range(200, 466):
            _adj[i, j] = 1.
    for i in range(466, 506):
        for j in range(466, 506):
            _adj[i, j] = 1.
    for i in range(506, 527):
        for j in range(506, 527):
            _adj[i, j] = 1.
    for i in range(527):
        _adj[i, i] = 0.
    # p is set to 0.2
    _adj = _adj * 0.2 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class CNN10(nn.Module):
    def __init__(self, args):
        super(CNN10, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock2(1, 64)
        self.conv_block2 = ConvBlock2(64, 128)
        self.conv_block3 = ConvBlock2(128, 256)
        self.conv_block4 = ConvBlock2(256, 512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, 527, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x):
        x = x.view((-1, 1, x.size(1), x.size(2)))
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        a = x1 + x2
        a = F.dropout(a, p=0.5, training=self.training)
        embedding = F.relu_(self.fc1(a))
        a = F.dropout(embedding, p=0.5, training=self.training)
        global_prob = torch.sigmoid(self.fc_audioset(a))
        global_prob = torch.clamp(global_prob, 1e-7, 1 - 1e-7)

        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        embedding = F.relu_(self.fc1(x))
        x = F.dropout(embedding, p=0.5, training=self.training)
        frame_prob = torch.sigmoid(self.fc_audioset(x))
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)
        #         global_prob = frame_prob.mean(dim = 1)
        return global_prob, frame_prob

    def predict(self, x, verbose=True, batch_size=100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i: i + batch_size])).cuda()
                output = self.forward(input)
                if not verbose: output = output[:1]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        return result if verbose else result[0]


class CRNN10(nn.Module):
    def __init__(self, args):
        super(CRNN10, self).__init__()

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=50, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock2(1, 64)
        self.conv_block2 = ConvBlock2(64, 128)
        self.conv_block3 = ConvBlock2(128, 256)
        self.conv_block4 = ConvBlock2(256, 512)
        self.rnn = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True, dropout=0.3,
                           bidirectional=True)

        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.fc_audioset = nn.Linear(1024, 527, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x, aug=True):
        x = x.view((-1, 1, x.size(1), x.size(2)))
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and aug:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        x = x.transpose(1, 2)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        a = x1 + x2
        a = F.dropout(a, p=0.5, training=self.training)
        embedding = F.relu_(self.fc1(a))
        a = F.dropout(embedding, p=0.5, training=self.training)
        global_prob = torch.sigmoid(self.fc_audioset(a))
        global_prob = torch.clamp(global_prob, 1e-7, 1 - 1e-7)

        return global_prob, global_prob

    def predict(self, x, verbose=True, batch_size=100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i: i + batch_size])).cuda()
                output = self.forward(input)
                if not verbose: output = output[:1]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        return result if verbose else result[0]

class semantic(nn.Module):
    def __init__(self, num_classes, audio_feature_dim, word_feature_dim, intermediary_dim=1024):
        super(semantic, self).__init__()
        self.num_classes = num_classes
        self.audio_feature_dim = audio_feature_dim
        self.word_feature_dim = word_feature_dim
        self.intermediary_dim = intermediary_dim
        self.fc_1 = nn.Linear(self.audio_feature_dim, self.intermediary_dim, bias=False)
        self.fc_2 = nn.Linear(self.word_feature_dim, self.intermediary_dim, bias=False)
        self.fc_3 = nn.Linear(self.intermediary_dim, self.intermediary_dim)
        self.fc_a = nn.Linear(self.intermediary_dim, 1)

    def forward(self, audio_feature_map, word_features):
        # bs,T,C
        batch_size = audio_feature_map.size()[0]
        convsize = audio_feature_map.size()[1]

        f_wh_feature = audio_feature_map.contiguous().view(batch_size*convsize, -1)
        f_wh_feature = self.fc_1(f_wh_feature).view(batch_size*convsize, 1, -1).repeat(1, self.num_classes, 1)

        f_wd_feature = self.fc_2(word_features).view(1, self.num_classes, 1024).repeat(batch_size*convsize,1,1)
        lb_feature = self.fc_3(torch.tanh(f_wh_feature*f_wd_feature).view(-1,1024))
        coefficient = self.fc_a(lb_feature)
        coefficient = torch.transpose(coefficient.view(batch_size, convsize, self.num_classes),1,2)

        coefficient = F.softmax(coefficient, dim=2)
        coefficient = torch.transpose(coefficient,1,2)
        coefficient = coefficient.view(batch_size, convsize, self.num_classes, 1).repeat(1,1,1,self.audio_feature_dim)
        audio_feature_map = audio_feature_map.view(batch_size, convsize, 1, self.audio_feature_dim).repeat(1, 1, self.num_classes, 1)* coefficient
        graph_net_input = torch.sum(audio_feature_map,1)
        return graph_net_input
class GCNN(nn.Module):
    def __init__(self, num_classes=527, in_channel=1024, t=0.3, adj_file=None):
        super(GCNN, self).__init__()
        self.num_classes = num_classes
        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 1024)
        self.relu = nn.LeakyReLU(0.2)
        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

    def forward(self, feature):
        adj = gen_adj(self.A).detach()
        x = self.gc1(feature, adj)
        x = self.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.gc2(x, adj)
        x = self.relu(x)
        return x
class Element_Wise_Layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Element_Wise_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        for i in range(self.in_features):
            self.weight[i].data.uniform_(-stdv, stdv)
        if self.bias is not None:
            for i in range(self.in_features):
                self.bias[i].data.uniform_(-stdv, stdv)
    def forward(self, input):
        x = input * self.weight
        x = torch.sum(x,2)
        if self.bias is not None:
            x = x + self.bias
        return x
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)
class CRNN10_GCN(nn.Module):
    def __init__(self, args):
        super(CRNN10_GCN, self).__init__()

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=50, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock2(1, 64)
        self.conv_block2 = ConvBlock2(64, 128)
        self.conv_block3 = ConvBlock2(128, 256)
        self.conv_block4 = ConvBlock2(256, 512)
        self.rnn = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True, dropout=0.3,
                           bidirectional=True)

        self.word_semantic = semantic(num_classes=527,
                                      audio_feature_dim=1024,
                                      word_feature_dim=300)
        with open('audioset_glove_word2vec.pkl', 'rb') as f:
            self.inp = pickle.load(f)
        self.graph_net = GCNN(num_classes=527, in_channel=1024, t=0.3, adj_file='audioset_adj.pkl')
        self.fc_output = nn.Linear(2048, 1024)
        self.classifiers = Element_Wise_Layer(527, 1024)
        self.init_model(
            model_pth='/data/dean/cmu-thesis/workspace/audioset/CRNN10_kd-batch128-ckpt1200-adam-lr1e-03-pat2-fac0.9-seed15213/model/checkpoint149.pt')
        self.init_weight()

    def init_model(self, model_pth=None):
        pre_model = CRNN10(args=None)
        pre_model.load_state_dict(torch.load(model_pth)['model'])
        for name, module in pre_model._modules.items():
            if name == 'bn0':
                self.bn0 = module
            if name == 'conv_block1':
                self.conv_block1 = module
            if name == 'conv_block2':
                self.conv_block2 = module
            if name == 'conv_block3':
                self.conv_block3 = module
            if name == 'conv_block4':
                self.conv_block4 = module
            if name == 'rnn':
                self.rnn = module

    def forward(self, x, aug=True):
        x = x.view((-1, 1, x.size(1), x.size(2)))
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        batch_size = x.size()[0]
        if self.training and aug:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        graph_net_input = self.word_semantic(x,torch.tensor(self.inp).cuda())
        graph_net_feature = self.graph_net(graph_net_input)
        output = torch.cat((graph_net_feature.view(batch_size * 527, -1),
                            graph_net_input.view(-1, 1024)), 1)
        output = self.fc_output(output)
        output = torch.tanh(output)
        output = output.contiguous().view(batch_size, 527, 1024)
        output = self.classifiers(output)
        global_prob = torch.sigmoid(output)
        global_prob = torch.clamp(global_prob, 1e-7, 1 - 1e-7)

        return global_prob, global_prob

    def predict(self, x, verbose=True, batch_size=100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i: i + batch_size])).cuda()
                output = self.forward(input)
                if not verbose: output = output[:1]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        return result if verbose else result[0]

class CRNN10_SED(nn.Module):
    def __init__(self, args):
        super(CRNN10_SED, self).__init__()
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=50, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock2(1, 64)
        self.conv_block2 = ConvBlock2(64, 128)
        self.conv_block3 = ConvBlock2(128, 256)
        self.conv_block4 = ConvBlock2(256, 512)
        self.rnn = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True, dropout=0.3,
                           bidirectional=True)
        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.fc_prob = nn.Linear(1024, 527)
        self.pooling = 'lin'
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_prob)

    def forward(self, x, aug=True):
        x = x.view((-1, 1, x.size(1), x.size(2)))
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and aug:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        x = F.dropout(x, p=0.5, training=self.training)
        embedding = F.relu_(self.fc1(x))
        x = F.dropout(embedding, p=0.5, training=self.training)
        frame_prob = torch.sigmoid(self.fc_prob(x))
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)

        if self.pooling == 'max':
            global_prob, _ = frame_prob.max(dim=1)
            return global_prob, frame_prob
        elif self.pooling == 'ave':
            global_prob = frame_prob.mean(dim=1)
            return global_prob, frame_prob
        elif self.pooling == 'lin':
            global_prob = (frame_prob * frame_prob).sum(dim=1) / frame_prob.sum(dim=1)
            return global_prob, frame_prob
        elif self.pooling == 'exp':
            global_prob = (frame_prob * frame_prob.exp()).sum(dim=1) / frame_prob.exp().sum(dim=1)
            return global_prob, frame_prob

    def predict(self, x, verbose=True, batch_size=100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i: i + batch_size])).cuda()
                output = self.forward(input)
                if not verbose: output = output[:1]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        return result if verbose else result[0]


class CNN10_MAT(nn.Module):
    def __init__(self, args):
        super(CNN10_MAT, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock2(1, 64)
        self.conv_block2 = ConvBlock2(64, 128)
        self.conv_block3 = ConvBlock2(128, 256)
        self.conv_block4 = ConvBlock2(256, 512)
        self.pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=25)
        self.mh1 = MultiHeadAttention(model_dim=512, num_heads=8, output_dim=512, dropout=0.2, share_weight=False)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, 527, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x):
        x = x.view((-1, 1, x.size(1), x.size(2)))
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        x = x.transpose(1, 2)
        x = self.pe(x)
        x, _, _ = self.mh1(x, x, x)
        x = x.transpose(1, 2)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        a = x1 + x2
        a = F.dropout(a, p=0.5, training=self.training)
        embedding = F.relu_(self.fc1(a))
        a = F.dropout(embedding, p=0.5, training=self.training)
        global_prob = torch.sigmoid(self.fc_audioset(a))
        global_prob = torch.clamp(global_prob, 1e-7, 1 - 1e-7)

        return global_prob, global_prob

    def predict(self, x, verbose=True, batch_size=100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i: i + batch_size])).cuda()
                output = self.forward(input)
                if not verbose: output = output[:1]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        return result if verbose else result[0]


class CNN14_MIL(nn.Module):
    def __init__(self, args):
        super(CNN14_MIL, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock2(1, 64)
        self.conv_block2 = ConvBlock2(64, 128)
        self.conv_block3 = ConvBlock2(128, 256)
        self.conv_block4 = ConvBlock2(256, 512)
        self.conv_block5 = ConvBlock2(512, 1024)
        self.conv_block6 = ConvBlock2(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, 527, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x):
        x = x.view((-1, 1, x.size(1), x.size(2)))
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        embedding = F.relu_(self.fc1(x))
        x = F.dropout(embedding, p=0.5, training=self.training)
        frame_prob = torch.sigmoid(self.fc_audioset(x))
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)
        global_prob = frame_prob.mean(dim=1)
        return global_prob, frame_prob

    def predict(self, x, verbose=True, batch_size=100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i: i + batch_size])).cuda()
                output = self.forward(input)
                if not verbose: output = output[:1]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        return result if verbose else result[0]


class CNN14(nn.Module):
    def __init__(self, args):
        super(CNN14, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock2(1, 64)
        self.conv_block2 = ConvBlock2(64, 128)
        self.conv_block3 = ConvBlock2(128, 256)
        self.conv_block4 = ConvBlock2(256, 512)
        self.conv_block5 = ConvBlock2(512, 1024)
        self.conv_block6 = ConvBlock2(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, 527, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x):
        x = x.view((-1, 1, x.size(1), x.size(2)))
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        a = x1 + x2
        a = F.dropout(a, p=0.5, training=self.training)
        embedding = F.relu_(self.fc1(a))
        a = F.dropout(embedding, p=0.5, training=self.training)
        global_prob = torch.sigmoid(self.fc_audioset(a))
        global_prob = torch.clamp(global_prob, 1e-7, 1 - 1e-7)

        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        embedding = F.relu_(self.fc1(x))
        x = F.dropout(embedding, p=0.5, training=self.training)
        frame_prob = torch.sigmoid(self.fc_audioset(x))
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)
        #         global_prob = frame_prob.mean(dim = 1)
        return global_prob, frame_prob

    def predict(self, x, verbose=True, batch_size=100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i: i + batch_size])).cuda()
                output = self.forward(input)
                if not verbose: output = output[:1]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        return result if verbose else result[0]


class CNN10_MIL(nn.Module):
    def __init__(self, args):
        super(CNN10_MIL, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock2(1, 64)
        self.conv_block2 = ConvBlock2(64, 128)
        self.conv_block3 = ConvBlock2(128, 256)
        self.conv_block4 = ConvBlock2(256, 512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, 527, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x):
        x = x.view((-1, 1, x.size(1), x.size(2)))
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        embedding = F.relu_(self.fc1(x))
        x = F.dropout(embedding, p=0.5, training=self.training)
        frame_prob = torch.sigmoid(self.fc_audioset(x))
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)
        global_prob = frame_prob.mean(dim=1)
        return global_prob, frame_prob

    def predict(self, x, verbose=True, batch_size=100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i: i + batch_size])).cuda()
                output = self.forward(input)
                if not verbose: output = output[:1]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        return result if verbose else result[0]


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNnet(nn.Module):
    def __init__(self, num_classes=527, in_channel=300, t=0.3, adj_file=None):
        super(GCNnet, self).__init__()
        self.num_classes = num_classes

        self.gc1 = GraphConvolution(in_channel, 256)
        self.gc2 = GraphConvolution(256, 512)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

    def forward(self, feature, inp):
        # inp = inp[0]

        adj = gen_adj(self.A).detach()
        inp = torch.tensor(inp).cuda()
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gc2(x, adj)
        cla = x
        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x, cla


class GCNnet_on(nn.Module):
    def __init__(self, num_classes=527, in_channel=300, t=0.3, adj_file=None):
        super(GCNnet_on, self).__init__()
        self.num_classes = num_classes

        self.gc1 = GraphConvolution(in_channel, 256)
        self.gc2 = GraphConvolution(256, 512)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A_on(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

    def forward(self, feature, inp):
        # inp = inp[0]

        adj = gen_adj(self.A).detach()
        inp = torch.tensor(inp).cuda()
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gc2(x, adj)
        cla = x
        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x, cla


class GCNnet_grad(nn.Module):
    def __init__(self, num_classes=527, in_channel=1024, t=0.3, adj_file=None):
        super(GCNnet_grad, self).__init__()
        self.num_classes = num_classes

        self.gc1 = GraphConvolution(in_channel, 512)
        self.gc2 = GraphConvolution(512, 1024)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

    def forward(self, feature, inp):
        # inp = inp[0]

        adj = gen_adj(self.A).detach()
        inp = torch.tensor(inp).cuda()
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gc2(x, adj)
        cla = x
        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x, cla


class gcnNet(nn.Module):
    def __init__(self, args):
        super(gcnNet, self).__init__()

        #         self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
        #             freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock2(1, 64)
        self.conv_block2 = ConvBlock2(64, 128)
        self.conv_block3 = ConvBlock2(128, 256)
        self.conv_block4 = ConvBlock2(256, 512)
        self.fc1 = nn.Linear(512, 512, bias=True)

        self.init_model(
            model_pth='/data/dean/cmu-thesis/workspace/audioset/CNN10-batch128-ckpt1000-adam-lr1e-03-pat2-fac0.9-seed15213/model/checkpoint151.pt')

        self.gcn = GCNnet(num_classes=527, in_channel=300, t=0.3, adj_file='audioset_adj.pkl')

        with open('audioset_glove_word2vec.pkl', 'rb') as f:
            self.inp = pickle.load(f)

    def init_model(self, model_pth=None):
        pre_model = CNN10(args=None)
        pre_model.load_state_dict(torch.load(model_pth)['model'])
        for name, module in pre_model._modules.items():
            if name == 'bn0':
                self.bn0 = module
            if name == 'conv_block1':
                self.conv_block1 = module
            if name == 'conv_block2':
                self.conv_block2 = module
            if name == 'conv_block3':
                self.conv_block3 = module
            if name == 'conv_block4':
                self.conv_block4 = module
            if name == 'fc1':
                self.fc1 = module

    def forward(self, x):
        x = x.view((-1, 1, x.size(1), x.size(2)))
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x, cla = self.gcn(x, self.inp)
        global_prob = torch.sigmoid(x)
        global_prob = torch.clamp(global_prob, 1e-7, 1 - 1e-7)

        return global_prob, cla

    def predict(self, x, verbose=True, batch_size=100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i: i + batch_size])).cuda()
                output = self.forward(input)
                if not verbose: output = output[:1]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        return result if verbose else result[0]


class gcnNet_on(nn.Module):
    def __init__(self, args):
        super(gcnNet_on, self).__init__()

        #         self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
        #             freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock2(1, 64)
        self.conv_block2 = ConvBlock2(64, 128)
        self.conv_block3 = ConvBlock2(128, 256)
        self.conv_block4 = ConvBlock2(256, 512)
        self.fc1 = nn.Linear(512, 512, bias=True)

        self.init_model(
            model_pth='/data/dean/cmu-thesis/workspace/audioset/CNN10-batch128-ckpt1000-adam-lr1e-03-pat2-fac0.9-seed15213/model/checkpoint151.pt')

        self.gcn = GCNnet_on(num_classes=527, in_channel=300, t=0.3, adj_file='audioset_adj.pkl')

        with open('audioset_glove_word2vec.pkl', 'rb') as f:
            self.inp = pickle.load(f)

    def init_model(self, model_pth=None):
        pre_model = CNN10(args=None)
        pre_model.load_state_dict(torch.load(model_pth)['model'])
        for name, module in pre_model._modules.items():
            if name == 'bn0':
                self.bn0 = module
            if name == 'conv_block1':
                self.conv_block1 = module
            if name == 'conv_block2':
                self.conv_block2 = module
            if name == 'conv_block3':
                self.conv_block3 = module
            if name == 'conv_block4':
                self.conv_block4 = module
            if name == 'fc1':
                self.fc1 = module

    def forward(self, x):
        x = x.view((-1, 1, x.size(1), x.size(2)))
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x, cla = self.gcn(x, self.inp)
        global_prob = torch.sigmoid(x)
        global_prob = torch.clamp(global_prob, 1e-7, 1 - 1e-7)

        return global_prob, cla

    def predict(self, x, verbose=True, batch_size=100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i: i + batch_size])).cuda()
                output = self.forward(input)
                if not verbose: output = output[:1]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        return result if verbose else result[0]


class GCN_grad(nn.Module):
    def __init__(self, args, fixed=False):
        super(GCN_grad, self).__init__()

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=50, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock2(1, 64)
        self.conv_block2 = ConvBlock2(64, 128)
        self.conv_block3 = ConvBlock2(128, 256)
        self.conv_block4 = ConvBlock2(256, 512)
        self.rnn = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True, dropout=0.3,
                           bidirectional=True)

        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.init_model(
            model_pth='/data/dean/cmu-thesis/workspace/audioset/CRNN10_aug-batch128-ckpt1200-adam-lr1e-03-pat2-fac0.9-seed15213/model/checkpoint151.pt')

        if fixed == True:
            for p in self.parameters():
                p.requires_grad = False

        self.gcn = GCNnet_grad(num_classes=527, in_channel=1024, t=0.3, adj_file='audioset_adj.pkl')

        with open('audioset_grad_word2vec.pkl', 'rb') as f:
            self.inp = pickle.load(f)

    def init_model(self, model_pth=None):
        pre_model = CRNN10(args=None)
        pre_model.load_state_dict(torch.load(model_pth)['model'])
        for name, module in pre_model._modules.items():
            if name == 'bn0':
                self.bn0 = module
            if name == 'conv_block1':
                self.conv_block1 = module
            if name == 'conv_block2':
                self.conv_block2 = module
            if name == 'conv_block3':
                self.conv_block3 = module
            if name == 'conv_block4':
                self.conv_block4 = module
            if name == 'rnn':
                self.rnn = module
            if name == 'fc1':
                self.fc1 = module

    def forward(self, x, aug=True):
        x = x.view((-1, 1, x.size(1), x.size(2)))
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and aug:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        x = x.transpose(1, 2)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        a = x1 + x2
        a = F.dropout(a, p=0.5, training=self.training)
        embedding = F.relu_(self.fc1(a))
        a = F.dropout(embedding, p=0.5, training=self.training)
        x, cla = self.gcn(a, self.inp)
        global_prob = torch.sigmoid(x)
        global_prob = torch.clamp(global_prob, 1e-7, 1 - 1e-7)

        return global_prob, global_prob

    def predict(self, x, verbose=True, batch_size=100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i: i + batch_size])).cuda()
                output = self.forward(input)
                if not verbose: output = output[:1]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        return result if verbose else result[0]


class ConvBlock(nn.Module):
    def __init__(self, n_input_feature_maps, n_output_feature_maps, kernel_size, batch_norm=False, pool_stride=None):
        super(ConvBlock, self).__init__()
        assert all(x % 2 == 1 for x in kernel_size)
        self.n_input = n_input_feature_maps
        self.n_output = n_output_feature_maps
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.pool_stride = pool_stride
        self.conv = nn.Conv2d(self.n_input, self.n_output, self.kernel_size,
                              padding=tuple(x // 2 for x in self.kernel_size), bias=~batch_norm)
        if batch_norm: self.bn = nn.BatchNorm2d(self.n_output)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm: x = self.bn(x)
        x = F.relu(x)
        if self.pool_stride is not None: x = F.max_pool2d(x, self.pool_stride)
        return x


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.__dict__.update(args.__dict__)  # Instill all args into self
        assert self.n_conv_layers % self.n_pool_layers == 0
        self.input_n_freq_bins = n_freq_bins = 64
        self.output_size = 527
        self.conv = []
        pool_interval = self.n_conv_layers // self.n_pool_layers
        n_input = 1
        for i in range(self.n_conv_layers):
            if (i + 1) % pool_interval == 0:  # this layer has pooling
                n_freq_bins //= 2
                n_output = self.embedding_size // n_freq_bins
                pool_stride = (2, 2) if i < pool_interval * 2 else (1, 2)
            else:
                n_output = self.embedding_size * 2 // n_freq_bins
                pool_stride = None
            layer = ConvBlock(n_input, n_output, self.kernel_size, batch_norm=self.batch_norm, pool_stride=pool_stride)
            self.conv.append(layer)
            self.__setattr__('conv' + str(i + 1), layer)
            n_input = n_output
        self.gru = nn.GRU(self.embedding_size, self.embedding_size // 2, 1, batch_first=True, bidirectional=True)
        self.fc_prob = nn.Linear(self.embedding_size, self.output_size)
        if self.pooling == 'att':
            self.fc_att = nn.Linear(self.embedding_size, self.output_size)
        # Better initialization
        nn.init.orthogonal_(self.gru.weight_ih_l0);
        nn.init.constant_(self.gru.bias_ih_l0, 0)
        nn.init.orthogonal_(self.gru.weight_hh_l0);
        nn.init.constant_(self.gru.bias_hh_l0, 0)
        nn.init.orthogonal_(self.gru.weight_ih_l0_reverse);
        nn.init.constant_(self.gru.bias_ih_l0_reverse, 0)
        nn.init.orthogonal_(self.gru.weight_hh_l0_reverse);
        nn.init.constant_(self.gru.bias_hh_l0_reverse, 0)
        nn.init.xavier_uniform_(self.fc_prob.weight);
        nn.init.constant_(self.fc_prob.bias, 0)
        if self.pooling == 'att':
            nn.init.xavier_uniform_(self.fc_att.weight);
            nn.init.constant_(self.fc_att.bias, 0)

    def forward(self, x):
        x = x.view((-1, 1, x.size(1), x.size(2)))  # x becomes (batch, channel, time, freq)
        for i in range(len(self.conv)):
            if self.dropout > 0: x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv[i](x)  # x becomes (batch, channel, time, freq)
        x = x.permute(0, 2, 1, 3).contiguous()  # x becomes (batch, time, channel, freq)
        x = x.view((-1, x.size(1), x.size(2) * x.size(3)))  # x becomes (batch, time, embedding_size)
        if self.dropout > 0: x = F.dropout(x, p=self.dropout, training=self.training)
        x, _ = self.gru(x)  # x becomes (batch, time, embedding_size)
        if self.dropout > 0: x = F.dropout(x, p=self.dropout, training=self.training)
        frame_prob = torch.sigmoid(self.fc_prob(x))  # shape of frame_prob: (batch, time, output_size)
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)
        if self.pooling == 'max':
            global_prob, _ = frame_prob.max(dim=1)
            return global_prob, frame_prob
        elif self.pooling == 'ave':
            global_prob = frame_prob.mean(dim=1)
            return global_prob, frame_prob
        elif self.pooling == 'lin':
            global_prob = (frame_prob * frame_prob).sum(dim=1) / frame_prob.sum(dim=1)
            return global_prob, frame_prob
        elif self.pooling == 'exp':
            global_prob = (frame_prob * frame_prob.exp()).sum(dim=1) / frame_prob.exp().sum(dim=1)
            return global_prob, frame_prob
        elif self.pooling == 'att':
            frame_att = F.softmax(self.fc_att(x), dim=1)
            global_prob = (frame_prob * frame_att).sum(dim=1)
            return global_prob, frame_prob, frame_att

    def predict(self, x, verbose=True, batch_size=100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i: i + batch_size])).cuda()
                output = self.forward(input)
                if not verbose: output = output[:1]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        return result if verbose else result[0]
