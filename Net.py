class CNN10_GL(nn.Module):
    def __init__(self, args):
        super(CNN10_GL, self).__init__()
        # Spec augmenter
        self.N = 5
        self.tau = 0.5
        self.ratio = 4
        self.spec_augmenter = SpecAugmentation(time_drop_width=50, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock2(1, 64)
        self.conv_block2 = ConvBlock2(64, 128)
        self.conv_block3 = ConvBlock2(128, 256)
        self.conv_block4 = ConvBlock2(256, 512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, 527, bias=True)

        self.fc1_L = nn.Linear(512, 512, bias=True)
        self.fc_audioset_L = nn.Linear(512, 527, bias=True)

        self.init_model(
            model_pth='/data/dean/cmu-thesis/workspace/audioset/CNN10_aug-batch128-ckpt1200-adam-lr1e-03-pat2-fac0.9-seed15213/model/checkpoint123.pt')

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
                self.fc1_L = module
            if name == 'fc_audioset':
                self.fc_audioset = module
                self.fc_audioset_L = module

    def forward(self, x, aug=True):
        x = x.view((-1, 1, x.size(1), x.size(2)))
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and aug:
            x = self.spec_augmenter(x)

        input_ = x.clone()
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
        global_prob = torch.clamp(global_prob, 1e-7, 1 - 1e-7)#bs,527
        sorted, indices = torch.sort(global_prob, dim=1, descending=True)
        indices = indices[:,:self.N]#bs,N

        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        embedding_L = F.relu_(self.fc1_L(x))
        x = F.dropout(embedding_L, p=0.5, training=self.training)
        frame_prob = torch.sigmoid(self.fc_audioset_L(x))
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)#bs,T,527

        frame_prob = frame_prob.transpose(1,2)#bs,527,T
        maps = torch.zeros(frame_prob.size(0),self.N,frame_prob.size(2)).cuda()#bs,N,T
        for i in range(indices.size(0)):
                maps[i] = torch.index_select(frame_prob[i],1,indices[i])
        regions = self.region_select(maps)
        local_regions = self.upsampling(regions,input_)
        local_prob = torch.zeros(global_prob.size(0),global_prob.size(1),self.N).cuda()#bs,N,T
        for i in range(self.N):
            local_ = local_regions[:,i,:,:]
            local_ = self.conv_block1(local_, pool_size=(2, 2), pool_type='avg')
            local_ = F.dropout(local_, p=0.2, training=self.training)
            local_ = self.conv_block2(local_, pool_size=(2, 2), pool_type='avg')
            local_ = F.dropout(local_, p=0.2, training=self.training)
            local_ = self.conv_block3(local_, pool_size=(1, 2), pool_type='avg')
            local_ = F.dropout(local_, p=0.2, training=self.training)
            local_ = self.conv_block4(local_, pool_size=(1, 2), pool_type='avg')
            local_ = F.dropout(local_, p=0.2, training=self.training)
            local_ = torch.mean(local_, dim=3)

            (x1, _) = torch.max(local_, dim=2)
            x2 = torch.mean(local_, dim=2)
            a = x1 + x2
            a = F.dropout(a, p=0.5, training=self.training)
            embedding = F.relu_(self.fc1(a))
            a = F.dropout(embedding, p=0.5, training=self.training)
            local_prob[:,:,i] = torch.sigmoid(self.fc_audioset(a))

        (local_prob, _) = torch.max(local_prob, dim=2)
        local_prob = torch.clamp(local_prob, 1e-7, 1 - 1e-7)  # bs,527

        return global_prob, local_prob

    def region_select(self, maps):
        regions = torch.zeros(maps.size(0),self.N,2, dtype=torch.int32).cuda()#bs,N,2
        for i in range(maps.size(0)):
            for j in range(maps.size(1)):
                map_ = maps[i,j]
                mask = (map_ > self.tau).nonzero()
                regions[i,j,0] = mask[0]
                regions[i,j,1] = mask[-1]
        return regions

    def upsampling(self,regions,x):
        local_regions = torch.zeros(x.size(0),self.N, x.size(2), x.size(3)).cuda()#bs,N,T,F
        for i in range(local_regions.size(0)):
            for j in range(local_regions.size(1)):
                local_regions[i,j,:,:] = x[i,0,regions[i,j,0]*self.ratio:regions[i,j,1]*self.ratio,:]
        return local_regions

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
