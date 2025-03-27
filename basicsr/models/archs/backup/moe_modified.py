class MOE(nn.Module):
    def __init__(self, N):
        super(MOE, self).__init__()
        self.N = N
        dropout_rate = 0.5

        #提取特征
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        #embedding
        self.layer1 = nn.Linear(16*3*3, self.N)#C*3*3
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(16*3*3, self.N)#C*3*3
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(16*3*3, self.N)#C*3*3
        self.dropout3 = nn.Dropout(dropout_rate)
        print("total parameter num of MOE: ", count_parameters(self), '-------------------------') # 4000

    def forward(self, x):
        # x.shape = 4,3,384,384
        B, C, H, W = x.shape
        assert C==3, 'channel of image isn\'t 3 !!!'
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        #固定输出大小的pooling
        B, _, H, W = x.shape
        kernel_size = (math.ceil(H / 3), math.ceil(W / 3))
        stride = (math.ceil(H / 3), math.ceil(W / 3))
        padding = (math.floor((kernel_size[0] * 3 - H + 1) / 2), math.floor((kernel_size[1] * 3 - W + 1) / 2))
        x = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)#  B  * N*C * 3 * 3
        # stx()
        x = x.view(B, 16*3*3)#二维

        #embedding
        q = torch.relu(self.layer1(x))
        q = self.dropout1(q) # self.N

        k = torch.relu(self.layer2(x))
        k = self.dropout2(k) # self.N

        v = torch.relu(self.layer3(x))
        v = self.dropout3(v) # self.N
        stx()

        router = torch.matmul(q.T, k)
        weights = torch.softmax(torch.matmul(router, v.T).T, dim=1) # self.N
        stx()
        return weights