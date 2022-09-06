import mindspore as ms
from mindspore import nn, ops, Tensor
from my_function import *
from mindspore import dataset as ds
from tqdm import tqdm

EPOCH = 8
BATCH_SIZE = 64
LR = 0.0001
drug_num = 1307
protein_num = 1996
indication_num = 3926
drug_feature = 1024  # ECFPs指纹
a1 = 0.00000001
a2 = 0.0001
op = ops.ReduceSum(keep_dims=True)

# 自定义网络
class AutoEncoder(nn.Cell):
    def __init__(self, size_x, size_y):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.SequentialCell([
            nn.Dense(size_x, 2048),
            nn.Dropout(0.2),         # 参数是扔掉的比例
            nn.GELU(),
            nn.BatchNorm1d(2048),
            nn.Dense(2048, 2048),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.BatchNorm1d(2048),
            nn.Dense(2048, protein_num),
            nn.Dropout(0.2)]
        )

        self.decoder = nn.SequentialCell([
            nn.Dense(protein_num, 4096),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.BatchNorm1d(4096),
            nn.Dense(4096, 4096),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.BatchNorm1d(4096),
            nn.Dense(4096, size_y),
            nn.Dropout(0.2)]
        )

    def construct(self, x):
        e0 = self.encoder(x)
        e1 = nn.Softmax()(e0)
        d0 = self.decoder(e1)
        d1 = nn.Softmax()(d0)
        return e1, d1
        
# 损失函数
class AEFSLoss(nn.Cell):
    def __init__(self):
        super().__init__() # 没有需要保存的参数和状态信息

    def construct(self, e, th, sr, sp, a):  # 定义前向的函数运算即可
        l2_1 = ops.MatMul()(e, e.T)
        l2_2 = ops.MatMul()(ops.Sqrt()(op(l2_1,1)), ops.Sqrt()(op(l2_1,1)).T)
        l2_3 = ops.TruncateDiv()(l2_1, l2_2) - sr
        l2 = op(ops.MatMul()(l2_3, l2_3)) / (e.shape[0] * e.shape[0])
        l3_1 = ops.MatMul()(e.T, e)
        l3_2 = ops.MatMul()(ops.Sqrt()(op(l3_1,1)), ops.Sqrt()(op(l3_1,1)).T)
        l3_3 = l3_1 / l3_2 - sp
        l3 = op(ops.MatMul()(l3_3, l3_3)) / (e.shape[1] * e.shape[1])
        return a * l2 + a * l3
        
# 损失网络
class MyWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn, loss_fn1):
        """实例化时传入前向网络和损失函数作为参数"""
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_func = loss_fn
        self.loss_func1 = loss_fn1

    def construct(self, data, batch_h, batch_y, batch_SR, SP, SD, a1, a2):
        """连接前向网络和损失函数"""
        encoded, decoded = self.backbone(data)
        loss1 = loss_func1(encoded, batch_h) + loss_func(encoded, batch_h, batch_SR, SP, a1)
        loss2 = loss_func1(decoded, batch_y) + loss_func(decoded, batch_y, batch_SR, SD, a2)
        loss = loss1 + loss2
        return loss

    def backbone_network(self):
        """要封装的骨干网络"""
        return self.backbone

if __name__ == '__main__':
    print("读取数据")
    train_drug_idx = np.loadtxt('dataset/train_idx.txt', dtype=int)
    train_drug_indications = np.loadtxt('dataset/train_RDA.txt', dtype=float32)
    train_drug_targets = np.loadtxt('dataset/train_DPI.txt', dtype=float32)
    train_drug_fps = np.loadtxt('dataset/train_fps.txt', dtype=float32)
    SP = np.loadtxt('dataset/SP.txt')
    SR = np.loadtxt('dataset/SR.txt')
    SD = np.loadtxt('dataset/SD.txt')
    print("处理数据")
    SP = Tensor(max_min_normalize(SP))
    SR = Tensor(max_min_normalize(SR))
    SD = Tensor(max_min_normalize(SD))
    print("初始化模型")
    input_data = ds.NumpySlicesDataset((train_drug_idx, train_drug_fps, train_drug_targets, train_drug_indications))
    train_loader = input_data.batch(BATCH_SIZE).repeat(EPOCH)

    AE = AutoEncoder(drug_feature, indication_num)
    optimizer = nn.Adam(AE.trainable_params(), learning_rate=LR)
    loss_func = AEFSLoss()
    loss_func1 = nn.MSELoss()
    net_with_criterion = MyWithLossCell(AE, loss_func, loss_func1)  # 构建损失网络
    train_net = nn.TrainOneStepCell(net_with_criterion, optimizer)  # 构建训练网络
    print("开始训练")
    x = 0
    for data in train_loader.create_dict_iterator():
        batch_id = data['column_0']
        batch_x = data['column_1']
        batch_h = data['column_2']
        batch_y = data['column_3']
        batch_SR = ms.numpy.zeros((batch_x.shape[0], batch_x.shape[0]))
        for m in range(batch_x.shape[0]):
            for n in range(batch_x.shape[0]):
                batch_SR[m, n] = SR[batch_id[m], batch_id[n]]

        train_net(batch_x, batch_h, batch_y, batch_SR, SP, SD, a1, a2)  # 执行训练，并更新权重
        loss = net_with_criterion(batch_x, batch_h, batch_y, batch_SR, SP, SD, a1, a2)  # 计算损失值
        print('Epoch:', (x//19), ' train loss: %.16f' % loss[0][0])
        x += 1

    ms.save_checkpoint(AE, "results/model.ckpt")