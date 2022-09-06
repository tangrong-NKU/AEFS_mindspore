import numpy as np
import mindspore as ms
from mindspore import nn, Tensor

EPOCH = 201
BATCH_SIZE = 64
LR = 0.0001
drug_num = 1307
protein_num = 1996
indication_num = 3926
drug_feature = 1024  # ECFPs指纹
a1 = 0.0000000001
a2 = 0.001

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


if __name__ ==  '__main__':
    test_drug_fps = Tensor(np.loadtxt('dataset/test_fps.txt', dtype=np.float32))

    param_dict = ms.load_checkpoint("results/model.ckpt")
    AE = AutoEncoder(drug_feature, indication_num)
    ms.load_param_into_net(AE, param_dict)
    preDTI, preRDA = AE(test_drug_fps)
    np.savetxt('results/y_pre_DPI.txt', preDTI.asnumpy(), fmt='%f')
