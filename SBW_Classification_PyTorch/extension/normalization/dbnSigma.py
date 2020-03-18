"""
Reference:  Decorrelated Batch Normalization, CVPR 2018

- Paper:
- Code: https://github.com/princeton-vl/DecorrelatedBN
      or  https://github.com/huangleiBuaa/DecorrelatedBN
"""
import torch.nn
from torch.nn import Parameter


__all__ = ['dbnSigma', 'DBNSigma']



class DBNSigma_Single(torch.nn.Module):
    def __init__(self, num_features, dim=4, eps=1e-3, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(DBNSigma_Single, self).__init__()
        # assert dim == 4, 'DBNSigma is not support 2D'
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        shape = [1] * dim
        shape[1] = self.num_features

        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        # running whiten matrix
        self.register_buffer('running_projection', torch.eye(num_features))


    def forward(self, X: torch.Tensor):
        x = X.transpose(0, 1).contiguous().view(self.num_features, -1)
        d, m = x.size()
        mean = x.mean(-1, keepdim=True) if self.training else self.running_mean
        xc = x - mean
        if self.training:
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean.data
            # calculate covariance matrix
            sigma = torch.addmm(self.eps, torch.eye(self.num_features).to(X), 1. / m, xc, xc.transpose(0, 1))
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * sigma.data
        else:
            sigma = self.running_projection
        # reciprocal of trace of Sigma: shape [g, 1, 1]
        u, eig, _ = sigma.svd()
        #eig, u = sigma.symeig(eigenvectors=True)
        scale = eig.rsqrt()
        wm = u.matmul(scale.diag()).matmul(u.t())
        xn = wm.mm(xc)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        return Xn

class DBNSigma(torch.nn.Module):
    def __init__(self, num_features, num_channels=16, dim=4, eps=1e-3, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(DBNSigma, self).__init__()
        # assert dim == 4, 'DBNSigma is not support 2D'
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.num_channels = num_channels
        num_groups = (self.num_features-1) // self.num_channels + 1 
        self.num_groups = num_groups
        self.DBNSigma_Groups = torch.nn.ModuleList(
            [DBNSigma_Single(num_features = self.num_channels, eps=eps, momentum=momentum) for _ in range(self.num_groups-1)]
        )
        num_channels_last=self.num_features - self.num_channels * (self.num_groups -1)
        self.DBNSigma_Groups.append(DBNSigma_Single(num_features = num_channels_last, eps=eps, momentum=momentum))

        print('DBNSigma by ZCA-------m_perGroup:' + str(self.num_channels) + '---nGroup:' + str(self.num_groups)+ '---MM:' + str(self.momentum) + '---Affine:' + str(affine))
        self.affine = affine
        self.dim = dim
        shape = [1] * dim
        shape[1] = self.num_features
        if self.affine:
            self.weight = Parameter(torch.Tensor(*shape))
            self.bias = Parameter(torch.Tensor(*shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor):
        X_splits = torch.split(X, self.num_channels, dim=1)
        X_hat_splits = []
        for i in range(self.num_groups):
            X_hat_tmp = self.DBNSigma_Groups[i](X_splits[i])
            X_hat_splits.append(X_hat_tmp)
        X_hat = torch.cat(X_hat_splits, dim=1)
        # affine
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return '{num_features}, num_channels={num_channels}, eps={eps}, ' \
               'momentum={momentum}, affine={affine}'.format(**self.__dict__)


if __name__ == '__main__':
    ItN = DBNSigma(8, num_channels=3, momentum=1, affine=False)
    print(ItN)
    ItN.train()
    x = torch.randn(32, 8, 4, 4)
    #x = torch.randn(32, 8)
    x.requires_grad_()
    y = ItN(x)
    z = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(z.matmul(z.t()) / z.size(1))

    y.sum().backward()
    print('x grad', x.grad.size())

    ItN.eval()
    y = ItN(x)
    z = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(z.matmul(z.t()) / z.size(1))
