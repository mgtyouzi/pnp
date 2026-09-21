import torch
import torch.nn as nn

class StyleRandomization(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()

        x = x.view(N, C, -1)
        idx_swap = torch.randperm(N)
        alpha = torch.rand(N, 1, 1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()

        if x.is_cuda:
                alpha = alpha.cuda()
        mean = alpha * mean + (1 - alpha) * mean[idx_swap]
        var = alpha * var + (1 - alpha) * var[idx_swap]
        x = x * (var + self.eps).sqrt() + mean
        x = x.view(N, C, H, W)

        return x, idx_swap

class StyleRandomization_multi_scale(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, feature, feature1, feature2, feature3, feature4, feature5):
        N, C, H, W = feature.size()
        N1,C1,H1,W1=feature1.size()
        N2, C2, H2, W2 = feature2.size()
        N3, C3, H3, W3 = feature3.size()
        N4, C4, H4, W4 = feature4.size()
        N5, C5, H5, W5 = feature5.size()

        if self.training:
            feature = feature.view(N, C, -1)
            feature1=feature1.view(N1, C1, -1)
            feature2 = feature2.view(N2, C2, -1)
            feature3 = feature3.view(N3, C3, -1)
            feature4 = feature4.view(N4, C4, -1)
            feature5 = feature5.view(N5, C5, -1)

            feature_mean = feature.mean(-1, keepdim=True)
            feature_var = feature.var(-1, keepdim=True)

            feature1_mean = feature1.mean(-1, keepdim=True)
            feature1_var = feature1.var(-1, keepdim=True)

            feature2_mean = feature2.mean(-1, keepdim=True)
            feature2_var = feature2.var(-1, keepdim=True)

            feature3_mean = feature3.mean(-1, keepdim=True)
            feature3_var = feature3.var(-1, keepdim=True)

            feature4_mean = feature4.mean(-1, keepdim=True)
            feature4_var = feature4.var(-1, keepdim=True)

            feature5_mean = feature5.mean(-1, keepdim=True)
            feature5_var = feature5.var(-1, keepdim=True)

            feature = (feature - feature_mean) / (feature_var + self.eps).sqrt()

            feature1 = (feature1 - feature1_mean) / (feature1_var + self.eps).sqrt()

            feature2 = (feature2 - feature2_mean) / (feature2_var + self.eps).sqrt()

            feature3 = (feature3 - feature3_mean) / (feature3_var + self.eps).sqrt()

            feature4 = (feature4 - feature4_mean) / (feature4_var + self.eps).sqrt()

            feature5 = (feature5 - feature5_mean) / (feature5_var + self.eps).sqrt()

            idx_swap = torch.randperm(N)
            alpha = torch.rand(N, 1, 1)

            if feature.is_cuda:
                alpha = alpha.cuda()
            feature_mean = alpha * feature_mean + (1 - alpha) * feature_mean[idx_swap]
            feature_var = alpha * feature_var + (1 - alpha) * feature_var[idx_swap]

            feature1_mean = alpha * feature1_mean + (1 - alpha) * feature1_mean[idx_swap]
            feature1_var = alpha * feature1_var + (1 - alpha) * feature1_var[idx_swap]

            feature2_mean = alpha * feature2_mean + (1 - alpha) * feature2_mean[idx_swap]
            feature2_var = alpha * feature2_var + (1 - alpha) * feature2_var[idx_swap]

            feature3_mean = alpha * feature3_mean + (1 - alpha) * feature3_mean[idx_swap]
            feature3_var = alpha * feature3_var + (1 - alpha) * feature3_var[idx_swap]

            feature4_mean = alpha * feature4_mean + (1 - alpha) * feature4_mean[idx_swap]
            feature4_var = alpha * feature4_var + (1 - alpha) * feature4_var[idx_swap]

            feature5_mean = alpha * feature5_mean + (1 - alpha) * feature5_mean[idx_swap]
            feature5_var = alpha * feature5_var + (1 - alpha) * feature5_var[idx_swap]

            feature = feature * (feature_var + self.eps).sqrt() + feature_mean
            feature = feature.view(N, C, H, W)

            feature1=feature1 * (feature1_var + self.eps).sqrt() + feature1_mean
            feature1 = feature1.view(N1, C1, H1, W1)

            feature2 = feature2 * (feature2_var + self.eps).sqrt() + feature2_mean
            feature2 = feature2.view(N2, C2, H2, W2)

            feature3 = feature3 * (feature3_var + self.eps).sqrt() + feature3_mean
            feature3 = feature3.view(N3, C3, H3, W3)

            feature4 = feature4 * (feature4_var + self.eps).sqrt() + feature4_mean
            feature4 = feature4.view(N4, C4, H4, W4)

            feature5 = feature5 * (feature5_var + self.eps).sqrt() + feature5_mean
            feature5 = feature5.view(N5, C5, H5, W5)

        else:
            idx_swap = torch.randperm(N)


        return feature,feature1,feature2,feature3,feature4,feature5

class StyleRandomization_multi_scale_3layer(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, feature, feature1, feature2):
        N, C, H, W = feature.size()
        N1,C1,H1,W1=feature1.size()
        N2, C2, H2, W2 = feature2.size()

        if self.training:
            feature = feature.view(N, C, -1)
            feature1=feature1.view(N1, C1, -1)
            feature2 = feature2.view(N2, C2, -1)

            feature_mean = feature.mean(-1, keepdim=True)
            feature_var = feature.var(-1, keepdim=True)

            feature1_mean = feature1.mean(-1, keepdim=True)
            feature1_var = feature1.var(-1, keepdim=True)

            feature2_mean = feature2.mean(-1, keepdim=True)
            feature2_var = feature2.var(-1, keepdim=True)

            feature = (feature - feature_mean) / (feature_var + self.eps).sqrt()

            feature1 = (feature1 - feature1_mean) / (feature1_var + self.eps).sqrt()

            feature2 = (feature2 - feature2_mean) / (feature2_var + self.eps).sqrt()

            idx_swap = torch.randperm(N)
            alpha = torch.rand(N, 1, 1)

            if feature.is_cuda:
                alpha = alpha.cuda()
            feature_mean = alpha * feature_mean + (1 - alpha) * feature_mean[idx_swap]
            feature_var = alpha * feature_var + (1 - alpha) * feature_var[idx_swap]

            feature1_mean = alpha * feature1_mean + (1 - alpha) * feature1_mean[idx_swap]
            feature1_var = alpha * feature1_var + (1 - alpha) * feature1_var[idx_swap]

            feature2_mean = alpha * feature2_mean + (1 - alpha) * feature2_mean[idx_swap]
            feature2_var = alpha * feature2_var + (1 - alpha) * feature2_var[idx_swap]

            feature = feature * (feature_var + self.eps).sqrt() + feature_mean
            feature = feature.view(N, C, H, W)

            feature1=feature1 * (feature1_var + self.eps).sqrt() + feature1_mean
            feature1 = feature1.view(N1, C1, H1, W1)

            feature2 = feature2 * (feature2_var + self.eps).sqrt() + feature2_mean
            feature2 = feature2.view(N2, C2, H2, W2)

        else:
            idx_swap = torch.randperm(N)


        return feature,feature1,feature2

class ContentRandomization_single(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()

        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()

            idx_swap = torch.randperm(N)
            x = x[idx_swap].detach()

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x


class ContentRandomization(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x_list):
        feature_len = len(x_list)
        N = x_list[0].size()[0]
        idx_swap = torch.randperm(N)
        x_list_new = []
        if N != 1:
            for i in range(feature_len):
                x = x_list[i]
                N, C, H, W = x.size()
                if self.training:
                    x = x.view(N, C, -1)
                    mean = x.mean(-1, keepdim=True)
                    var = x.var(-1, keepdim=True)

                    x = (x - mean) / (var + self.eps).sqrt()
                    x = x[idx_swap].detach()

                    x = x * (var + self.eps).sqrt() + mean
                    x = x.view(N, C, H, W)

                    x_list_new.append(x)

            return x_list_new
        else:
            return x_list
