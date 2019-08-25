import torch.nn as nn


class PCBPool(object):
    def __init__(self, args):
        self.args = args
        self.pool = nn.AdaptiveAvgPool2d(1) if args.max_or_avg == 'avg' else nn.AdaptiveMaxPool2d(1)

    def __call__(self, in_dict):
        feat = in_dict['feat']
        assert feat.size(2) % self.args.num_parts == 0
        stripe_h = int(feat.size(2) / self.args.num_parts)
        feat_list = []
        for i in range(self.args.num_parts):
            # shape [N, C]
            local_feat = self.pool(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :])
            # shape [N, C]
            if not self.args.local_conv:
                local_feat = local_feat.view(local_feat.size(0), -1)
            feat_list.append(local_feat)
        out_dict = {'feat_list': feat_list}
        return out_dict

class PCBPool_nine(object):
    def __init__(self, args):
        self.args = args
        self.pool = nn.AdaptiveAvgPool2d(1) if args.max_or_avg == 'avg' else nn.AdaptiveMaxPool2d(1)

    def __call__(self, in_dict):
        feat = in_dict['feat']
        assert feat.size(2) % self.args.num_parts == 0
        stripe_h = int(feat.size(2) / self.args.num_parts)
        feat_list = []
        for i in range(self.args.num_parts):
            # shape [N, C]
            local_feat = self.pool(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :])
            # shape [N, C]
            if not self.args.local_conv:
                local_feat = local_feat.view(local_feat.size(0), -1)

            feat_list.append(local_feat)

        for i in range(2):
            # shape [N, C]
            local_feat = self.pool(feat[:, :, i * stripe_h: (3*(i+1)) * stripe_h, :])
            if not self.args.local_conv:
                local_feat = local_feat.view(local_feat.size(0), -1)
            feat_list.append(local_feat)

        local_feat = self.pool(feat)
        if not self.args.local_conv:
            local_feat = local_feat.view(local_feat.size(0), -1)
        feat_list.append(local_feat)

        out_dict = {'feat_list': feat_list}
        return out_dict
