import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from .lavaszSoftmax import lovasz_softmax
# med_frq = [0.000000, 0.452448, 0.637584, 0.377464, 0.585595,
#            0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
#            2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
#            0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
#            1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
#            4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
#            3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
#            0.750738, 4.040773,2.154782,0.771605,0.781544,0.377464]

class lovaszSoftmax(nn.Module):
    def __init__(self,classes='present',per_image=False,ignore_index=None):
        super(lovaszSoftmax, self).__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image
        self.classes = classes
    def forward(self,output,target):
        if not isinstance(output, tuple):
            output = (output,)
        loss = 0
        for item in output:

            h, w = item.size(2), item.size(3)
            # 变换大小需要4维
            label = F.interpolate(target.unsqueeze(1).float(), size=(h, w))
            logits = F.softmax(item,dim=1)
            loss += lovasz_softmax(logits,label.squeeze(1),ignore=self.ignore_index,per_image=self.per_image,classes=self.classes)
        return loss/len(output)

class MscCrossEntropyLoss(nn.Module):
    #

    def __init__(self, weight=None, ignore_index=-100, reduction='mean', gate_gt=None):
        super(MscCrossEntropyLoss, self).__init__()

        self.weight = weight
        self.gate_gt=gate_gt
        self.ignore_index = ignore_index
        self.reduction = reduction




    def forward(self, input, target):
        # list={}
        # # dominant loss
        # for i,item in enumerate(input):
        #     list[i]=item
        # dom_loss = F.cross_entropy(list[0], target,weight=self.weight,ignore_index=self.ignore_index, reduction=self.reduction)
        # # aux. loss
        # loss2_1 = F.cross_entropy(list[1], target,weight=self.weight,ignore_index=self.ignore_index, reduction=self.reduction)
        # loss3_1 = F.cross_entropy(list[2], target,weight=self.weight,ignore_index=self.ignore_index, reduction=self.reduction)
        # loss4_1 = F.cross_entropy(list[3], target,weight=self.weight,ignore_index=self.ignore_index, reduction=self.reduction)
        # loss5_1 = F.cross_entropy(list[4], target,weight=self.weight,ignore_index=self.ignore_index, reduction=self.reduction)
        # loss2_2 = F.cross_entropy(list[5], target,weight=self.weight,ignore_index=self.ignore_index, reduction=self.reduction)
        # loss3_2 = F.cross_entropy(list[6], target,weight=self.weight,ignore_index=self.ignore_index, reduction=self.reduction)
        # loss4_2 = F.cross_entropy(list[7], target,weight=self.weight,ignore_index=self.ignore_index, reduction=self.reduction)
        # loss5_2 = F.cross_entropy(list[8], target,weight=self.weight,ignore_index=self.ignore_index, reduction=self.reduction)
        # # regression
        # #我不会伪标签的用法，在此省略
        # #reg_loss = F.smooth_l1_loss(list[9], self.gate_gt) * 2
        # loss = dom_loss + 0.8 * (loss2_1 * 1 + loss3_1 * 0.8 + loss4_1 * 0.6 + loss5_1 * 0.4) + 0.8 * (loss2_2 * 1 + loss3_2 * 0.8 + loss4_2 * 0.6 + loss5_2 * 0.4)
        # return loss

        if not isinstance(input, tuple):
            input = (input,)

        loss = 0
        # weight = [0.2,0.4,0.6,0.8]

        # h,w = target.size()[1:]


        for item in input:
            h, w = item.size(2), item.size(3)



            item_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w))
            # item_target = F.interpolate(item, size=(h, w))

            # print('item_target', item_target.shape)
            # print('2_item', item.shape)
            # print('item_target.squeeze', item_target.squeeze(1).long().shape)

            # loss += F.cross_entropy(item, item_target.squeeze(1).long(), weight=self.weight,
            #             ignore_index=self.ignore_index, reduction=self.reduction)

            loss += F.cross_entropy(item, item_target.squeeze(1).long(),weight=self.weight,
                                    ignore_index=self.ignore_index, reduction=self.reduction)

            # loss += F.cross_entropy(item, item_target.long(),weight=self.weight,
            #                         ignore_index=self.ignore_index, reduction=self.reduction)

            # loss += F.cross_entropy(item_target, target.long(), weight=self.weight,
            #                         ignore_index=self.ignore_index, reduction=self.reduction)

            #对输入的一个batch求loss的平均

        return loss / len(input)
class BCrossEntropyLoss2(nn.Module):
    #

    def __init__(self, weight=None, ignore_index=-100, reduction='mean',gate_gt=None):
        super(BCrossEntropyLoss2, self).__init__()

        self.weight = weight
        self.gate_gt=gate_gt
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):

        if not isinstance(input, tuple):
            input = (input,)

        loss = 0
        # weight = [0.2,0.4,0.6,0.8]

        # h,w = target.size()[1:]


        for item in input:
            h, w = item.size(2), item.size(3)

            item_target = F.interpolate(target.float(), size=(h, w))

            loss += F.binary_cross_entropy_with_logits(torch.nn.functional.log_softmax(item, 1), item_target, weight=self.weight, size_average=True)

            #对输入的一个batch求loss的平均
        return loss / len(input)

if __name__ == '__main__':
    import torch
    # depth = torch.randn(6,3,480,640)
    score = torch.randn(6,1)
    sof = nn.Softmax(dim=0)
    out = sof(score)
    print(out)
    # print(score.shape)
    # print(score)
    # score = score[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1,3,480,640)
    # out = torch.mul(depth,score[:,0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1,3,480,640))
    # print(score.shape)
    # print(score)
    # torch.randn(6,3,480,640)
    # print(out)
    # out = out.view(3,480,640)
    # print(out)

    # predict = torch.randn((2, 21, 512, 512))
    # gt = torch.randint(0, 255, (2, 512, 512))

    # loss_function = MscCrossEntropyLoss()
    # result = loss_function(predict, gt)
    # print(result)
