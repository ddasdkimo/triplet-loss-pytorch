import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import wideresnet
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
import json


class Inference:
    def __init__(self):
        global model,key2
        num_classes = 100
        text_file = open('resnet50_'+str(num_classes) +
                         '_allergy_336_lr0001_triplet_base_lables_map.txt', "r")
        t = text_file.read()
        key = json.loads(t)
        key2 = dict(zip(key.values(), key.keys()))

        model = wideresnet.resnet50(pretrained=True, num_classes=1000)
        state_dict = torch.load(
            'resnet50_'+str(num_classes)+'_allergy_336_lr0001_triplet_base_best.pth.tar')
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(2048, 256)
        model.classifier = nn.Linear(256, num_classes)
        model.load_state_dict(state_dict["state_dict"])
        print(model)
        model.eval()

    def inference(self, path):
        global model,key2
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform1 = transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(334),
            transforms.ToTensor(),
            normalize,
        ])
        img = Image.open(path)
        img1 = transform1(img)
        input_var = torch.autograd.Variable(img1.cuda())
        input_var = input_var[np.newaxis, :]
        input_var = input_var.cuda()
        output = model(input_var.cpu())
        output = model.classifier(output)
        topk = (1, 1)
        maxk = max(topk)
        c, pred = output.topk(maxk, 1, True, True)
        torch.max(output, 1)
        pred = pred.t()

        b = F.normalize(output[0], p=2, dim=0)

        print(key2[int(pred[0][0])])
