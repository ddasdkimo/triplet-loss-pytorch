import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import wideresnet
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

num_classes = 3
model = wideresnet.resnet50(pretrained=True, num_classes=1000)
state_dict = torch.load(
    'resnet50_'+str(num_classes)+'_allergy_336_lr0001_triplet_base_best.pth.tar')
model.avgpool = nn.AdaptiveAvgPool2d(1)
model.fc = nn.Linear(2048, 256)
model.classifier = nn.Linear(256, num_classes)
model.load_state_dict(state_dict["state_dict"])
print(model)
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform1 = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(334),
    transforms.ToTensor(),
    normalize,
])

# transform1 = transforms.Compose([
#     transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
# ]
# )

# numpy.ndarray
# img = cv2.imread("datasets/cats/cat.1.jpg")
# img = Image.open("datasets/cats/cat.1.jpg")
img = Image.open("datasets/giraffe/2.png")

# img = Image.open("datasets/dogs/dog.1.jpg")
img1 = transform1(img)

input_var = torch.autograd.Variable(img1.cuda())
input_var = input_var[np.newaxis,:]
input_var = input_var.cuda()
output = model(input_var.cpu())
output = model.classifier(output)
topk=(1,1)
maxk = max(topk)
c, pred = output.topk(maxk, 1, True, True)
pred = pred.t()
