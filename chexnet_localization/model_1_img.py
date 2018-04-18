import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

CKPT_PATH = 'model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

file_path = "D:/ChestXray-NIHCC/images/00030606_006.png"



def main() :

    cudnn.benchmark = True

    # initialize and load the model
    model = DenseNet121(N_CLASSES).cuda()
    model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])



    image = Image.open(file_path).convert('RGB')

    transform=transforms.Compose([transforms.Resize(256),
                                    transforms.TenCrop(224),
                                    transforms.Lambda
                                    (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                    transforms.Lambda
                                    (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                ])

    inp = transform (image)
    n_crops, c, h, w = inp.size()
    input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
    output = model(input_var)
    output_mean = output.view(1, n_crops, -1).mean(1)
    print (output_mean)


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

if __name__ == '__main__':
    main()