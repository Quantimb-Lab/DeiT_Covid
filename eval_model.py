import torch
import covid_dataset as cvr
from torchvision import transforms
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import os
from timm.models import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import glob

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

root_path = './student_model/'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
BATCH_SIZE = 1
models_path = glob.glob('./student_model/small_prettrain_AUG/checkpoint.pth')

for model_path in models_path:
    if 'small_chexnet' not in model_path:
        print(10 * '=')
        print(model_path)
        print(10 * '=')
        PATH_TO_MODEL = model_path
        # PATH_TO_MODEL = "./covid_results/checkpoint"
        PATH_TO_IMAGES = "./dataset/images/"
        FINDINGS = ['Normal',
                    'Pneumonia',
                    'COVID']

        checkpoint = torch.load(PATH_TO_MODEL)

        model = create_model(
                'deit_small_distilled_patch16_224',
                pretrained=False,
                num_classes=3,
                drop_rate=0,
                drop_path_rate=0.1,
                drop_block_rate=None,
            )

        model.load_state_dict(checkpoint['model'], strict=True)

        model = model.cuda()


        mean = [0.5604332, 0.5608471, 0.5612673]
        std = [0.22447398, 0.22461453, 0.22464663]
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD
        model.eval()
        data_transforms = {
            'val': transforms.Compose([
                transforms.Scale(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        }

        # create dataloader
        dataset = cvr.COVIDDataset(
            path_to_images=PATH_TO_IMAGES,
            fold="test",
            transform=data_transforms['val'])
        dataloader = torch.utils.data.DataLoader(
            dataset, BATCH_SIZE, shuffle=False, num_workers=8)

        correct = 0
        pred_list = []
        label_list = []
        # iterate over dataloader
        for i, data in enumerate(dataloader):
            if True:
                inputs, labels, _ = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                true_labels = labels.cpu().data.numpy()
                batch_size = true_labels.shape
                outputs = model(inputs)
                probs_np = softmax(outputs.cpu().data.numpy())
                labels = labels.cpu().data.numpy()
                pred_list.append(probs_np[0])
                label_list.append(labels[0])
            else:
                break


        pred_list = np.array(pred_list)
        label_list = np.array(label_list)

     

        print(confusion_matrix(np.argmax(label_list,axis=1), np.argmax(pred_list,axis=1)))
        print(f'Accuracy:{np.sum(np.argmax(label_list,axis=1) == np.argmax(pred_list,axis=1))*100.0/153}')