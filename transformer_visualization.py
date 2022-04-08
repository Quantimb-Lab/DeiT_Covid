from timm.models import create_model
import torch
import cv2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook


def single_image_handler(PATH_TO_IMAGE, transform):
    img1 = Image.open(PATH_TO_IMAGE)
    img1 = img1.convert('RGB')
    x = transform(img1)
    x = x.cuda()
    return x.unsqueeze(0)


def plot_attention_map(original_img, att_map):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Original')
    ax2.set_title('Attention Map Last Layer')
    _ = ax1.imshow(original_img)
    _ = ax2.imshow(att_map)
    plt.show()

def create_deit_model(PATH_TO_MODEL):
    model = create_model(
        'deit_small_distilled_patch16_224',
        pretrained=False,
        num_classes=3,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )
    checkpoint = torch.load(PATH_TO_MODEL)

    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.cuda()
    model.eval()
    return model


def attention_score(att_mat, img, get_mask):
    # att_mat = torch.stack(att_mat).squeeze(1)
    att_mat = torch.mean(att_mat, dim=1)
    residual_att = torch.eye(att_mat.size(1)).cuda()
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 2:].reshape(grid_size, grid_size).detach().numpy()
    if get_mask:
        mask = mask/mask.max()
        result = cv2.resize(mask, (224, 224))
    else:
        mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
        result = (mask * img).astype("uint8")

    return result




mean = [0.5604332, 0.5608471, 0.5612673]
std = [0.22447398, 0.22461453, 0.22464663]

PATH_TO_MODEL = './student_model/small_prettrain_no_AUG/checkpoint.pth'
PATHS_TO_IMAGE = glob.glob("./data/test/*/*")


model = create_deit_model(PATH_TO_MODEL)
data_transforms = {
            'val': transforms.Compose([
                transforms.Scale(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        }


for layer in range(12):
    print(20 * '=')
    print(f'Layer: {layer}')
    print(20 * '=')
    if os.path.exists(f'./viz/AttentionMap/layer{layer}/'):
        pass
    else:
        os.mkdir(f'./viz/AttentionMap/layer{layer}/')

    model.blocks[layer].attn.attn_drop.register_forward_hook(get_features('attn_drop'))
    for PATH_TO_IMAGE in PATHS_TO_IMAGE:
        filename =PATH_TO_IMAGE.split('/')[-1]
        print(filename)
        x = single_image_handler(PATH_TO_IMAGE, data_transforms['val'])
        original_img = x.cpu().data.numpy()[0]
        original_img = np.swapaxes(original_img, 0, 2).swapaxes(0, 1)
        features = {}

        output = model(x)
        original_img = (original_img - np.min(original_img))/(np.max(original_img) - np.min(original_img))

        att_map = attention_score(features['attn_drop'],original_img, 1)

        # att_map = (att_map - np.min(att_map))/(np.max(att_map) - np.min(att_map))

        fig, ax = plt.subplots()
        ax.imshow(original_img)
        ax.imshow(att_map, alpha=0.4, cmap='jet')
        ax.set_axis_off()
        plt.savefig(f'./viz/AttentionMap/layer{layer}/{filename}', aspect='auto')
        plt.close()

