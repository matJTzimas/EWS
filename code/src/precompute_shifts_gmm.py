import os
from os.path import join
from utils import get_transform, load_model, prep_for_plot, remove_axes, prep_args, normalize, unnorm
from modules import FeaturePyramidNet, DinoFeaturizer, sample
from data import ContrastiveSegDataset
import hydra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib.colors import ListedColormap
from PIL import Image
from torchvision import transforms as T
import yaml
from sklearn.cluster import KMeans
from tqdm import tqdm
from types import SimpleNamespace
from sklearn.mixture import GaussianMixture
import time 
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def img_transform(img,res=224):
    
    w ,h = img.size
    resize = T.Resize(res, Image.NEAREST)
    cropper = T.CenterCrop(res)
    comp = T.Compose([cropper, T.ToTensor(), normalize])
    resize_image = resize(img)
    w_ , h_ = resize_image.size
    cropped_image = comp(resize_image)

    return cropped_image

@ignore_warnings(category=ConvergenceWarning)
def shift_from_sim(sim): 
    flattened_tensor = sim.flatten().reshape(-1, 1).cpu().numpy()
    
    # Apply Gaussian Mixture Model with 2 components
    # gmm = GaussianMixture(n_components=2,means_init=[[0.04],[0.25]],random_state=0)
    gmm = GaussianMixture(
        means_init=[[0.04],[0.3]],
        n_components=2,            # Only two components
        covariance_type="diag",     # Diagonal covariance, faster for 1D
        tol=1e-3,                   # Default tolerance, balanced speed/accuracy
        reg_covar=1e-6,             # Regularization to handle numerical stability
        max_iter=50,                # Reduced max iterations for faster results
        n_init=1,                   # Single initialization to save time
        init_params="random",       # KMeans initialization for faster convergence
        random_state=42             # Ensures reproducibility
    )
    gmm.fit(flattened_tensor)
    
    # Get the centroids (means of the two Gaussian components)
    centroids = gmm.means_.flatten()
    # print(centroids)
    return np.mean(centroids)

file_path = './configs/train_config.yaml'

# Reading the YAML file
with open(file_path, 'r') as file:
    config_dict = yaml.safe_load(file)
cfg = SimpleNamespace(**config_dict)

print(f'Computing contrastive loss shifts for {cfg.dir_dataset_name} Dataset')
res = cfg.res
patch_size = cfg.dino_patch_size
feats_res = res // patch_size 
net = DinoFeaturizer(cfg.dim, cfg)
net = net.eval().cuda()

train_img_dir = os.path.join(cfg.pytorch_data_dir, cfg.dir_dataset_name,'imgs/train')
train_imgs = sorted(os.listdir(train_img_dir))

shifts = [] 
for cnt, img_path in enumerate(train_imgs):
    print(f'Image {cnt} / {len(train_imgs)}')
    raw_img = Image.open(os.path.join(train_img_dir,img_path))
    raw_img = raw_img.convert("RGB")
    tsf_img = img_transform(raw_img,res=res)
    feats, _, _ = net(tsf_img.unsqueeze(0).cuda())
    feats = feats.squeeze(0)

    patch_shift = torch.zeros((feats.shape[1],feats.shape[2]))

    for i in tqdm(range(feats.shape[1])): 
        for j in range(feats.shape[2]):
            
            curr_vector = feats[:, i, j]
            sim = torch.einsum('chw,c->hw',F.normalize(feats,dim=0),F.normalize(curr_vector,dim=0))
            curr_shift = shift_from_sim(sim)
            patch_shift[i][j] = curr_shift

    shifts.append(patch_shift)


shifts = torch.stack(shifts, dim=0)
torch.save(shifts,f'../saved_models/{cfg.dir_dataset_name}_shifts.pt')
print(f'Precompute shifts have been computed. Tensor size{shifts.size()}')



