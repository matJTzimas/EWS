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
from torchvision import transforms as T
import random
from sklearn.cluster import KMeans
import yaml
from types import SimpleNamespace
from PIL import Image, ImageOps
from scipy.ndimage import minimum_filter

cityscapes_categories = ['flat','construction','object','nature','sky','human','vehicle']

def shift_from_sim(sim): 
    h, w = sim.shape[0], sim.shape[1]  # Example size
    similarity_tensor = sim.cpu()  # Replace with your actual tensor

    # Step 2: Flatten the tensor (convert it to a 1D array)
    flattened_tensor = similarity_tensor.flatten().numpy()  # Convert to NumPy for KMeans

    # Step 3: Apply K-Means clustering with 2 clusters
    kmeans = KMeans(n_clusters=2,init=np.array([0.05, 0.3]).reshape(-1,1),n_init=1)
    kmeans.fit(flattened_tensor.reshape(-1, 1))  # Reshape for clustering

    # Get the cluster labels for each element
    cluster_labels = kmeans.labels_
    # print(kmeans.cluster_centers_)

    # Step 4: Reshape the cluster labels back to (h, w)
    clustered_tensor = cluster_labels.reshape(h, w)
    val, ind = torch.max(sim.flatten(),dim=0)

    if cluster_labels[ind] == 0 : 
        updated_cluster_labels = (1 - cluster_labels)
    else : 
        updated_cluster_labels = cluster_labels

    clustered_tensor = updated_cluster_labels.reshape(h, w)

    cluster_sim = clustered_tensor * similarity_tensor.numpy()
    new_cluster_sim = np.where(cluster_sim>0.01, cluster_sim, cluster_sim+1)
    nonnegclust = []
    for i in range(cluster_sim.shape[0]):
        for j in range(cluster_sim.shape[1]): 
            if cluster_sim[i][j] > 0.01 : 
                nonnegclust.append(cluster_sim[i][j])

    nonnegclust = np.array(nonnegclust)
    return np.min(nonnegclust)

#resize to res and res center cropping
def res_crop_getpoint(p, img, res): 

    xp, yp = p
    w ,h = img.size
    resize = T.Resize(res, Image.NEAREST)
    cropper = T.CenterCrop(res)
    comp = T.Compose([cropper, T.ToTensor(), normalize])

    resize_image = resize(img)
    w_ , h_ = resize_image.size


    cropped_image = comp(resize_image)
    _, w__, h__ = cropped_image.size()

    if w > h : 
        yp_ = yp * h_ // h
        xp_ = xp * w_ // w

        dif = (w_ - res )//2 
        xp_ = xp_ - dif 
        
    elif h > w :
        yp_ = yp * h_ // h
        xp_ = xp * w_ // w

        dif = (h_ - res )//2 
        yp_ = yp_ - dif 
    
    else :
        yp_ = yp * h_ // h
        xp_ = xp * w_ // w

    return (xp_, yp_) , cropped_image


def getPatch(p, patch_size):
    x, y = p
    return x//patch_size, y//patch_size

def pstate(image, p):
    width, height = image.size

    # Determine the size of the square (smallest dimension)
    square_size = min(width, height)

    # Calculate the coordinates to crop the image from the center
    left = (width - square_size) // 2
    top = (height - square_size) // 2
    right = (width + square_size) // 2
    bottom = (height + square_size) // 2

    state = p[1] > left and p[1] < right and p[0] > top and p[0] < bottom
    
    return state

def generate_annotationsv2(cfg):

    random_points = cfg.random_points
    random_imgs = cfg.random_imgs

    annotations = dict()

    train_gt_dir = os.path.join(cfg.pytorch_data_dir, cfg.dir_dataset_name,'labels/train')
    train_gts = sorted(os.listdir(train_gt_dir))
    # gt_random_index = np.random.randint(low=0, high=len(train_gts),size=random_imgs)

    # while len(list(dict.fromkeys(gt_random_index))) != random_imgs :
    #     gt_random_index = np.random.randint(low=0, high=len(train_gts),size=random_imgs)

    print(f'FINDING {random_imgs} to annotate')

    class_info = []
    while len(class_info) < random_points * random_imgs : 
        gt_random_index = np.random.randint(low=0, high=len(train_gts),size=1)

        index = gt_random_index[0]
        # print(index) 
        gt_path = os.path.join(train_gt_dir,train_gts[index])
        gt = Image.open(gt_path)
        gt = ImageOps.exif_transpose(gt)
        mask_array = np.array(gt)
        if np.sum(mask_array) == 0 :
            print(f'{gt_path} not containing the object')
            continue
        mask_array = minimum_filter(mask_array, size=8)
        ones_coordinates = np.argwhere(mask_array == 1)

        init_size = 8
        while np.sum(mask_array) < 16 :
            mask_array = np.array(gt)     
            mask_array = minimum_filter(mask_array, size=init_size)
            ones_coordinates = np.argwhere(mask_array == 1)
            init_size -= 5 
            print(f'mask sum -> {np.sum(mask_array)}')
            print(f'{gt_path} new min filter size {init_size} ')

        ones_coordinates_val = []
        for op in ones_coordinates : 
            if pstate(gt,op) == True : 
                ones_coordinates_val.append(op)

        if len(ones_coordinates_val) < random_points : 
            continue

        print(index,gt_path) 

        valid_coordinates = [] 
        for i in tqdm(range(random_points)):
            p = (-1,-1)
            while not pstate(gt,p) :
                random_coordinates = random.sample(list(map(tuple, ones_coordinates_val)), 1)
                p = random_coordinates[0]
            valid_coordinates.append((p[1],p[0]))

        for rcord in valid_coordinates :
            if cfg.dir_dataset_name == 'fireStego':
                gt_path = gt_path.replace('labels','imgs').replace('gt','rgb')
            elif cfg.dir_dataset_name == 'CityscapesOne' : 
                if "gtCoarse" in gt_path : 
                    gt_path = gt_path.replace('labels','imgs').replace('gtCoarse_labelIds','leftImg8bit')
                else : 
                    gt_path = gt_path.replace('labels','imgs').replace('gtFine_labelIds','leftImg8bit')
            else : 
                gt_path = gt_path.replace('labels','imgs')

            class_info.append([gt_path,rcord[0],rcord[1]])

    annotations['class'] = class_info
    return annotations


def get_sim(net, res, patch_size, raw_img, tag_raw_img, p=(100,100)):

    (target_x, target_y), img = res_crop_getpoint((p[0], p[1]), raw_img, res)
    feats, _, _  = net(img.unsqueeze(0).cuda())
    patch_i, patch_j = getPatch((target_x, target_y), patch_size)
    gt_vectors = [] 
    
    gt_vector = feats[:,:,patch_j,patch_i].squeeze(0)
    gt_vectors.append(gt_vector)

    feats = feats.squeeze(0)

    (target_x, target_y), tag_img = res_crop_getpoint((p[0], p[1]), tag_raw_img, res)
    tag_feats, _, _  = net(tag_img.unsqueeze(0).cuda())

    tag_feats = tag_feats.squeeze(0)

    sim = torch.einsum('chw,c->hw',F.normalize(tag_feats,dim=0),F.normalize(gt_vector,dim=0))
    return sim.cpu()

def my_app():
    file_path = './configs/train_config.yaml'

    # Reading the YAML file
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    cfg = SimpleNamespace(**config_dict)

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

 


    if cfg.generate_points :
        annotations = generate_annotationsv2(cfg)
    else :
        annotations = cfg.annotations

    res = cfg.res
    patch_size = cfg.dino_patch_size
    feats_res = res // patch_size 
    net = DinoFeaturizer(cfg.dim, cfg)
    net = net.eval().cuda()
    random_imgs = cfg.random_imgs
    random_points = cfg.random_points
    point_data = dict()
    cls_mask = []
    
    # for each class
    for cls_cnt, pcls in enumerate(annotations.keys()) : 
        curr_cls = annotations[pcls]
        annot_points = torch.zeros((len(curr_cls),2))
        annot_patch_coords = torch.zeros((len(curr_cls),2)) 
        annot_feat_vectors = torch.zeros((len(curr_cls),cfg.vit_dim)) 
        annot_ref_feats = torch.zeros((len(curr_cls),cfg.vit_dim, feats_res, feats_res))
        sims = torch.zeros((random_imgs,random_points,feats_res,feats_res))

        imgs_paths = [] 
        # for each annotation per class
        for cnt ,annot in enumerate(curr_cls):
            
            point_img_path = annot[0]
            print(point_img_path)
            print("Annotated point on current image", annot[1],annot[2])
            imgs_paths.append(point_img_path)
            orig_x = annot_points[cnt,0] = annot[1]
            orig_y = annot_points[cnt,1] = annot[2]
            
            raw_img = Image.open(point_img_path)
            raw_img = ImageOps.exif_transpose(raw_img)

            raw_img = raw_img.convert("RGB")
            (target_x, target_y), img = res_crop_getpoint((orig_x, orig_y), raw_img, res)
            feats, _, _ = net(img.unsqueeze(0).cuda())
            
            annot_ref_feats[cnt,:,:,:] = feats.squeeze(0)
            
            patch_i, patch_j = getPatch((target_x, target_y), patch_size)
            annot_patch_coords[cnt,0] = patch_i
            annot_patch_coords[cnt,1] = patch_j
            vector_point = feats[:, :, patch_j, patch_i]

            curr_sim = torch.einsum("chw,c->hw", F.normalize(feats.squeeze(0),dim=0), F.normalize(vector_point.squeeze(0),dim=0))

            annot_feat_vectors[cnt,:] = vector_point.squeeze(0)
            cls_mask.append(cls_cnt)
            sims[cnt//random_points, cnt%random_points,:,:] = curr_sim
            
        single_img_paths = list(dict.fromkeys(imgs_paths))
        repeated_imgs = torch.zeros(len(single_img_paths))

        # find how many times each annotated image is repeated 
        for i , i_img in enumerate(single_img_paths):
            for j_img in imgs_paths: 
                if i_img == j_img : 
                    repeated_imgs[i] += 1 
        
        # find for each annot image the closest image and compute the shift based on that

        i_cnt = 0
        gt_shifts = []
        for step in repeated_imgs:
            step = int(step)
            curr_img = imgs_paths[i_cnt]
            print(f'curr_img -> {curr_img}')
            
            for i in range(0,1):    
                
                raw_img = Image.open(curr_img)
                raw_img = raw_img.convert("RGB")
                raw_img = ImageOps.exif_transpose(raw_img)
                
                tag_raw_img = raw_img
                # tag_raw_img = Image.open(pos_img_path)
                # tag_raw_img = tag_raw_img.convert("RGB")
                # tag_raw_img = ImageOps.exif_transpose(tag_raw_img)


                sims = [] 
                for point_i in  annot_points[i_cnt:i_cnt+step,:] : 
                    sim = get_sim(net, res, patch_size, raw_img,tag_raw_img,(int(point_i[0]),int(point_i[1])))
                    sims.append(sim)

                sims = torch.stack(sims,dim=0)
                sim = torch.mean(sims,dim=0)

                curr_gt_shift = shift_from_sim(sim)
                gt_shifts.append(curr_gt_shift)
                print('Current shift of the image: ', curr_gt_shift)
            i_cnt += step

            
        gt_shift = np.mean(gt_shifts) 

    
        print(f'Ground truth shift (b_proto): {gt_shift}')
        point_data[pcls]= {
            'feature_vectors' : annot_feat_vectors,
            'point' : annot_points, 
            'patch_coord' : annot_patch_coords,
            'ref_feats': annot_ref_feats,
            'shift': gt_shift
        }
            
    
    sum_feature_vectors = [point_data[pcls]['feature_vectors'] for pcls in annotations.keys()]
    sum_point = [point_data[pcls]['point'] for pcls in annotations.keys()]
    sum_patch_coord = [point_data[pcls]['patch_coord'] for pcls in annotations.keys()] 
    sum_ref_feats = [point_data[pcls]['ref_feats'] for pcls in annotations.keys()]
    sum_shifts = [point_data[pcls]['shift'] for pcls in annotations.keys()]
    sum_feature_vectors = torch.cat(sum_feature_vectors, dim=0)

    sum_point = torch.cat(sum_point, dim=0)
    sum_patch_coord = torch.cat(sum_patch_coord, dim=0)
    sum_ref_feats = torch.cat(sum_ref_feats, dim=0)
    
    point_data['sum'] = {
        'feature_vectors' : sum_feature_vectors,
        'point': sum_point,
        'patch_coord': sum_patch_coord,
        'ref_feats' : sum_ref_feats,
        'class_mask' :  cls_mask,
        'shifts' : torch.tensor(sum_shifts)
    }
        
        
    
    torch.save(point_data, f'../saved_models/Kvectors_{cfg.dir_dataset_name}_{cfg.category_class}.pt')
if __name__ == "__main__" : 
    # prep_args()
    my_app()