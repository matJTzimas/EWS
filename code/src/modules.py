import torch

from utils import *
import torch.nn.functional as F
import dino.vision_transformer as vits
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class DinoFeaturizer(nn.Module):

    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        patch_size = self.cfg.dino_patch_size
        self.patch_size = patch_size
        self.feat_type = self.cfg.dino_feat_type
        arch = self.cfg.model_type
        self.model = vits.__dict__[arch](
            patch_size=patch_size,
            num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().cuda()
        self.dropout = torch.nn.Dropout2d(p=.1)

        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        if cfg.pretrained_weights is not None:
            state_dict = torch.load(cfg.pretrained_weights, map_location="cpu")
            state_dict = state_dict["teacher"]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            # state_dict = {k.replace("projection_head", "mlp"): v for k, v in state_dict.items()}
            # state_dict = {k.replace("prototypes", "last_layer"): v for k, v in state_dict.items()}

            msg = self.model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(cfg.pretrained_weights, msg))
        else:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)

        if arch == "vit_small":
            self.n_feats = 384
        else:
            self.n_feats = 768
            
        self.cluster1BG = self.make_clusterer(self.n_feats) # BackGround cluster 1 
        self.cluster1C = self.make_clusterer(self.n_feats) # classes cluster 1 
        
        self.proj_type = cfg.projection_type
        if self.proj_type == "nonlinear":
            self.cluster2BG = self.make_nonlinear_clusterer(self.n_feats)
            self.cluster2C = self.make_nonlinear_clusterer(self.n_feats)

        curr = sum(p.numel() for p in self.cluster1BG.parameters())
        curr += sum(p.numel() for p in self.cluster2C.parameters())
        print(f'CNN FOR DINO DISTRILATION PARAMETERS {curr*2 }')
            

    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))  # ,

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def forward(self, img, n=1, return_class_feat=False):
        self.model.eval()
        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            # get selected layer activations
            feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            if self.feat_type == "feat":
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            elif self.feat_type == "KK":
                image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
                B, H, I, J, D = image_k.shape
                image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            else:
                raise ValueError("Unknown feat type:{}".format(self.feat_type))

            if return_class_feat:
                return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)

        if self.proj_type is not None:
            codeBG = self.cluster1BG(self.dropout(image_feat))
            codeC = self.cluster1C(self.dropout(image_feat))
            
            if self.proj_type == "nonlinear":
                codeBG += self.cluster2BG(self.dropout(image_feat))
                codeC += self.cluster2C(self.dropout(image_feat))
                
        else:
            codeBG = image_feat
            codeC = image_feat

        if self.cfg.dropout:
            return self.dropout(image_feat), codeBG, codeC
        else:
            return image_feat, codeBG, codeC
        


class ResizeAndClassify(nn.Module):

    def __init__(self, dim: int, size: int, n_classes: int):
        super(ResizeAndClassify, self).__init__()
        self.size = size
        self.predictor = torch.nn.Sequential(
            torch.nn.Conv2d(dim, n_classes, (1, 1)),
            torch.nn.LogSoftmax(1))

    def forward(self, x):
        return F.interpolate(self.predictor.forward(x), self.size, mode="bilinear", align_corners=False)

######
class ClusterLookup(nn.Module):

    def __init__(self, dim: int, n_classes: int,cfg):
        super(ClusterLookup, self).__init__()
        self.n_classes = n_classes
        self.dim = dim


        point_data = torch.load(f'../saved_models/Kvectors_{cfg.dir_dataset_name}_{cfg.category_class}.pt')
        sum_data = point_data['sum']
        sum_feature_vectors = sum_data['feature_vectors'].cuda()
        class_vector = torch.mean(sum_feature_vectors,dim=0,keepdim=True)
        bg_vector = torch.randn(1, dim).cuda()
        init_clusters = torch.cat((bg_vector,class_vector),dim=0)
        self.clusters = torch.nn.Parameter(init_clusters)

        # frozen prototype
        # point_data = torch.load(f'../saved_models/Kvectors_{cfg.dir_dataset_name}.pt')
        # sum_data = point_data['sum']
        # sum_feature_vectors = sum_data['feature_vectors'].cuda()
        # # Compute class_vector and freeze it by registering as a buffer
        # class_vector = torch.mean(sum_feature_vectors, dim=0, keepdim=True)
        # self.register_buffer("class_vector", class_vector)

        # # Initialize bg_vector as a trainable parameter
        # bg_vector = torch.randn(1, dim).cuda()
        # self.bg_vector = torch.nn.Parameter(bg_vector)

        # # Concatenate the frozen class_vector and trainable bg_vector
        # init_clusters = torch.cat((self.bg_vector, self.class_vector), dim=0)

        # self.clusters = init_clusters

        # self.clusters = torch.nn.Parameter(torch.randn(n_classes, dim))
        self.super_loss = torch.nn.MSELoss()
        

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, x, alpha, is_training=True,  log_probs=False):
        normed_clusters = F.normalize(self.clusters, dim=1)
        normed_features = F.normalize(x, dim=1)
        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)

        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) \
                .permute(0, 3, 1, 2).to(torch.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)

        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        
        if log_probs:
            return nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs
######




class compute_probs(nn.Module): 
    def __init__(self, n_classes: int,cfg):
        super(compute_probs, self).__init__()
        self.mode = cfg.mode
        self.dim = cfg.dim
        self.n_classes = n_classes -1 # exclude background
        self.point_data = torch.load(f'../saved_models/Kvectors_{cfg.dir_dataset_name}_{cfg.category_class}.pt')
        self.dcls = self.point_data.keys()
        self.annot_list = [self.point_data[ccls] for ccls in self.dcls]
        
        self.sum_data = self.point_data['sum']
        self.sum_feature_vectors = self.sum_data['feature_vectors'].cuda()
        self.sum_point = self.sum_data['point'].cuda()
        self.sum_patch_coord = self.sum_data['patch_coord'].long().cuda()
        self.sum_ref_feats = self.sum_data['ref_feats'].cuda()
        self.cls_mask = self.sum_data['class_mask']
        self.num_vectors = len(self.cls_mask)

    def select_prototype(self, vectors): 

        distances = torch.zeros(vectors.shape[0])

        for i in range(vectors.shape[0]):
            curr_vec = vectors[i,:]
            curr_vec = curr_vec[None,:].repeat(vectors.shape[0],1)
            eucl_dist = (curr_vec - vectors).pow(2).sum(1).sqrt().sum()
            distances[i] = eucl_dist


        val, ind = torch.max(distances,dim=0)
        prototype_vector = vectors[ind,:]

        return prototype_vector, ind
        
    def generate_prototypes(self, ref_code_c, code, mode='mean'):
        code_vectors = ref_code_c[[i for i in range(self.num_vectors)], :, self.sum_patch_coord[:,1], self.sum_patch_coord[:,0]]
        
        prototypes = torch.zeros((self.n_classes, self.dim)).cuda() # change to be sych with trainconfig


        repeat = torch.zeros((self.n_classes)).cuda()
        for i in range(self.n_classes): 
            for vcls in self.cls_mask:
                if i == vcls :
                    repeat[i] += 1

        if mode == 'median':
            proto_list = [[] for _ in range(self.n_classes)]
            for i, vcls in enumerate(self.cls_mask): 
                proto_list[vcls].append(code_vectors[i,:])
            
            for i in range(self.n_classes): 
                if repeat[i] == 0 : 
                    continue
                cls_vectors = torch.stack(proto_list[i],dim=0)
                curr_prototype, ind = self.select_prototype(cls_vectors)

                prototypes[i,:] = curr_prototype

        elif mode == 'mean':  
            for i, vcls in enumerate(self.cls_mask):
                prototypes[vcls,:] += code_vectors[i,:]
            
            for i in range(self.n_classes): 
                if repeat[i] == 0 : 
                    continue
                prototypes[i,:] = prototypes[i,:] / repeat[i]

        elif mode == 'max':
            prototypes = torch.zeros((code.shape[0], self.n_classes, self.dim)).cuda() # change to be sych with trainconfig

            proto_list = [[] for _ in range(self.n_classes)]
            for i, vcls in enumerate(self.cls_mask): 
                proto_list[vcls].append(code_vectors[i,:])
            
            for i in range(self.n_classes): 
                if repeat[i] == 0 : 
                    continue
                cls_vectors = torch.stack(proto_list[i],dim=0)
                code_map = torch.einsum("bchw,nc->bnhw", norm(code), norm(cls_vectors))
                code_map_max = code_map.reshape(code_map.shape[0],code_map.shape[1]*code_map.shape[2]*code_map.shape[3])
                _, ind = code_map_max.max(dim=1)
                ind = ind // (code_map.shape[2]*code_map.shape[3])
                for b in range(code.shape[0]):
                    prototypes[b,i,:] = cls_vectors[ind[b],:]
        elif mode == 'softmax':
            prototypes = torch.zeros((code.shape[0], self.n_classes, self.dim)).cuda() # change to be sych with trainconfig

            proto_list = [[] for _ in range(self.n_classes)]
            for i, vcls in enumerate(self.cls_mask): 
                proto_list[vcls].append(code_vectors[i,:])
            
            for i in range(self.n_classes): 
                if repeat[i] == 0 : 
                    continue
                cls_vectors = torch.stack(proto_list[i],dim=0)
                code_map = torch.einsum("bchw,nc->bnhw", norm(code), norm(cls_vectors))
                code_map_max = code_map.reshape(code_map.shape[0],code_map.shape[1], code_map.shape[2]*code_map.shape[3])
                val, ind = code_map_max.max(dim=2)
                weights = torch.softmax(val,dim=1)
                weights = weights[:,:,None].repeat(1,1,cls_vectors.shape[1])
                weighted_vectors = weights * cls_vectors
                prototypes[:,i,:] = weighted_vectors.sum(dim=1)
        else : 
            print('WRONG PROTOTYPE')

        return prototypes, repeat
        
    def forward(self, code_C, code_BG, ref_code_c, ref_code_bg):
        
        code_BG = norm(code_BG)
        code_C = norm(code_C)

        prototypes, repeat = self.generate_prototypes(ref_code_c, code_C, mode=self.mode)

        if self.mode == 'max' or self.mode == 'softmax':
            prob_C = torch.einsum("bchw, bcv->bvhw", code_C, norm(prototypes.permute(0,2,1)))
        else : 
            prob_C = torch.einsum("bchw, vc->bvhw", code_C, norm(prototypes))

        all_prob_map = torch.sum(prob_C, dim=1)
            
        flat_prob_all = all_prob_map.view(all_prob_map.shape[0], -1)
        min_BG_value, min_BG_index = torch.min(flat_prob_all, dim=1) # find the coords of the min

        min_BG_value, min_BG_index = torch.topk(-flat_prob_all, dim=1,k=1) # find the coords of the min
        # print(min_BG_value)
        i_x, i_y = min_BG_index // all_prob_map.shape[2] , min_BG_index % all_prob_map.shape[2]


        batch_index = torch.tensor([i for i in range(code_BG.shape[0])]).cuda()


        probs_bg = [] 
        for bg in range(i_x.shape[1]) : 
            min_BG_index = torch.stack((batch_index, i_x[:,bg],i_y[:,bg]), dim=1)

            min_vector_BG = code_BG[min_BG_index[:,0], :, min_BG_index[:,1], min_BG_index[:,2]]
            prob_BG = torch.einsum("bchw, bc->bhw", code_BG, min_vector_BG)
            prob_BG = prob_BG[:,None,:,:]
            probs_bg.append(prob_BG)
        probs_bg = torch.cat(probs_bg,dim=1)
        
        probs_bg = torch.max(probs_bg,dim=1).values
        # probs_bg = torch.mean(probs_bg,dim=1,keepdim=False)


        prob_BG = probs_bg[:,None,:,:]

        probs = torch.cat((prob_BG, prob_C), dim=1)
        
        cluster_probs = F.one_hot(torch.argmax(probs, dim=1), self.n_classes + 1) \
                 .permute(0, 3, 1, 2).to(torch.float32)
        
        cluster_probs = F.softmax(probs,dim=1)
        return cluster_probs

class FeaturePyramidNet(nn.Module):

    @staticmethod
    def _helper(x):
        # TODO remove this hard coded 56
        return F.interpolate(x, 56, mode="bilinear", align_corners=False).unsqueeze(-1)

    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)),
            LambdaLayer(FeaturePyramidNet._helper))

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)),
            LambdaLayer(FeaturePyramidNet._helper))

    def __init__(self, granularity, cut_model, dim, continuous):
        super(FeaturePyramidNet, self).__init__()
        self.layer_nums = [5, 6, 7]
        self.spatial_resolutions = [7, 14, 28, 56]
        self.feat_channels = [2048, 1024, 512, 3]
        self.extra_channels = [128, 64, 32, 32]
        self.granularity = granularity
        self.encoder = NetWithActivations(cut_model, self.layer_nums)
        self.dim = dim
        self.continuous = continuous
        self.n_feats = self.dim

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        assert granularity in {1, 2, 3, 4}
        self.cluster1 = self.make_clusterer(self.feat_channels[0])
        self.cluster1_nl = self.make_nonlinear_clusterer(self.feat_channels[0])

        if granularity >= 2:
            # self.conv1 = DoubleConv(self.feat_channels[0], self.extra_channels[0])
            # self.conv2 = DoubleConv(self.extra_channels[0] + self.feat_channels[1], self.extra_channels[1])
            self.conv2 = DoubleConv(self.feat_channels[0] + self.feat_channels[1], self.extra_channels[1])
            self.cluster2 = self.make_clusterer(self.extra_channels[1])
        if granularity >= 3:
            self.conv3 = DoubleConv(self.extra_channels[1] + self.feat_channels[2], self.extra_channels[2])
            self.cluster3 = self.make_clusterer(self.extra_channels[2])
        if granularity >= 4:
            self.conv4 = DoubleConv(self.extra_channels[2] + self.feat_channels[3], self.extra_channels[3])
            self.cluster4 = self.make_clusterer(self.extra_channels[3])

    def c(self, x, y):
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)
        low_res_feats = feats[self.layer_nums[-1]]

        all_clusters = []

        # all_clusters.append(self.cluster1(low_res_feats) + self.cluster1_nl(low_res_feats))
        all_clusters.append(self.cluster1(low_res_feats))

        if self.granularity >= 2:
            # f1 = self.conv1(low_res_feats)
            # f1_up = self.up(f1)
            f1_up = self.up(low_res_feats)
            f2 = self.conv2(self.c(f1_up, feats[self.layer_nums[-2]]))
            all_clusters.append(self.cluster2(f2))
        if self.granularity >= 3:
            f2_up = self.up(f2)
            f3 = self.conv3(self.c(f2_up, feats[self.layer_nums[-3]]))
            all_clusters.append(self.cluster3(f3))
        if self.granularity >= 4:
            f3_up = self.up(f3)
            final_size = self.spatial_resolutions[-1]
            f4 = self.conv4(self.c(f3_up, F.interpolate(
                x, (final_size, final_size), mode="bilinear", align_corners=False)))
            all_clusters.append(self.cluster4(f4))

        avg_code = torch.cat(all_clusters, 4).mean(4)

        if self.continuous:
            clusters = avg_code
        else:
            clusters = torch.log_softmax(avg_code, 1)

        return low_res_feats, clusters


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def average_norm(t):
    return t / t.square().sum(1, keepdim=True).sqrt().mean()


def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)


@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size


def sample_nonzero_locations(t, target_size):
    nonzeros = torch.nonzero(t)
    coords = torch.zeros(target_size, dtype=nonzeros.dtype, device=nonzeros.device)
    n = target_size[1] * target_size[2]
    for i in range(t.shape[0]):
        selected_nonzeros = nonzeros[nonzeros[:, 0] == i]
        if selected_nonzeros.shape[0] == 0:
            selected_coords = torch.randint(t.shape[1], size=(n, 2), device=nonzeros.device)
        else:
            selected_coords = selected_nonzeros[torch.randint(len(selected_nonzeros), size=(n,)), 1:]
        coords[i, :, :, :] = selected_coords.reshape(target_size[1], target_size[2], 2)
    coords = coords.to(torch.float32) / t.shape[1]
    coords = coords * 2 - 1
    return torch.flip(coords, dims=[-1])


class ContrastiveCorrelationLoss(nn.Module):

    def __init__(self, cfg, ):
        super(ContrastiveCorrelationLoss, self).__init__()
        self.cfg = cfg
        self.mode = cfg.mode
        self.dim = cfg.dim
        self.n_classes = cfg.dir_dataset_n_classes - 1 
        self.point_data = torch.load(f'../saved_models/Kvectors_{cfg.dir_dataset_name}_{cfg.category_class}.pt')
        self.dcls = self.point_data.keys()
        self.annot_list = [self.point_data[ccls] for ccls in self.dcls]
        
        self.sum_data = self.point_data['sum']
        self.sum_feature_vectors = self.sum_data['feature_vectors'].cuda()
        self.sum_point = self.sum_data['point']
        self.sum_patch_coord = self.sum_data['patch_coord'].long().cuda()
        self.sum_ref_feats = self.sum_data['ref_feats'].cuda()
        self.cls_mask = self.sum_data['class_mask']
        self.shifts = self.sum_data['shifts']
        self.num_vectors = len(self.cls_mask)

    def select_prototype(self, vectors): 

        distances = torch.zeros(vectors.shape[0])

        for i in range(vectors.shape[0]):
            curr_vec = vectors[i,:]
            curr_vec = curr_vec[None,:].repeat(vectors.shape[0],1)
            eucl_dist = (curr_vec - vectors).pow(2).sum(1).sqrt().sum()
            distances[i] = eucl_dist


        val, ind = torch.max(distances,dim=0)
        prototype_vector = vectors[ind,:]

        return prototype_vector, ind
        
    def generate_prototypes(self, ref_code_c, code, mode='mean'):
        code_vectors = ref_code_c[[i for i in range(self.num_vectors)], :, self.sum_patch_coord[:,1], self.sum_patch_coord[:,0]]
        
        prototypes = torch.zeros((self.n_classes, self.dim)).cuda() # change to be sych with trainconfig


        repeat = torch.zeros((self.n_classes)).cuda()
        for i in range(self.n_classes): 
            for vcls in self.cls_mask:
                if i == vcls :
                    repeat[i] += 1

        if mode == 'median':
            proto_list = [[] for _ in range(self.n_classes)]
            for i, vcls in enumerate(self.cls_mask): 
                proto_list[vcls].append(code_vectors[i,:])
            
            for i in range(self.n_classes): 
                if repeat[i] == 0 : 
                    continue
                cls_vectors = torch.stack(proto_list[i],dim=0)
                curr_prototype, ind = self.select_prototype(cls_vectors)

                prototypes[i,:] = curr_prototype

        elif mode == 'mean':  
            for i, vcls in enumerate(self.cls_mask):
                prototypes[vcls,:] += code_vectors[i,:]
            
            for i in range(self.n_classes): 
                if repeat[i] == 0 : 
                    continue
                prototypes[i,:] = prototypes[i,:] / repeat[i]

        elif mode == 'max':
            prototypes = torch.zeros((code.shape[0], self.n_classes, self.dim)).cuda() # change to be sych with trainconfig

            proto_list = [[] for _ in range(self.n_classes)]
            for i, vcls in enumerate(self.cls_mask): 
                proto_list[vcls].append(code_vectors[i,:])
            
            for i in range(self.n_classes): 
                if repeat[i] == 0 : 
                    continue
                cls_vectors = torch.stack(proto_list[i],dim=0)
                code_map = torch.einsum("bchw,nc->bnhw", norm(code), norm(cls_vectors))
                code_map_max = code_map.reshape(code_map.shape[0],code_map.shape[1]*code_map.shape[2]*code_map.shape[3])
                _, ind = code_map_max.max(dim=1)
                ind = ind // (code_map.shape[2]*code_map.shape[3])
                for b in range(code.shape[0]):
                    prototypes[b,i,:] = cls_vectors[ind[b],:]

        elif mode == 'softmax':
            prototypes = torch.zeros((code.shape[0], self.n_classes, self.dim)).cuda() # change to be sych with trainconfig

            proto_list = [[] for _ in range(self.n_classes)]
            for i, vcls in enumerate(self.cls_mask): 
                proto_list[vcls].append(code_vectors[i,:])
            
            for i in range(self.n_classes): 
                if repeat[i] == 0 : 
                    continue
                cls_vectors = torch.stack(proto_list[i],dim=0)
                code_map = torch.einsum("bchw,nc->bnhw", norm(code), norm(cls_vectors))
                code_map_max = code_map.reshape(code_map.shape[0],code_map.shape[1], code_map.shape[2]*code_map.shape[3])
                val, ind = code_map_max.max(dim=2)
                weights = torch.softmax(val,dim=1)
                weights = weights[:,:,None].repeat(1,1,cls_vectors.shape[1])
                weighted_vectors = weights * cls_vectors
                prototypes[:,i,:] = weighted_vectors.sum(dim=1)
        else : 
            print('WRONG PROTOTYPE')
        
        return prototypes, repeat
                     
    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = tensor_correlation(norm(f1), norm(f2))

            if self.cfg.pointwise:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))

        shift = shift[:,:,:,:,None]
        shift = shift.repeat(1,1,1,cd.shape[2],cd.shape[3])

        if self.cfg.zero_clamp:
            min_val = 0.0
        else:
            min_val = -9999.0

        if self.cfg.stabalize:
            loss = - cd.clamp(min_val, .8) * (fd - shift)
        else:
            loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd
    

    def shift_from_sim(self, sim): 
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

    def generate_class_maps(self, class_maps, repeat, mode='mean'):
        
        final_map = torch.zeros((class_maps.shape[0], self.n_classes, class_maps.shape[2],class_maps.shape[3]))
        
        cnt = 0
     
        for i, step in enumerate(repeat):
            step = int(step.item()) 
            temp_tensor = class_maps[:,cnt:cnt+step,:,:]

            if mode == 'mean':
                temp_tensor = torch.mean(temp_tensor,dim=1,keepdim=False)
            else : 
                temp_tensor = torch.max(temp_tensor,dim=1).values
            # print(temp_tensor.size())
            final_map[:,i,:,:] = temp_tensor
            cnt = cnt+step
        return final_map
    
    def class_supervised_helper(self, feats, code, ref_code, shift,mtype): 


        norm_f1 = norm(feats)
        
        
        # Get the prototypes from the reference code
        with torch.no_grad(): 
            
            prototype, repeat = self.generate_prototypes(ref_code,code,mode=self.mode)

            class_feats_map = torch.einsum("bchw,nc->bnhw", norm_f1, norm(self.sum_feature_vectors))             
            max_class_maps = self.generate_class_maps(class_feats_map, repeat,mode='max')
            mean_class_maps = self.generate_class_maps(class_feats_map, repeat,mode='mean')

            if mtype == 'c':
                shift = torch.tensor(0.24)
                shift = shift.cuda()
            else : 
                shift = torch.tensor(0.24)
                shift = shift.cuda()

            shift = self.shifts.cuda()

            # shift = torch.zeros(max_class_maps.size())
            # for i in range(max_class_maps.shape[0]):
            #     for j in range(max_class_maps.shape[1]):
            #         curr_shift = self.shift_from_sim(mean_class_maps[i,j,:,:])
            #         # print(curr_shift)
            #         shift[i,j,:,:] = curr_shift

            # print('max ', torch.mean(max_class_maps.flatten()))
            # print('mean', torch.mean(mean_class_maps.flatten()))

            # shift = shift.cuda()

        if self.mode == 'max' or self.mode == 'softmax':
            code_map = torch.einsum("bchw,bcn->bnhw", norm(code), norm(prototype.permute(0,2,1)))
        else : 
            code_map = torch.einsum("bchw,nc->bnhw", norm(code), norm(prototype))

        # self.save_feature_maps_as_images(torch.clone(code_map.detach())) 
        
        if self.cfg.zero_clamp: # default true 
            min_val = 0.0
        else:
            min_val = -9999.0
        
        loss = - code_map.clamp(min_val) * (mean_class_maps.cuda() - shift) 
        
        return loss, code_map
    
    
    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2
    
    def save_histograms(self, fire_map, output_dir='/media/FastData/dtzimas/STEGOS/RESULTS/histograms', bins=20):
        # os.makedirs(output_dir, exist_ok=True)
        
        batch_size, number_of_examples, height, width = fire_map.size()
        
        # Loop through each feature map
        for i in range(batch_size):
            for j in range(number_of_examples):
                feature_map = fire_map[i, j, :, :].cpu().numpy().flatten()
                hist, bin_edges = np.histogram(feature_map, bins=bins)
                
                plt.figure()
                plt.hist(feature_map, bins=bin_edges)
                plt.title(f'Histogram for batch {i}, example {j}')
                plt.xlabel('Pixel Value')
                plt.ylabel('Frequency')
                
                # Save the histogram
                hist_path = os.path.join(output_dir, f'histogram_batch_{i}_example_{j}.png')
                plt.savefig(hist_path)
                plt.close()
                
                
    def save_feature_maps_as_images(self, tensor, output_dir='/media/FastData/dtzimas/STEGOS/RESULTS/feature_maps'):
        os.makedirs(output_dir, exist_ok=True)
        
        batch_size, number_of_examples, height, width = tensor.size()
        
        # Loop through each feature map
        for i in range(batch_size):
            for j in range(number_of_examples):
                feature_map = tensor[i, j, :, :].cpu().numpy()
                
                plt.figure()
                plt.imshow(feature_map, cmap='gray')
                plt.title(f'Feature map for batch {i}, example {j}')
                plt.colorbar()
                
                # Save the feature map as an image
                image_path = os.path.join(output_dir, f'feature_map_batch_{i}_example_{j}.png')
                plt.savefig(image_path)
                plt.close()
                
                
    def forward(self,
                orig_feats: torch.Tensor, 
                orig_code_BG: torch.Tensor, 
                orig_code_C: torch.Tensor, 
                ref_code_c: torch.Tensor, ref_code_bg: torch.Tensor,
                shifts : torch.Tensor
                ):
        
        # feature samples = 11 
        coord_shape = [orig_feats.shape[0], self.cfg.feature_samples, self.cfg.feature_samples, 2]

        # default false
        if self.cfg.use_salience:
            coords1_reg = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            coords2_reg = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            mask = (torch.rand(coord_shape[:-1], device=orig_feats.device) > .1).unsqueeze(-1).to(torch.float32)
            coords1 = coords1_nonzero * mask + coords1_reg * (1 - mask)
            coords2 = coords2_nonzero * mask + coords2_reg * (1 - mask)
        else:
            #giati kanei sample me vasi ena random coordinates megethous 11
            coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            #[0,1] *2-1 -> [-1,1] gia to grid sample
        
        
        #compute the sampled 
        # feats = sample(orig_feats, coords1)
        # code_BG = sample(orig_code_BG, coords1)
        # code_C = sample(orig_code_C, coords1)
        # shifts = sample(shifts[:,:,:,None].permute(0,3,1,2),coords1)
        # shifts = shifts.permute(0,2,3,1)

        # # feats -> [btach, 384 , sample size, sample size]
        # feats_pos = sample(orig_feats_pos, coords2)
        # code_pos_BG = sample(orig_code_pos_BG, coords2)
        # code_pos_C = sample(orig_code_pos_C, coords2)

        shifts = shifts[:,:,:,None]

        if self.cfg.pos_intra_weight > 0 :

            pos_intra_loss_C, pos_intra_cd_C = self.helper(
                orig_feats, orig_feats, orig_code_C, orig_code_C, shifts)
        else : 
            pos_intra_loss_C = pos_intra_cd_C = torch.tensor(0.0, requires_grad=True, device='cuda')
        
        if self.cfg.class_intra_weight > 0 :
            
            class_intra_loss_C, class_intra_cd_C = self.class_supervised_helper(
                orig_feats, orig_code_C, ref_code_c, self.cfg.class_intra_shift,'c'
            )
            
            class_intra_loss_BG, class_intra_cd_BG = self.class_supervised_helper(
                orig_feats, orig_code_BG, ref_code_bg, self.cfg.class_intra_shift,'bg'
            )
        else : 
            class_intra_loss_BG = class_intra_cd_BG = torch.tensor(0.0, requires_grad=True, device='cuda')
            class_intra_loss_C = class_intra_cd_C = torch.tensor(0.0, requires_grad=True, device='cuda')

            # gmm = GaussianMixture(n_components=2, random_state=42)
            # gmm.fit(class_intra_cd_BG.flatten().reshape(-1,1).cpu().detach().numpy())
            # means = gmm.means_.flatten()

            # print(f'Background dif : {abs(means[0] - means[1])}')

        
        return (pos_intra_loss_C.mean(),
                pos_intra_cd_C,
                class_intra_loss_BG.mean(),
                class_intra_cd_BG,
                class_intra_loss_C.mean(),
                class_intra_cd_C)

class Decoder(nn.Module):
    def __init__(self, code_channels, feat_channels):
        super().__init__()
        self.linear = torch.nn.Conv2d(code_channels, feat_channels, (1, 1))
        self.nonlinear = torch.nn.Sequential(
            torch.nn.Conv2d(code_channels, code_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(code_channels, code_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(code_channels, feat_channels, (1, 1)))

    def forward(self, x):
        return self.linear(x) + self.nonlinear(x)


class NetWithActivations(torch.nn.Module):
    def __init__(self, model, layer_nums):
        super(NetWithActivations, self).__init__()
        self.layers = nn.ModuleList(model.children())
        self.layer_nums = []
        for l in layer_nums:
            if l < 0:
                self.layer_nums.append(len(self.layers) + l)
            else:
                self.layer_nums.append(l)
        self.layer_nums = set(sorted(self.layer_nums))

    def forward(self, x):
        activations = {}
        for ln, l in enumerate(self.layers):
            x = l(x)
            if ln in self.layer_nums:
                activations[ln] = x
        return activations


class ContrastiveCRFLoss(nn.Module):

    def __init__(self, n_samples, alpha, beta, gamma, w1, w2, shift):
        super(ContrastiveCRFLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.w1 = w1
        self.w2 = w2
        self.n_samples = n_samples
        self.shift = shift

    def forward(self, guidance, clusters):
        device = clusters.device
        assert (guidance.shape[0] == clusters.shape[0])
        assert (guidance.shape[2:] == clusters.shape[2:])
        h = guidance.shape[2]
        w = guidance.shape[3]

        coords = torch.cat([
            torch.randint(0, h, size=[1, self.n_samples], device=device),
            torch.randint(0, w, size=[1, self.n_samples], device=device)], 0)

        selected_guidance = guidance[:, :, coords[0, :], coords[1, :]]
        coord_diff = (coords.unsqueeze(-1) - coords.unsqueeze(1)).square().sum(0).unsqueeze(0)
        guidance_diff = (selected_guidance.unsqueeze(-1) - selected_guidance.unsqueeze(2)).square().sum(1)

        sim_kernel = self.w1 * torch.exp(- coord_diff / (2 * self.alpha) - guidance_diff / (2 * self.beta)) + \
                     self.w2 * torch.exp(- coord_diff / (2 * self.gamma)) - self.shift

        selected_clusters = clusters[:, :, coords[0, :], coords[1, :]]
        cluster_sims = torch.einsum("nka,nkb->nab", selected_clusters, selected_clusters)
        return -(cluster_sims * sim_kernel)
