from utils import *
from modules import *
from data import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
import torch.multiprocessing
import seaborn as sns
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
from torch.optim.lr_scheduler import StepLR
import time
from crf import dense_crf
import json 

torch.multiprocessing.set_sharing_strategy('file_system')

def get_class_labels(dataset_name):
    if dataset_name.startswith("cityscapes"):
        return [
            'road', 'sidewalk', 'parking', 'rail track', 'building',
            'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'caravan', 'trailer', 'train',
            'motorcycle', 'bicycle']
    elif dataset_name == "cocostuff27":
        return [
            "electronic", "appliance", "food", "furniture", "indoor",
            "kitchen", "accessory", "animal", "outdoor", "person",
            "sports", "vehicle", "ceiling", "floor", "food",
            "furniture", "rawmaterial", "textile", "wall", "window",
            "building", "ground", "plant", "sky", "solid",
            "structural", "water"]
    elif dataset_name == "voc":
        return [
            'background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif dataset_name == "potsdam":
        return [
            'roads and cars',
            'buildings and clutter',
            'trees and vegetation']
    elif dataset_name == "directory" :
        #  return [
        #     'background',
        #     'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        #     'bus', 'car', 'cat', 'chair', 'cow',
        #     'diningtable', 'dog', 'horse', 'motorbike', 'person',
        #     'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'] 
        # return ['Background', 'Fire', 'Flood']
        return ['Background', 'bird']
        
    else:
        raise ValueError("Unknown Dataset {}".format(dataset_name))


class LitUnsupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        self.fire_target = torch.load(f'../saved_models/Kvectors_{cfg.dir_dataset_name}_{cfg.category_class}.pt')
        
        if not cfg.continuous:
            dim = n_classes
        else:
            dim = cfg.dim

        data_dir = join(cfg.output_root, "data")
        if cfg.arch == "feature-pyramid":
            cut_model = load_model(cfg.model_type, data_dir).cuda()
            self.net = FeaturePyramidNet(cfg.granularity, cut_model, dim, cfg.continuous)
        elif cfg.arch == "dino":
            self.net = DinoFeaturizer(dim, cfg)
        else:
            raise ValueError("Unknown arch {}".format(cfg.arch))

        # self.train_cluster_probe = ClusterLookup(dim, n_classes)
        
        self.compute_result = compute_probs(n_classes=n_classes,cfg=cfg)
        self.cluster_probe = ClusterLookup(384 , n_classes + cfg.extra_clusters,cfg)
        self.linear_probe = nn.Conv2d(dim, n_classes, (1, 1))

        self.decoder = nn.Conv2d(dim, self.net.n_feats, (1, 1))

        self.Ucluster_metrics = UnsupervisedMetrics(
            "test/UNSUPERVISED/", n_classes, cfg.extra_clusters, True)
        self.cluster_metrics = UnsupervisedMetrics(
            "test/EWS/", n_classes, cfg.extra_clusters, True)
        self.linear_metrics = UnsupervisedMetrics(
            "test/SUPERVISED/", n_classes, 0, False)

        self.test_cluster_metrics = UnsupervisedMetrics(
            "final/cluster/", n_classes, cfg.extra_clusters, True)
        self.test_linear_metrics = UnsupervisedMetrics(
            "final/linear/", n_classes, 0, False)

        self.linear_probe_loss_fn = torch.nn.CrossEntropyLoss()
        self.crf_loss_fn = ContrastiveCRFLoss(
            cfg.crf_samples, cfg.alpha, cfg.beta, cfg.gamma, cfg.w1, cfg.w2, cfg.shift)

        self.contrastive_corr_loss_fn = ContrastiveCorrelationLoss(cfg)
        for p in self.contrastive_corr_loss_fn.parameters():
            p.requires_grad = False

        self.automatic_optimization = False
        
        if self.cfg.dataset_name.startswith("cityscapes"):
            self.label_cmap = create_cityscapes_colormap()
        else:
            self.label_cmap = create_pascal_label_colormap()
        
        self.point_data = torch.load(f'../saved_models/Kvectors_{cfg.dir_dataset_name}_{cfg.category_class}.pt')
        self.dcls = self.point_data.keys()
        self.annot_list = [self.point_data[ccls] for ccls in self.dcls]
        
        self.sum_data = self.point_data['sum']
        self.sum_feature_vectors = self.sum_data['feature_vectors']
        self.sum_point = self.sum_data['point']
        self.sum_patch_coord = self.sum_data['patch_coord']
        self.sum_ref_feats = self.sum_data['ref_feats'].cuda()
        self.cls_mask = self.sum_data['class_mask']

        self.cluster_miou = 0
        self.Ucluster_miou = 0
        self.linear_miou = 0
        self.cluster_acc = 0
        self.Ucluster_acc = 0
        self.linear_acc = 0


        self.val_steps = 0
    
        self.stop = 0 
        self.save_hyperparameters()
        
        
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)[1]
    
    def _apply_crf(self,tup):
        return dense_crf(tup[0], tup[1])
    
    def batched_crf(self, img_tensor, prob_tensor): # TODO this works since pool is not utilized
        outputs = [self._apply_crf((img, prob)) for img, prob in zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu())]
        return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)
    
    def recursive_chmod(self, path, mode):
        for root, dirs, files in os.walk(path):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                os.chmod(dir_path, mode)
            for file_name in files:
                file_path = os.path.join(root, file_name)
                os.chmod(file_path, mode)
        # Change the root directory permissions at the end
        os.chmod(path, mode)
    
    def training_step(self, batch, batch_idx):

        # training_step defined the train loop.
        # It is independent of forward
        net_optim, linear_probe_optim, cluster_probe_optim = self.optimizers()


        net_optim.zero_grad()
        linear_probe_optim.zero_grad()

        with torch.no_grad():
            ind = batch["ind"] # batch index 
            img = batch["img"] # batch current img x 
            img_aug = batch["img_aug"] # augmentated version of current x 
            coord_aug = batch["coord_aug"] # 
            label = batch["label"] # gt map of current x 
            shifts = batch['shifts']


        # updated version for 2 map 
        feats, code_BG, code_C = self.net(img)

            
        # den kserw    
        log_args = dict(sync_dist=False, rank_zero_only=True)

        # default den xrisimopoietai varaei error if true 
        if self.cfg.use_true_labels:
            signal = one_hot_feats(label + 1, self.n_classes + 1)
        else:
            signal = feats

        loss = 0

        # bool gia save logs se current epoch
        should_log_hist = (self.cfg.hist_freq is not None) and \
                          (self.global_step % self.cfg.hist_freq == 0) and \
                          (self.global_step > 0)



        # sum ref feats einai stakarismena ta vit feats poy antistoixoun sta annotation images  
        ref_code_cls = self.net.cluster1C(self.sum_ref_feats)
        ref_code_cls += self.net.cluster2C(self.sum_ref_feats)
        
        ref_code_bg = self.net.cluster1BG(self.sum_ref_feats)
        ref_code_bg += self.net.cluster2BG(self.sum_ref_feats)
        
        # OK here 
        
        # Probably this code does not require any changes
        if self.cfg.correspondence_weight > 0:
            
            (
                pos_intra_loss, pos_intra_cd,
                class_intra_loss_BG, class_intra_cd_BG,
                class_intra_loss_C, class_intra_cd_C
            ) = self.contrastive_corr_loss_fn(
                signal, 
                code_BG,
                code_C,
                ref_code_cls, ref_code_bg,
                shifts
            )

            if should_log_hist:
                self.logger.experiment.add_histogram("intra_cd", pos_intra_cd, self.global_step)

                
            
            # print(f'train pos intra loss -> {pos_intra_loss.mean()}')
            pos_intra_loss = pos_intra_loss.mean()
            class_intra_loss_BG = class_intra_loss_BG.mean()
            class_intra_loss_C = class_intra_loss_C.mean()
            
            self.log('loss/pos_intra', pos_intra_loss, **log_args)

            loss += (self.cfg.class_intra_weight * class_intra_loss_BG )
            
            loss += (
                     self.cfg.pos_intra_weight * pos_intra_loss +
                     self.cfg.class_intra_weight * class_intra_loss_C) * self.cfg.correspondence_weight




        flat_label = label.reshape(-1)
        mask = (flat_label >= 0) & (flat_label < self.n_classes)

        dcode_C = torch.clone(code_C.detach())
        dcode_BG = torch.clone(code_BG.detach())
        
        
        
        # UNSUPERVISED AND SUPERVISED RESULTS. THIS IS NOT FOR OUR CLUSTER HEAD BUT FOR COMPETITORS
        ## few parameters for the supervised segmentation task -> test how algorithm work
        linear_logits = self.linear_probe(dcode_C)
        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, self.n_classes)
        
        linear_loss = self.linear_probe_loss_fn(linear_logits[mask], flat_label[mask]).mean()
        loss += linear_loss
        self.log('loss/linear', linear_loss, **log_args)

        # COMPUTE PROB SCORES
        # cluster_probs = self.compute_result(torch.clone(code_C.detach()), torch.clone(code_BG.detach()), torch.clone(ref_code_cls.detach()),torch.clone(ref_code_bg.detach()))
        cluster_loss, cluster_probs = self.cluster_probe(feats, None)
        loss += cluster_loss
        self.log('loss/cluster', cluster_loss, **log_args)
        self.log('loss/total', loss, **log_args)


        self.manual_backward(loss)
        net_optim.step()

        cluster_probe_optim.step()
        linear_probe_optim.step()


        if self.cfg.reset_probe_steps is not None and self.global_step == self.cfg.reset_probe_steps:
            print("RESETTING PROBES")
            self.linear_probe.reset_parameters()
            self.cluster_probe.reset_parameters()
            self.trainer.optimizers[1] = torch.optim.Adam(list(self.linear_probe.parameters()), lr=5e-2)
            self.trainer.optimizers[2] = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-2)

        if self.global_step % 2000 == 0 and self.global_step > 0:
            print("RESETTING TFEVENT FILE")
            # Make a new tfevent file
            self.logger.experiment.close()
            self.logger.experiment._get_file_writer()



        return loss

    def on_train_start(self):
        tb_metrics = {
            **self.linear_metrics.compute(),
            **self.cluster_metrics.compute(),
            **self.Ucluster_metrics.compute(),
        }
        self.logger.log_hyperparams(self.cfg, tb_metrics)

    def validation_step(self, batch, batch_idx):
        

        img = batch["img"]
        label = batch["label"]
        self.net.eval()
        with torch.no_grad():

            feats, code_BG, code_C = self.net(img)
            
            feats =  F.interpolate(feats, label.shape[-2:], mode='bilinear', align_corners=False)
            code_C = F.interpolate(code_C, label.shape[-2:], mode='bilinear', align_corners=False)
            code_BG = F.interpolate(code_BG, label.shape[-2:], mode='bilinear', align_corners=False)


            ref_code_c = self.net.cluster1C(self.sum_ref_feats)
            ref_code_c += self.net.cluster2C(self.sum_ref_feats)

            ref_code_bg = self.net.cluster1BG(self.sum_ref_feats)
            ref_code_bg += self.net.cluster2BG(self.sum_ref_feats)


            linear_preds = self.linear_probe(code_C)
            linear_preds = linear_preds.argmax(1)
            self.linear_metrics.update(linear_preds, label)

            cluster_preds = self.compute_result(torch.clone(code_C.detach()), torch.clone(code_BG.detach()), torch.clone(ref_code_c.detach()),torch.clone(ref_code_bg.detach()))

            # print(f'cluster preds',cluster_preds.permute(0,2,3,1))
            cluster_preds_softmax = F.softmax(cluster_preds,dim=1)
            # cluster_preds = self.batched_crf(img, cluster_preds_softmax).cuda()

            cluster_preds = cluster_preds.argmax(1)
            self.cluster_metrics.update(cluster_preds, label)

            
            cluster_loss, cluster_probs = self.cluster_probe(feats, 1)
            cluster_probs = cluster_probs.argmax(1)


            self.Ucluster_metrics.update(cluster_probs, label)


            return {
                'img': img[:self.cfg.n_images].detach().cpu(),
                'linear_preds': linear_preds[:self.cfg.n_images].detach().cpu(),
                "cluster_preds": cluster_preds[:self.cfg.n_images].detach().cpu(),
                "Ucluster_probs": cluster_probs[:self.cfg.n_images].detach().cpu(),
                "label": label[:self.cfg.n_images].detach().cpu()}

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        with torch.no_grad():
            tb_metrics = {
                **self.linear_metrics.compute(),
                **self.cluster_metrics.compute(),
                **self.Ucluster_metrics.compute(),
            }

            if self.trainer.is_global_zero:
                #output_num = 0
                output_num = random.randint(0, len(outputs) -1)
                output = {k: v.detach().cpu() for k, v in outputs[output_num].items()}

                fig, ax = plt.subplots(5, self.cfg.n_images, figsize=(self.cfg.n_images * 3, 5 * 3))
                for i in range(self.cfg.n_images):
                    ax[0, i].imshow(prep_for_plot(output["img"][i]))
                    ax[1, i].imshow(self.label_cmap[output["label"][i]][0])
                    ax[2, i].imshow(self.label_cmap[output["linear_preds"][i]])
                    ax[3, i].imshow(self.label_cmap[self.cluster_metrics.map_clusters(output["cluster_preds"][i])])
                    ax[4, i].imshow(self.label_cmap[self.Ucluster_metrics.map_clusters(output["Ucluster_probs"][i])])
                ax[0, 0].set_ylabel("Image", fontsize=16)
                ax[1, 0].set_ylabel("GT", fontsize=16)
                ax[2, 0].set_ylabel("SUPERVISED", fontsize=16)
                ax[3, 0].set_ylabel("EWS", fontsize=16)
                ax[4, 0].set_ylabel("UNSUPERVISED", fontsize=16)

                remove_axes(ax)
                plt.tight_layout()
                add_plot(self.logger.experiment, "plot_labels", self.global_step)

                if self.cfg.has_labels:
                    fig = plt.figure(figsize=(13, 10))
                    ax = fig.gca()
                    hist = self.cluster_metrics.histogram.detach().cpu().to(torch.float32)
                    hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
                    sns.heatmap(hist.t(), annot=False, fmt='g', ax=ax, cmap="Blues")
                    ax.set_xlabel('Predicted labels')
                    ax.set_ylabel('True labels')
                    names = get_class_labels(self.cfg.dataset_name)
                    if self.cfg.extra_clusters:
                        names = names + ["Extra"]
                    ax.set_xticks(np.arange(0, len(names)) + .5)
                    ax.set_yticks(np.arange(0, len(names)) + .5)
                    ax.xaxis.tick_top()
                    ax.xaxis.set_ticklabels(names, fontsize=14)
                    ax.yaxis.set_ticklabels(names, fontsize=14)
                    colors = [self.label_cmap[i] / 255.0 for i in range(len(names))]
                    [t.set_color(colors[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
                    [t.set_color(colors[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
                    # ax.yaxis.get_ticklabels()[-1].set_color(self.label_cmap[0] / 255.0)
                    # ax.xaxis.get_ticklabels()[-1].set_color(self.label_cmap[0] / 255.0)
                    plt.xticks(rotation=90)
                    plt.yticks(rotation=0)
                    ax.vlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_xlim())
                    ax.hlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_ylim())
                    plt.tight_layout()
                    add_plot(self.logger.experiment, "conf_matrix", self.global_step)

                    all_bars = torch.cat([
                        self.cluster_metrics.histogram.sum(0).cpu(),
                        self.cluster_metrics.histogram.sum(1).cpu()
                    ], axis=0)
                    ymin = max(all_bars.min() * .8, 1)
                    ymax = all_bars.max() * 1.2

                    fig, ax = plt.subplots(1, 2, figsize=(2 * 5, 1 * 4))
                    ax[0].bar(range(self.n_classes),
                              self.cluster_metrics.histogram.sum(0).cpu(),
                              tick_label=names,
                              color=colors)
                    ax[0].set_ylim(ymin, ymax)
                    ax[0].set_title("Label Frequency")
                    ax[0].set_yscale('log')
                    ax[0].tick_params(axis='x', labelrotation=90)

                    ax[1].bar(range(self.n_classes + self.cfg.extra_clusters),
                              self.cluster_metrics.histogram.sum(1).cpu(),
                              tick_label=names,
                              color=colors)
                    ax[1].set_ylim(ymin, ymax)
                    ax[1].set_title("Cluster Frequency")
                    ax[1].set_yscale('log')
                    ax[1].tick_params(axis='x', labelrotation=90)

                    plt.tight_layout()
                    add_plot(self.logger.experiment, "label frequency", self.global_step)

            if self.global_step > 2:
                self.log_dict(tb_metrics)

                # if self.trainer.is_global_zero and self.cfg.azureml_logging:
                #     from azureml.core.run import Run
                #     run_logger = Run.get_context()
                #     for metric, value in tb_metrics.items():
                #         run_logger.log(metric, value)

            
            if tb_metrics["test/EWS/mIoU"] > self.cluster_miou : 
                self.cluster_miou = tb_metrics["test/EWS/mIoU"]
                self.cluster_acc = tb_metrics["test/EWS/Accuracy"]
                self.stop = 0

            if tb_metrics["test/UNSUPERVISED/mIoU"] > self.Ucluster_miou : 
                self.Ucluster_miou = tb_metrics["test/UNSUPERVISED/mIoU"]
                self.Ucluster_acc = tb_metrics["test/UNSUPERVISED/Accuracy"]

            if tb_metrics["test/SUPERVISED/mIoU"] > self.linear_miou : 
                self.linear_miou = tb_metrics["test/SUPERVISED/mIoU"]
                self.linear_acc = tb_metrics["test/SUPERVISED/Accuracy"]

            print(f'\nUNSUPERVISED -> {tb_metrics["test/UNSUPERVISED/mIoU"]} || max D+P miou {self.Ucluster_miou}')
            print(f'EWS -> {tb_metrics["test/EWS/mIoU"]} || max EWS miou {self.cluster_miou}')
            print(f'SUPERVISED -> {tb_metrics["test/SUPERVISED/mIoU"]} || max SS miou {self.linear_miou}')

            self.linear_metrics.reset()
            self.cluster_metrics.reset()
            self.Ucluster_metrics.reset()

            self.stop += 1 
            if self.stop > 5 : 
                sys.exit(1)


    def configure_optimizers(self):
        main_params = list(self.net.parameters())

        if self.cfg.rec_weight > 0:
            main_params.extend(self.decoder.parameters())

        net_optim = torch.optim.Adam(main_params, lr=self.cfg.lr)
        # net_optim = torch.optim.SGD(main_params, lr=self.cfg.lr)


        linear_probe_optim = torch.optim.Adam(list(self.linear_probe.parameters()), lr=5e-3)
        cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-2)

        return net_optim, linear_probe_optim, cluster_probe_optim

@hydra.main(version_base=None, config_path="configs", config_name="train_config.yaml")
def my_app(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    # print('Omega: ',OmegaConf.to_yaml(cfg))
    pytorch_data_dir = cfg.pytorch_data_dir
    data_dir = join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    checkpoint_dir = join(cfg.output_root, "checkpoints")
    json_dir = join(cfg.output_root, "json_dir")

    if cfg.experiment_name == "auto": 
        prefix = "{}/{}_{}_{}IMGS_{}POINTS_".format(cfg.log_dir, cfg.dir_dataset_name, cfg.category_class, cfg.random_imgs, cfg.random_points)
        name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
        cfg.full_name = prefix

    else: 
        prefix = "{}/{}_{}".format(cfg.log_dir, cfg.dir_dataset_name, cfg.experiment_name)
        name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
        cfg.full_name = prefix

    print('\n\n\n---------------------------------------------------------------------------------------\n')
    print(f'EXPERIMENT {cfg.dir_dataset_name} IMGS {cfg.random_imgs} + POINTS {cfg.random_points} + SEED {cfg.seed}')
    print('\n---------------------------------------------------------------------------------------\n\n\n')

    # print(data_dir)
    # print(cfg.output_root)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    seed_everything(seed=0)



    geometric_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=cfg.res, scale=(0.8, 1.0))
    ])
    photometric_transforms = T.Compose([
        T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        T.RandomGrayscale(.2),
        T.RandomApply([T.GaussianBlur((5, 5))])
    ])

    sys.stdout.flush()

    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=get_transform(cfg.res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.res, True, cfg.loader_crop_type),
        cfg=cfg,
        aug_geometric_transform=geometric_transforms,
        aug_photometric_transform=photometric_transforms,
        num_neighbors=1,
        mask=True,
        pos_images=True,
        pos_labels=True
    )

    if cfg.dataset_name == "voc":
        val_loader_crop = None
    else:
        val_loader_crop = "center"

    # print("checking the dataset name : ", pytorch_data_dir)
    val_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(320, False, val_loader_crop),
        target_transform=get_transform(320, True, val_loader_crop),
        mask=True,
        cfg=cfg,
    )

    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)


    val_batch_size = cfg.batch_size

    val_loader = DataLoader(val_dataset, val_batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    model = LitUnsupervisedSegmenter(train_dataset.n_classes, cfg)

    tb_logger = TensorBoardLogger(
        join(log_dir, name),
        default_hp_metric=False
    )

    # if cfg.submitting_to_aml:
    #     gpu_args = dict(gpus=1, val_check_interval=250)

    #     if gpu_args["val_check_interval"] > len(train_loader):
    #         gpu_args.pop("val_check_interval")

    # else:
    gpu_args = dict(gpus=-1, accelerator='ddp', val_check_interval=cfg.val_freq)
    # gpu_args = dict(gpus=1, accelerator='ddp', val_check_interval=cfg.val_freq)

    if gpu_args["val_check_interval"] > len(train_loader) // 4:
        gpu_args.pop("val_check_interval")

    trainer = Trainer(
        log_every_n_steps=cfg.scalar_log_freq,
        logger=tb_logger,
        max_steps=cfg.max_steps,
        callbacks=[
            ModelCheckpoint(
                dirpath=join(checkpoint_dir, name),
                every_n_train_steps=400,
                save_top_k=2,
                monitor="test/EWS/mIoU",
                mode="max",
            )
        ],
        enable_progress_bar=False,
        **gpu_args
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    prep_args()
    my_app()
