import random
from random import sample
import argparse
import numpy as np
import os
import h5py
import itertools
from sympy import arg
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
import scipy.spatial.distance as SSD
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
import datasets.mvtec as mvtec

import timm
from utils import compute_pca, pca_reduction

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('-d', '--data_path', type=str, default='D:/dataset/mvtec_anomaly_detection')
    parser.add_argument('-s', '--save_path', type=str, default='./mvtec_result')
    parser.add_argument('-a', '--arch', type=str, choices=['resnet18', 'wide_resnet50_2',
     'efficientnetv2_m_in21ft1k', 'efficientnetv2_xl_in21ft1k', 
     'efficientnet_b5_ns', 'efficientnet_b6_ns', 'efficientnet_b7_ns', 
     'efficientnet_l2_ns_475'], default='efficientnetv2_m_in21ft1k')
    parser.add_argument('-r', '--reduce_dim', action='store_true')
    parser.add_argument('-p', '--pca', action="store_true", help="Enable pca")
    parser.add_argument('-n', '--npca', action="store_true", help="Enable npca")
    parser.add_argument('-v', '--variance_threshold', type=float, default=0.99, help="Variance threshold to apply")
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--save_gpu_memory', action='store_true', help='In case of gpu OOM')
    return parser.parse_args()


def main():

    args = parse_args()

    # load model
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    elif args.arch == 'efficientnetv2_m_in21ft1k':
        model = timm.create_model('tf_efficientnetv2_m_in21ft1k', pretrained=True)
        t_d = (80 + 160 + 304) # (48 + 80 + 160 + 176 + 304) features1,2,3,4,5
        d = 100
    elif args.arch == 'efficientnetv2_xl_in21ft1k':
        model = timm.create_model('tf_efficientnetv2_xl_in21ft1k', pretrained=True)
        t_d = (192 + 256 + 512) # (64 + 96 + 192 + 256 + 512) features1,2,3,4,5
        d = 100
    elif args.arch == 'efficientnet_b5_ns':
        model = timm.create_model('tf_efficientnet_b5_ns', pretrained=True)
        t_d = (40 + 64 + 176) # (40 + 64 + 128 + 176 + 304) features1,2,3,4,5
        d = 100
    elif args.arch == 'efficientnet_b6_ns':
        model = timm.create_model('tf_efficientnet_b6_ns', pretrained=True)
        t_d = (40 + 72 + 200) # (40 + 72 + 144 + 200 + 344) features1,2,3,4,5
        d = 100
    elif args.arch == 'efficientnet_b7_ns':
        model = timm.create_model('tf_efficientnet_b7_ns', pretrained=True)
        t_d = (48 + 80 + 224) # (48 + 80 + 160 + 224 + 384) features1,2,3,4,5
        d = 100
    elif args.arch == 'efficientnet_l2_ns_475':
        model = timm.create_model('tf_efficientnet_l2_ns_475', pretrained=True)
        t_d = (104 + 344 + 480) # (104 + 176 + 344 + 480 + 824) features1,2,3,4,5
        d = 550

    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    if args.reduce_dim:
        idx = torch.tensor(sample(range(0, t_d), d))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    if 'resnet' in args.arch:
        model.layer1[-1].register_forward_hook(hook)
        model.layer2[-1].register_forward_hook(hook)
        model.layer3[-1].register_forward_hook(hook)
    elif args.arch == 'efficientnetv2_m_in21ft1k':
        model.blocks[2][-1].register_forward_hook(hook)
        model.blocks[3][-1].register_forward_hook(hook)
        model.blocks[5][-1].register_forward_hook(hook)
    elif args.arch == 'efficientnet_b7_ns':
        model.blocks[1][-1].register_forward_hook(hook)
        model.blocks[3][-1].register_forward_hook(hook)
        model.blocks[4][-1].register_forward_hook(hook)
    elif 'efficientnet' in args.arch:
        model.blocks[1][-1].register_forward_hook(hook)
        # model.blocks[2][-1].register_forward_hook(hook)
        model.blocks[3][-1].register_forward_hook(hook)
        model.blocks[4][-1].register_forward_hook(hook)
        # model.blocks[2][0].register_forward_hook(hook)
        # model.blocks[3][0].register_forward_hook(hook)
        # model.blocks[4][0].register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []
    use_gpu = args.use_gpu

    for class_name in mvtec.CLASS_NAMES:

        train_dataset = mvtec.MVTecDataset(args, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True)
        test_dataset = mvtec.MVTecDataset(args, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, f'temp_{args.arch}', f'train_{class_name}.hdf5')
        if args.pca:
            train_feature_filepath = os.path.join(args.save_path, f'temp_{args.arch}', f'train_{class_name}_pca.hdf5')
        elif args.npca:
            train_feature_filepath = os.path.join(args.save_path, f'temp_{args.arch}', f'train_{class_name}_npca.hdf5')
        if not os.path.exists(train_feature_filepath):
            for (x, _, _) in tqdm(train_dataloader, f'| feature extraction | train | {class_name} |'):
                # model prediction
                with torch.no_grad():
                    _ = model(x.to(device))
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())
                # initialize hook outputs
                outputs = []
            if args.npca or args.pca:
                # calculate npca or pca
                train_pca_features = []
                for i,train_output in enumerate(train_outputs.values()):
                    train_ouput_cat = torch.cat(train_output, 0)
                    b,c,h,w = train_ouput_cat.size()
                    train_pca_features.append(train_ouput_cat.permute(0,2,3,1).reshape(b*h*w,c))
                pca_mean, pca_components = compute_pca(args,
                    train_pca_features,
                    variance_threshold=args.variance_threshold,
                )
                del train_pca_features, train_ouput_cat
                for i,k in enumerate(train_outputs.keys()):
                    outputs_reduced = []
                    for batch in train_outputs[k]:
                        reduced = pca_reduction(batch, pca_mean[i], pca_components[i], device)
                        outputs_reduced.append(reduced)
                    train_outputs[k] = torch.cat(outputs_reduced, 0).cpu().detach()
            else:
                for k, v in train_outputs.items():
                    train_outputs[k] = torch.cat(v, 0)

            # Embedding concat
            embedding_vectors = train_outputs['layer1']
            for layer_name in ['layer2', 'layer3']:
                embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

            # randomly select d dimension
            if args.reduce_dim:
                embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            # calculate multivariate Gaussian distribution
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            mean = torch.mean(embedding_vectors, dim=0).numpy()
            _cov = torch.zeros(C, C).numpy()
            cov_inv = torch.zeros(C, C, H * W).numpy()
            I = np.identity(C)
            for i in range(H * W):
                # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
                _cov = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
                if use_gpu:
                    _cov = torch.Tensor(_cov).to(device)
                    cov_inv[:, :, i] = torch.linalg.inv(_cov).cpu().numpy()
                else:    
                    cov_inv[:, :, i] =  np.linalg.inv(_cov)
            # save learned distribution
            train_outputs = [mean, cov_inv]
            with h5py.File(train_feature_filepath, 'w') as f:
                f.create_dataset("mean", data=mean)
                f.create_dataset("cov_inv", data=cov_inv)
                if args.npca or args.pca:
                    for i in range(len(pca_mean)):
                        f.create_dataset(f"pca_mean_{i}", data=pca_mean[i].cpu().numpy())
                        f.create_dataset(f"pca_components_{i}", data=pca_components[i].cpu().numpy())
            del mean, _cov, cov_inv
        else:
            print(f'load train set feature from: {train_feature_filepath}')
            with h5py.File(train_feature_filepath, 'r') as f:
                train_outputs = [f['mean'][()], f['cov_inv'][()]]
                if args.npca or args.pca:
                    pca_mean = [torch.Tensor(f[f'pca_mean_{i}'][()]) for i in range(len(test_outputs))]
                    pca_components = [torch.Tensor(f[f'pca_components_{i}'][()]) for i in range(len(test_outputs))]

        gt_list = []
        gt_mask_list = []
        test_imgs = []

        # extract test set features
        for (x, y, mask) in tqdm(test_dataloader, f'| feature extraction | test | {class_name} |'):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
                # print(v.shape)
            # initialize hook outputs
            outputs = []
        if args.npca or args.pca:
            for i,k in enumerate(test_outputs.keys()):
                outputs_reduced = []
                for batch in test_outputs[k]:
                    reduced = pca_reduction(batch, pca_mean[i], pca_components[i], device)
                    outputs_reduced.append(reduced)
                test_outputs[k] = torch.cat(outputs_reduced, 0).cpu().detach()
        else:
            for k, v in test_outputs.items():
                test_outputs[k] = torch.cat(v, 0)
        
        # Embedding concat
        embedding_vectors = test_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

        # randomly select d dimension
        if args.reduce_dim:
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        
        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        if use_gpu:
            embedding_vectors = embedding_vectors.view(B, C, H * W).to(device)
            dist_list = torch.zeros(size=(H*W, B))
            mean = torch.Tensor(train_outputs[0]).to(device)
            cov_inv = torch.Tensor(train_outputs[1]).to(device)
            if args.save_gpu_memory:
                for i in range(H * W):
                    delta = embedding_vectors[:, :, i] - mean[:, i]
                    m_dist = torch.sqrt(torch.diag(torch.mm(torch.mm(delta, cov_inv[:, :, i]), delta.t())).clamp(0))
                    dist_list[i] = m_dist
                dist_list = dist_list.cpu().numpy()
                dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
                dist_list = torch.tensor(dist_list)
            else:
                delta = (embedding_vectors - mean).permute(2, 0, 1)
                dist_list = (torch.matmul(delta, cov_inv.permute(2, 0, 1)) * delta).sum(2).permute(1, 0)
                dist_list = dist_list.reshape(B, H, W)
                dist_list = dist_list.clamp(0).sqrt().cpu()
        else:
            embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
            dist_list = []
            for i in range(H * W):
                mean = train_outputs[0][:, i]
                # dist = [mahalanobis(sample[:, i], mean, train_outputs[1][:, :, i]) for sample in embedding_vectors]
                dist = SSD.cdist(embedding_vectors[:,:,i], mean[None, :], metric='mahalanobis', VI=train_outputs[1][:, :, i])
                dist = list(itertools.chain(*dist))            
                dist_list.append(dist)
            dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
            dist_list = torch.tensor(dist_list)
        # upsample
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                  align_corners=False).squeeze().numpy()
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append(img_roc_auc)
        print('image ROCAUC: %.3f' % (img_roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
        
        # get optimal threshold
        gt_mask = np.asarray(gt_mask_list)
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        save_dir = args.save_path + '/' + f'pictures_{args.arch}'
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = 255.
    vmin = 0.
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax = ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none', norm=norm)
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


if __name__ == '__main__':
    main()
