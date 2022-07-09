import os
import pdb
import pickle
import torch
from mmdet.models import DETECTORS
from .two_stage import TwoStage3DDetector
import torch.nn as nn
from model_utils.minkunet import MinkUNet34C
import MinkowskiEngine as ME
from visdom import Visdom
import cv2
import numpy as np
import hv_cuda


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


import hv_cuda
def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


class HVFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, xyz, scale, obj, res, num_rots, corners):
        ctx.save_for_backward(points, xyz, scale, obj, res, num_rots, corners)
        # t = time()
        outputs = hv_cuda.forward(points, xyz, scale, obj, res, num_rots, corners)
        # print(time() - t)
        grid_obj, grid_rot, grid_scale = outputs
        return grid_obj, grid_rot, grid_scale
        
class HoughVotingModule(nn.Module):
    def __init__(self, res=0.03, num_rots=36, nms_size=0.15, thresh=0, num_proposal=256, no_grad=True):
        super().__init__()
        self.res = torch.tensor(res, dtype=torch.float32, device='cuda')
        self.num_rots = torch.tensor(num_rots, dtype=torch.int32, device='cuda')
        self.no_grad = no_grad
        self.nms_size_grid = int(nms_size // res)
        self.num_proposal = num_proposal
        self.thresh = thresh
    
    def forward(self, pc : torch.Tensor, xyz : torch.Tensor, scale : torch.Tensor, prob : torch.Tensor, corners : torch.Tensor, vote_points : torch.Tensor, pow=0.5):

        with torch.set_grad_enabled(not self.no_grad):
            hv_map, _, hv_scale = HVFunction.apply(pc, xyz.contiguous(), scale.contiguous(), prob.contiguous(), self.res, self.num_rots, corners)

        candidates = []
        probs = []
        scales = []

        hv_map_y = hv_map.max(1)[0] + 1e-7
        hv_map_y = torch.pow(hv_map_y, pow)
        hv_map_yidx = torch.argmax(hv_map, 1)

        dist = hv_map_y.reshape(-1)
        if (not torch.all(torch.isfinite(dist))) or (dist.sum() < 1e-7):
            dist = torch.ones_like(dist)

        cnt = 0
        loc = []
        sample_vals = []
        scales = []
        while cnt < self.num_proposal:
            sample = torch.multinomial(dist, int(self.num_proposal * 1.5), replacement=True)  # hopefully only one trial is enough
            sample_val = dist[sample]
            sample_idx = unravel_index(sample, hv_map_y.shape)
            # pdb.set_trace()
            world_loc = torch.stack([sample_idx[0], hv_map_yidx[sample_idx[0], sample_idx[1]], sample_idx[1]], -1) * self.res + corners[0]
            scale = hv_scale[sample_idx[0], hv_map_yidx[sample_idx[0], sample_idx[1]], sample_idx[1], :]
            dist2seed = torch.min(torch.cdist(world_loc, vote_points), -1)[0]

            # rejection sampling
            if torch.sum(dist2seed < 0.3) == 0:
                loc.append(world_loc)
                sample_vals.append(sample_val)
                scales.append(scale)
            else:
                loc.append(world_loc[dist2seed < 0.3])
                sample_vals.append(sample_val[dist2seed < 0.3])
                scales.append(scale[dist2seed < 0.3])
            cnt += loc[-1].shape[0]
        loc = torch.cat(loc)
        sample_vals = torch.cat(sample_vals)
        scales = torch.cat(scales)
        candidates = loc[:self.num_proposal]
        sample_vals = sample_vals[:self.num_proposal]
        scales = scales[:self.num_proposal]

        probs = torch.zeros_like(candidates)[..., 0]
        return candidates, probs, scales


hv = HoughVotingModule(res=0.05, nms_size=0.3, thresh=0, num_proposal=512, num_rots=60)
mink_net = MinkUNet34C(3, 8).cuda().eval()
mink_net.load_state_dict(torch.load('sunrgbd/checkpoint.pth')['model_state_dict'])


@DETECTORS.register_module()
class BRNetCanon(TwoStage3DDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None
    ):
        super(BRNetCanon, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained
        )

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_bboxes_ignore=None):
        
        # hv = self.hv
        # mink_net = self.mink_net
        points_cat = torch.stack(points)

        feats_dict = self.extract_feat(points_cat)
        seed_points = feats_dict['fp_xyz'][-1]
        seed_features = feats_dict['fp_features'][-1]
        vote_points, vote_features, vote_offset = self.rpn_head.vote_module(seed_points, seed_features)
        feats_dict['vote_points'], feats_dict['vote_features'], feats_dict['vote_offset'] = vote_points, vote_features, vote_offset

        if self.test_cfg.rpn.sample_mod == 'custom':
            border = 0.
            # forward mink
            with torch.no_grad():
                coords = []
                feats = []
                for i in range(len(points)):
                    pc = points[i][..., [0, 2, 1]]
                    coord, feat = ME.utils.sparse_quantize(
                        pc.cpu(),
                        features=pc.cpu(),
                        quantization_size=0.03,
                        device='cuda')
                    coords.append(coord.float())
                    feats.append(feat)
                scan_input = ME.SparseTensor(torch.cat(feats), ME.utils.batched_coordinates(coords), device='cuda')
                scan_output = mink_net(scan_input)
                coords, feats = scan_output.decomposed_coordinates_and_features
                
                proposals = []
                probs = []
                scales = []
                for coord, feat, img_meta, vote_point in zip(coords, feats, img_metas, vote_points):
                    xyz, scale, prob = feat[..., :3], feat[..., 3:6], torch.softmax(feat[..., 6:], dim=-1)[..., 1]
                    scale = torch.exp(scale)
                    pc = coord * 0.03
                    corner = torch.stack([torch.min(pc, 0)[0], torch.max(pc, 0)[0]])
                    corner[0][0] -= border
                    corner[0][2] -= border
                    corner[1][0] += border
                    corner[1][2] += border

                    proposal, prob, scale = hv.forward(pc, xyz, scale, prob, corner, vote_point[..., [0, 2, 1]], pow=0.5)
                    proposals.append(proposal[..., [0, 2, 1]])
                    probs.append(prob)
                    scales.append(scale[..., [0, 2, 1]])
                    
            feats_dict['proposals'] = torch.stack(proposals)
            feats_dict['probs'] = torch.stack(probs)
            feats_dict['scales'] = torch.stack(scales)

        losses = dict()
        if self.with_rpn:
            rpn_outs = self.rpn_head(feats_dict, self.train_cfg.rpn.sample_mod)
            feats_dict.update(rpn_outs)

            rpn_loss_inputs = (points, gt_bboxes_3d, gt_labels_3d,
                               pts_semantic_mask, pts_instance_mask, img_metas)
            rpn_losses = self.rpn_head.loss(
                rpn_outs,
                *rpn_loss_inputs,
                gt_bboxes_ignore=gt_bboxes_ignore,
                ret_target=True
            )
            feats_dict['targets'] = rpn_losses.pop('targets')
            losses.update(rpn_losses)

            # Generate rpn proposals
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            proposal_inputs = (points, rpn_outs, img_metas)
            proposal_list = self.rpn_head.get_bboxes(
                *proposal_inputs, use_nms=proposal_cfg.use_nms
            )
            feats_dict['proposal_list'] = proposal_list
        else:
            raise NotImplementedError

        roi_losses = self.roi_head.forward_train(
            feats_dict, img_metas, points,
            gt_bboxes_3d, gt_labels_3d,
            pts_semantic_mask,
            pts_instance_mask,
            gt_bboxes_ignore
        )
        losses.update(roi_losses)

        return losses

    def simple_test(self, points, img_metas, imgs=None,
        rescale=None):
        
        points_cat = torch.stack(points)

        feats_dict = self.extract_feat(points_cat)
        seed_points = feats_dict['fp_xyz'][-1]
        seed_features = feats_dict['fp_features'][-1]
        vote_points, vote_features, vote_offset = self.rpn_head.vote_module(seed_points, seed_features)
        feats_dict['vote_points'], feats_dict['vote_features'], feats_dict['vote_offset'] = vote_points, vote_features, vote_offset
        
        if self.test_cfg.rpn.sample_mod == 'custom':
            border = 0.
            # forward mink
            with torch.no_grad():
                coords = []
                feats = []
                for i in range(len(points)):
                    pc = points[i][..., [0, 2, 1]]
                    coord, feat = ME.utils.sparse_quantize(
                        pc.cpu(),
                        features=pc.cpu(),
                        quantization_size=0.03,
                        device='cuda')
                    coords.append(coord.float())
                    feats.append(feat)
                # import pdb; pdb.set_trace()
                scan_input = ME.SparseTensor(torch.cat(feats), ME.utils.batched_coordinates(coords), device='cuda')
                scan_output = mink_net(scan_input)

                coords, feats = scan_output.decomposed_coordinates_and_features
                proposals = []
                probs = []
                scales = []
                for coord, feat, img_meta, vote_point in zip(coords, feats, img_metas, vote_points):
                    xyz, scale, prob = feat[..., :3], feat[..., 3:6], torch.softmax(feat[..., 6:], dim=-1)[..., 1]
                    scale = torch.exp(scale)
                    pc = coord * 0.03
                    corner = torch.stack([torch.min(pc, 0)[0], torch.max(pc, 0)[0]])
                    corner[0][0] -= border
                    corner[0][2] -= border
                    corner[1][0] += border
                    corner[1][2] += border

                    proposal, prob, scale = hv.forward(pc, xyz, scale, prob, corner, vote_point[..., [0, 2, 1]], pow=0.5)
                    proposals.append(proposal[..., [0, 2, 1]])
                    probs.append(prob)
                    scales.append(scale[..., [0, 2, 1]])
            feats_dict['proposals'] = torch.stack(proposals)
            feats_dict['probs'] = torch.stack(probs)
            feats_dict['scales'] = torch.stack(scales)

        if self.with_rpn:
            proposal_cfg = self.test_cfg.rpn
            rpn_outs = self.rpn_head(feats_dict, proposal_cfg.sample_mod)
            feats_dict.update(rpn_outs)
            # Generate rpn proposals
            proposal_list = self.rpn_head.get_bboxes(
                points, rpn_outs, img_metas, use_nms=proposal_cfg.use_nms)
            feats_dict['proposal_list'] = proposal_list
        else:
            raise NotImplementedError

        return self.roi_head.simple_test(
            feats_dict, img_metas, points_cat)
