from utils.calc_map import eval_det_multiprocessing, get_iou_obb
from utils.dataloader import ScanNetXYZProbMultiDataset, SceneNNDataset
import torch
import hydra
from utils.minkunet import MinkUNet34C
import logging
import numpy as np
from tqdm import tqdm
import MinkowskiEngine as ME
import hv_cuda
import torch.nn as nn
import os

SCENENN = False

logger = logging.getLogger(__name__)

thresh_high = 60
thresh_low = 10
valid_ratio = 0.2
elimination = 2


class HVFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, xyz, scale, obj, res, num_rots):
        ctx.save_for_backward(points, xyz, scale, obj, res, num_rots)
        outputs = hv_cuda.forward(points, xyz, scale, obj, res, num_rots)
        grid_obj, grid_rot, grid_scale = outputs
        return grid_obj, grid_rot, grid_scale
    
    @staticmethod
    def backward(ctx, grad_obj, grad_rot, grad_scale):
        d_points = d_res = d_num_rots = None
        points, xyz, scale, obj, res, num_rots = ctx.saved_tensors
        outputs = hv_cuda.backward(grad_obj.contiguous(), points, xyz, scale, obj, res, num_rots)
        d_xyz_labels, d_scale_labels, d_obj_labels = outputs
        return d_points, d_xyz_labels, d_scale_labels, d_obj_labels, d_res, d_num_rots


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


class HoughVoting(torch.nn.Module):
    def __init__(self, res=0.03, num_rots=120):
        super().__init__()
        # dtype?
        self.res = torch.tensor(res, dtype=torch.float32).cuda()
        self.num_rots = torch.tensor(num_rots, dtype=torch.int32).cuda()
        
    def forward(self, points, xyz, scale, obj):
        return HVFunction.apply(points, xyz, scale, obj, self.res, self.num_rots)


def collate_fn(batch):
    id_scans, coords, feats, xyz_labels, scale_labels, class_labels = list(zip(*batch))

    # Generate batched coordinates
    coords_batch = ME.utils.batched_coordinates(coords)

    # Concatenate all lists
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    xyz_labels_batch = torch.from_numpy(np.concatenate(xyz_labels, 0)).float()
    scale_labels_batch = torch.from_numpy(np.concatenate(scale_labels, 0)).float()
    class_labels_batch = torch.from_numpy(np.concatenate(class_labels, 0)).long()
    
    return id_scans, coords_batch, feats_batch, xyz_labels_batch, scale_labels_batch, class_labels_batch
        
        
def nms(boxes, scores, overlap_threshold):
    I = np.argsort(scores)
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)
        suppress = [last-1]
        for pos in range(last-1):
            j = I[pos]
            o = get_iou_obb(boxes[i], boxes[j])
            if (o>overlap_threshold):
                suppress.append(pos)
        I = np.delete(I,suppress)
    return pick


def compute_map(pred_map_cls, gt_map_cls, ovthresh=0.5):
    rec, prec, ap = eval_det_multiprocessing(pred_map_cls, gt_map_cls, ovthresh=ovthresh, get_iou_func=get_iou_obb)
    ret_dict = {} 
    for key in sorted(ap.keys()):
        clsname = str(key)
        ret_dict['%s Average Precision'%(clsname)] = ap[key]
    ret_dict['mAP'] = np.mean(list(ap.values()))
    rec_list = []
    for key in sorted(ap.keys()):
        clsname = str(key)
        try:
            ret_dict['%s Recall'%(clsname)] = rec[key][-1]
            rec_list.append(rec[key][-1])
        except:
            ret_dict['%s Recall'%(clsname)] = 0
            rec_list.append(0)
    ret_dict['AR'] = np.mean(rec_list)
    return ret_dict


idx2name = {
    0: 'others',
    1: '03211117',
    2: '04379243',
    3: '02808440',
    4: '02747177',
    5: '04256520',
    6: '03001627',
    7: '02933112',
    8: '02871439'
}

name2catname = {
    '03211117': 'display',
    '04379243': 'table',
    '02808440': 'bathtub',
    '02747177': 'trashbin',
    '04256520': 'sofa',
    '02933112': 'cabinet',
    '02871439': 'bookshelf',
    'others': 'others',
    '03001627': 'chair',
}


@hydra.main(config_name='config', config_path='config')
def main(cfg):
    cfg.category = 'all'
    
    if SCENENN:
        val_dataset = SceneNNDataset(cfg, training=False, augment=False)
    else:
        val_dataset = ScanNetXYZProbMultiDataset(cfg, training=False, augment=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, collate_fn=collate_fn, shuffle=True, batch_size=1, num_workers=cfg.num_workers)

    logger.info('Start testing...')
    
    nclasses = 9
    # each class predict xyz and scale independently
    model = MinkUNet34C(6 if cfg.use_xyz else 3, 6 * nclasses + nclasses + 1)
    model.load_state_dict(torch.load(hydra.utils.to_absolute_path('pretrained/joint.pth')))

    hv = HoughVoting(cfg.scannet_res)
    
    model = model.cuda()
    # validation
    model.eval()
    
    pred_map_cls = {}
    gt_map_cls = {}
    cnt = 0
    for scan_ids, scan_points, scan_feats, _, _, _ in tqdm(val_dataloader):
        cnt += 1
        id_scan = scan_ids[0]
        
        feats = scan_feats.reshape(-1, 6 if cfg.use_xyz else 3)  # recenter to [-1, 1]?
        feats[:, -3:] = feats[:, -3:] * 2. - 1.
        scan_input = ME.SparseTensor(feats, scan_points, device='cuda')
        with torch.no_grad():
            scan_output = model(scan_input)
        
        scan_output_xyz = scan_output.F[:, :3 * nclasses]
        scan_output_scale = scan_output.F[:, 3 * nclasses:6 * nclasses]
        scan_output_class = scan_output.F[:, 6 * nclasses:]
        
        class_label_idx = scan_output_class.argmax(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3)
        class_label_idx[class_label_idx == nclasses] = 0
        scan_output_xyz = torch.gather(scan_output_xyz.reshape(-1, nclasses, 3), 1, class_label_idx)[:, 0]
        scan_output_scale = torch.gather(scan_output_scale.reshape(-1, nclasses, 3), 1, class_label_idx)[:, 0]
        
        curr_points = scan_points[:, 1:]

        xyz_pred = scan_output_xyz
        if cfg.log_scale:
            scale_pred = torch.exp(scan_output_scale)
        else:
            scale_pred = scan_output_scale
        class_pred = torch.argmax(scan_output_class[..., :-1], dim=-1)
        prob_pred = torch.max(torch.softmax(scan_output_class, dim=-1)[..., :-1], dim=-1)[0]

        with torch.no_grad():
            grid_obj, grid_rot, grid_scale = hv(curr_points.to('cuda') * cfg.scannet_res, xyz_pred.contiguous(), scale_pred.contiguous(), prob_pred.contiguous())

        map_scene = []            
        boxes = []
        scores = []
        probs = []
        classes = []
        scan_points = curr_points.to('cuda') * cfg.scannet_res
        corners = torch.stack([torch.min(scan_points, 0)[0], torch.max(scan_points, 0)[0]])
        l, h, w = 2, 2, 2
        bbox_raw = torch.from_numpy(np.array([[l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]]).T).float()
        while True:
            cand = torch.stack(unravel_index(torch.argmax(grid_obj), grid_obj.shape))
            cand_world = torch.tensor([corners[0, 0] + cfg.scannet_res * cand[0], corners[0, 1] + cfg.scannet_res * cand[1], corners[0, 2] + cfg.scannet_res * cand[2]]).cuda()

            if grid_obj[cand[0], cand[1], cand[2]].item() < thresh_high:
                break
            
            grid_obj[max(cand[0]-elimination,0):cand[0]+elimination+1, max(cand[1]-elimination,0):cand[1]+elimination+1, max(cand[2]-elimination,0):cand[2]+elimination+1] = 0
            
            rot_vec = grid_rot[cand[0], cand[1], cand[2]]
            rot = torch.atan2(rot_vec[1], rot_vec[0])
            rot_mat_full = torch.tensor([[torch.cos(rot), 0, -torch.sin(rot)], [0, 1, 0], [torch.sin(rot), 0, torch.cos(rot)]]).cuda()
            scale_full = grid_scale[cand[0], cand[1], cand[2]]
            
            # fast filtering
            bbox = (rot_mat_full @ torch.diag(scale_full) @ bbox_raw.cuda().T).T
            bounding_vol = (torch.stack([torch.min(bbox, 0)[0], torch.max(bbox, 0)[0]]) / cfg.scannet_res).int()
            cand_coords = torch.stack(torch.meshgrid(torch.arange(bounding_vol[0, 0], bounding_vol[1, 0] + 1), torch.arange(bounding_vol[0, 1], bounding_vol[1, 1] + 1), torch.arange(bounding_vol[0, 2], bounding_vol[1, 2] + 1)), -1).reshape(-1, 3).cuda()
            cand_coords = cand_coords + cand
            cand_coords = torch.max(torch.min(cand_coords, torch.tensor(grid_obj.shape).cuda() - 1), torch.tensor([0, 0, 0]).cuda())

            coords_inv = (((cand_coords - cand) * cfg.scannet_res) @ rot_mat_full) / scale_full
            bbox_mask = (-1 < coords_inv[:, 0]) & (coords_inv[:, 0] < 1) \
                            & (-1 < coords_inv[:, 1]) & (coords_inv[:, 1] < 1) \
                                & (-1 < coords_inv[:, 2]) & (coords_inv[:, 2] < 1)
            bbox_coords = cand_coords[bbox_mask]
            
            coords_inv_world = ((scan_points - cand_world) @ rot_mat_full) / scale_full
            bbox_mask_world = (-1 < coords_inv_world[:, 0]) & (coords_inv_world[:, 0] < 1) \
                    & (-1 < coords_inv_world[:, 1]) & (coords_inv_world[:, 1] < 1) \
                    & (-1 < coords_inv_world[:, 2]) & (coords_inv_world[:, 2] < 1)

            # back project elimination: current off   
            # prob_delta = torch.zeros_like(prob_pred)
            # prob_delta[bbox_mask_world] = prob_pred[bbox_mask_world]
            # if not torch.all(prob_delta == 0):
            #     grid_obj_delta, _, _ = hv(scan_points.cuda(), xyz_pred.contiguous(), scale_pred.contiguous(), prob_delta.contiguous())
            #     grid_obj -= grid_obj_delta

            grid_obj[bbox_coords[:, 0], bbox_coords[:, 1], bbox_coords[:, 2]] = 0
                    
            mask = prob_pred[bbox_mask_world] > 0.3
            if torch.sum(mask) < valid_ratio * torch.sum(bbox_mask_world) or torch.sum(bbox_mask_world) < thresh_low:
                continue
            
            gt_coords = coords_inv_world[bbox_mask_world][mask]
            error = torch.mean(torch.norm(xyz_pred[bbox_mask_world][mask] - gt_coords, dim=-1) * prob_pred[bbox_mask_world][mask]).item()
            
            if error > 0.3:
                continue

            elems, counts = torch.unique(class_pred[bbox_mask_world][mask], return_counts=True)
            best_class_idx = elems[torch.argmax(counts)].item()
            
            probmax = torch.max(prob_pred[bbox_mask_world])
            bbox = (rot_mat_full @ torch.diag(scale_full) @ bbox_raw.cuda().T).T + cand_world
            boxes.append(bbox.cpu().numpy())
            scores.append(probmax.item())
            probs.append(probmax.item())
            classes.append(best_class_idx)
            
        boxes = np.array(boxes)
        scores = np.array(scores)
        probs = np.array(probs)
        classes = np.array(classes)

        if len(classes) > 0:
            for i in range(nclasses):
                if SCENENN and name2catname[idx2name[i]] not in ['cabinet', 'chair', 'table', 'sofa', 'display']:
                    continue
                if (classes == i).sum() > 0:
                    boxes_cls = boxes[classes == i]
                    scores_cls = scores[classes == i]
                    probs_cls = probs[classes == i]
                    pick = nms(boxes_cls, scores_cls, 0.3)
                    for j in pick:
                        map_scene.append((name2catname[idx2name[i]], boxes_cls[j], probs_cls[j]))
        
        pred_map_cls[id_scan] = map_scene
        
        # read ground truth
        lines = open(os.path.join(os.path.join(cfg.data.scene_nn_root, 'results_gt') if SCENENN else cfg.data.gt_path, '{}.txt'.format(id_scan))).read().splitlines()
        map_scene = []
        for line in lines:
            tx, ty, tz, ry, sx, sy, sz = [float(v) for v in line.split(' ')[:7]]
            if not SCENENN:
                category = name2catname[line.split(' ')[-1]]
            else:
                category = line.split(' ')[-1]
            if SCENENN and category == 'desk':
                category = 'table'
            if SCENENN and category == 'television':
                category = 'display'
            bbox = (np.array([[np.cos(ry), 0, -np.sin(ry)], [0, 1, 0], [np.sin(ry), 0, np.cos(ry)]]) @ np.diag([sx, sy, sz]) @ bbox_raw.numpy().T).T + np.array([tx, ty, tz])
            bbox_mat = np.eye(4)
            bbox_mat[:3, :3] = np.array([[np.cos(ry), 0, -np.sin(ry)], [0, 1, 0], [np.sin(ry), 0, np.cos(ry)]]) @ np.diag([sx, sy, sz])
            bbox_mat[:3, 3] = np.array([tx, ty, tz])
            map_scene.append((category, bbox))
        
        gt_map_cls[id_scan] = map_scene
    
    for thresh in [0.25, 0.5]:
        print(thresh)
        ret_dict = compute_map(pred_map_cls, gt_map_cls, thresh)
        for k in range(nclasses):
            name = name2catname[idx2name[k]]
            logger.info('{} Recall: {}'.format(name, ret_dict['{} Recall'.format(name)]))
            logger.info('{} Average Precision: {}'.format(name, ret_dict['{} Average Precision'.format(name)]))
        logger.info('mean Average Precision: {}'.format(ret_dict['mAP']))
        
if __name__ == "__main__":
    main()