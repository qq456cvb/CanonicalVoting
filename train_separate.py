import torch
import hydra
from utils.dataloader import ScanNetXYZProbSymDataset
from utils.minkunet import MinkUNet34C
import logging
import numpy as np
from tqdm import tqdm
import MinkowskiEngine as ME
import hv_cuda
import hydra.utils as utils
import torch.nn as nn
from utils.calc_map import eval_det_multiprocessing, get_iou_obb
import os
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
    
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def collate_fn(batch):
    id_scans, coords, feats, xyz_labels, scale_labels, obj_labels, class_labels = list(zip(*batch))

    # Generate batched coordinates
    coords_batch = ME.utils.batched_coordinates(coords)

    # Concatenate all lists
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    scale_labels_batch = torch.from_numpy(np.concatenate(scale_labels, 0)).float()
    obj_labels_batch = torch.from_numpy(np.concatenate(obj_labels, 0)).long()
    class_labels_batch = torch.from_numpy(np.concatenate(class_labels, 0)).long()
    xyz_labels_batch = []
    for xyz_label in xyz_labels:
        xyz_label_th = []
        for segments_idx, xyzs in xyz_label:
            xyz_label_th.append([torch.from_numpy(segments_idx).long(), [torch.from_numpy(xyz).float() for xyz in xyzs]])
        xyz_labels_batch.append(xyz_label_th)
    
    return id_scans, coords_batch, feats_batch, xyz_labels_batch, scale_labels_batch, obj_labels_batch, class_labels_batch


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (ME.MinkowskiBatchNorm)):
            m.momentum = bn_momentum
    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))
        

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
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


@hydra.main(config_name='config', config_path='config')
def main(cfg):
    global BN_MOMENTUM_INIT
    global BN_MOMENTUM_MAX
    global BN_DECAY_STEP
    global BN_DECAY_RATE
    global BASE_LEARNING_RATE
    global LR_DECAY_STEPS
    global LR_DECAY_RATES
    BN_MOMENTUM_INIT = 0.5
    BN_MOMENTUM_MAX = 0.001
    BN_DECAY_STEP = cfg.opt.bn_decay_step
    BN_DECAY_RATE = cfg.opt.bn_decay_rate
    BASE_LEARNING_RATE = cfg.opt.learning_rate
    LR_DECAY_STEPS = [int(x) for x in cfg.opt.lr_decay_steps.split(',')]
    LR_DECAY_RATES = [float(x) for x in cfg.opt.lr_decay_rates.split(',')]

    train_dataset = ScanNetXYZProbSymDataset(cfg, training=True, augment=cfg.augment)
    val_dataset = ScanNetXYZProbSymDataset(cfg, training=False, augment=False)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=cfg.num_workers, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, collate_fn=collate_fn, shuffle=True, batch_size=1, num_workers=cfg.num_workers)

    logger.info('Start training...')
    
    # each class predict xyz and scale independently
    model = MinkUNet34C(6 if cfg.use_xyz else 3, 8)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=cfg.weight_decay
    )
    bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
    bnm_scheduler = BNMomentumScheduler(model, bn_lambda=bn_lbmd, last_epoch=cfg.start_epoch-1)
    
    hv = HoughVoting(cfg.scannet_res)
    
    obj_criterion = torch.nn.CrossEntropyLoss()
    model = model.cuda()
    xyz_weights = torch.tensor([float(x) for x in cfg.xyz_component_weights.split(',')]).cuda()
    
    meter = AverageMeter()
    losses = {}
    for epoch in range(cfg.start_epoch, cfg.max_epoch + 1):
        # Training
        adjust_learning_rate(optimizer, epoch)
        bnm_scheduler.step() # decay BN momentum
        
        model.train()
        meter.reset()
        
        with tqdm(enumerate(train_dataloader)) as t:
            for i, data in t:
                _, scan_points, scan_feats, scan_xyz_labels, scan_scale_labels, scan_obj_labels, _ = data
                mask = scan_obj_labels == 1
                if not torch.any(mask):
                    continue
                
                feats = scan_feats.reshape(-1, 6 if cfg.use_xyz else 3) # recenter to [-1, 1] ?
                feats[:, -3:] = feats[:, -3:] * 2. - 1.
                scan_input = ME.SparseTensor(feats, scan_points, device='cuda')
                scan_output = model(scan_input)
                
                scan_output_xyz = scan_output.F[:, :3]
                scan_output_scale = scan_output.F[:, 3:6]
                scan_output_obj = scan_output.F[:, 6:8]


                loss_obj = obj_criterion(scan_output_obj, scan_obj_labels.cuda())
                losses['loss_obj'] = loss_obj
                
                
                loss_xyz = torch.zeros(()).cuda()
                loss_scale = torch.zeros(()).cuda()
                if cfg.log_scale:
                    scan_scale_target = torch.log(scan_scale_labels[mask].cuda())
                else:
                    scan_scale_target = scan_scale_labels[mask].cuda()
                    
                loss_scale = torch.mean((scan_output_scale[mask] - scan_scale_target) ** 2 * xyz_weights)
                
                loss_xyz = []
                for j in range(cfg.batch_size):
                    batch_idx = scan_points[:, 0] == j
                    curr_points = scan_points[:, 1:][batch_idx]
                    
                    for segments_idx, xyzs in scan_xyz_labels[j]:
                        pred = scan_output_xyz[segments_idx.cuda()]
                        loss_xyz_obj = []
                        for xyz in xyzs:
                            loss_xyz_obj.append(torch.mean((pred - xyz.cuda()) ** 2 * xyz_weights))
                        loss_xyz_obj = torch.min(torch.stack(loss_xyz_obj))
                        loss_xyz.append(loss_xyz_obj)
                        
                loss_xyz = torch.mean(torch.stack(loss_xyz))
                
                loss_xyz *= cfg.xyz_factor
                loss_scale *= cfg.scale_factor
                
                losses['loss_xyz'] = loss_xyz
                losses['loss_scale'] = loss_scale
                
                loss = torch.sum(torch.stack(list(losses.values())))

                optimizer.zero_grad()
                loss.backward()
                
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                
                meter.update(loss.item())
                
                t.set_postfix(loss=meter.avg, **dict([(k, v.item()) for (k, v) in losses.items()]))
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'epoch{}.pth'.format(epoch))
        
        if epoch % 10 == 0:
            # validation
            model.eval()
            meter.reset()
            logger.info('epoch {} validation'.format(epoch))
            pred_map_cls = {}
            gt_map_cls = {}
            cnt = 0
            for scan_ids, scan_points, scan_feats, scan_xyz_labels, scan_scale_labels, scan_obj_labels, _ in tqdm(val_dataloader):
                mask = scan_obj_labels == 1
                if not torch.any(mask):
                    continue
                cnt += 1
                id_scan = scan_ids[0]
                
                feats = scan_feats.reshape(-1, 6 if cfg.use_xyz else 3)  # recenter to [-1, 1]?
                feats[:, -3:] = feats[:, -3:] * 2. - 1.
                scan_input = ME.SparseTensor(feats, scan_points, device='cuda')
                with torch.no_grad():
                    scan_output = model(scan_input)
                
                scan_output_xyz = scan_output.F[:, :3]
                scan_output_scale = scan_output.F[:, 3:6]
                scan_output_obj = scan_output.F[:, 6:8]

                loss_obj = obj_criterion(scan_output_obj, scan_obj_labels.cuda())
                losses['loss_obj'] = loss_obj
                
                loss_xyz = torch.zeros(())
                loss_scale = torch.zeros(())
                
                if cfg.log_scale:
                    scan_scale_target = torch.log(scan_scale_labels[mask].cuda())
                else:
                    scan_scale_target = scan_scale_labels[mask].cuda()
                    
                loss_scale = torch.mean((scan_output_scale[mask] - scan_scale_target) ** 2 * xyz_weights)
                
                loss_xyz = []
                for j in range(1):
                    batch_idx = scan_points[:, 0] == j
                    curr_points = scan_points[:, 1:][batch_idx]
                    for segments_idx, xyzs in scan_xyz_labels[j]:
                        pred = scan_output_xyz[segments_idx.cuda()]
                        loss_xyz_obj = []
                        for xyz in xyzs:
                            loss_xyz_obj.append(torch.mean((pred - xyz.cuda()) ** 2 * xyz_weights))
                        loss_xyz_obj = torch.min(torch.stack(loss_xyz_obj))
                        loss_xyz.append(loss_xyz_obj)
                        
                loss_xyz = torch.mean(torch.stack(loss_xyz))
                
                loss_xyz *= cfg.xyz_factor
                loss_scale *= cfg.scale_factor
                
                losses['loss_xyz'] = loss_xyz
                losses['loss_scale'] = loss_scale
                
                curr_points = scan_points[:, 1:]
                
                xyz_pred = scan_output_xyz
                prob_pred = torch.softmax(scan_output_obj, dim=-1)[:, 1]
                if cfg.log_scale:
                    scale_pred = torch.exp(scan_output_scale)
                else:
                    scale_pred = scan_output_scale
                
                with torch.no_grad():
                    grid_obj, grid_rot, grid_scale = hv(curr_points.to('cuda') * cfg.scannet_res, xyz_pred.contiguous(), scale_pred.contiguous(), prob_pred.contiguous())
                        
                map_scene = []            
                boxes = []
                scores = []
                probs = []
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

                    grid_obj[bbox_coords[:, 0], bbox_coords[:, 1], bbox_coords[:, 2]] = 0
                    
                    coords_inv_world = ((scan_points - cand_world) @ rot_mat_full) / scale_full
                    bbox_mask_world = (-1 < coords_inv_world[:, 0]) & (coords_inv_world[:, 0] < 1) \
                            & (-1 < coords_inv_world[:, 1]) & (coords_inv_world[:, 1] < 1) \
                            & (-1 < coords_inv_world[:, 2]) & (coords_inv_world[:, 2] < 1)
                            
                    mask = prob_pred[bbox_mask_world] > 0.3
                    if torch.sum(mask) < valid_ratio * torch.sum(bbox_mask_world) or torch.sum(bbox_mask_world) < thresh_low:
                        continue
                    
                    gt_coords = coords_inv_world[bbox_mask_world][mask]
                    error = torch.mean(torch.norm(xyz_pred[bbox_mask_world][mask] - gt_coords, dim=-1) * prob_pred[bbox_mask_world][mask]).item()
                    
                    if error > 0.3:
                        continue
                    
                    probmax = torch.max(prob_pred[bbox_mask_world])
                    bbox = (rot_mat_full @ torch.diag(scale_full) @ bbox_raw.cuda().T).T + cand_world
                    boxes.append(bbox.cpu().numpy())
                    scores.append(probmax.item())
                    probs.append(probmax.item())

                pick = nms(np.array(boxes), np.array(scores), 0.3)
                for i in pick:
                    map_scene.append((cfg.category, boxes[i], probs[i]))
                
                pred_map_cls[id_scan] = map_scene
                
                # read ground truth
                lines = open(os.path.join(cfg.data.gt_path, '{}.txt'.format(id_scan))).read().splitlines()
                map_scene = []
                for line in lines:
                    tx, ty, tz, ry, sx, sy, sz = [float(v) for v in line.split(' ')[:7]]
                    category = line.split(' ')[-1]
                    bbox = (np.array([[np.cos(ry), 0, -np.sin(ry)], [0, 1, 0], [np.sin(ry), 0, np.cos(ry)]]) @ np.diag([sx, sy, sz]) @ bbox_raw.numpy().T).T + np.array([tx, ty, tz])
                    bbox_mat = np.eye(4)
                    bbox_mat[:3, :3] = np.array([[np.cos(ry), 0, -np.sin(ry)], [0, 1, 0], [np.sin(ry), 0, np.cos(ry)]]) @ np.diag([sx, sy, sz])
                    bbox_mat[:3, 3] = np.array([tx, ty, tz])
                    map_scene.append((category, bbox))
                
                gt_map_cls[id_scan] = map_scene
                
                loss = torch.sum(torch.stack(list(losses.values())))
                
                meter.update(loss.item())
                losses_numeral = dict([(k, v.item()) for (k, v) in losses.items()])
                logger.info(', '.join([k + ': {' + k + '}' for k in losses_numeral.keys()]).format(**losses_numeral))
            
            ret_dict = compute_map(pred_map_cls, gt_map_cls)
            if cfg.category != 'all':
                logger.info('{} Recall: {}'.format(cfg.category, ret_dict['{} Recall'.format(cfg.category)]))
                logger.info('{} Average Precision: {}'.format(cfg.category, ret_dict['{} Average Precision'.format(cfg.category)]))
            

if __name__ == "__main__":
    main()