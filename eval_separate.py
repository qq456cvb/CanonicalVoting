import numpy as np
from tqdm import tqdm
import torch
from utils.dataloader import ScanNetXYZProbDataset
from utils.minkunet import MinkUNet34C
from time import time
import hv_cuda
import logging
import os
logger = logging.getLogger(__name__)
import MinkowskiEngine as ME
from train_separate import collate_fn
from time import time
import hydra
from utils.calc_map import eval_det_multiprocessing, get_iou_obb


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


class HoughVoting(torch.nn.Module):
    def __init__(self, res=0.03, num_rots=120):
        super().__init__()
        # dtype?
        self.res = torch.tensor(res, dtype=torch.float32).cuda()
        self.num_rots = torch.tensor(num_rots, dtype=torch.int32).cuda()
        
    def forward(self, points, xyz, scale, obj):
        return HVFunction.apply(points, xyz, scale, obj, self.res, self.num_rots)
    
    
    
def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


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

all_categories = ['02747177', '02808440', '02871439', '02933112', '03001627', '03211117', '04256520', '04379243', 'others']
color_palette = [
    (31, 119, 180), 		# 
    (255, 187, 120),		# 
    (188, 189, 34), 		# 
    (140, 86, 75),  		# 
    (255, 152, 150),		# 
    (214, 39, 40),  		# 
    (197, 176, 213),		# 
    (148, 103, 189),		# 
    (196, 156, 148),		# 
    (23, 190, 207), 		# 
    (178, 76, 76),  
    (247, 182, 210),	
]


@hydra.main(config_path='config/config.yaml', strict=False)
def main(cfg):
    cfg.category = 'all'
    print(cfg.pretty())
    
    val_dataset = ScanNetXYZProbDataset(cfg, training=False, augment=False)
    print(len(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, collate_fn=collate_fn, shuffle=False, batch_size=1, num_workers=10)

    logger.info('Start testing...')
    
    all_models = {}
    for category in all_categories:
        model = MinkUNet34C(6 if cfg.use_xyz else 3, 8)
        model.load_state_dict(torch.load('pretrained/separate/{}.pth'.format('{:08d}'.format(cfg.category) if category != 'others' else 'others')))
        model = model.cuda()
        model.eval()
        
        all_models[category] = model

    hv = HoughVoting(cfg.scannet_res)
    
    # validation
    l, h, w = 2, 2, 2
    bbox_raw = np.array([[l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]]).T
    bbox_raw = torch.from_numpy(bbox_raw).float().cuda()
    scannet_res = cfg.scannet_res 
    
    pred_map_cls = {}
    gt_map_cls = {}
    cnt = 0
    with tqdm(val_dataloader) as t:
        for scan_ids, scan_points, scan_feats, scan_xyz_labels, scan_scale_labels, scan_obj_labels, scan_class_labels in t:
            id_scan = scan_ids[0]
            cnt += 1
            
            feats = scan_feats.reshape(-1, 6 if cfg.use_xyz else 3) # recenter to [-1, 1] ?
            feats[:, -3:] = feats[:, -3:] * 2. - 1.
            scan_input = ME.SparseTensor(feats, coords=scan_points).to('cuda')
            
            bundle = {}
            for category in all_categories:
                model = all_models[category]
                with torch.no_grad():
                    scan_output = model(scan_input)

                scan_output_xyz = scan_output.F[:, :3]
                scan_output_scale = scan_output.F[:, 3:6]
                scan_output_obj = scan_output.F[:, 6:8]

                xyz_pred = scan_output_xyz
                prob_pred = torch.softmax(scan_output_obj, dim=-1)[:, 1]
                
                if cfg.log_scale:
                    scale_pred = torch.exp(scan_output_scale)
                else:
                    scale_pred = scan_output_scale
                
                with torch.no_grad():
                    grid_obj, grid_rot, grid_scale = hv(scan_points[:, 1:].to('cuda') * cfg.scannet_res, xyz_pred.contiguous(), scale_pred.contiguous(), prob_pred.contiguous())

                bundle[category] = (grid_obj, grid_rot, grid_scale, xyz_pred, prob_pred, scale_pred)
              
            scan_points = scan_points[:, 1:] * cfg.scannet_res
            scan_rgb = scan_feats[:, -3:]
            corners = torch.stack([torch.min(scan_points, 0)[0], torch.max(scan_points, 0)[0]])
            
            t.set_postfix(id_scan=str(scan_ids[0]))
            elimination = 2
            
            map_scene = []
            for idx, category in enumerate(all_categories):
                boxes = []
                scores = []
                probs = []
                
                grid_obj, grid_rot, grid_scale, xyz_pred, prob_pred, scale_pred = bundle[category]
                
                while True:
                    cand = torch.stack(unravel_index(torch.argmax(grid_obj), grid_obj.shape))
                    
                    if grid_obj.cpu().numpy()[cand[0], cand[1], cand[2]] < 60:
                        break
                    
                    grid_obj[max(cand[0]-elimination,0):cand[0]+elimination, max(cand[1]-elimination,0):cand[1]+elimination, max(cand[2]-elimination,0):cand[2]+elimination] = 0
                    
                    rot_vec = grid_rot[cand[0], cand[1], cand[2]]
                    rot = torch.atan2(rot_vec[1], rot_vec[0])
                    rot_mat_full = torch.tensor([[torch.cos(rot), 0, -torch.sin(rot)], [0, 1, 0], [torch.sin(rot), 0, torch.cos(rot)]]).cuda()
                    scale_full = grid_scale[cand[0], cand[1], cand[2]]
                    
                    cand_world = torch.tensor([corners[0, 0] + cfg.scannet_res * cand[0], corners[0, 1] + cfg.scannet_res * cand[1], corners[0, 2] + cfg.scannet_res * cand[2]]).cuda()
                    
                    # fast filtering
                    bbox = (rot_mat_full @ torch.diag(scale_full) @ bbox_raw.T).T
                    bounding_vol = (torch.stack([torch.min(bbox, dim=0)[0], torch.max(bbox, dim=0)[0]]) / scannet_res).int()
                    cand_coords = torch.stack(torch.meshgrid(torch.arange(bounding_vol[0, 0], bounding_vol[1, 0] + 1), torch.arange(bounding_vol[0, 1], bounding_vol[1, 1] + 1), torch.arange(bounding_vol[0, 2], bounding_vol[1, 2] + 1)), -1).reshape(-1, 3)
                    cand_coords = cand_coords.cuda() + cand
                    cand_coords = torch.max(torch.min(cand_coords, torch.tensor(grid_obj.shape).cuda() - 1), torch.tensor([0, 0, 0]).long().cuda())

                    coords_inv = (((cand_coords - cand) * scannet_res) @ rot_mat_full) / scale_full
                    bbox_mask = (-1 < coords_inv[:, 0]) & (coords_inv[:, 0] < 1) \
                                    & (-1 < coords_inv[:, 1]) & (coords_inv[:, 1] < 1) \
                                        & (-1 < coords_inv[:, 2]) & (coords_inv[:, 2] < 1)
                    bbox_coords = cand_coords[bbox_mask]
                    
                    coords_inv_world = ((scan_points.cuda() - cand_world) @ rot_mat_full) / scale_full
                    bbox_mask_world = (-1 < coords_inv_world[:, 0]) & (coords_inv_world[:, 0] < 1) \
                            & (-1 < coords_inv_world[:, 1]) & (coords_inv_world[:, 1] < 1) \
                            & (-1 < coords_inv_world[:, 2]) & (coords_inv_world[:, 2] < 1)

                    # vote elimination: current off        
                    # prob_delta = torch.zeros_like(prob_pred)
                    # prob_delta[bbox_mask_world] = prob_pred[bbox_mask_world]
                    # if not torch.all(prob_delta == 0):
                    #     grid_obj_delta, _, _ = hv(scan_points.cuda(), xyz_pred.contiguous(), scale_pred.contiguous(), prob_delta.contiguous())
                    #     grid_obj -= grid_obj_delta

                    grid_obj[bbox_coords[:, 0], bbox_coords[:, 1], bbox_coords[:, 2]] = 0
                    
                    mask = prob_pred[bbox_mask_world] > 0.3
                    if torch.sum(mask) < 0.2 * torch.sum(bbox_mask_world) or torch.sum(bbox_mask_world) < 10:
                        continue
                    
                    gt_coords = coords_inv_world[bbox_mask_world][mask]
                    error = torch.mean(torch.norm(xyz_pred[bbox_mask_world][mask] - gt_coords, dim=-1) * prob_pred[bbox_mask_world][mask])
                    
                    if error > 0.3:
                        continue

                    probmax = torch.max(prob_pred[bbox_mask_world])
                    bbox = (rot_mat_full @ torch.diag(scale_full) @ bbox_raw.cuda().T).T + cand_world

                    boxes.append(bbox.cpu().numpy())
                    scores.append(probmax.item())
                    probs.append(probmax.item())
                
                pick = nms(np.array(boxes), np.array(scores), 0.3)
                for i in pick:
                    map_scene.append((category, boxes[i], probs[i]))
                    
            pred_map_cls[id_scan] = map_scene
            
            # read ground truth
            lines = open(os.path.join(cfg.data.gt_path, '{}.txt'.format(id_scan))).read().splitlines()
            map_scene = []
            for line in lines:
                tx, ty, tz, ry, sx, sy, sz = [float(v) for v in line.split(' ')[:7]]
                category = line.split(' ')[-1]
                if cfg.category != 'all' and category != cfg.category:
                    continue
                bbox = (np.array([[np.cos(ry), 0, -np.sin(ry)], [0, 1, 0], [np.sin(ry), 0, np.cos(ry)]]) @ np.diag([sx, sy, sz]) @ bbox_raw.cpu().numpy().T).T + np.array([tx, ty, tz])
                bbox_mat = np.eye(4)
                bbox_mat[:3, :3] = np.array([[np.cos(ry), 0, -np.sin(ry)], [0, 1, 0], [np.sin(ry), 0, np.cos(ry)]]) @ np.diag([sx, sy, sz])
                bbox_mat[:3, 3] = np.array([tx, ty, tz])
                map_scene.append((category, bbox))
            
            gt_map_cls[id_scan] = map_scene
            
        for thresh in [0.25, 0.5]:
            logger.info('thresh: {}'.format(thresh))
            ret_dict = compute_map(pred_map_cls, gt_map_cls, ovthresh=thresh)
            
            for category in all_categories:
                logger.info('{} Recall: {}'.format(category, ret_dict['{} Recall'.format(category)]))
                logger.info('{} Average Precision: {}'.format(category, ret_dict['{} Average Precision'.format(category)]))
                

        
if __name__ == "__main__":
    main()