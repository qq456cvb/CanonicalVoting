import torch.utils.data as torchdata
import numpy as np
import os
import json
from plyfile import PlyData
import quaternion
import MinkowskiEngine as ME
import pickle
import collections
import h5py


def get_top8_classes_mapping():                                                                                                                                                                                                                                                                                           
    top = collections.defaultdict(lambda : 0)
    top["03211117"] = 1
    top["04379243"] = 2
    top["02808440"] = 3
    top["02747177"] = 4
    top["04256520"] = 5
    top["03001627"] = 6
    top["02933112"] = 7
    top["02871439"] = 8
    return top


def calc_Mbbox_no_rot(model):
    trs_obj = model["trs"]
    bbox_obj = np.asarray(model["bbox"], dtype=np.float64)
    center_obj = np.asarray(model["center"], dtype=np.float64)
    trans_obj = np.asarray(trs_obj["translation"], dtype=np.float64)
    rot_obj = np.asarray(trs_obj["rotation"], dtype=np.float64)
    q_obj = np.quaternion(rot_obj[0], rot_obj[1], rot_obj[2], rot_obj[3])
    scale_obj = np.asarray(trs_obj["scale"], dtype=np.float64)

    tcenter1 = np.eye(4)
    tcenter1[0:3, 3] = center_obj
    rot1 = np.eye(4)
    rot1[0:3, 0:3] = quaternion.as_rotation_matrix(q_obj)
    trans1 = np.eye(4)
    trans1[0:3, 3] = np.linalg.inv(rot1[0:3, 0:3]) @ trans_obj
    scale1 = np.eye(4)
    scale1[0:3, 0:3] = np.diag(scale_obj)
    bbox1 = np.eye(4)
    bbox1[0:3, 0:3] = np.diag(bbox_obj)
    M = trans1.dot(scale1).dot(tcenter1).dot(bbox1)
    return M


def calc_Mbbox(model):
    trs_obj = model["trs"]
    bbox_obj = np.asarray(model["bbox"], dtype=np.float64)
    center_obj = np.asarray(model["center"], dtype=np.float64)
    trans_obj = np.asarray(trs_obj["translation"], dtype=np.float64)
    rot_obj = np.asarray(trs_obj["rotation"], dtype=np.float64)
    q_obj = np.quaternion(rot_obj[0], rot_obj[1], rot_obj[2], rot_obj[3])
    scale_obj = np.asarray(trs_obj["scale"], dtype=np.float64)

    tcenter1 = np.eye(4)
    tcenter1[0:3, 3] = center_obj
    trans1 = np.eye(4)
    trans1[0:3, 3] = trans_obj
    rot1 = np.eye(4)
    rot1[0:3, 0:3] = quaternion.as_rotation_matrix(q_obj)
    scale1 = np.eye(4)
    scale1[0:3, 0:3] = np.diag(scale_obj)
    bbox1 = np.eye(4)
    bbox1[0:3, 0:3] = np.diag(bbox_obj)
    M = trans1.dot(rot1).dot(scale1).dot(tcenter1).dot(bbox1)
    return M


def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M 


def apply_trans(pc, trans):
    return (trans @ np.concatenate([pc, np.ones((pc.shape[0], 1))], -1).T).T[:, :3]


class ScanNetXYZProbMultiDataset(torchdata.Dataset):
    def __init__(self, cfg, training, augment):
        self.cfg = cfg
        self.annotations = json.load(open(self.cfg.data.scan2cad))
        with open(self.cfg.data.train_split if training else self.cfg.data.val_split) as f:
            valid_ids = f.read().splitlines()
        
        self.annotations = [annotation for annotation in self.annotations if annotation['id_scan'] in valid_ids]
        self.segments = pickle.load(open(self.cfg.data.train_segments if training else self.cfg.data.val_segments, 'rb'))
        self.mean_size = np.zeros([3,])
        self.catid2idx = get_top8_classes_mapping()
        
        # filter annotations
        if self.cfg.category != 'all':
            if self.cfg.category == 'others': 
                valid_annotations = [annotation for annotation in self.annotations if len([model for model in annotation['aligned_models'] if self.catid2idx[model['catid_cad']] == 0]) > 0]
            else:
                valid_annotations = [annotation for annotation in self.annotations if len([model for model in annotation['aligned_models'] if model['catid_cad'] == self.cfg.category]) > 0]
        else:
            valid_annotations = self.annotations
            
        self.annotations = valid_annotations
        self.training = training
        self.augment = augment

    def __len__(self):
        return len(self.annotations)
    
    # to do, random rotationv
    def __getitem__(self, index):
        annotation = self.annotations[index]
        id_scan = annotation['id_scan']
        
        segments = self.segments[id_scan]
        scan_file = os.path.join(self.cfg.data.scannet, 'scans', id_scan, id_scan + "_vh_clean_2.ply")
        assert(np.all(np.abs(np.array(annotation["trs"]["scale"]) - 1.) < 1e-7))
        
        # y := z, z := -y, original z up direction
        Mscan = make_M_from_tqs(annotation["trs"]["translation"], annotation["trs"]["rotation"], annotation["trs"]["scale"])
        assert os.path.exists(scan_file), scan_file + " does not exist."

        with open(scan_file, 'rb') as read_file:
            mesh_scan = PlyData.read(read_file)
        
        pcd = np.stack([mesh_scan["vertex"]['x'], mesh_scan["vertex"]['y'], mesh_scan["vertex"]['z']], -1)
        scan_rgb = np.stack([mesh_scan["vertex"]["red"], mesh_scan["vertex"]["green"], mesh_scan["vertex"]["blue"]], -1)
        scan_rgb = (scan_rgb / 255.).astype(np.float32)
        
        scan_points = (Mscan @ np.concatenate([pcd, np.ones((pcd.shape[0], 1))], -1).T).T[:, :3]

        aligned_models = annotation['aligned_models']
        for i in range(len(aligned_models)):
            aligned_models[i]['segments'] = segments[i]
            
        if self.cfg.category != 'all':
            if self.cfg.category == 'others':
                valid_models = [model for model in annotation['aligned_models'] if self.catid2idx[model['catid_cad']] == 0]
            else:
                valid_models = [model for model in annotation['aligned_models'] if model['catid_cad'] == self.cfg.category]
        else:
            valid_models = aligned_models
            
        if len(valid_models) == 0:
            return self.__getitem__(np.random.randint(len(self)))
            
        if self.augment:
            augment_mat = np.eye(4)
            
            if self.cfg.augment_color:
                scan_rgb *= (1+0.4*np.random.random(3)-0.2) # brightness change for each channel
                scan_rgb += (0.1*np.random.random(3)-0.05) # color shift for each channel
                scan_rgb += np.expand_dims((0.05*np.random.random(scan_points.shape[0])-0.025), -1) # jittering on each pixel
                scan_rgb = np.clip(scan_rgb, 0, 1)
            
            rot_angle = np.random.randint(4) * np.pi / 2. + (np.random.random() - 0.5) * 2. * np.pi / 9.
            rot_mat = np.array([[np.cos(rot_angle), 0, -np.sin(rot_angle)], [0, 1, 0], [np.sin(rot_angle), 0, np.cos(rot_angle)]])

            scan_points = scan_points @ rot_mat.T
            augment_mat[:3, :3] = rot_mat @ augment_mat[:3, :3]
          
        scan_points = scan_points.astype(np.float32)
        scan_xyz_labels = np.zeros_like(scan_points, dtype=np.float32)
        scan_scale_labels = np.zeros_like(scan_points, dtype=np.float32)
        scan_class_labels = np.ones((scan_points.shape[0],), dtype=np.int32) * 9
        
        # may be could use dense xyz, assign each point to each closest object center
        for model in valid_models:
            if np.min(np.asarray(model['trs']["scale"], dtype=np.float32)) < 1e-3:  # some label is singular, thus invalid
                continue
            unit2scan_scale = np.diag(np.asarray(model['trs']["scale"], dtype=np.float32)) @ np.diag(np.asarray(model["bbox"], dtype=np.float32))
            
            Mbbox = calc_Mbbox(model)
            
            if self.augment:
                Mbbox = augment_mat @ Mbbox
            
            segment_points_inv = apply_trans(scan_points[model['segments']], np.linalg.inv(Mbbox))
            scan_xyz_labels[model['segments']] = segment_points_inv
            scan_scale_labels[model['segments']] = np.diag(unit2scan_scale)
            scan_class_labels[model['segments']] = self.catid2idx[model['catid_cad']]
            

        if self.cfg.use_xyz:
            scan_feats = np.concatenate([scan_points, scan_rgb], -1)
        else:
            scan_feats = scan_rgb

        feats = np.concatenate([scan_feats, scan_xyz_labels, scan_scale_labels], -1)
        discrete_idx = ME.utils.sparse_quantize(
            np.ascontiguousarray(scan_points),
            quantization_size=self.cfg.scannet_res,
            return_index=True)[1]
        
        discrete_coords = np.floor(scan_points[discrete_idx] / self.cfg.scannet_res).astype(np.float32)
        unique_feats = feats[discrete_idx]
        unique_class_labels = scan_class_labels[discrete_idx]
        
        unique_scan_feats = unique_feats[:, :scan_feats.shape[1]]
        unique_xyz_labels = unique_feats[:, scan_feats.shape[1]:scan_feats.shape[1] + scan_xyz_labels.shape[1]]
        unique_scale_labels = unique_feats[:, -scan_scale_labels.shape[1]:]
        
        return id_scan, discrete_coords, unique_scan_feats, unique_xyz_labels, unique_scale_labels, unique_class_labels


class SceneNNDataset(torchdata.Dataset):
    train_list = ['005', '014', '015', '016', '025', '036', '038', '041', '045', '047', '052', '054', '057', '061', '062', '066', '071', '073', '078', '080', '084', '087', '089', '096', '098', '109', '201', '202', '209', '217', '223', '225', '227', '231', '234', '237', '240', '243', '249', '251', '255', '260', '263', '265', '270', '276', '279', '286', '294', '308', '522', '609', '613', '614', '623', '700']
    test_list = ['011', '021', '065', '032', '093', '246', '086', '069', '206', '252', '273', '527', '621', '076', '082', '049', '207', '213', '272', '074']
    target_classes = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'desk', 'television']  # dummy
    def __init__(self, cfg, training, augment):
        self.cfg = cfg

        self.training = training
        self.augment = augment
        self.root = self.cfg.data.scenenn

        self.room_list = SceneNNDataset.train_list + SceneNNDataset.test_list
        self.annotations = json.load(open(os.path.join(self.cfg.scene_nn_root, 'full_annotations.json'), 'r'))

        valid_ids = SceneNNDataset.train_list + SceneNNDataset.test_list
        
        self.annotations = [annotation for annotation in self.annotations if annotation['id_scan'] in valid_ids]

        if self.cfg.category != 'all' and not self.cfg.evaluate:
            if self.cfg.category == 'table':
                valid_annotations = [annotation for annotation in self.annotations if len([model for model in annotation['aligned_models'] if model['nyu_name'] in ['table', 'desk']]) > 0]
            else:
                valid_annotations = [annotation for annotation in self.annotations if len([model for model in annotation['aligned_models'] if model['nyu_name'] == self.cfg.category]) > 0]
        else:
            valid_annotations = self.annotations
        
        self.annotations = valid_annotations
        self.segments = pickle.load(open(os.path.join(self.cfg.scene_nn_root, 'scenenn_segments.pkl'), 'rb'))

    def __len__(self):
        return len(self.annotations)
    
    # to do, random rotation
    def __getitem__(self, index):
        annotation = self.annotations[index]
        id_scan = annotation['id_scan']
        
        segments = self.segments[id_scan]

        assert(np.all(np.abs(np.array(annotation["trs"]["scale"]) - 1.) < 1e-7))
        
        # y := z, z := -y, original z up direction
        Mscan = make_M_from_tqs(annotation["trs"]["translation"], annotation["trs"]["rotation"], annotation["trs"]["scale"])
        f = h5py.File(os.path.join(self.cfg.scene_nn_root, 'scenenn_seg/scenenn_seg_{}.hdf5'.format(id_scan)))
        data = f['data'][:]

        pcd = data[:, :, -3:].reshape(-1, 3)
        rgb = data[:, :, -6:-3].reshape(-1, 3)

        pcd = pcd[:, [0, 2, 1]]  # scene nn to scannet coordinates
        pcd[:, 1] = -pcd[:, 1]
        
        _, indices = np.unique(pcd, axis=0, return_index=True)
        pcd = pcd[indices].astype(np.float32)
        scan_rgb = rgb[indices].astype(np.float32)

        scan_points = (Mscan @ np.concatenate([pcd, np.ones((pcd.shape[0], 1))], -1).T).T[:, :3]

        aligned_models = annotation['aligned_models']
        for i in range(len(aligned_models)):
            aligned_models[i]['segments'] = segments[i]

        if self.cfg.category != 'all' and not self.cfg.evaluate:
            if self.cfg.category == 'table':
                valid_models = [model for model in annotation['aligned_models'] if model['nyu_name'] in ['table', 'desk']]
            else:
                valid_models = [model for model in annotation['aligned_models'] if model['nyu_name'] == self.cfg.category]
        else:
            valid_models = aligned_models
            
        assert(len(valid_models) > 0)
        
        if self.augment:
            augment_mat = np.eye(4)
            rot_angle = np.random.randint(4) * np.pi / 2. + (np.random.random() - 0.5) * 2. * np.pi / 9.
            rot_mat = np.array([[np.cos(rot_angle), 0, -np.sin(rot_angle)], [0, 1, 0], [np.sin(rot_angle), 0, np.cos(rot_angle)]])

            scan_points = scan_points @ rot_mat.T
            augment_mat[:3, :3] = rot_mat @ augment_mat[:3, :3]
                
        discrete_idx = ME.utils.sparse_quantize(
            np.ascontiguousarray(scan_points),
            quantization_size=self.cfg.scannet_res,
            return_index=True)[1]

        scan_points = scan_points[discrete_idx]
        scan_rgb = scan_rgb[discrete_idx]
        idx_mapping = dict([(j, i) for i, j in enumerate(discrete_idx)])
        discrete_coords = np.floor(scan_points / self.cfg.scannet_res).astype(np.float32)

        scan_xyz_labels = np.zeros_like(scan_points, dtype=np.float32)
        scan_scale_labels = np.zeros_like(scan_points, dtype=np.float32)
        scan_class_labels = np.zeros((scan_points.shape[0],), dtype=np.int32)

        scan_xyz_labels = []

        plots = scan_points[:,:3]
        labels = np.ones((plots.shape[0]), dtype=np.int)
        for model in valid_models:
            unit2scan_scale = np.diag(np.asarray(model['trs']["scale"], dtype=np.float32)) @ np.diag(np.asarray(model["bbox"], dtype=np.float32))
            Mbbox = calc_Mbbox(model)

            if self.augment:
                Mbbox = augment_mat @ Mbbox

            segments = np.array([idx_mapping[i] for i in model['segments'] if i in idx_mapping])
            segment_points_inv = apply_trans(segments, np.linalg.inv(Mbbox))
            scan_scale_labels[segments] = np.diag(unit2scan_scale)
            scan_class_labels[segments] = SceneNNDataset.target_classes.index(model['nyu_name'])
            scan_xyz_labels[segments] = segment_points_inv

            l,w,h = 2, 2, 2
            x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
            y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
            z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
            
            bbox2 = apply_trans(np.stack([x_corners,y_corners,z_corners], 0).T, Mbbox)

        if self.cfg.use_xyz:
            scan_feats = np.concatenate([scan_points, scan_rgb], -1)
        else:
            scan_feats = scan_rgb
        
        return id_scan, discrete_coords, scan_feats, scan_xyz_labels, scan_scale_labels, scan_class_labels


class ScanNetXYZProbSymDataset(torchdata.Dataset):
    def __init__(self, cfg, training, augment):
        self.cfg = cfg
        self.annotations = json.load(open(self.cfg.data.scan2cad))
        with open(self.cfg.data.train_split if training else self.cfg.data.val_split) as f:
            valid_ids = f.read().splitlines()
        
        self.annotations = [annotation for annotation in self.annotations if annotation['id_scan'] in valid_ids]
        self.segments = pickle.load(open(self.cfg.data.train_segments if training else self.cfg.data.val_segments, 'rb'))
        self.mean_size = np.zeros([3,])
        self.catid2idx = get_top8_classes_mapping()
        
        # filter annotations
        if self.cfg.category != 'all':
            if self.cfg.category == 'others':
                valid_annotations = [annotation for annotation in self.annotations if len([model for model in annotation['aligned_models'] if self.catid2idx[model['catid_cad']] == 0]) > 0]
            else:
                valid_annotations = [annotation for annotation in self.annotations if len([model for model in annotation['aligned_models'] if model['catid_cad'] == self.cfg.category]) > 0]
        else:
            valid_annotations = self.annotations
            
        self.annotations = valid_annotations
        self.training = training
        self.augment = augment

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        annotation = self.annotations[index]
        id_scan = annotation['id_scan']
        
        segments = self.segments[id_scan]
        scan_file = os.path.join(self.cfg.data.scannet, 'scans', id_scan, id_scan + "_vh_clean_2.ply")
        assert(np.all(np.abs(np.array(annotation["trs"]["scale"]) - 1.) < 1e-7))
        
        # y := z, z := -y, original z up direction
        Mscan = make_M_from_tqs(annotation["trs"]["translation"], annotation["trs"]["rotation"], annotation["trs"]["scale"])
        assert os.path.exists(scan_file), scan_file + " does not exist."

        with open(scan_file, 'rb') as read_file:
            mesh_scan = PlyData.read(read_file)
        
        pcd = np.stack([mesh_scan["vertex"]['x'], mesh_scan["vertex"]['y'], mesh_scan["vertex"]['z']], -1)
        scan_rgb = np.stack([mesh_scan["vertex"]["red"], mesh_scan["vertex"]["green"], mesh_scan["vertex"]["blue"]], -1)
        
        scan_points = (Mscan @ np.concatenate([pcd, np.ones((pcd.shape[0], 1))], -1).T).T[:, :3]

        aligned_models = annotation['aligned_models']
        for i in range(len(aligned_models)):
            aligned_models[i]['segments'] = segments[i]
            
        if self.cfg.category != 'all':
            if self.cfg.category == 'others':
                valid_models = [model for model in annotation['aligned_models'] if self.catid2idx[model['catid_cad']] == 0]
            else:
                valid_models = [model for model in annotation['aligned_models'] if model['catid_cad'] == self.cfg.category]
        else:
            valid_models = aligned_models
            
        if len(valid_models) == 0:
            return self.__getitem__(np.random.randint(len(self)))
        
        if self.augment:
            augment_mat = np.eye(4)
            
            if self.cfg.augment_color:
                scan_rgb *= (1+0.4*np.random.random(3)-0.2) # brightness change for each channel
                scan_rgb += (0.1*np.random.random(3)-0.05) # color shift for each channel
                scan_rgb += np.expand_dims((0.05*np.random.random(scan_points.shape[0])-0.025), -1) # jittering on each pixel
                scan_rgb = np.clip(scan_rgb, 0, 1)
            
            rot_angle = np.random.randint(4) * np.pi / 2. + (np.random.random() - 0.5) * 2. * np.pi / 9.
            rot_mat = np.array([[np.cos(rot_angle), 0, -np.sin(rot_angle)], [0, 1, 0], [np.sin(rot_angle), 0, np.cos(rot_angle)]])

            scan_points = scan_points @ rot_mat.T
            augment_mat[:3, :3] = rot_mat @ augment_mat[:3, :3]
        
        scan_points = scan_points.astype(np.float32)
        
        discrete_idx = ME.utils.sparse_quantize(
            np.ascontiguousarray(scan_points),
            quantization_size=self.cfg.scannet_res,
            return_index=True)[1]
        
        scan_points = scan_points[discrete_idx]
        scan_rgb = scan_rgb[discrete_idx]
        idx_mapping = dict([(j, i) for i, j in enumerate(discrete_idx)])
        discrete_coords = np.floor(scan_points / self.cfg.scannet_res).astype(np.float32)
        
        scan_rgb = (scan_rgb / 255.).astype(np.float32)
        scan_scale_labels = np.zeros_like(scan_points, dtype=np.float32)
        scan_obj_labels = np.zeros((scan_points.shape[0],), dtype=np.int32)
        scan_class_labels = np.zeros((scan_points.shape[0],), dtype=np.int32)
        
        def roty(angle):
            return np.array([[np.cos(angle), 0, -np.sin(angle), 0], [0, 1, 0, 0], [np.sin(angle), 0, np.cos(angle), 0], [0, 0, 0, 1]])

        scan_xyz_labels = []
        for model in valid_models:
            if np.min(np.asarray(model['trs']["scale"], dtype=np.float32)) < 1e-3:  # some label is singular, thus invalid
                continue
            unit2scan_scale = np.diag(np.asarray(model['trs']["scale"], dtype=np.float32)) @ np.diag(np.asarray(model["bbox"], dtype=np.float32))
            
            Mbbox = calc_Mbbox(model)
            sym = model['sym']
            Mbboxes = [Mbbox]
            if sym == '__SYM_ROTATE_UP_2':
                Mbboxes.append(Mbbox @ roty(np.pi))
            elif sym == '__SYM_ROTATE_UP_4':
                Mbboxes.append(Mbbox @ roty(np.pi / 2))
                Mbboxes.append(Mbbox @ roty(np.pi))
                Mbboxes.append(Mbbox @ roty(-np.pi / 2))
            elif sym == '__SYM_ROTATE_UP_INF':
                for i in range(1, 36):
                    Mbboxes.append(Mbbox @ roty(2 * np.pi / 36 * i))
            
            if self.augment:
                for i in range(len(Mbboxes)):
                    Mbboxes[i] = augment_mat @ Mbboxes[i]
                
            segments_idx = np.array([idx_mapping[i] for i in model['segments'] if i in idx_mapping])
            xyzs = []
            obj_points = scan_points[segments_idx]
            for Mbbox in Mbboxes:
                segment_points_inv = apply_trans(obj_points, np.linalg.inv(Mbbox))
                xyzs.append(segment_points_inv)
            scan_scale_labels[segments_idx] = np.diag(unit2scan_scale)
            scan_obj_labels[segments_idx] = 1
            scan_class_labels[segments_idx] = self.catid2idx[model['catid_cad']]
            
            scan_xyz_labels.append([segments_idx, xyzs])

        if self.cfg.use_xyz:
            scan_feats = np.concatenate([scan_points, scan_rgb], -1)
        else:
            scan_feats = scan_rgb
        
        return id_scan, discrete_coords, scan_feats, scan_xyz_labels, scan_scale_labels, scan_obj_labels, scan_class_labels


import hydra
from tqdm import tqdm
@hydra.main(config_path='config/config.yaml')
def main(cfg):
    cfg.category = 'all'
    ds = ScanNetXYZProbMultiDataset(cfg, training=True, augment=True)
    print(len(ds))
    for d in tqdm(ds):
        import pdb; pdb.set_trace()
        pass

if __name__ == '__main__':
    main()