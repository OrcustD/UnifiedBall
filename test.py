import os
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Shuttlecock_Trajectory_Dataset, data_dir
from utils.general import *
from utils.metric import *


pred_types = ['TP', 'TN', 'FP1', 'FP2', 'FN']
pred_types_map = {pred_type: i for i, pred_type in enumerate(pred_types)}
inpaintnet_eval_types = ['inpaint', 'reconstruct', 'baseline']


def get_ensemble_weight(seq_len, eval_mode):
    """ Get weight for temporal ensemble.

        Args:
            seq_len (int): Length of input sequence
            eval_mode (str): Mode of temporal ensemble
                Choices:
                    - 'average': Return uniform weight
                    - 'weight': Return positional weight
        
        Returns:
            weight (torch.Tensor): Weight for temporal ensemble
    """

    if eval_mode == 'average':
        weight = torch.ones(seq_len) / seq_len
    elif eval_mode == 'weight':
        weight = torch.ones(seq_len)
        for i in range(math.ceil(seq_len/2)):
            weight[i] = (i+1)
            weight[seq_len-i-1] = (i+1)
        weight = weight / weight.sum()
    else:
        raise ValueError('Invalid mode')
    
    return weight

def predict_location(heatmap, mode='center'):
    """ Get coordinates from the heatmap.

        Args:
            heatmap (numpy.ndarray): A single heatmap with shape (H, W)
            mode (str): Mode of prediction
                Choices:
                    - 'center': Return center coordinate of the bounding box with max area
                    - 'bbox': Return bounding box of the contour with max area

        Returns:
            cxcy (Tuple[int, int]): center coordinate of the the bounding box with max area
            x1y1x2y2 (Tuple[int, int, int, int]): bounding box of the contour with max area
    """
    if np.amax(heatmap) == 0:
        # No respond in heatmap
        if mode == 'center':
            return 0, 0
        if mode == 'bbox':
            return 0, 0, 0, 0
    else:
        # Find all respond area in the heapmap
        (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in cnts]

        # Find largest area amoung all contours
        max_area_idx = 0
        max_area = rects[0][2] * rects[0][3]
        for i in range(1, len(rects)):
            area = rects[i][2] * rects[i][3]
            if area > max_area:
                max_area_idx = i
                max_area = area
        x, y, w, h = rects[max_area_idx]

        if mode == 'center':
            return int(x+w/2), int(y+h/2)
        if mode == 'bbox':
            return int(x), int(y), int(x+w), int(y+h)

def evaluate(indices, y_true=None, y_pred=None, c_true=None, c_pred=None, tolerance=4., img_scaler=(1, 1)):
    """ Predict and output the result of each frame.

        Args:
            indices (torch.Tensor) - Indices with shape (N, L, 2)
            y_true (torch.Tensor, optional) - Ground-truth heatmap sequences with shape (N, L, H, W)
            y_pred (torch.Tensor, optional) - Predicted heatmap sequences with shape (N, L, H, W)
            c_true (torch.Tensor, optional) - Ground-truth coordinate sequences with shape (N, L, 2)
            c_pred (torch.Tensor, optional) - Predicted coordinate sequences with shape (N, L, 2)
            tolerance (float) - Tolerance for FP1
            img_scaler (Tuple[float, float]) - Scaler of input image size to original image size
        
        Returns:
            pred_dict (Dict) - Prediction result
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Type':[]}
    """
    
    pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Type':[]}

    """ TODO: save detection bbox
        detection_list = []
        Detection Format
        [{
            "image_id": int,
            "category_id": int,
            "bbox": [x,y,width,height],
            "score": float,
        }]
    """
    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy().tolist() if torch.is_tensor(indices) else indices.numpy().tolist()

    # Transform input for heatmap prediction
    if y_true is not None and y_pred is not None:
        assert c_true is None and c_pred is None, 'Invalid input'
        y_pred = y_pred > 0.5
        y_true = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else y_true
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_true = to_img_format(y_true) # (N, L, H, W)
        y_pred = to_img_format(y_pred) # (N, L, H, W)
    
    # Transform input for coordinate prediction
    if c_true is not None and c_pred is not None:
        assert y_true is None and y_pred is None, 'Invalid input'
        c_true = c_true.detach().cpu().numpy() if torch.is_tensor(c_true) else c_true
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred
        c_true[:, :, 0] = c_true[:, :, 0] * WIDTH
        c_true[:, :, 1] = c_true[:, :, 1] * HEIGHT
        c_pred[:, :, 0] = c_pred[:, :, 0] * WIDTH
        c_pred[:, :, 1] = c_pred[:, :, 1] * HEIGHT

    for n in range(batch_size):
        prev_d_i = [-1, -1] # for ignoring the same frame in sequence
        for f in range(seq_len):
            d_i = indices[n][f]
            if d_i != prev_d_i:
                if c_true is not None and c_pred is not None:
                    # Predict from coordinate
                    c_t = c_true[n][f]
                    c_p = c_pred[n][f]
                    cx_pred, cy_pred = int(c_p[0]), int(c_p[1])
                    vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                    if np.amax(c_p) == 0 and np.amax(c_t) == 0:
                        # True Negative: prediction is no ball, and ground truth is no ball
                        pred_dict['Type'].append(pred_types_map['TN'])
                    elif np.amax(c_p) > 0 and np.amax(c_t) == 0:
                        # False Positive 2: prediction is ball existing, but ground truth is no ball
                        pred_dict['Type'].append(pred_types_map['FP2'])
                    elif np.amax(c_p) == 0 and np.amax(c_t) > 0:
                        # False Negative: prediction is no ball, but ground truth is ball existing
                        pred_dict['Type'].append(pred_types_map['FN'])
                    elif np.amax(c_p) > 0 and np.amax(c_t) > 0:
                        # Both prediction and ground truth are ball existing
                        cx_true, cy_true = int(c_t[0]), int(c_t[1])
                        dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                        if dist > tolerance:
                            # False Positive 1: prediction is ball existing, but is too far from ground truth
                            pred_dict['Type'].append(pred_types_map['FP1'])
                        else:
                            # True Positive
                            pred_dict['Type'].append(pred_types_map['TP'])
                    else:
                        raise ValueError(f'Invalid input: {c_p}, {c_t}')
                elif y_true is not None and y_pred is not None:
                    # Predict from heatmap
                    y_t = y_true[n][f]
                    y_p = y_pred[n][f]
                    cx_pred, cy_pred = predict_location(to_img(y_p))
                    vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                    if np.amax(y_p) == 0 and np.amax(y_t) == 0:
                        # True Negative: prediction is no ball, and ground truth is no ball
                        pred_dict['Type'].append(pred_types_map['TN'])
                    elif np.amax(y_p) > 0 and np.amax(y_t) == 0:
                        # False Positive 2: prediction is ball existing, but ground truth is no ball
                        pred_dict['Type'].append(pred_types_map['FP2'])
                    elif np.amax(y_p) == 0 and np.amax(y_t) > 0:
                        # False Negative: prediction is no ball, but ground truth is ball existing
                        pred_dict['Type'].append(pred_types_map['FN'])
                    elif np.amax(y_p) > 0 and np.amax(y_t) > 0:
                        # Both prediction and ground truth are ball existing
                        # Find center coordinate of the contour with max area as prediction
                        cx_true, cy_true = predict_location(to_img(y_t))
                        dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                        if dist > tolerance:
                            # False Positive 1: prediction is ball existing, but is too far from ground truth
                            pred_dict['Type'].append(pred_types_map['FP1'])
                        else:
                            # True Positive
                            pred_dict['Type'].append(pred_types_map['TP'])
                    else:
                        raise ValueError('Invalid input')
                else:
                    raise ValueError('Invalid input')
                pred_dict['Frame'].append(int(d_i[1]))
                pred_dict['X'].append(int(cx_pred*img_scaler[0]))
                pred_dict['Y'].append(int(cy_pred*img_scaler[1]))
                pred_dict['Visibility'].append(vis_pred)
                prev_d_i = d_i
            else:
                break
    
    return pred_dict

def generate_inpaint_mask(pred_dict, y_max, y_th_ratio=0.04):
    """ Generate inpaint mask form predicted trajectory.

        Args:
            pred_dict (Dict): Prediction result
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
            y_max (int): Max y coordinate of the original image size
            y_th_ratio (float): Threshold ratio for y coordinate
        
        Returns:
            inpaint_mask (List): Inpaint mask
    """
    y = np.array(pred_dict['Y'])
    vis_pred = np.array(pred_dict['Visibility'])
    inpaint_mask = np.zeros_like(y)
    i = 0 # index that ball start to disappear
    j = 0 # index that ball start to appear
    threshold = y_max * y_th_ratio
    while j < len(vis_pred):
        while i < len(vis_pred) and vis_pred[i] == 1:
            i += 1
        j = i
        while j < len(vis_pred) and vis_pred[j] == 0:
            j += 1
        if j == i:
            break
        elif i == 0 and y[j] > threshold:
            # start from the first frame that ball disappear
            inpaint_mask[:j] = 1
        elif (i > 1 and y[i-1] > threshold) and (j < len(vis_pred) and y[j] > threshold):
            inpaint_mask[i:j] = 1
        else:
            # ball is out of the field of camera view 
            pass
        i = j
    
    return inpaint_mask.tolist()

# Only for training evaluation, won't save the result
def get_eval_res(pred_dict):
    """ Parse prediction result and get evaluation result.

        Args:
            pred_dict (Dict): Prediction result
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Type':[]}
        
        Returns:
            res (numpy.ndarray): Evaluation result
                Format: np.array([TP, TN, FP1, FP2, FN])
    """

    type_res = np.array(pred_dict['Type'])
    res = np.zeros(5)
    for pred_type in pred_types:
        res[pred_types_map[pred_type]] += int((type_res == pred_types_map[pred_type]).sum())

    return res

def eval_tracknet(model, data_loader, param_dict):
    """ Evaluate TrackNet model.

        Args:
            model (nn.Module): TrackNet model
            data_loader (torch.utils.data.DataLoader): DataLoader for evaluation
            param_dict (Dict): Parameters
                param_dict['verbose'] (bool): Whether to show progress bar
                param_dict['tolerance'] (int): Tolerance for FP1
            
        Returns:
            (float): Average loss
            res_dict (Dict): Evaluation result
                Format:{'TP': TP, 'TN': TN,
                        'FP1': FP1, 'FP2': FP2, 'FN': FN,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'miss_rate': miss_rate}
    """

    model.eval()
    losses = []
    confusion_matrix = np.zeros(5) # TP, TN, FP1, FP2, FN
    if param_dict['verbose']:
        data_prob = tqdm(data_loader)
    else:
        data_prob = data_loader
    
    for step, (i, x, y, _, _) in enumerate(data_prob):
        x, y = x.float().cuda(), y.float().cuda()
        with torch.no_grad():
            y_pred = model(x)

        loss = WBCELoss(y_pred, y)
        losses.append(loss.item())

        pred_dict = evaluate(i, y_true=y, y_pred=y_pred, tolerance=param_dict['tolerance'])
        confusion_matrix += get_eval_res(pred_dict)
        
        if param_dict['verbose']:
            TP, TN, FP1, FP2, FN = confusion_matrix
            data_prob.set_description(f'Evaluation')
            data_prob.set_postfix(TP=TP, TN=TN, FP1=FP1, FP2=FP2, FN=FN)
    
    TP, TN, FP1, FP2, FN = confusion_matrix
    accuracy, precision, recall, f1, miss_rate = get_metric(TP, TN, FP1, FP2, FN)
    res_dict = {'TP': TP, 'TN': TN,
                'FP1': FP1, 'FP2': FP2, 'FN': FN,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'miss_rate': miss_rate}
    
    return float(np.mean(losses)), res_dict

def eval_inpaintnet(model, data_loader, param_dict):
    """ Evaluate TrackNet model.

        Args:
            model (nn.Module): TrackNet model
            data_loader (torch.utils.data.DataLoader): DataLoader for evaluation
            param_dict (Dict): Parameters
                param_dict['verbose'] (bool): Whether to show progress bar
                param_dict['tolerance'] (int): Tolerance for FP1
            
        Returns:
            (float): Average loss
            res_dict (Dict): Evaluation result
                Format:{'TP': TP, 'TN': TN,
                        'FP1': FP1, 'FP2': FP2, 'FN': FN,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'miss_rate': miss_rate}
    """

    model.eval()
    losses = []
    confusion_matrix = {eval_type: np.zeros(5) for eval_type in inpaintnet_eval_types} # TP, TN, FP1, FP2, FN
    if param_dict['verbose']:
        data_prob = tqdm(data_loader)
    else:
        data_prob = data_loader

    for step, (i, coor_pred, coor, _, _, inpaint_mask) in enumerate(data_prob):
        coor_pred, coor, inpaint_mask = coor_pred.float().cuda(), coor.float().cuda(), inpaint_mask.float().cuda()
        
        with torch.no_grad():
            coor_inpaint = model(coor_pred, inpaint_mask)
            coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask)
            
            loss = nn.MSELoss()(coor_inpaint * inpaint_mask, coor * inpaint_mask)
            losses.append(loss.item())

            # Thresholding
            th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
            coor_inpaint[th_mask] = 0.
        
        for eval_type in inpaintnet_eval_types:
            if eval_type == 'inpaint':
                pred_dict = evaluate(i, c_true=coor, c_pred=coor_inpaint, tolerance=param_dict['tolerance'])
            elif eval_type == 'reconstruct':
                pred_dict = evaluate(i, c_true=coor_pred, c_pred=coor_inpaint, tolerance=param_dict['tolerance'])
            elif eval_type == 'baseline':
                pred_dict = evaluate(i, c_true=coor, c_pred=coor_pred, tolerance=param_dict['tolerance'])
            else:
                raise ValueError('Invalid eval_type')
            confusion_matrix[eval_type] += get_eval_res(pred_dict)
        
        if param_dict['verbose']:
            TP, TN, FP1, FP2, FN = confusion_matrix['inpaint']
            data_prob.set_description(f'Evaluation')
            data_prob.set_postfix(TP=TP, TN=TN, FP1=FP1, FP2=FP2, FN=FN)
    
    res_dict = {}
    for eval_type in inpaintnet_eval_types:
        TP, TN, FP1, FP2, FN = confusion_matrix[eval_type]
        accuracy, precision, recall, f1, miss_rate = get_metric(TP, TN, FP1, FP2, FN)
        res_dict[eval_type] = {'TP': TP, 'TN': TN,
                               'FP1': FP1, 'FP2': FP2, 'FN': FN,
                               'accuracy': accuracy,
                               'precision': precision,
                               'recall': recall,
                               'f1': f1,
                               'miss_rate': miss_rate}
    
    return float(np.mean(losses)), res_dict

# For testing evaluation
def get_test_res(pred_dict, drop=False):
    """ Parse prediction result and get evaluation result.

        Args:
            pred_dict (Dict): Prediction result
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Type':[]}
            drop (bool): Whether to drop the frames in the drop frame range

        Returns:
            res_dict (Dict): Evaluation result
                Format: {'TP': TP, 'TN': TN,
                         'FP1': FP1, 'FP2': FP2, 'FN': FN,
                         'accuracy': accuracy,
                         'precision': precision,
                         'recall': recall,
                         'f1': f1,
                         'miss_rate': miss_rate}
    """

    res_dict = {pred_type: 0 for pred_type in pred_types}
    for rally_key, pred in pred_dict.items():
        if drop:
            drop_frame_dict = json.load(open(f'{data_dir}/drop_frame.json'))
            start_f, end_f = drop_frame_dict['start'], drop_frame_dict['end']
            type_res = np.array(pred['Type'])[start_f[rally_key]:end_f[rally_key]]
        else:
            type_res = np.array(pred['Type'])
        
        # Calculate metrics
        for pred_type in pred_types:
            res_dict[pred_type] += int((type_res == pred_types_map[pred_type]).sum())
    
    TP, TN, FP1, FP2, FN = res_dict['TP'], res_dict['TN'], res_dict['FP1'], res_dict['FP2'], res_dict['FN']
    accuracy, precision, recall, f1, miss_rate = get_metric(TP, TN, FP1, FP2, FN)
    res_dict = {'TP': TP, 'TN': TN,
                'FP1': FP1, 'FP2': FP2, 'FN': FN,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'miss_rate': miss_rate}
    
    return res_dict

def test(model, split, param_dict, save_inpaint_mask=False):
    """ Test model on all the rallies in the split.

        Args:
            model (nn.Module): TrackNet model
            split (str): Split for testing
                Choices: 'train', 'val', 'test'
            param_dict (Dict): Parameters
            save_inpaint_mask (bool): Whether to save inpaint mask to '{data_dir}/match{match_id}/predicted_csv/{rally_id}_ball.csv'

        Returns:
            pred_dict (Dict): Evaluation result
                Format: {'{match_id}_{rally_id}': {
                            'TP': TP, 'TN': TN,
                            'FP1': FP1, 'FP2': FP2, 'FN': FN,
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'miss_rate': miss_rate}, ...
                        }
    """

    # Rally-based test
    pred_dict = {}
    rally_dirs = get_rally_dirs(data_dir, split)
    rally_dirs = [os.path.join(data_dir, rally_dir) for rally_dir in rally_dirs]
    for rally_dir in rally_dirs:
        # Parse rally directory to form rally key
        file_format_str = os.path.join('{}', 'frame', '{}')
        match_dir, rally_id = parse.parse(file_format_str, rally_dir)
        match_id = match_dir.split('match')[-1]
        rally_key = f'{match_id}_{rally_id}'

        # Test
        tmp_pred = test_rally(model, rally_dir, param_dict)
        pred_dict[rally_key] = tmp_pred

        if save_inpaint_mask:
            if not os.path.exists(os.path.join(match_dir, 'predicted_csv')):
                os.makedirs(os.path.join(match_dir, 'predicted_csv'))
            csv_file = os.path.join(match_dir, 'predicted_csv',f'{rally_id}_ball.csv')
            write_pred_csv(tmp_pred, save_file=csv_file, save_inpaint_mask=save_inpaint_mask)
    
    return pred_dict

def test_rally(model, rally_dir, param_dict):
    """ Test model on a single rally.

        Args:
            model (Tuple[nn.Module, nn.Module]): TrackNet model
            rally_dir (str): Directory of the rally
            param_dict (Dict): Parameters
                param_dict['eval_mode'] (str): Mode of temporal ensemble
                param_dict['tolerance'] (int): Tolerance for FP1
                param_dict['bg_mode'] (str): Mode of background
                param_dict['batch_size'] (int): Batch size
                param_dict['num_workers'] (int): Number of workers
                param_dict['tracknet_seq_len'] (int): Length of input sequence for TrackNet
                param_dict['inpaintnet_seq_len'] (int): Length of input sequence for InpaintNet

        Returns:
            pred_dict (Dict): Evaluation result
                Format: {'TP': TP, 'TN': TN,
                         'FP1': FP1, 'FP2': FP2, 'FN': FN,
                         'accuracy': accuracy,
                         'precision': precision,
                         'recall': recall,
                         'f1': f1,
                         'miss_rate': miss_rate}
    """

    tracknet, inpaintnet = model
    w, h = Image.open(os.path.join(rally_dir, '0.png')).size
    w_scaler, h_scaler = WIDTH / w, HEIGHT / h

    # Test on TrackNet
    tracknet.eval()
    seq_len = param_dict['tracknet_seq_len']
    pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Type':[]}

    if param_dict['eval_mode'] == 'nonoverlap':
        # Create dataset with non-overlap sampling
        dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='heatmap', bg_mode=param_dict['bg_mode'], rally_dir=rally_dir, padding=True)
        data_loader = DataLoader(dataset, batch_size=param_dict['batch_size'], shuffle=False, num_workers=param_dict['num_workers'], drop_last=False)
        
        data_prob = tqdm(data_loader) if param_dict['verbose'] else data_loader
        for step, (i, x, y, _, _) in enumerate(data_prob):
            x = x.float().cuda()
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()
            
            # Predict
            tmp_pred = evaluate(i, y_true=y, y_pred=y_pred, tolerance=param_dict['tolerance'], img_scaler=(w_scaler, h_scaler))
            for key in tmp_pred.keys():
                pred_dict[key].extend(tmp_pred[key])
    else:
        # Create dataset with overlap sampling for temporal ensemble
        dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='heatmap', bg_mode=param_dict['bg_mode'], rally_dir=rally_dir)
        data_loader = DataLoader(dataset, batch_size=param_dict['batch_size'], shuffle=False, num_workers=param_dict['num_workers'], drop_last=False)

        num_batch = len(data_loader) # for handling edge cases
        weight = get_ensemble_weight(seq_len, param_dict['eval_mode'])

        # Init buffer parameters
        buffer_size = seq_len - 1
        batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
        frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
        y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)

        data_prob = tqdm(data_loader) if param_dict['verbose'] else data_loader
        for step, (i, x, y, _, _) in enumerate(data_prob):
            x = x.float().cuda()
            b_size, seq_len = i.shape[0], i.shape[1]
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()
            
            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
            ensemble_y = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)
            ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

            if step == num_batch-1:
                # Last batch
                y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)
                count = buffer_size
                for b in range(b_size+buffer_size):
                    if b >= b_size:
                        # Last input sequence
                        y_pred = y_pred_buffer[batch_i+b, frame_i].sum(0)
                        y_pred /= count
                        frame_idx = seq_len-count
                        ensemble_i = torch.cat((ensemble_i, i[-1][frame_idx].reshape(1, 1, 2)), dim=0)
                        ensemble_y = torch.cat((ensemble_y, y[-1][frame_idx].reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                        count -= 1
                    else:
                        # General case
                        y_pred = (y_pred_buffer[batch_i+b, frame_i] * weight[:, None, None]).sum(0)
                        ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                        ensemble_y = torch.cat((ensemble_y, y[b][0].reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
            else:
                for b in range(b_size):
                    if step == 0 and b < buffer_size:
                        # First batch
                        y_pred = y_pred_buffer[batch_i+b, frame_i].sum(0)
                        y_pred /= (b+1)
                    else:
                        # General case
                        y_pred = (y_pred_buffer[batch_i+b, frame_i] * weight[:, None, None]).sum(0)
                    
                    ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                    ensemble_y = torch.cat((ensemble_y, y[b][0].reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                    ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                    
            # Predict
            tmp_pred = evaluate(ensemble_i, y_true=ensemble_y, y_pred=ensemble_y_pred,
                               tolerance=param_dict['tolerance'], img_scaler=(w_scaler, h_scaler))
            for key in tmp_pred.keys():
                pred_dict[key].extend(tmp_pred[key])

            # Update buffer, keep last predictions for ensemble in next iteration
            y_pred_buffer = y_pred_buffer[-(seq_len-1):]
    
    # Test on TrackNetV3 (TrackNet + InpaintNet)
    if inpaintnet is not None:
        inpaintnet.eval()
        seq_len = param_dict['inpaintnet_seq_len']
        pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Type':[]}

        if param_dict['eval_mode'] == 'nonoverlap':
            # Create dataset with non-overlap sampling
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='coordinate', rally_dir=rally_dir, padding=True)
            data_loader = DataLoader(dataset, batch_size=param_dict['batch_size'], shuffle=False, num_workers=param_dict['num_workers'], drop_last=False)

            data_prob = tqdm(data_loader) if param_dict['verbose'] else data_loader
            for step, (i, coor_pred, coor, _, _, inpaint_mask) in enumerate(data_prob):
                coor_pred, coor, inpaint_mask = coor_pred.float(), coor.float(), inpaint_mask.float()
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.cuda(), inpaint_mask.cuda()).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask) # replace predicted coordinates with inpainted coordinates
                
                # Thresholding
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.
                
                # Predict
                tmp_pred = evaluate(i, c_true=coor, c_pred=coor_inpaint, tolerance=param_dict['tolerance'], img_scaler=(w_scaler, h_scaler))
                for key in tmp_pred.keys():
                    pred_dict[key].extend(tmp_pred[key])
        else:
            # Create dataset with overlap sampling for temporal ensemble
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='coordinate', rally_dir=rally_dir)
            data_loader = DataLoader(dataset, batch_size=param_dict['batch_size'], shuffle=False, num_workers=param_dict['num_workers'], drop_last=False)

            num_batch = len(data_loader)
            weight = get_ensemble_weight(seq_len, param_dict['eval_mode'])

            # Init buffer params
            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len) # [0, 1, 2, 3, 4, 5, 6, 7]
            frame_i = torch.arange(seq_len-1, -1, -1) # [7, 6, 5, 4, 3, 2, 1, 0]
            coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)

            data_prob = tqdm(data_loader) if param_dict['verbose'] else data_loader
            for step, (i, coor_pred, coor, _, _, inpaint_mask) in enumerate(data_prob):
                coor_pred, coor, inpaint_mask = coor_pred.float(), coor.float(), inpaint_mask.float()
                b_size = i.shape[0]
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.cuda(), inpaint_mask.cuda()).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1-inpaint_mask) # replace predicted coordinates with inpainted coordinates
                
                # Thresholding
                th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.

                coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
                ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)
                
                if step == num_batch-1:
                    # Last batch
                    coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                    coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)
                    count = buffer_size
                    for b in range(b_size+buffer_size):
                        if b >= b_size:
                            # Last input sequence
                            coor_inpaint = coor_inpaint_buffer[batch_i+b, frame_i].sum(0)
                            coor_inpaint /= count
                            frame_idx = seq_len-count
                            ensemble_i = torch.cat((ensemble_i, i[-1][frame_idx].view(1, 1, 2)), dim=0)
                            ensemble_coor = torch.cat((ensemble_coor, coor[-1][frame_idx].view(1, 1, 2)), dim=0)
                            ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)
                            count -= 1
                        else:
                            # General case
                            coor_inpaint = (coor_inpaint_buffer[batch_i+b, frame_i] * weight[:, None]).sum(0)
                            ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                            ensemble_coor = torch.cat((ensemble_coor, coor[b][0].view(1, 1, 2)), dim=0)
                            ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)
                else:
                    for b in range(b_size):
                        if step == 0 and b < buffer_size:
                            # First batch
                            coor_inpaint = coor_inpaint_buffer[batch_i+b, frame_i].sum(0)
                            coor_inpaint /= (b+1)
                        else:
                            # General case
                            coor_inpaint = (coor_inpaint_buffer[batch_i+b, frame_i] * weight[:, None]).sum(0)
                        
                        ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                        ensemble_coor = torch.cat((ensemble_coor, coor[b][0].view(1, 1, 2)), dim=0)
                        ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)

                # Thresholding
                th_mask = ((ensemble_coor_inpaint[:, :, 0] < COOR_TH) & (ensemble_coor_inpaint[:, :, 1] < COOR_TH))
                ensemble_coor_inpaint[th_mask] = 0.

                # Predict
                tmp_pred = evaluate(ensemble_i, c_true=ensemble_coor, c_pred=ensemble_coor_inpaint,
                                    tolerance=param_dict['tolerance'], img_scaler=(w_scaler, h_scaler))
                for key in tmp_pred.keys():
                    pred_dict[key].extend(tmp_pred[key])

                # Update buffer, keep last predictions for ensemble in next iteration
                coor_inpaint_buffer = coor_inpaint_buffer[-(seq_len-1):]

        return pred_dict
    else:
        pred_dict['Inpaint_Mask'] = generate_inpaint_mask(pred_dict, y_max=h, y_th_ratio=0.04)

        return pred_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracknet_file', type=str, help='file path of the TrackNet model checkpoint')
    parser.add_argument('--inpaintnet_file', type=str, default='', help='file path of the InpaintNet model checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='dataset split for testing')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for testing')
    parser.add_argument('--tolerance', type=float, default=4, help='difference tolerance of center distance between prediction and ground truth in input size')
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'], help='evaluation mode')
    parser.add_argument('--video_file', type=str, default='', help='file path of the video with label (must in dataset directory with same data_dir)')
    parser.add_argument('--output_pred', action='store_true', default=False, help='whether to output detail prediction result for error analysis')
    parser.add_argument('--save_dir', type=str, default='output', help='directory to save the evaluation result')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    param_dict = vars(args)
    param_dict['num_workers'] = args.batch_size if args.batch_size <= 16 else 16
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Load parameter
    print(f'Loading checkpoint...')
    if args.tracknet_file:
        tracknet_ckpt = torch.load(args.tracknet_file)
        param_dict['tracknet_seq_len'] = tracknet_ckpt['param_dict']['seq_len']
        param_dict['bg_mode'] = tracknet_ckpt['param_dict']['bg_mode']
        tracknet = get_model('TrackNet', seq_len=param_dict['tracknet_seq_len'], bg_mode=param_dict['bg_mode']).cuda()
        tracknet.load_state_dict(tracknet_ckpt['model'])
        model = (tracknet, None)
    
    if args.inpaintnet_file:
        assert args.tracknet_file
        inpaintnet_ckpt = torch.load(args.inpaintnet_file)
        param_dict['inpaintnet_seq_len'] = inpaintnet_ckpt['param_dict']['seq_len']
        inpaintnet = get_model('InpaintNet').cuda()
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
        model = (tracknet, inpaintnet)

    if args.video_file:
        # Evaluation on video
        print(f'Test on video {args.video_file} ...')
        file_format_str = os.path.join('{}', 'video', '{}.mp4')
        match_dir, rally_id = parse.parse(file_format_str, args.video_file)
        rally_dir = os.path.join(match_dir, 'frame', rally_id)

        # Load label
        csv_file = os.path.join(match_dir, 'corrected_csv', f'{rally_id}_ball.csv') if 'test' in rally_dir else os.path.join(match_dir, 'csv', f'{rally_id}_ball.csv')
        assert os.path.exists(csv_file), f'{csv_file} does not exist.'
        label_df = pd.read_csv(csv_file, encoding='utf8').sort_values(by='Frame').fillna(0)

        # Predict label
        pred_dict = test_rally(model, rally_dir, param_dict)

        # Write results
        out_video_file = os.path.join(args.save_dir, f'{rally_id}.mp4')
        out_csv_file = os.path.join(args.save_dir, f'{rally_id}_ball.csv')
        frame_list, fps, (w, h) = generate_frames(args.video_file)
        write_pred_video(frame_list, dict(fps=fps, shape=(w, h)), pred_dict, label_df=label_df, save_file=out_video_file)
        write_pred_csv(pred_dict, save_file=out_csv_file)
    else:
        # Evaluation on dataset

        eval_analysis_file = os.path.join(args.save_dir, f'{args.split}_eval_analysis_{args.eval_mode}.json') # for error analysis interface
        eval_res_file = os.path.join(args.save_dir, f'{args.split}_eval_res_{args.eval_mode}.json')

        start_time = time.time()
        print(f'Split: {args.split}')
        print(f'Evaluation mode: {args.eval_mode}')
        print(f'Tolerance Value: {args.tolerance}')
        
        pred_dict = test(model, args.split, param_dict)
        if args.split == 'test':
            # Drop samples which is not in the effective trajectory
            res_dict = get_test_res(pred_dict, drop=True)
        else:
            res_dict = get_test_res(pred_dict, drop=False)
        
        with open(eval_res_file, 'w') as f:
            json.dump(res_dict, f, indent=2)

        if args.output_pred:
            eval_dict = dict(param_dict=param_dict, pred_dict=pred_dict)
            with open(eval_analysis_file, 'w') as f:
                json.dump(eval_dict, f, indent=2)
