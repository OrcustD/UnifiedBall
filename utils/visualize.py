import os
import cv2
import parse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.general import *

def write_to_tb(model_type, tb_writer, losses, val_res, epoch, step):
    """ Write training and validation results to tensorboard. 

        Args:
            model_type (str): Model type
                Choices:'TrackNet', 'InpaintNet'
            tb_writer (tensorboard.SummaryWriter): Tensorboard writer
            losses (Tuple[float, float]): Training and validation losses
            val_res (dict):Validation results
            epoch (int): Current epoch

        Returns:
            None
    """
    tb_writer.add_scalars(f"Val_Loss/WBCE", {model_type: losses[1]}, step)
    tb_writer.add_scalars(f"Metric/Accurcy", {model_type: val_res['accuracy']}, epoch)
    tb_writer.add_scalars(f"Metric/Precision", {model_type: val_res['precision']}, epoch)
    tb_writer.add_scalars(f"Metric/Recall", {model_type: val_res['recall']}, epoch)
    tb_writer.add_scalars(f"Metric/F1-score", {model_type: val_res['f1']}, epoch)
    tb_writer.add_scalars(f"Metric/Miss_Rate", {model_type: val_res['miss_rate']}, epoch)
    tb_writer.add_scalars(f"Results/TP", {model_type: val_res['TP']}, epoch)
    tb_writer.add_scalars(f"Results/TN", {model_type: val_res['TN']}, epoch)
    tb_writer.add_scalars(f"Results/FP1", {model_type: val_res['FP1']}, epoch)
    tb_writer.add_scalars(f"Results/FP2", {model_type: val_res['FP2']}, epoch)
    tb_writer.add_scalars(f"Results/FN", {model_type: val_res['FN']}, epoch)
    tb_writer.flush()

def plot_median_files(data_dir):
    """ Plot median frames of each match and rally and save to '{data_dir}/median'. 
    
        Args:
            data_dir (str): Data root directory
    """

    rally_dirs = []
    if not os.path.exists(os.path.join(data_dir, 'median')):
        os.makedirs(os.path.join(data_dir, 'median'))
    
    for split in ['train', 'test', 'val']:
        match_dirs = list_dirs(os.path.join(data_dir, split))
        # For each match
        for match_dir in match_dirs:
            file_format_str = os.path.join('{}', 'match{}')
            _, match_id = parse.parse(file_format_str, match_dir)
            if os.path.exists(os.path.join(data_dir, split, f'match{match_id}', 'median.npz')):
                # median = np.load(os.path.join(data_dir, split, f'match{match_id}', 'median.npz'))['median'][..., ::-1] # BGR to RGB
                median = np.load(os.path.join(data_dir, split, f'match{match_id}', 'median.npz'))['median']
                cv2.imwrite(os.path.join(data_dir, 'median', f'{split}_m{match_id}.{IMG_FORMAT}'), median)
            rally_dirs = list_dirs(os.path.join(match_dir, 'frame'))
            # For each rally
            for rally_dir in rally_dirs:
                file_format_str = os.path.join('{}', 'frame', '{}')
                _, rally_id = parse.parse(file_format_str, rally_dir)
                if os.path.exists(os.path.join(rally_dir, 'median.npz')):
                    # median = np.load(os.path.join(rally_dir, 'median.npz'))['median'][..., ::-1] # BGR to RGB
                    median = np.load(os.path.join(rally_dir, 'median.npz'))['median']
                    cv2.imwrite(os.path.join(data_dir, 'median', f'{split}_m{match_id}_r{rally_id}.{IMG_FORMAT}'), median)

def plot_heatmap_pred_sample(x, y, y_pred, c, bg_mode, save_dir, curr_step=None, exp_name=None, output_type='gif'):
    """ Visualize input and output of TrackNet and save as a gif. Including 4 subplots:
            Top left: Frames sequence with ball coordinate marked
            Top right: Ground-truth heatmap sequence
            Bottom left: Predicted heatmap sequence
            Bottom right: Predicted heatmap sequence with thresholding

        Args:
            x (numpy.ndarray): Frame sequence with shape (L, H, W)
            y (numpy.ndarray): Ground-truth heatmap sequence with shape (L, H, W)
            y_pred (numpy.ndarray): Predicted heatmap sequence with shape (L, H, W)
            c (numpy.ndarray): Ground-truth ball coordinate sequence with shape (L, 2)
            bg_mode (str): Background mode of TrackNet
            save_dir (str): Save directory
        
        Returns:
            None
    """

    imgs = []

    # Thresholding
    y_map = y_pred > 0.5

    # Convert input and output to image format
    x = to_img(x)
    y = to_img(y)
    y_p = to_img(y_pred)
    y_m = to_img(y_map)
    
    # Write image sequence to gif
    os.makedirs(f'{save_dir}/{exp_name}', exist_ok=True)
    
    for f in range(c.shape[0]):
        # Convert grayscale image to BGR image for concatenation
        tmp_x = cv2.cvtColor(x[f], cv2.COLOR_GRAY2BGR) if bg_mode == 'subtract' else x[f]
        tmp_y = cv2.cvtColor(y[f], cv2.COLOR_GRAY2BGR)
        tmp_pred = cv2.cvtColor(y_p[f], cv2.COLOR_GRAY2BGR)
        tmp_map = cv2.cvtColor(y_m[f], cv2.COLOR_GRAY2BGR)
        assert tmp_x.shape == tmp_y.shape == tmp_pred.shape == tmp_map.shape

        # Mark ground-truth label
        cv2.circle(tmp_x, (int(c[f][0] * WIDTH), int(c[f][1] * HEIGHT)), 2, (0, 0, 255), -1)

        # Concatenate 4 subplots
        up_img = cv2.hconcat([tmp_x, tmp_y])
        down_img = cv2.hconcat([tmp_pred, tmp_map])
        img = cv2.vconcat([up_img, down_img])

        # Cast cv image to PIL image for saving gif format
        if output_type == 'gif':
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            imgs.append(img)
            if curr_step:
                imgs[0].save(f'{save_dir}/{exp_name}/{curr_step}.gif', format='GIF', save_all=True, append_images=imgs[1:], duration=1000, loop=0)
            else:
                imgs[0].save(f'{save_dir}/{exp_name}/all.gif', format='GIF', save_all=True, append_images=imgs[1:], duration=1000, loop=0)
        else:
            save_name = f'{save_dir}/{exp_name}/{curr_step}.jpg' if curr_step else f'{save_dir}/{exp_name}/all.jpg'
            cv2.imwrite(save_name, img)

def plot_traj_pred_sample(coor_gt, coor_inpaint, inpaint_mask, save_dir=''):
    """ Visualize input and output of InpaintNet and save as a image.

        Args:
            coor_gt (numpy.ndarray): Ground-truth trajectory with shape (L, 2)
            coor_inpaint (numpy.ndarray): Inpainted trajectory with shape (L, 2)
            inpaint_mask (numpy.ndarray): Inpainting mask with shape (L, 1)
            save_dir (str): Save directory
        
        Returns:
            None
    """

    # Create an empty image
    img = np.ones((HEIGHT, WIDTH, 3), dtype='uint8')

    # Mark ground-truth and predicted coordinate in the trajectory
    for f in range(coor_gt.shape[0]):
        img = cv2.circle(img, (int(coor_gt[f][0] * WIDTH), int(coor_gt[f][1] * HEIGHT)), 2, (0, 0, 255), -1)
        if inpaint_mask[f] == 1:
            img = cv2.circle(img, (int(coor_inpaint[f][0] * WIDTH), int(coor_inpaint[f][1] * HEIGHT)), 2, (0, 255, 0), -1)
    
    cv2.imwrite(f'{save_dir}/cur_pred_InpaintNet.{IMG_FORMAT}', img)

def plot_diff_hist(pred_dict_base, pred_dict_refine, split, save_dir, data_dir):
    """ Plot difference histogram. (difference is calculated in input space)

        Args:
            pred_dict_base (Dict): Baseline prediction dictionary
            pred_dict_refine (Dict): Refined prediction dictionary
            split (str): Split name
            save_dir (str): Save directory

        Returns:
            None
    
    """
    plt.rcParams.update({'font.size': 16})
    
    pred_types = ['TP', 'TN', 'FP1', 'FP2', 'FN']
    pred_types_map = {i: pred_type for i, pred_type in enumerate(pred_types)}

    drop_frame_dict = json.load(open(os.path.join(data_dir, 'drop_frame.json')))
    rally_keys = drop_frame_dict['map']
    start_frame, end_frame = drop_frame_dict['start'], drop_frame_dict['end']

    for err_type in ['FP1', 'FP2']:
        refine_diff, baseline_diff = [], []
        for rally_key in rally_keys:
            pred_base = pred_dict_base[rally_key]
            pred_refine = pred_dict_refine[rally_key]
            match_id, rally_id = rally_key.split('_')[0], '_'.join(rally_key.split('_')[1:])
            start_f, end_f = start_frame[rally_key], end_frame[rally_key]
            w, h = Image.open(os.path.join(data_dir, split, f'match{match_id}', 'frame', rally_id, f'0.{IMG_FORMAT}')).size
            w_scaler, h_scaler = w/WIDTH, h/HEIGHT

            # Load ground truth
            csv_file = os.path.join(data_dir, split, f'match{match_id}', 'corrected_csv' if split == 'test' else 'csv', f'{rally_id}_ball.csv')
            label_df = pd.read_csv(csv_file, encoding='utf8')
            x, y, vis = np.array(label_df['X']), np.array(label_df['Y']),np.array(label_df['Visibility'])

            # Load predicted trajectory
            x_pred, y_pred, vis_pred = np.array(pred_base['X']), np.array(pred_base['Y']), np.array(pred_base['Visibility'])
            err_type_pred = np.array(pred_base['Type'])

            # Load refined trajectory
            x_refine, y_refine, vis_refine = np.array(pred_refine['X']), np.array(pred_refine['Y']), np.array(pred_refine['Visibility'])
            err_type_refine = np.array(pred_refine['Type'])

            for frame_i in range(start_f, end_f):
                if pred_types_map[err_type_pred[frame_i]] == 'FP1' and err_type == 'FP1':
                    cx_true, cy_true = int(x[frame_i]/w_scaler), int(y[frame_i]/h_scaler)
                    cx_pred, cy_pred = int(x_pred[frame_i]/w_scaler), int(y_pred[frame_i]/h_scaler)
                    diff = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                    baseline_diff.append(diff)
                elif pred_types_map[err_type_pred[frame_i]] == 'FP2' and err_type == 'FP2':
                    prev_offset, next_offset = 1, 1
                    # Search for nearest visible frame
                    while frame_i-prev_offset >= 0 and vis[frame_i-prev_offset] != 1:
                        prev_offset += 1
                    while frame_i+next_offset < len(x_pred) and vis[frame_i+next_offset] != 1:
                        next_offset += 1
                    cx_pred, cy_pred = int(x_pred[frame_i]/w_scaler), int(y_pred[frame_i]/h_scaler)
                    cx_true, cy_true = int(x[frame_i-prev_offset]/w_scaler), int(y[frame_i-prev_offset]/h_scaler)
                    diff_1 = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2)) # diff with previous frame
                    cx_true, cy_true = int(x[frame_i+next_offset]/w_scaler), int(y[frame_i+next_offset]/h_scaler)
                    diff_2 = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2)) # diff with next frame
                    diff = min(diff_1, diff_2)
                    baseline_diff.append(diff)
                else:
                    pass
                

            for frame_i in range(start_f, end_f):
                if pred_types_map[err_type_refine[frame_i]] == 'FP1' and err_type == 'FP1':
                    cx_true, cy_true = int(x[frame_i]/w_scaler), int(y[frame_i]/h_scaler)
                    cx_pred, cy_pred = int(x_refine[frame_i]/w_scaler), int(y_refine[frame_i]/h_scaler)
                    diff = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                    refine_diff.append(diff)
                elif pred_types_map[err_type_refine[frame_i]] == 'FP2' and err_type == 'FP2':
                    prev_offset, next_offset = 1, 1
                    # Search for nearest visible frame
                    while frame_i-prev_offset >= 0 and vis[frame_i-prev_offset] != 1:
                        prev_offset += 1
                    while frame_i+next_offset < len(x_refine) and vis[frame_i+next_offset] != 1:
                        next_offset += 1
                    cx_pred, cy_pred = int(x_refine[frame_i]/w_scaler), int(y_refine[frame_i]/h_scaler)
                    cx_true, cy_true = int(x[frame_i-prev_offset]/w_scaler), int(y[frame_i-prev_offset]/h_scaler)
                    diff_1 = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                    cx_true, cy_true = int(x[frame_i+next_offset]/w_scaler), int(y[frame_i+next_offset]/h_scaler)
                    diff_2 = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                    diff = min(diff_1, diff_2)
                    refine_diff.append(diff)
                else:
                    pass

        refine_diff, baseline_diff = np.array(refine_diff), np.array(baseline_diff)
        max_diff = max(math.ceil(np.max(refine_diff)), math.ceil(np.max(baseline_diff)))
        bins_bound = [b for b in range(0, max_diff, 4)]

        plt.figure(figsize=(12, 4))
        plt.title(f'{err_type} Sample\nCoordinate Difference Histogram')
        counts, _, _ = plt.hist(refine_diff, bins=bins_bound, label='refine')
        _, _, _ = plt.hist(baseline_diff, bins=bins_bound, label='baseline')
        if max(counts) > 10:
            plt.yticks(np.arange(0, max(counts), 10)) 
        plt.grid(b=True, axis='y')
        if err_type == 'FP1':
            plt.xlabel('Difference between predicted and ground truth coordinate (pixel)')
        else:
            plt.xlabel('Difference between predicted and nearest ground truth coordinate (pixel)')
        plt.ylabel('Sample Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{err_type}_diff.{IMG_FORMAT}'))
        plt.clf()