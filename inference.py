import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pandas as pd
import cv2
import glob

from dataset_ball import UniBall_Dataset
from test import predict_location
from utils.general import get_model, to_img_format, generate_frames, to_img, write_pred_video

def predict(y_pred=None, img_scaler=(1, 1), WIDTH=640, HEIGHT=480, start_frame=0):
    """ Predict coordinates from heatmap or inpainted coordinates. 

        Args:
            indices (torch.Tensor): indices of input sequence with shape (N, L, 2)
            y_pred (torch.Tensor, optional): predicted heatmap sequence with shape (N, L, H, W)
            c_pred (torch.Tensor, optional): predicted inpainted coordinates sequence with shape (N, L, 2)
            img_scaler (Tuple): image scaler (w_scaler, h_scaler)

        Returns:
            pred_dict (Dict): dictionary of predicted coordinates
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
    """

    pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

    # Transform input for heatmap prediction
    if y_pred is not None:
        y_pred = y_pred > 0.5
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred)
    
    N = y_pred.shape[0]
    for i in range(N):
        y_p = y_pred[i][0]
        bbox_pred = predict_location(to_img(y_p))
        cx_pred, cy_pred = int(bbox_pred[0]+bbox_pred[2]/2), int(bbox_pred[1]+bbox_pred[3]/2)
        cx_pred, cy_pred = int(cx_pred*img_scaler[0]), int(cy_pred*img_scaler[1])
        vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
        pred_dict['Frame'].append(int(i+start_frame))
        pred_dict['X'].append(cx_pred)
        pred_dict['Y'].append(cy_pred)
        pred_dict['Visibility'].append(vis_pred)    
    return pred_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='TrackNet', choices=['TrackNet', 'VideoMamba'], help='model type')
    parser.add_argument('--model_path', type=str, default=None, help='checkpoint path')
    parser.add_argument('--video_dir', type=str, default=None, help='input videos dir')
    parser.add_argument('--video_path', type=str, default=None, help='input video path')
    parser.add_argument('--frame_dir', type=str, default=None, help='input frames dir')
    parser.add_argument('--image_format', type=str, default='jpg', help='image format')
    parser.add_argument('--video_format', type=str, default='mp4', help='video format')
    parser.add_argument('--save_dir', type=str, default='pred_result', help='directory to save the prediction result')
    parser.add_argument('--output_video', action='store_true', default=True, help='whether to output video with predicted trajectory')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    args.last_only = True
    
    return args

def remove_ddp_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict

def main():
    args = get_args()
    num_workers = args.batch_size if args.batch_size <= 16 else 16
    
    print(f'Load {args.model_name}...')
    assert args.model_path is not None, 'Please provide model path'
    assert os.path.exists(args.model_path), f'No checkpoint found in {args.model_path}'
    ckpt = torch.load(args.model_path)
    param_dict = ckpt['param_dict']
    seq_len = param_dict['seq_len']
    bg_mode = param_dict['bg_mode']
    model = get_model(args.model_name, seq_len, bg_mode)

    model.load_state_dict(remove_ddp_prefix(ckpt['model']))
    WIDTH = 512
    HEIGHT = 288
    model = model.cuda()
    model.eval()

    videos = []
    if args.video_dir is not None:
        videos = glob.glob(os.path.join(args.video_dir, f'*.{args.video_format}'))
        args.save_dir = os.path.join(args.save_dir, os.path.basename(args.video_dir))
        os.makedirs(args.save_dir, exist_ok=True)
    elif args.video_path is not None:
        videos = [args.video_path]
    
    for video in tqdm(videos):
        start = time.time()
        frame_list = generate_frames(video)
        h, w = frame_list[0].shape[:2]
        img_scaler = (w/WIDTH, h/HEIGHT)
        video_name = '.'.join(os.path.basename(video).split(".")[:-1])
        print(f'Predicting {video_name}...')
        median_path = os.path.join(args.save_dir, video_name+'_median.npz')
        if not os.path.exists(median_path):
            print(f'No median frame found for {video_name}, Creating...')
            median = np.median(frame_list, 0)
            np.savez(median_path, median=median)
            cv2.imwrite(os.path.join(args.save_dir, f'{video_name}_median.png'), median)
        else:
            median = np.load(median_path)['median']

        dataset = UniBall_Dataset(seq_len=seq_len, sliding_step=1, data_mode='heatmap', bg_mode=bg_mode,
                                                frame_arr=np.array(frame_list)[:, :, :, ::-1], median=median)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Img_scaler': img_scaler, 'Img_shape': (w, h)}

        start_frame = 0
        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().cuda()
            b_size, seq_len = i.shape[0], i.shape[1]
            with torch.no_grad():
                y_pred = model(x).detach().cpu()
                y_pred = y_pred[:, -1:, :, :]
                result = predict(y_pred, img_scaler, WIDTH, HEIGHT, start_frame=start_frame)
                start_frame += b_size
                for key in result:
                    pred_dict[key].extend(result[key])
        
        
        out_csv_file = os.path.join(args.save_dir, f'{video_name}.csv')
        pred_df = pd.DataFrame({'Frame': pred_dict['Frame'],
                                'Visibility': pred_dict['Visibility'],
                                'X': pred_dict['X'],
                                'Y': pred_dict['Y']})
        pred_df.to_csv(out_csv_file, index=False)
        print(f'Prediction for {video_name} done.\nFPS: {len(frame_list)/(time.time()-start):.2f}s')

        if args.output_video:
            out_video_file = os.path.join(args.save_dir, f'{video_name}.mp4')
            write_pred_video(video, pred_dict, save_file=out_video_file, traj_len=seq_len)
        

    print('Done.')

if __name__ == '__main__':
    main()