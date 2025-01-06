import os
import cv2
import parse
import numpy as np
import pandas as pd
# from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.data_io import load_images
import json

class UniBall_Dataset(Dataset):
    """ Shuttlecock_Trajectory_Dataset
            Dataset description: https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw
    """
    def __init__(self,
        root_dir='data/tabletennis',
        split='train',
        seq_len=8,
        sliding_step=1,
        data_mode='heatmap',
        bg_mode='',
        frame_alpha=-1,
        height=288,
        width=512,
        frame_arr=None,
        padding=False,
        debug=False,
        SIGMA=2.5,
        median=None,
        img_format='jpg',
        heatmap_mode='gaussian',
    ):
        """ Initialize the dataset

            Args:
                root_dir (str): File path of root directory of the dataset
                split (str): Split of the dataset, 'train', 'test' or 'val'
                seq_len (int): Length of the input sequence
                sliding_step (int): Sliding step of the sliding window during the generation of input sequences
                data_mode (str): Data mode
                    Choices:
                        - 'heatmap':Return TrackNet input data
                        - 'coordinate': Return InpaintNet input data
                bg_mode (str): Background mode
                    Choices:
                        - '': Return original frame sequence
                        - 'subtract': Return the difference frame sequence
                        - 'subtract_concat': Return the frame sequence with RGB and difference frame channels
                        - 'concat': Return the frame sequence with background as the first frame
                frame_alpha (float): Frame mixup alpha
                rally_dir (str): Rally directory
                frame_arr (numpy.ndarray): Frame sequence for TrackNet inference
                padding (bool): Padding the last frame if the frame sequence is shorter than the input sequence
                debug (bool): Debug mode
                HEIGHT (int): Height of the image for input.
                WIDTH (int): Width of the image for input.
                SIGMA (int): Sigma of the Gaussian heatmap which control the label size.
                median (numpy.ndarray): Median image
        """

        assert split in ['train', 'test', 'val'], f'Invalid split: {split}, should be train, test or val'
        assert data_mode in ['heatmap', 'coordinate'], f'Invalid data_mode: {data_mode}, should be heatmap or coordinate'
        assert bg_mode in ['', 'subtract', 'subtract_concat', 'concat'], f'Invalid bg_mode: {bg_mode}, should be "", subtract, subtract_concat or concat'

        # Image size
        self.metainfo = {}
        if os.path.exists(f'{root_dir}/info/metainfo.json'):
            self.metainfo = json.load(open(f'{root_dir}/info/metainfo.json'))

        self.HEIGHT, self.WIDTH = height, width
        self.img_format = img_format

        # Gaussian heatmap parameters
        self.heatmap_mode = heatmap_mode
        self.mag = 1
        self.sigma = SIGMA

        self.root_dir = root_dir
        self.split = split
        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.data_mode = data_mode
        self.bg_mode = bg_mode
        self.frame_alpha = frame_alpha

        # Data for inference
        self.frame_arr = frame_arr
        self.pred_dict = None
        self.padding = padding and self.sliding_step == self.seq_len

        # Initialize the input data
        if self.frame_arr is not None:
            # assert self.data_mode == 'heatmap', f'Invalid data_mode: {self.data_mode}, frame_arr only for heatmap mode' 
            self.data_dict, self.img_config = self._gen_input_from_frame_arr()
            if self.bg_mode:
                if median is None:
                    median = np.median(self.frame_arr, 0)
                if self.bg_mode == 'concat':
                    median = cv2.resize(median, (self.WIDTH, self.HEIGHT))
                    # median = Image.fromarray(median.astype('uint8'))
                    # median = np.array(median.resize(size=(self.WIDTH, self.HEIGHT)))
                    self.median = np.moveaxis(median, -1, 0)
                else:
                    self.median = median
        else:
            # Generate rally image configuration file
            self.rally_dict = self._get_rally_dict()
            img_config_file = os.path.join(self.root_dir, f'img_config_{self.HEIGHT}x{self.WIDTH}_{self.split}.npz')
            if not os.path.exists(img_config_file):
                self._gen_rally_img_congif_file(img_config_file)
            img_config = np.load(img_config_file)
            self.img_config = {key: img_config[key] for key in img_config.keys()}
            
            # For training and evaluation
            # Generate and load input file 
            os.makedirs(os.path.join(self.root_dir, 'cache'), exist_ok=True)
            input_file = os.path.join(self.root_dir, 'cache', f'data_l{self.seq_len}_s{self.sliding_step}_{self.data_mode}_{self.split}.npz')
            if not os.path.exists(input_file):
                self._gen_input_file(file_name=input_file)
            data_dict = np.load(input_file)
            self.data_dict = {key: data_dict[key] for key in data_dict.keys()}
            if debug:
                num_data = 256
                for key in self.data_dict.keys():
                    self.data_dict[key] = self.data_dict[key][:num_data]
    
    def _get_rally_dict(self):
        """ Return the rally index-path mapping dictionary from split. """
        datalist = json.load(open(os.path.join(self.root_dir, 'info', f'{self.split}.json')))
        rally_dirs = [os.path.join(self.root_dir, 'all', match, 'frame', rally) for match, rally in datalist]
        rally_dict = {'i2p':{i: rally_dir for i, rally_dir in enumerate(rally_dirs)},
                      'p2i':{rally_dir: i for i, rally_dir in enumerate(rally_dirs)}}
        return rally_dict

    def _get_rally_i(self, rally_dir):
        """ Return the corresponding rally index of the rally directory. """
        if rally_dir not in self.rally_dict['p2i'].keys():
            return None
        else:
            return self.rally_dict['p2i'][rally_dir]
    
    def _gen_rally_img_congif_file(self, file_name):
        """ Generate rally image configuration file. """
        if 'image_shape' in self.metainfo:
            num_rally = len(self.rally_dict['i2p'])
            h, w, c = self.metainfo['image_shape']
            w_scaler, h_scaler = w / self.WIDTH, h / self.HEIGHT
            img_scaler = [(w_scaler, h_scaler)] * num_rally
            img_shape = [(w, h)] * num_rally
        else:
            img_scaler = [] # (num_rally, 2)
            img_shape = [] # (num_rally, 2)
            for rally_i, rally_dir in tqdm(self.rally_dict['i2p'].items()):
                # w, h = Image.open(os.path.join(rally_dir, f'0000.{self.img_format}')).size
                h, w, c = cv2.imread(os.path.join(rally_dir, f'0000.{self.img_format}')).shape
                if (w != 1920) or (h != 1080):
                    print(f'{rally_dir} is ({w}, {h}), not (1920, 1080)!')
                w_scaler, h_scaler = w / self.WIDTH, h / self.HEIGHT
                img_scaler.append((w_scaler, h_scaler))
                img_shape.append((w, h))
        
        np.savez(file_name, img_scaler=img_scaler, img_shape=img_shape)
            
    def _gen_input_file(self, file_name):
        """ Generate input file for training and evaluation. """
        print('Generate input file...')
        id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
        frame_file = np.array([]).reshape(0, self.seq_len)
        coor = np.array([], dtype=np.float32).reshape(0, 1, 2)
        vis = np.array([], dtype=np.float32).reshape(0, 1)

        # Generate input sequences from each rally
        for rally_i, rally_dir in tqdm(self.rally_dict['i2p'].items()):
            data_dict = self._gen_input_from_rally_dir(rally_dir)
            id = np.concatenate((id, data_dict['id']), axis=0)
            frame_file = np.concatenate((frame_file, data_dict['frame_file']), axis=0)
            coor = np.concatenate((coor, data_dict['coor']), axis=0)
            vis = np.concatenate((vis, data_dict['vis']), axis=0)
        
        np.savez(file_name, id=id, frame_file=frame_file, coor=coor, vis=vis)

    def _gen_input_from_rally_dir(self, rally_dir):
        """ Generate input sequences from a rally directory. """

        rally_i = self._get_rally_i(rally_dir)
        
        file_format_str = os.path.join('{}', 'frame', '{}')
        match_dir, rally_id = parse.parse(file_format_str, rally_dir)

        # Read label csv file
        csv_file = os.path.join(match_dir, 'csv', f'{rally_id}_ball.csv')
        
        assert os.path.exists(csv_file), f'{csv_file} does not exist.'
        label_df = pd.read_csv(csv_file, encoding='utf8').sort_values(by='Frame').fillna(0)
        fids, x, y, v = np.array(label_df['Frame']), np.array(label_df['X']), np.array(label_df['Y']), np.array(label_df['Visibility'])

        id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
        frame_file = np.array([]).reshape(0, self.seq_len)
        coor = np.array([], dtype=np.float32).reshape(0, 1, 2)
        vis = np.array([], dtype=np.float32).reshape(0, 1)
        
        # Construct input sequences, the labeled frames is the last frame of each input sequence
        for i, fid in enumerate(fids):
            first_index = fid - (self.seq_len-1) * self.sliding_step
            if first_index < 0:
                continue
            # if first_index < 1:
            #     continue
            tmp_idx, tmp_frames, tmp_coor, tmp_vis = [], [], [], []
            # Construct a single input sequence
            for curr_i in range(first_index, fid+1, self.sliding_step):
                tmp_idx.append((rally_i, curr_i))
                tmp_frames.append(os.path.join(rally_dir, f'{curr_i:04d}.{self.img_format}'))
            tmp_coor.append((x[i], y[i]))
            tmp_vis.append(v[i])
            
            # Append the input sequence
            if len(tmp_frames) == self.seq_len:
                id = np.concatenate((id, [tmp_idx]), axis=0)
                frame_file = np.concatenate((frame_file, [tmp_frames]), axis=0)
                coor = np.concatenate((coor, [tmp_coor]), axis=0)
                vis = np.concatenate((vis, [tmp_vis]), axis=0)
        
        return dict(id=id, frame_file=frame_file, coor=coor, vis=vis)

    def _gen_input_from_frame_arr(self):
        """ Generate input sequences from a frame array. """

        # Calculate the image scaler
        h, w, _ = self.frame_arr[0].shape
        h_scaler, w_scaler = h / self.HEIGHT, w / self.WIDTH

        id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
        for i in range(0, len(self.frame_arr), self.sliding_step):
            first_index = i - self.seq_len * self.sliding_step
            if first_index < 0:
                continue
            tmp_idx = []
            # Construct a single input sequence
            for curr_i in range(first_index, i, self.sliding_step):
                tmp_idx.append((0, curr_i))
            if len(tmp_idx) == self.seq_len:
                # Append the input sequence
                id = np.concatenate((id, [tmp_idx]), axis=0)
        return dict(id=id), dict(img_scaler=(w_scaler, h_scaler), img_shape=(w, h))
    
    def _get_heatmap(self, cx, cy):
        """ Generate a Gaussian heatmap centered at (cx, cy). """
        if cx == cy == 0:
            return np.zeros((1, self.HEIGHT, self.WIDTH))
        if self.heatmap_mode == 'hard':
            x, y = np.meshgrid(np.linspace(1, self.WIDTH, self.WIDTH), np.linspace(1, self.HEIGHT, self.HEIGHT))
            heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
            heatmap[heatmap <= self.sigma**2] = 1.
            heatmap[heatmap > self.sigma**2] = 0.
        elif self.heatmap_mode == 'gaussian':
            x = np.arange(self.WIDTH)
            y = np.arange(self.HEIGHT)
            xx, yy = np.meshgrid(x, y)
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            heatmap = np.exp(-0.5 * (dist / self.sigma) ** 2)
            heatmap = heatmap / np.max(heatmap)
        else:
            raise NotImplementedError
        heatmap = heatmap * self.mag
        return heatmap.reshape(1, self.HEIGHT, self.WIDTH)
    
    def _load_heatmap_sample(self, idx):
        """
        Load one heatmap sample of the given index. 
        """
        data_idx = self.data_dict['id'][idx]
        frame_file = self.data_dict['frame_file'][idx]
        coor = self.data_dict['coor'][idx]
        vis = self.data_dict['vis'][idx]
        w, h = self.img_config['img_shape'][data_idx[0][0]]
        w_scaler, h_scaler = self.img_config['img_scaler'][data_idx[0][0]]

        # Read median image
        if self.bg_mode:
            file_format_str = os.path.join('{}', 'frame', '{}','{}.'+self.img_format)
            match_dir, rally_id, _ = parse.parse(file_format_str, frame_file[0])
            median_file = os.path.join(match_dir, 'median.npz') if os.path.exists(os.path.join(match_dir, 'median.npz')) else os.path.join(match_dir, 'frame', rally_id, 'median.npz')
            assert os.path.exists(median_file), f'{median_file} does not exist.'
            median_img = np.load(median_file)['median']

        frames = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
        heatmaps = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
        
        # Read image and generate heatmap
        imgs = np.array(load_images(frame_file))
        if self.bg_mode == 'subtract':
            medians = np.array([median_img] * imgs.shape[0])
            imgs = np.sum(np.absolute(imgs - medians), 3).astype('uint8')
        elif self.bg_mode == 'subtract_concat':
            medians = np.array([median_img] * imgs.shape[0])
            diff_imgs = np.sum(np.absolute(imgs - medians), 3).astype('uint8')
            imgs = np.concatenate((imgs, diff_imgs), axis=3)
        else:
            imgs = imgs
        
        imgs = np.array([cv2.resize(img, (self.WIDTH, self.HEIGHT)) for img in imgs])
        imgs = np.moveaxis(imgs, -1, 1)
            
        heatmap = np.array([self._get_heatmap(int(coor[i][0]/w_scaler), int(coor[i][1]/h_scaler)) for i in range(coor.shape[0])])
        heatmap = heatmap.reshape(-1, self.HEIGHT, self.WIDTH)
        frames = imgs
        
        if self.bg_mode == 'concat':
            median_img = cv2.resize(median_img, (self.WIDTH, self.HEIGHT))
            median_img = np.moveaxis(median_img, -1, 0).reshape(1, -1, self.HEIGHT, self.WIDTH)
            frames = np.concatenate((median_img, frames), axis=0)

        # Normalization
        frames /= 255.
        coor[:, 0] = coor[:, 0] / w
        coor[:, 1] = coor[:, 1] / h
        frames = frames.reshape(-1, self.HEIGHT, self.WIDTH)

        # return data_idx, frames, heatmap, np.array([coor]), np.array([vis])
        return data_idx, frames, heatmap, coor, vis
        
    def __len__(self):
        """ Return the number of data in the dataset. """
        return len(self.data_dict['id'])

    def __getitem__(self, idx):
        """ Return the data of the given index.

            For training and evaluation:
                Return data_idx, frames, heatmaps, tmp_coor, tmp_vis

            For inference:
                Return data_idx, frames
        """
        if self.frame_arr is not None:
            data_idx = self.data_dict['id'][idx] # (L,)
            imgs = self.frame_arr[data_idx[:, 1], ...] # (L, H, W, 3)

            if self.bg_mode:
                median_img = self.median
            
            # Process the frame sequence
            frames = np.array([]).reshape(0, self.HEIGHT, self.WIDTH)
            for i in range(self.seq_len):
                # img = Image.fromarray(imgs[i])
                img = imgs[i]
                if self.bg_mode == 'subtract':
                    # img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8'))
                    # img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                    img = np.sum(np.absolute(img - median_img), 2)
                    img = cv2.resize(img, (self.WIDTH, self.HEIGHT))
                    img = img.reshape(1, self.HEIGHT, self.WIDTH)
                elif self.bg_mode == 'subtract_concat':
                    # diff_img = Image.fromarray(np.sum(np.absolute(img - median_img), 2).astype('uint8'))
                    # diff_img = np.array(diff_img.resize(size=(self.WIDTH, self.HEIGHT)))
                    diff_img = np.sum(np.absolute(img - median_img), 2)
                    diff_img = cv2.resize(diff_img, (self.WIDTH, self.HEIGHT))
                    diff_img = diff_img.reshape(1, self.HEIGHT, self.WIDTH)
                    # img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                    img = cv2.resize(img, (self.WIDTH, self.HEIGHT))
                    img = np.moveaxis(img, -1, 0)
                    img = np.concatenate((img, diff_img), axis=0)
                else:
                    # img = np.array(img.resize(size=(self.WIDTH, self.HEIGHT)))
                    img = cv2.resize(img, (self.WIDTH, self.HEIGHT))
                    img = np.moveaxis(img, -1, 0)
                
                frames = np.concatenate((frames, img), axis=0)
            
            if self.bg_mode == 'concat':
                frames = np.concatenate((median_img, frames), axis=0)
            
            # Normalization
            frames /= 255.

            return data_idx, frames

        elif self.data_mode == 'heatmap':
            if self.frame_alpha > 0:
                # TODO: mixup augmentation
                raise NotImplementedError
            else:
                data_idx, frames, heatmaps, coor, vis = self._load_heatmap_sample(idx)
                return data_idx, frames, heatmaps, coor, vis
        else:
            raise NotImplementedError

