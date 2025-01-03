import os
import cv2
import numpy as np
import parse
from utils.general import list_dirs
from utils.data_io import load_images_in_parallel
from tqdm import tqdm
import json

def vis_median_frame(median):
    cv2.imwrite('median.png', median[..., ::-1])  # Convert RGB to BGR for saving


def get_match_median(match_dir, vis_dir):
    """ Generate and save the match median frame to the corresponding match directory.

        Args:
            match_dir (str): File path of match directory
                Format: '{data_dir}/{split}/match{match_id}'
            
        Returns:
            None
    """

    if os.path.exists(os.path.join(match_dir, 'median.npz')):
        return
    medians = []

    # For each rally in the match
    rally_dirs = list_dirs(os.path.join(match_dir, 'frame'))
    for rally_dir in tqdm(rally_dirs):
        file_format_str = os.path.join('{}', 'frame', '{}')
        match_id, rally_id = parse.parse(file_format_str, rally_dir)
        match_id = os.path.basename(match_id)

        # Load rally median, if not exist, generate it
        if not os.path.exists(os.path.join(rally_dir, 'median.npz')):
            get_rally_median(rally_dir)

        frame = np.load(os.path.join(rally_dir, 'median.npz'))['median']
        medians.append(frame)
    
    # Calculate the median of all rally medians
    median = np.median(np.array(medians), 0)
    np.savez(os.path.join(match_dir, 'median.npz'), median=median) # Must be lossless, do not save as image format
    cv2.imwrite(f'{vis_dir}/{match_id}_median.png', median[..., ::-1])  # Convert RGB to BGR for saving

def get_rally_median(image_dir):
    """ Generate and save the rally median frame to the corresponding rally directory.

        Args:
            video_file (str): File path of video file
                Format: '{data_dir}/{split}/match{match_id}/video/{rally_id}.mp4'
        
        Returns:
            None
    """
    
    frames = []

    # Get all .jpg file paths under image_dir
    
    frames = load_images_in_parallel(image_dir, format='RGB')
    
    # Calculate the median of all frames
    median = np.median(np.array(frames), 0)
    np.savez(os.path.join(image_dir, 'median.npz'), median=median) # Must be lossless, do not save as image format

def main(data_dir, vis_dir):
    """ Generate and save the median frame for each rally and match.

        Args:
            data_dir (str): File path of data directory
                Format: '{data_dir}/{split}'
        
        Returns:
            None
    """

    # For each match
    match_dirs = list_dirs(data_dir)
    for match_dir in tqdm(match_dirs, desc='Generating median frames'):
        get_match_median(match_dir, vis_dir)

if __name__ == "__main__":
    data_dir = 'data/tabletennis/all'
    vis_dir = 'data/tabletennis/median'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    main(data_dir, vis_dir)