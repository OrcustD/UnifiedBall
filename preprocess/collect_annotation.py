import os
import json
from tqdm import tqdm
import pandas as pd


def get_valid_annotations(annot_path):
    annot = json.load(open(annot_path, 'r'))
    frame_id = int(os.path.basename(annot_path).split('.')[0])
    result = annot["step_1"]["result"]
    if len(result) == 1:
        # Frame, Visibility, X, Y
        return [frame_id, 1, round(result[0]["x"]), round(result[0]["y"])]
    elif len(result) == 0:
        return [frame_id, 0, 0, 0]
    else:
        return None

def collect_csv(round_dir, output_dir, match_to_idxs):
    annot_files = os.listdir(round_dir)
    video_name = os.path.basename(round_dir)
    match_name = '_'.join(video_name.split('_')[:-1])
    round_name = video_name.split('_')[-1]
    round_results = []
    for annot_file in annot_files:
        frame_annot = get_valid_annotations(os.path.join(round_dir, annot_file))
        if frame_annot is not None:
            round_results.append(frame_annot)
    if len(round_results) > 0:
        output_dir = os.path.join(output_dir, match_to_idxs[match_name], 'csv')
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(round_results, columns=['Frame', 'Visibility', 'X', 'Y']).sort_values(by='Frame').to_csv(os.path.join(output_dir, f'{round_name}_ball.csv'), index=False)

def collect_csv_all(annot_dir, output_dir, match_to_idxs):
    round_dirs = os.listdir(annot_dir)
    for round_dir in tqdm(round_dirs):
        collect_csv(os.path.join(annot_dir, round_dir), output_dir, match_to_idxs)

if __name__ == '__main__':
    sport = 'tabletennis'
    annot_dir = f'data/annotations_raw/{sport}'
    output_dir = f'data/{sport}/all'
    match_to_idxs = json.load(open(f'data/{sport}/match_to_idxs.json', 'r'))
    video_names = os.listdir(annot_dir)
    collect_csv_all(annot_dir, output_dir, match_to_idxs)
    

