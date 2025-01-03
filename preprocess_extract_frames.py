import os
import json
from tqdm import tqdm
import subprocess

def extract_frames_from_video(video_path, output_dir):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ffmpeg_path = "ffmpeg"
    # 使用 ffmpeg 提取帧
    command = [
        ffmpeg_path, 
        "-i", 
        video_path,
        "-start_number", "0",
        os.path.join(output_dir, "%04d.jpg")
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def extract_frames_from_all_videos(src, dst, match_to_idxs):
    # 遍历 src 目录下的所有文件
    video_files = []
    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith(('.mp4')):  # 根据需要添加更多视频格式
                video_files.append(os.path.join(root, file))
    # 使用 tqdm 显示进度条
    for video_path in tqdm(video_files, desc="Extracting frames"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        match_name = '_'.join(video_name.split('_')[:-1])
        round_name = video_name.split('_')[-1]
        output_dir = os.path.join(dst, match_to_idxs[match_name], 'frame', round_name)
        extract_frames_from_video(video_path, output_dir)

def build_match_rounds(video_names):
    matches = {}
    match_idxs = 1
    match_to_idxs = {}
    for video_name in video_names:
        match_name = '_'.join(video_name.split('_')[:-1])
        if match_name not in matches:
            matches[match_name] = []
        if match_name not in match_to_idxs:
            match_to_idxs[match_name] = f'match{match_idxs}'
            match_idxs += 1
        matches[match_name].append(video_name)
    # matches = dict(sorted(matches.items(), key=lambda item: len(item[1]), reverse=False))
    return matches, match_to_idxs

if __name__ == '__main__':
    sport = 'tabletennis'
    annot_dir = f'data/annotations_raw/{sport}'
    # video_dir = f'/nas/shared/sport/donglinfeng/unified_ball_clips/{sport}'
    video_dir = f'data/video_raw/{sport}'
    output_dir = f'data/{sport}/all'
    os.makedirs(output_dir, exist_ok=True)

    video_names = os.listdir(annot_dir)
    matches, match_to_idxs = build_match_rounds(video_names)
    json.dump(match_to_idxs, open(f'data/{sport}/match_to_idxs.json', 'w'), indent=4)
    extract_frames_from_all_videos(video_dir, output_dir, match_to_idxs)



