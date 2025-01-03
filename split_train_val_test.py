import os
import random
import glob
import json
from utils.general import list_dirs

def split_train_val_test(data_dir, val_ratio=0.1, test_ratio=0.05):
    # Get all match directories
    match_dirs = list_dirs(data_dir)
    
    # Initialize lists for train, val, and test sets
    train_set = []
    val_set = []
    test_set = []

    # Split matches into trainval and test sets
    matches = sorted([os.path.basename(match_dir) for match_dir in match_dirs])
    # random.shuffle(matches)
    num_test_matches = int(len(matches) * test_ratio)
    test_matches = matches[:num_test_matches]
    trainval_matches = matches[num_test_matches:]

    # Process test matches
    for match_name in test_matches:
        rally_files = sorted(glob.glob(os.path.join(data_dir, match_name, 'csv', '*.csv')))
        for rally_file in rally_files:
            rally_id = os.path.basename(rally_file).split('_')[0]
            test_set.append((match_name, rally_id))

    # Process trainval matches
    for match_name in trainval_matches:
        rally_files = sorted(glob.glob(os.path.join(data_dir, match_name, 'csv', '*.csv')))
        # random.shuffle(rally_files)
        num_val_rallies = int(len(rally_files) * val_ratio)
        val_rallies = rally_files[:num_val_rallies]
        train_rallies = rally_files[num_val_rallies:]

        for rally_file in val_rallies:
            rally_id = os.path.basename(rally_file).split('_')[0]
            val_set.append((match_name, rally_id))

        for rally_file in train_rallies:
            rally_id = os.path.basename(rally_file).split('_')[0]
            train_set.append((match_name, rally_id))

    return train_set, val_set, test_set

if __name__ == "__main__":
    # Example usage
    data_dir = 'data/tabletennis/all'
    save_dir = 'data/tabletennis/info'
    os.makedirs(save_dir, exist_ok=True)
    train_set, val_set, test_set = split_train_val_test(data_dir, val_ratio=0.1, test_ratio=0.05)
    print(f"Train set: {len(train_set)}\nVal set: {len(val_set)}\nTest set: {len(test_set)}")
    # Save the train, val, and test sets as JSON files
    with open(os.path.join(save_dir, 'train.json'), 'w') as f:
        json.dump(train_set, f, indent=4)

    with open(os.path.join(save_dir, 'val.json'), 'w') as f:
        json.dump(val_set, f, indent=4)

    with open(os.path.join(save_dir, 'test.json'), 'w') as f:
        json.dump(test_set, f, indent=4)