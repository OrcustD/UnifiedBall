nohup python train.py --data_type UniBall --data_dir data/tabletennis --model_name TrackNet --seq_len 4 --epochs 30 --batch_size 10 --bg_mode concat --alpha -1 --save_dir exp_tt_0102 --verbose --last_only > exp_tt_0102.log 2>&1 &