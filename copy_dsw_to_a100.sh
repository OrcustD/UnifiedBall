#!/bin/bash
rclone='/cpfs01/shared/sport/wuhao/tools/rclone/rclone'
$rclone copy --progress --ignore-existing $1 a100:pjlab-lingjun-gvlab-a100/donglinfeng/$1 