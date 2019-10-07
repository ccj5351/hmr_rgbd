#!/bin/bash
#export DISPLAY=127.0.0.1:0.0

#nvidia-docker run --rm -v /mnt/interns/changjiang/DensePose_SK/:/home -it changjiang/hmr bash
#nvidia-docker run --rm -v /mnt/interns/changjiang/DensePose_SK/:/home -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix changjiang/hmr bash
#NV_GPU=2 nvidia-docker run --cpus=4 --rm -e="DISPLAY" -v="$HOME/.Xauthority:/root/.Xauthority:rw" --net=host \
NV_GPU=4,5,6,7 nvidia-docker run --rm -e="DISPLAY" -v="$HOME/.Xauthority:/root/.Xauthority:rw" --net=host \
	-v /mnt/interns/changjiang/DensePose_SK/hmr/:/home/hmr/ \
	-v /mnt/interns/changjiang/git/:/home/git/ \
	-v /mnt/interns/changjiang/PatientPositioning/Datasets/:/home/hmr/datasets \
	-v /home/UIIUSA/changjiang/.bashrc:/root/.bashrc \
	-v /home/UIIUSA/changjiang/lib/bin/:/root/lib/bin \
	-it changjiang/hmr bash

# 
# -v /etc/localtime:/etc/localtime:ro \

#docker run --runtime=nvidia --rm -v /mnt/UIIUSA/srikrishna/Projects/PatientPositioning/Demo/DensePose:/home/BaseCode -it densepose bash

#NV_GPU=2 nvidia-docker run --rm -e="DISPLAY" -v="$HOME/.Xauthority:/root/.Xauthority:rw" --net=host \
#	-v /mnt/interns/changjiang/DensePose_SK/hmr/:/home/hmr/ \
#	-v /mnt/interns/changjiang/PatientPositioning/Datasets/:/home/hmr/datasets \
#	-v /home/UIIUSA/changjiang/.bashrc:/root/.bashrc \
#	-v /home/UIIUSA/changjiang/lib/bin/:/root/lib/bin \
#	-p 0.0.0.0:6009:6009 \
#	-it changjiang/hmr bash
