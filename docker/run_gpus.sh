#!/bin/bash
#export DISPLAY=127.0.0.1:0.0

#nvidia-docker run --rm -v /mnt/interns/changjiang/DensePose_SK/:/home -it changjiang/hmr bash
#nvidia-docker run --rm -v /mnt/interns/changjiang/DensePose_SK/:/home -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix changjiang/hmr bash
echo "Using GPU-ID: $1"
#exit
NV_GPU=$1 nvidia-docker run --rm -e="DISPLAY" -v="$HOME/.Xauthority:/root/.Xauthority:rw" --net=host \
	-v /mnt/interns/changjiang/DensePose_SK/hmr/:/home/hmr/ \
	-v /mnt/interns/changjiang/PatientPositioning/Datasets/:/home/hmr/datasets \
	-v /home/UIIUSA/changjiang/.bashrc:/root/.bashrc \
	-v /home/UIIUSA/changjiang/lib/bin/:/root/lib/bin \
	-it changjiang/hmr bash

#docker run --runtime=nvidia --rm -v /mnt/UIIUSA/srikrishna/Projects/PatientPositioning/Demo/DensePose:/home/BaseCode -it densepose bash
