cate="03_01"
ctype="c0008"

cate="27_08"
ctype="c0010"

cate="36_08"
ctype="c0010"

cate="36_14"
ctype="c0011"

cate="36_20"
ctype="c0011"

fname="surreal-small/val_run0/${cate}/${cate}_${ctype}"
#fname="surreal-small/01_01_100_samples/01_01/01_01_${ctype}"
depth_fname="./data/${fname}_depth.mat"
info_fname="./data/${fname}_info.mat"

#pretrained_model="./logs/debug3dloss-flip/HMR_3DSUP_surreal_small_resnet_fc3_dropout_Elr1e-05_kp-weight60_Dlr1e-04_Jul02_1620/model.ckpt-5150"
#pretrained_model="./logs/logs_saved/HMR_3DSUP_surreal_small_resnet_fc3_dropout_Elr1e-05_kp-weight60_Dlr1e-04_Jul10_0243/model.ckpt-1476"
#pretrained_model="./logs/logs_saved/HMR_3DSUP_surreal_small_resnet_fc3_dropout_Elr1e-05_kp-weight60_Jul20_1718/model.ckpt-23350"
pretrained_model="./logs/surreal_train_data_2dLoss_smplLoss/HMR_3DSUP_surreal_resnet_fc3_dropout_Elr1e-05_kp-weight60_Dlr1e-04_Jul21_0408/model.ckpt-919500"
#pretrained_model="./models/model.ckpt-667589"
result_dir="./results/surreal_train"
t_beg=0
t_end=1
flag=false
#flag=true
if [ "$flag" = true ]; then
	CUDA_VISIBLE_DEVICES=$1 python demo_depth.py --depth_fname=$depth_fname --info_fname=$info_fname \
		--result_dir=$result_dir --t_beg=$t_beg --t_end=$t_end \
		--load_path=$pretrained_model
fi

#exit
# cad60
depth_fname="./data/cad-60-small/arranging_objects-0510175411-RGB_1_dep_extracted.pfm"
image_fname="./data/cad-60-small/arranging_objects-0510175411-RGB_1_img_extracted.pfm"


pretrained_model="./logs/surreal_cad/cad60-sur27k/HMR_3DSUP_cad_60_120-surreal_resnet_fc3_dropout_Elr1e-05_kp-weight60_Dlr1e-04_Aug08_1329/model.ckpt-166299"

flag=false
#flag=true

#cad120
declare -a depth_fnames=(
                         "./data/cad-60-small/Person1-0512164529-RGB_1_dep_extracted.pfm"
                         "./data/cad-60-small/Person1-0512164529-RGB_2_dep_extracted.pfm"
                         "./data/cad-60-small/Person1-0512164529-RGB_26_dep_extracted.pfm"
                         "./data/cad-60-small/Person1-0512164529-RGB_1001_dep_extracted.pfm"
                         "./data/cad-60-small/Person1-0512170134-RGB_1069_dep_extracted.pfm"
                         "./data/cad-60-small/Person1-0512173312-RGB_339_dep_extracted.pfm"

                         "./data/cad-120-small/Subject4_rgbd_images/taking_medicine-1130145737-RGB_114_dep_extracted.pfm" 
                         "./data/cad-120-small/Subject4_rgbd_images/making_cereal-1130144242-RGB_763_dep_extracted.pfm"
                         "./data/cad-120-small/Subject4_rgbd_images/cleaning_objects-0510172333-RGB_138_dep_extracted.pfm"
                         "./data/cad-120-small/Subject4_rgbd_images/arranging_objects-0510173051-RGB_26_dep_extracted.pfm"
                         "./data/cad-120-small/Subject4_rgbd_images/arranging_objects-0510173051-RGB_2_dep_extracted.pfm"
                         "./data/cad-120-small/Subject4_rgbd_images/arranging_objects-0510173051-RGB_1_dep_extracted.pfm"
                        )

declare -a genders=( 
                   'm' 'm' 'm' 'm' 'm' 'm' 
                   'm' 'm' 'm' 'm' 'm' 'm' 
                   )

declare -a image_fnames=(
                         "./data/cad-60-small/Person1-0512164529-RGB_1_img_extracted.pfm"
                         "./data/cad-60-small/Person1-0512164529-RGB_2_img_extracted.pfm"
                         "./data/cad-60-small/Person1-0512164529-RGB_26_img_extracted.pfm"
                         "./data/cad-60-small/Person1-0512164529-RGB_1001_img_extracted.pfm"
                         "./data/cad-60-small/Person1-0512170134-RGB_1069_img_extracted.pfm"
                         "./data/cad-60-small/Person1-0512173312-RGB_339_img_extracted.pfm"


                         "./data/cad-120-small/Subject4_rgbd_images/taking_medicine-1130145737-RGB_114_img_extracted.pfm" 
                         "./data/cad-120-small/Subject4_rgbd_images/making_cereal-1130144242-RGB_763_img_extracted.pfm"
                         "./data/cad-120-small/Subject4_rgbd_images/cleaning_objects-0510172333-RGB_138_img_extracted.pfm"
                         "./data/cad-120-small/Subject4_rgbd_images/arranging_objects-0510173051-RGB_26_img_extracted.pfm"
                         "./data/cad-120-small/Subject4_rgbd_images/arranging_objects-0510173051-RGB_2_img_extracted.pfm"
                         "./data/cad-120-small/Subject4_rgbd_images/arranging_objects-0510173051-RGB_1_img_extracted.pfm"
                         )

for idx in $(seq 0 11)
do
	if (( "$idx" > 5 )); then
		DATA_TYPE='cad120'
  else
		DATA_TYPE='cad60'
	fi
	#echo $DATA_TYPE
	#echo $idx
	#echo ${genders[idx]}
  result_dir="./results/surreal_cad_train/${DATA_TYPE}/"
	if [ "$flag" = true ]; then
		CUDA_VISIBLE_DEVICES=$1 python demo_depth.py --depth_fname=${depth_fnames[idx]} --image_fname=${image_fnames[idx]} \
			--gender=${genders[idx]} --data_type=${DATA_TYPE} \
			--result_dir=$result_dir --load_path=$pretrained_model
	fi
done

#--------------------------
# evaluation many images
#--------------------------

#flag=false
flag=true

#cad120
declare -a h5_fnames=(
                        "./results/eval-mpjpepa-thred-90mm-5-samples.h5"
												"./datasets/tf_datasets/cad_60_120_h5/cad-120/Subject4_rgbd_images/eval-mpjpepa-thred-90mm-3000-samples.h5"
												"./datasets/tf_datasets/surreal_h5/cmu/val/run0/eval-mpjpepa-1000-samples.h5"
												"./datasets/tf_datasets/surreal_h5/cmu/val/run1/eval-mpjpepa-1000-samples.h5"
                        )

declare -a data_types=(
                        "cad120-sub4-eval"
                        "cad120-sub4-eval"
												"surreal-val-run0-eval"
												"surreal-val-run1-eval"
                        )
#for idx in $(seq 0 1)
for idx in 3
do
	MODEL_TYPE='hmr'
	pretrained_model="./models/model.ckpt-667589"
	#MODEL_TYPE='ours'
	#pretrained_model="./logs/surreal_cad/cad60-sur27k/HMR_3DSUP_cad_60_120-surreal_resnet_fc3_dropout_Elr1e-05_kp-weight60_Dlr1e-04_Aug08_1329/model.ckpt-166299"
	DATA_TYPE="${data_types[idx]}-${MODEL_TYPE}"
	echo $DATA_TYPE
  result_dir="./results/surreal_cad_train/${DATA_TYPE}/"
	if [ "$flag" = true ]; then
		CUDA_VISIBLE_DEVICES=$1 python demo_depth.py \
			--eval_model_type=${MODEL_TYPE} \
			--h5_filename=${h5_fnames[idx]} \
			--data_type=${DATA_TYPE} \
			--result_dir=$result_dir --load_path=$pretrained_model
	fi
done
