# TODO: Replace with where you downloaded your resnet_v2_50.
#PRETRAINED=/scratch1/projects/tf_datasets/models/resnet_v2_50/resnet_v2_50.ckpt
PRETRAINED=models/resnet_v2_50.ckpt
# TODO: Replace with where you generated tf_record!
#DATA_DIR=/scratch1/storage/hmr_release_files/test_tf_datasets/
DATA_DIR=datasets/tf_datasets/


#####################
#flag=false
flag=true
DATASETS="surreal_smpl_joints3d_pair"
#DATASETS="surreal_smpl_joints3d_pair_small_100"
#DATASETS="surreal_cam"
#DATASETS="surreal_small"
#DATASETS="mpi_inf_3dhp"
#PRETRAINED='./logs/surreal_smpl_train/joints3d_to_smpl_regressor_Jul24_1250/model.ckpt-172000'

#BATCH_SIZE=100
#BATCH_SIZE=80
BATCH_SIZE=1024
#BATCH_SIZE=16
#LOGS='logs/debug-poseflip-onfly'
LOGS='logs/surreal_smpl_train'
#LOGS='logs/tmp'

#EPO_NUM=5000
EPO_NUM=50
#IS_TRAINING=false
#IS_TRAINING=true
TASK_TYPE='test'
#TOY_DATA_TYPE='cad-120-small-1'
TOY_DATA_TYPE='cad-any-1-sample'
#TOY_DATA_TYPE='mosh-hd5-small-1-x'

#PRETRAINED='./logs/surreal_smpl_train/joints3d_to_smpl_regressor_Jul24_1250/'
#PRETRAINED='logs/surreal_smpl_train/joints3d_to_smpl_regressor_Jul24_1752/'
PRETRAINED=''
#PRETRAINED='logs/surreal_smpl_train/joints3d_cycle_loss_weight_1.0_Jul25_1633'
#PRETRAINED='logs/surreal_smpl_train/joints3d_to_smpl_regressor_Jul24_1807/'
#PRETRAINED='logs/surreal_smpl_train/joints3d_cycle_loss_weight_1.0_RsLoss_Jul26_0237/'
RESULT_DIR='./results/smpl_regressor/'
#WEIGHT_JOINTS_3D=100.0
WEIGHT_JOINTS_3D=1.0
METHOD_UNIQUE_NAME="joints3d_cycle_loss_weight_${WEIGHT_JOINTS_3D}_RsLoss"
JOINT_TYPE='lsp'

if [ "$flag" = true ]; then
	CMD="python -m src.pose_perceptron --lr 1e-4 \
		--pretrained_model_path=${PRETRAINED} \
		--data_dir ${DATA_DIR} \
		--batch_size ${BATCH_SIZE} \
		--datasets ${DATASETS} \
		--epoch ${EPO_NUM} \
		--task_type=${TASK_TYPE} \
		--result_dir=${RESULT_DIR} \
		--weight_joints3d=${WEIGHT_JOINTS_3D} \
		--method_name=${METHOD_UNIQUE_NAME} \
		--joint_type=${JOINT_TYPE} \
		--toy_data_type=${TOY_DATA_TYPE} \
		--log_dir ${LOGS}"
	echo $CMD
  $CMD
fi
