# TODO: Replace with where you downloaded your resnet_v2_50.
#PRETRAINED=/scratch1/projects/tf_datasets/models/resnet_v2_50/resnet_v2_50.ckpt
PRETRAINED=models/resnet_v2_50.ckpt
# TODO: Replace with where you generated tf_record!
#DATA_DIR=/scratch1/storage/hmr_release_files/test_tf_datasets/
DATA_DIR=datasets/tf_datasets/

#HAS_DEPTH=True
#HAS_DEPTH=False
#CMD="python -m src.main --d_lr 1e-4 --e_lr 1e-5 --log_img_step 1000 --pretrained_model_path=${PRETRAINED} --data_dir ${DATA_DIR} --e_loss_weight 60. --batch_size=64 --use_3d_label True --e_3d_weight 60. --datasets lsp,lsp_ext,mpii,h36m,coco,mpi_inf_3dhp --epoch 75 --log_dir logs"
#CMD="python -m src.main --d_lr 1e-4 --e_lr 1e-5 --log_img_step 1000 --pretrained_model_path=${PRETRAINED} --data_dir ${DATA_DIR} --e_loss_weight 60. --batch_size=64 --use_3d_label True --e_3d_weight 60. --datasets coco --epoch 10 --log_dir logs"

#####################
#flag=false
flag=true
DATASETS="cad_60_120,surreal"
DATASETS="cad_small,surreal_small"
DATASETS="cad_60_120,surreal_small"
DATASETS="surreal_27k"
DATASETS="cad_60_120,surreal_27k"
DATASETS="cad_small,surreal_small"
DATASETS="cad_60_120,surreal"

#-----------------------------
# datasets used here --------
#-----------------------------
DATASETS="cad_small,surreal_small"

#DATASETS="cad_60_120"
#DATASETS="surreal"
#DATASETS="surreal_cam"
#DATASETS="surreal_small"
#DATASETS="mpi_inf_3dhp"
USE_3D_LABEL=true
#USE_3D_LABEL=false
HAS_DEPTH=true
#HAS_DEPTH=false
PRETRAINED=models/resnet_v2_50.ckpt
#PRETRAINED=''
#BATCH_SIZE=64
#BATCH_SIZE=80
BATCH_SIZE=64
#LOGS='logs/debug-poseflip-onfly'
LOGS='logs/surreal_train_data_2dLoss_smplLoss'
LOGS='logs/surreal_cad/cad60P2-sur100'
LOGS='logs/surreal_cad/cad60-sur27k'


LOGS='logs/surreal_cad/depth/tmp'
LP=''
#LP='logs/surreal_train_data_2dLoss_smplLoss/HMR_3DSUP_surreal_resnet_fc3_dropout_Elr1e-05_kp-weight60_Dlr1e-04_Jul21_0408'
#LP='logs/surreal_cad/small-10/HMR_3DSUP_cad_small-surreal_small_resnet_fc3_dropout_Elr1e-05_kp-weight60_Dlr1e-04_Aug07_0258'

#LP='./logs/surreal_cad/cad60-sur27k/HMR_3DSUP_cad_60_120-surreal_resnet_fc3_dropout_Elr1e-05_kp-weight60_Dlr1e-04_Aug08_1329'
#LP='logs/surreal_cad/depth/tmp/HMR_3DSUP_cad_small-surreal_small_resnet_fc3_dropout_Elr1e-05_kp-weight60_Dlr1e-04_Aug14_1525'


#-------------------------------------
# formal training on large dataset
#-------------------------------------
LOG_IMG_STEP=800
EPO_NUM=5


#-------------------------------------
# small training on small dataset
#-------------------------------------
EPO_NUM=100000
LOG_IMG_STEP=100

#ENCODER_ONLY=true
ENCODER_ONLY=false
JOINT_TYPE='lsp'

SPLIT='train'
HAS_DEPTH_LOSS=false
HAS_DEPTH_LOSS=true
#SPLIT=''

if [ "$flag" = true ]; then
	CMD="python -m src.main --d_lr 1e-4 --e_lr 1e-5 \
		--log_img_step ${LOG_IMG_STEP} \
		--pretrained_model_path=${PRETRAINED} \
		--load_path=${LP}
		--data_dir ${DATA_DIR} \
		--e_loss_weight 60. \
		--batch_size ${BATCH_SIZE} \
		--e_3d_weight_js3d 0. \
		--e_3d_weight_smpl 60. \
		--e_weight_depth 0.001 \
		--has_depth_loss=${HAS_DEPTH_LOSS} \
		--datasets ${DATASETS} \
		--joint_type=${JOINT_TYPE} \
		--has_depth=${HAS_DEPTH} \
		--use_3d_label=${USE_3D_LABEL} \
		--epoch ${EPO_NUM} \
		--log_dir ${LOGS} \
		--split=${SPLIT} \
		--encoder_only=${ENCODER_ONLY}"
	echo $CMD
  CUDA_VISIBLE_DEVICES=$1 $CMD
fi
# To pick up training/training from a previous model, set LP
# LP='logs/<WITH_YOUR_TRAINED_MODEL>'
# CMD="python -m src.main --d_lr 1e-4 --e_lr 1e-5 --log_img_step 1000 --load_path=${LP} --e_loss_weight 60. --batch_size=64 --use_3d_label True --e_3d_weight 60. --datasets lsp lsp_ext mpii h36m coco mpi_inf_3dhp --epoch 75"


# To pick up training/training from a previous model, set LP
# LP='logs/<WITH_YOUR_TRAINED_MODEL>'
# CMD="python -m src.main --d_lr 1e-4 --e_lr 1e-5 --log_img_step 1000 --load_path=${LP} --e_loss_weight 60. --batch_size=64 --use_3d_label True --e_3d_weight 60. --datasets lsp lsp_ext mpii h36m coco mpi_inf_3dhp --epoch 75"
