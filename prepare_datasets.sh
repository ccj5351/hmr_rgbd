# ---------------------------
# ----- SET YOUR PATH!! -----
# ---------------------------
# This is where you want all of your tf_records to be saved:
DATA_DIR=./datasets/tf_datasets

# This is the directory that contains README.txt
#LSP_DIR=./datasets/lsp_dataset
LSP_DIR=./datasets/LSP/lsp

# This is the directory that contains README.txt
LSP_EXT_DIR=./datasets/LSP/lspet

# This is the directory that contains 'images' and 'annotations'
MPII_DIR=./datasets/MPII

# This is the directory that contains README.txt
COCO_DIR=./datasets/COCO

# This is the directory that contains README.txt, S1..S8, etc
MPI_INF_3DHP_DIR=./datasets/mpi_inf_3dhp

## Mosh
# This is the path to the directory that contains neutrSMPL_* directories
MOSH_DIR=./datasets/MoSh/neutrMosh

## CAD 60/120
CAD_DIR=./datasets/cad-60-120

## SURREAL
SURREAL_DIR=./datasets/surreal


## Mosh smpl_joints3d_pair
MOSH_H5_DIR=./datasets/MoSh/data/mosh_gen
# ---------------------------


# ---------------------------
# Run each command below from this directory. I advice to run each one independently.
# ---------------------------

# ----- LSP -----
#CUDA_VISIBLE_DEVICES=2 python -m src.datasets.lsp_to_tfrecords --img_directory $LSP_DIR --output_directory $DATA_DIR/lsp
#python -m src.datasets.lsp_to_tfrecords --img_directory $LSP_DIR --output_directory $DATA_DIR/lsp

# ----- LSP-extended -----
#CUDA_VISIBLE_DEVICES=2 python -m src.datasets.lsp_to_tfrecords --img_directory $LSP_EXT_DIR --output_directory $DATA_DIR/lsp_ext
#python -m src.datasets.lsp_to_tfrecords --img_directory $LSP_EXT_DIR --output_directory $DATA_DIR/lsp_ext

# ----- MPII -----
#CUDA_VISIBLE_DEVICES=2 python -m src.datasets.mpii_to_tfrecords --img_directory $MPII_DIR --output_directory $DATA_DIR/mpii
#python -m src.datasets.mpii_to_tfrecords --img_directory $MPII_DIR --output_directory $DATA_DIR/mpii

# ----- COCO -----
#CUDA_VISIBLE_DEVICES=2 python -m src.datasets.coco_to_tfrecords --data_directory $COCO_DIR --output_directory $DATA_DIR/coco
#python -m src.datasets.coco_to_tfrecords --data_directory $COCO_DIR --output_directory $DATA_DIR/coco

# ----- MPI-INF-3DHP -----
#CUDA_VISIBLE_DEVICES=2 python -m src.datasets.mpi_inf_3dhp_to_tfrecords --data_directory $MPI_INF_3DHP_DIR --output_directory $DATA_DIR/mpi_inf_3dhp
#python -m src.datasets.mpi_inf_3dhp_to_tfrecords --data_directory $MPI_INF_3DHP_DIR --output_directory $DATA_DIR/mpi_inf_3dhp

# ----- Mosh data, for each dataset -----
# CMU:
#CUDA_VISIBLE_DEVICES=2 python -m src.datasets.smpl_to_tfrecords --data_directory $MOSH_DIR --output_directory $DATA_DIR/mocap_neutrMosh --dataset_name 'neutrSMPL_CMU'
#python -m src.datasets.smpl_to_tfrecords --data_directory $MOSH_DIR --output_directory $DATA_DIR/mocap_neutrMosh --dataset_name 'neutrSMPL_CMU'

# H3.6M:
#CUDA_VISIBLE_DEVICES=2 python -m src.datasets.smpl_to_tfrecords --data_directory $MOSH_DIR --output_directory $DATA_DIR/mocap_neutrMosh --dataset_name 'neutrSMPL_H3.6'
#python -m src.datasets.smpl_to_tfrecords --data_directory $MOSH_DIR --output_directory $DATA_DIR/mocap_neutrMosh --dataset_name 'neutrSMPL_H3.6'

# jointLim:
#CUDA_VISIBLE_DEVICES=2 python -m src.datasets.smpl_to_tfrecords --data_directory $MOSH_DIR --output_directory $DATA_DIR/mocap_neutrMosh --dataset_name 'neutrSMPL_jointLim'
#python -m src.datasets.smpl_to_tfrecords --data_directory $MOSH_DIR --output_directory $DATA_DIR/mocap_neutrMosh --dataset_name 'neutrSMPL_jointLim'

# ----- CAD60-120 -----
flag=false
#flag=true
if [ "$flag" = true ]; then
	#TASK_TYPE='joints_annotation_from_densepose'
	TASK_TYPE='joints_annotation_from_cad_gt'
	SAVE_TO_H5FILES=True
	#python -m src.datasets.cad60_120_to_tfrecords --img_directory $CAD_DIR --output_directory $DATA_DIR/cad_60_120 --task_type_cad $TASK_TYPE
	CUDA_VISIBLE_DEVICES=$1 python -m src.datasets.cad60_120_to_tfrecords \
		--img_directory $CAD_DIR --output_directory $DATA_DIR/cad_60_120_h5 \
		--task_type_cad $TASK_TYPE \
		--is_save_to_h5files=${SAVE_TO_H5FILES}
fi

# ----- SURREAL  -----

#flag=false
flag=true
if [ "$flag" = true ]; then
	#python -m src.datasets.surreal_to_tfrecords --img_directory $SURREAL_DIR --output_directory $DATA_DIR/surreal_debug
	#TASK_TYPE='add_smpl_joints3d_pair'
	TASK_TYPE='add_to_tfrecord'
	SAVE_TO_H5FILES=True
	#CUDA_VISIBLE_DEVICES=$1 python -m src.datasets.surreal_to_tfrecords --img_directory $SURREAL_DIR --output_directory $DATA_DIR/surreal_smpl_joints3d_pair_small_valid_5 --task_type=$TASK_TYPE
	CUDA_VISIBLE_DEVICES=$1 python -m src.datasets.surreal_to_tfrecords \
		--img_directory $SURREAL_DIR --output_directory $DATA_DIR/surreal_h5 --task_type=$TASK_TYPE \
		--is_save_to_h5files=${SAVE_TO_H5FILES}
fi

# ---- Mosh h5 annotation ----
#python -m src.datasets.mosh_to_tfrecords --pathToMoSh $MOSH_H5_DIR
