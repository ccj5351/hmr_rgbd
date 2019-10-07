depth_fname="./data/surreal-small/01_01_100_samples/01_01/01_01_c0001_depth.mat"
info_fname="./data/surreal-small/01_01_100_samples/01_01/01_01_c0001_info.mat"
result_dir="./results/surreal_debug"
t_beg=0
t_end=1

python -m src.load_data_4_inference --depth_fname=$depth_fname --info_fname=$info_fname \
	--result_dir=$result_dir --t_beg=$t_beg --t_end=$t_end

