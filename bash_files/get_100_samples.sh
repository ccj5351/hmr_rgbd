cd /mnt/interns/changjiang/PatientPositioning/Datasets/surreal/cmu-small/samples/01_01_100_samples/01_01/
pwd
for idx in 1 2 3 4 5 6 7 8 10 11 12
do
	tmpdir="./01_01_c$(printf "%04d" "$idx")"
	for i in  $(seq 0 100)
	do
		src="${tmpdir}/frame_$(printf "%03d" "$i").jpg"
		if test -f "$src"; then
			echo "$src exist"
			dst="./100-samples/c$(printf "%04d" "$idx")_frame_$(printf "%03d" "$i").jpg"
		  echo $dst
			cp $src $dst
		fi
	done
done
