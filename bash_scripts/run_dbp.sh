CUDA_VISIBLE_DEVICES=$0 python3 model/run_dbp.py \
	--file_dir datasets/dbp/$3 \
	#--rate 0.3 \
	#--lr .0005 \
	#--epochs 3000 \
	#--hidden_units "300,300,150" \
	#--check_point 25  \
	--bsize 3000 \
	--il \
	--il_start 500 \
	--semi_learn_step 5 \
	--csls \
	--csls_k 3 \
	--seed 0 
