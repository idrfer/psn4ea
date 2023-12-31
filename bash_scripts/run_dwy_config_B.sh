CUDA_VISIBLE_DEVICES=$0 python3 src/run_dwy.py \
	--file_dir datasets/dwy/ \
	--rate 0.5 \
	--lr .0005 \
	--epochs 3500 \
	--hidden_units "500,500,300" \
	--csls \
	--csls_k 3 \
	--check_point 50  \
	--bsize 6500  \
	--semi_learn_step 5 \
	--il \
	--il_start 500 \
	--seed 0
