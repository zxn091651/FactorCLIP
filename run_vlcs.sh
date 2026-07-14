python vlcs.py \
        --source_domains "LABELME,PASCAL,SUN" \
        --target_domain "CALTECH" \
        --shots 1 \
        --config configs/vlcs.yaml \
        --output_dir "./experiments_mini/VLCS" \
        --data_root "./datasets/VLCS"
wait

python vlcs.py \
        --source_domains "LABELME,PASCAL,SUN" \
        --target_domain "CALTECH" \
        --shots 5 \
        --config configs/vlcs.yaml \
        --output_dir "./experiments_mini/VLCS" \
        --data_root "./datasets/VLCS"
wait

python vlcs.py \
        --source_domains "CALTECH,PASCAL,SUN" \
        --target_domain "LABELME" \
        --shots 1 \
        --config configs/vlcs.yaml \
        --output_dir "./experiments_mini/VLCS" \
        --data_root "./datasets/VLCS"
wait

python vlcs.py \
        --source_domains "CALTECH,PASCAL,SUN" \
        --target_domain "LABELME" \
        --shots 5 \
        --config configs/vlcs.yaml \
        --output_dir "./experiments_mini/VLCS" \
        --data_root "./datasets/VLCS"
wait

python vlcs.py \
        --source_domains "CALTECH,LABELME,SUN" \
        --target_domain "PASCAL" \
        --shots 1 \
        --config configs/vlcs.yaml \
        --output_dir "./experiments_mini/VLCS" \
        --data_root "./datasets/VLCS"
wait

python vlcs.py \
        --source_domains "CALTECH,LABELME,SUN" \
        --target_domain "PASCAL" \
        --shots 5 \
        --config configs/vlcs.yaml \
        --output_dir "./experiments_mini/VLCS" \
        --data_root "./datasets/VLCS"
wait

python vlcs.py \
        --source_domains "CALTECH,LABELME,PASCAL" \
        --target_domain "SUN" \
        --shots 1 \
        --config configs/vlcs.yaml \
        --output_dir "./experiments_mini/VLCS" \
        --data_root "./datasets/VLCS"
wait

python vlcs.py \
        --source_domains "CALTECH,LABELME,PASCAL" \
        --target_domain "SUN" \
        --shots 5 \
        --config configs/vlcs.yaml \
        --output_dir "./experiments_mini/VLCS" \
        --data_root "./datasets/VLCS"
wait
