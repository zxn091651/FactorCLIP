python multi.py \
        --source_domains "amazon,visda,stl10" \
        --target_domain "clipart" \
        --shots 1 \
        --config configs/multi.yaml \
        --output_dir "./experiments_mini/Multi" \
        --data_root "./datasets"
wait

python multi.py \
        --source_domains "amazon,visda,stl10" \
        --target_domain "clipart" \
        --shots 5 \
        --config configs/multi.yaml \
        --output_dir "./experiments_mini/Multi" \
        --data_root "./datasets"
wait

python multi.py \
        --source_domains "amazon,visda,stl10" \
        --target_domain "painting" \
        --shots 1 \
        --config configs/multi.yaml \
        --output_dir "./experiments_mini/Multi" \
        --data_root "./datasets"
wait

python multi.py \
        --source_domains "amazon,visda,stl10" \
        --target_domain "painting" \
        --shots 5 \
        --config configs/multi.yaml \
        --output_dir "./experiments_mini/Multi" \
        --data_root "./datasets"
wait

python multi.py \
        --source_domains "amazon,visda,stl10" \
        --target_domain "real" \
        --shots 1 \
        --config configs/multi.yaml \
        --output_dir "./experiments_mini/Multi" \
        --data_root "./datasets"
wait

python multi.py \
        --source_domains "amazon,visda,stl10" \
        --target_domain "real" \
        --shots 5 \
        --config configs/multi.yaml \
        --output_dir "./experiments_mini/Multi" \
        --data_root "./datasets"
wait

python multi.py \
        --source_domains "amazon,visda,stl10" \
        --target_domain "sketch" \
        --shots 1 \
        --config configs/multi.yaml \
        --output_dir "./experiments_mini/Multi" \
        --data_root "./datasets"
wait

python multi.py \
        --source_domains "amazon,visda,stl10" \
        --target_domain "sketch" \
        --shots 5 \
        --config configs/multi.yaml \
        --output_dir "./experiments_mini/Multi" \
        --data_root "./datasets"
wait
