python pacs.py \
        --source_domains "art_painting,cartoon,photo" \
        --target_domain "sketch" \
        --shots 1 \
        --config configs/pacs.yaml \
        --output_dir "./experiments_mini/PACS" \
        --data_root "./datasets/PACS"
wait

python pacs.py \
        --source_domains "art_painting,cartoon,photo" \
        --target_domain "sketch" \
        --shots 5 \
        --config configs/pacs.yaml \
        --output_dir "./experiments_mini/PACS" \
        --data_root "./datasets/PACS"
wait

python pacs.py \
        --source_domains "art_painting,cartoon,sketch" \
        --target_domain "photo" \
        --shots 1 \
        --config configs/pacs.yaml \
        --output_dir "./experiments_mini/PACS" \
        --data_root "./datasets/PACS"
wait

python pacs.py \
        --source_domains "art_painting,cartoon,sketch" \
        --target_domain "photo" \
        --shots 5 \
        --config configs/pacs.yaml \
        --output_dir "./experiments_mini/PACS" \
        --data_root "./datasets/PACS"
wait

python pacs.py \
        --source_domains "art_painting,photo,sketch" \
        --target_domain "cartoon" \
        --shots 1 \
        --config configs/pacs.yaml \
        --output_dir "./experiments_mini/PACS" \
        --data_root "./datasets/PACS"
wait

python pacs.py \
        --source_domains "art_painting,photo,sketch" \
        --target_domain "cartoon" \
        --shots 5 \
        --config configs/pacs.yaml \
        --output_dir "./experiments_mini/PACS" \
        --data_root "./datasets/PACS"
wait

python pacs.py \
        --source_domains "cartoon,photo,sketch" \
        --target_domain "art_painting" \
        --shots 1 \
        --config configs/pacs.yaml \
        --output_dir "./experiments_mini/PACS" \
        --data_root "./datasets/PACS"
wait

python pacs.py \
        --source_domains "cartoon,photo,sketch" \
        --target_domain "art_painting" \
        --shots 5 \
        --config configs/pacs.yaml \
        --output_dir "./experiments_mini/PACS" \
        --data_root "./datasets/PACS"
wait
