python office.py \
        --source_domains "clipart,product,realworld" \
        --target_domain "art" \
        --shots 1 \
        --config configs/office_final.yaml \
        --output_dir "./experiments_mini/OfficeHome" \
        --data_root "./datasets/OfficeHome"
wait

python office.py \
        --source_domains "clipart,product,realworld" \
        --target_domain "art" \
        --shots 5 \
        --config configs/office_final.yaml \
        --output_dir "./experiments_mini/OfficeHome" \
        --data_root "./datasets/OfficeHome"
wait

python office.py \
        --source_domains "art,product,realworld" \
        --target_domain "clipart" \
        --shots 1 \
        --config configs/office_final.yaml \
        --output_dir "./experiments_mini/OfficeHome" \
        --data_root "./datasets/OfficeHome"
wait

python office.py \
        --source_domains "art,product,realworld" \
        --target_domain "clipart" \
        --shots 5 \
        --config configs/office_final.yaml \
        --output_dir "./experiments_mini/OfficeHome" \
        --data_root "./datasets/OfficeHome"
wait

python office.py \
        --source_domains "art,clipart,realworld" \
        --target_domain "product" \
        --shots 1 \
        --config configs/office_final.yaml \
        --output_dir "./experiments_mini/OfficeHome" \
        --data_root "./datasets/OfficeHome"
wait

python office.py \
        --source_domains "art,clipart,realworld" \
        --target_domain "product" \
        --shots 5 \
        --config configs/office_final.yaml \
        --output_dir "./experiments_mini/OfficeHome" \
        --data_root "./datasets/OfficeHome"
wait

python office.py \
        --source_domains "art,clipart,product" \
        --target_domain "realworld" \
        --shots 1 \
        --config configs/office_final.yaml \
        --output_dir "./experiments_mini/OfficeHome" \
        --data_root "./datasets/OfficeHome"
wait

python office.py \
        --source_domains "art,clipart,product" \
        --target_domain "realworld" \
        --shots 5 \
        --config configs/office_final.yaml \
        --output_dir "./experiments_mini/OfficeHome" \
        --data_root "./datasets/OfficeHome"
wait
