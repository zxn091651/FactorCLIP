python minidomainnet.py \
        --source_domains "clipart,painting,sketch" \
        --target_domain "real" \
        --shots 1 \
        --config configs/minidomainnet.yaml \
        --output_dir "./experiments_mini/miniDomainNet" \
        --data_root "./datasets/miniDomainNet"
wait

python minidomainnet.py \
        --source_domains "clipart,painting,sketch" \
        --target_domain "real" \
        --shots 5 \
        --config configs/minidomainnet.yaml \
        --output_dir "./experiments_mini/miniDomainNet" \
        --data_root "./datasets/miniDomainNet"
wait
