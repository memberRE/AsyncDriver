python data_generation/data_process.py \
--data_path ./nuplan/dataset/nuplan-v1.1/splits/trainval/ \
--map_path ./nuplan/dataset/maps/ \
--save_path ./asyncdriver_data/ \
--scenarios_per_type 5000 \
--scenario_cache ./scenario_cache.pkl \
--start_s $1