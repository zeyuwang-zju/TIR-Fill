python extract_edge.py \
--imgfile_path /home/data/wangzeyu/FLIR_ADAS_1_3/train/thermal_8_bit/ \
--save_path /home/data/wangzeyu/FLIR_ADAS_1_3/train/edge/ \
--low_threshold 80 \
--high_threshold 160

python extract_edge.py \
--imgfile_path /home/data/wangzeyu/FLIR_ADAS_1_3/val/thermal_8_bit_256_256/ \
--save_path /home/data/wangzeyu/FLIR_ADAS_1_3/val/edge_256_256/ \
--low_threshold 80 \
--high_threshold 160