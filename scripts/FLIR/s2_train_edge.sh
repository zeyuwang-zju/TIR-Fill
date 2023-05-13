nohup python train_edge.py \
--dataset_name FLIR \
--seed 0 \
--device cuda:1 \
--image_root \
--edge_root \
--mask_root \
--loadsize 288 \
--cropsize 256 \
--batch_size 16 \
--num_workers 4 \
--num_epochs 1000 \
--lr 1e-3 \
--beta1 0.5 \
--beta2 0.9 \
--fm_loss_weight 1 \
--sample_step 100 \
--sample_size 4 \
--edge_ckpt_path None \
>> FLIR_edge.out &