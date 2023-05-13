nohup python train_generator.py \
--dataset_name FLIR \
--seed 0 \
--device cuda:1 \
--image_root \
--edge_root  \
--mask_root  \
--loadsize 288 \
--cropsize 256 \
--batch_size 4 \
--num_workers 8 \
--num_epochs 1000 \
--lr 1e-4 \
--beta1 0.5 \
--beta2 0.9 \
--sample_step 100 \
--sample_size 4 \
--edge_ckpt_path \
--gen_ckpt_path None \
>> FLIR_gen.out &