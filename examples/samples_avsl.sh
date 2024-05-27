# resnet50 cub200 proxyanchor(AVSL) 
python examples/demo.py --data_path <path-to-data> --save_path <path-to-log> --device 0 --batch_size 180 --test_batch_size 180 --setting avsl_proxyanchor --feature_dim_list 512 1024 2048 --embeddings_dim 512 --avsl_m 0.5 --topk_corr 128 --prob_gamma 10 --index_p 2 --pa_pos_margin 1.8 --pa_neg_margin 2.2 --pa_alpha 16 --final_pa_pos_margin 1.8 --final_pa_neg_margin 2.2 --final_pa_alpha 16 --num_classes 100 --use_proxy --wd 0.0001 --gamma 0.5 --step 10 --dataset cub200 --model resnet50 --splits_to_eval test --warm_up 5 --warm_up_list embedder collector --loss0_weight=0.5 --loss1_weight=1 --loss2_weight=0.5 --lr_collector=0.00011 --lr_embedder=0.00055 --lr_trunk=0.00001 \
--save_name proxy-anchor-resnet50-cub200-avsl

# resnet50 cars196 proxyanchor(AVSL) 
python examples/demo.py --data_path <path-to-data> --save_path <path-to-log> --device 0 --batch_size 180 --test_batch_size 180 --setting avsl_proxyanchor --feature_dim_list 512 1024 2048 --embeddings_dim 512 --avsl_m 0.5 --topk_corr 128 --prob_gamma 10 --index_p 2 --pa_pos_margin 1.8 --pa_neg_margin 2.2 --pa_alpha 16 --final_pa_pos_margin 1.8 --final_pa_neg_margin 2.2 --final_pa_alpha 16 --num_classes 98 --use_proxy --wd 0.0001 --gamma 0.5 --step 5 --dataset cars196 --model resnet50 --splits_to_eval test --warm_up 5 --warm_up_list embedder collector --loss0_weight=1 --loss1_weight=4 --loss2_weight=4 --lr_collector=0.09918213648327627 --lr_embedder=0.00016294139202611374 --lr_trunk=0.00025230934618157737 \
--save_name proxy-anchor-resnet50-cars196-avsl

# resnet50 online_products proxyanchor(AVSL)
python examples/demo.py --data_path <path-to-data> --save_path <path-to-log> --device 0 --batch_size 180 --test_batch_size 180 --setting avsl_proxyanchor --feature_dim_list 512 1024 2048 --embeddings_dim 512 --avsl_m 0.5 --topk_corr 128 --prob_gamma 10 --index_p 2 --loss0_weight 1.0 --loss1_weight 4.0 --loss2_weight 4.0 --pa_pos_margin 1.8 --pa_neg_margin 2.4 --pa_alpha 16 --final_pa_pos_margin 1.8 --final_pa_neg_margin 2.2 --final_pa_alpha 16 --num_classes 11318 --use_proxy --wd 0.0001 --gamma 0.25 --step 15 --lr_trunk 0.0006 --lr_embedder 0.0006 --lr_collector 0.06 --dataset online_products --delete_old --model resnet50 --splits_to_eval test --warm_up 5 --warm_up_list embedder collector --not_freeze_bn --test_split_num 100 --interval 5 --k_list 1 10 100 --k 101 \
--save_name proxy-anchor-resnet50-online-avsl
