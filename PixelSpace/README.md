###select CIFAR-10 as example
### supernet train
CUDA_VISIBLE_DEVICES=4,5,6,7 python main_distill.py --config cifar10.yml --exp exp_dis_nas_id2 --doc cifar_dis   --super_train
### search for the best subnet:
CUDA_VISIBLE_DEVICES=4,5,6,7 python main_distill.py --config cifar10.yml --exp exp_dis_nas --doc cifar_dis --nas_search
### train the subnet
CUDA_VISIBLE_DEVICES=2,3 python main_distill.py --config cifar10.yml --exp exp_retrain_alone  --doc cifar_dis  --stand_alone_train
### sample with the trained subnet
CUDA_VISIBLE_DEVICES=0 python main_distill.py --config cifar10.yml --exp exp_retrain_alone --doc cifar_dis --sample --stand_alone_sample --fid --timesteps 20 --eta 0  --skip_type quad  --sample_type dpm





