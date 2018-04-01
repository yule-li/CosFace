MODEL_DIR=models/model-20180309-083949.ckpt-60000
CUDA_VISIBLE_DEVICES=1 python test/test.py dataset/lfw-112X96 ${MODEL_DIR} --lfw_file_ext jpg --network_type sphere_network
