V="v1"
if test ${V} = "v1"
then
  MODEL_DIR=models/model-20180309-083949.ckpt-60000
  TEST_DATA=dataset/lfw-112X96
  EMBEDDING_SIZE=512
  FC_BN=''
  PREWHITEN='--prewhiten'
  IMAGE_WIDTH=96
else
  MODEL_DIR=models/model-20180626-205832.ckpt-60000
  TEST_DATA=dataset/lfw-112x112
  EMBEDDING_SIZE=1024  
  FC_BN='--fc_bn'
  PREWHITEN=''
  IMAGE_WIDTH=112
fi
IMAGE_HEIGHT=112
CUDA_VISIBLE_DEVICES=1 python test/test.py ${TEST_DATA} ${MODEL_DIR} --lfw_file_ext jpg --network_type sphere_network --embedding_size ${EMBEDDING_SIZE} ${FC_BN} ${PREWHITEN} --image_height ${IMAGE_HEIGHT} --image_width ${IMAGE_WIDTH}
