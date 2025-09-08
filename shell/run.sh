datapath=N:/MVTec-AD_DiffAD
augpath=N:/DiffusionAD-main/dtd-r1.0.1/dtd/images
classes=('wood')
flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))


cd ..
python -X utf8 main.py \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
  net \
    -b wideresnet50 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 100 \
    --eval_epochs 1 \
    --dsc_layers 2 \
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --mining 1 \
    --noise 0.015 \
    --radius 0.75 \
    --p 0.5 \
    --step 20 \
    --limit 392 \
  dataset \
    --distribution 0 \
    --mean 0.5 \
    --std 0.1 \
    --fg 0 \
    --rand_aug 1 \
    --downsampling 16\
    --batch_size 1 \
    --resize 288 \
    --imagesize 288 "${flags[@]}" mvtec $datapath $augpath

read -p "Press any key to continue..."
