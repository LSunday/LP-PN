dataset=huffpost
data_path="data/huffpost.json"
n_train_class=20
n_val_class=5
n_test_class=16

# dataset=amazon
# data_path="data/amazon.json"
# n_train_class=10
# n_val_class=5
# n_test_class=9

# dataset=reuters
# data_path="data/reuters.json"
# n_train_class=15
# n_val_class=5
# n_test_class=11

# dataset=20newsgroup
# data_path="data/20news.json"
# n_train_class=8
# n_val_class=5
# n_test_class=7







python src/main.py \
    --cuda 0 \
    --way 5 \
    --shot 1 \
    --query 25 \
    --mode train \
    --dataset=$dataset \
    --data_path=$data_path \
    --n_train_class=$n_train_class \
    --n_val_class=$n_val_class \
    --n_test_class=$n_test_class \
    --embedding bilstm \
    --train_episodes 100 \
    --n 1 \
    --lr_g 1e-3 \
    --lr_d 1e-3 \
    --patience 20 \
    --seed 123 \
    --rn 30 \
    --g 5 \
    --k 5 \


python src/main.py \
    --cuda 1 \
    --way 5 \
    --shot 5 \
    --query 25 \
    --mode train \
    --dataset=$dataset \
    --data_path=$data_path \
    --n_train_class=$n_train_class \
    --n_val_class=$n_val_class \
    --n_test_class=$n_test_class \
    --embedding bilstm \
    --train_episodes 100 \
    --n 1 \
    --lr_g 1e-3 \
    --lr_d 1e-3 \
    --patience 20 \
    --seed 123 \
    --rn 30 \
    --g 5 \
    --k 11 \

