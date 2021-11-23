datasets=("WN2" "WN11" "WN17" "WN21")
model="MLP"
len_segs=(300 400 500)
hidden_sizes=128
optimizer="SGD"
num_epoch=10000
lr=0.1
for dataset in "${datasets[@]}"; do
  for len_seg in "${len_segs[@]}"; do
  python3 baseline_test.py --dataset "$dataset"  --model "$model" \
                           --len_seg "$len_seg"  --optimizer "$optimizer" --dim_feature "$hidden_sizes" \
                           --num_epoch $num_epoch --lr $lr  --last_time_step

  done
done