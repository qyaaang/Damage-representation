datasets=("WN2" "WN11" "WN17" "WN21")
alphas=(5 8 10 15)
model="MLP"
len_segs=(300 400 500)
hidden_sizes=128
optimizer="SGD"
num_epoch=10000
lr=0.1
idx=0
for dataset in "${datasets[@]}"; do
  for len_seg in "${len_segs[@]}"; do
  printf "\033[1;32mDataset:\t%s\nLength of segments:\t%s\n\033[0m" \
           "$dataset" "$len_seg"
  python3 baseline_test.py --dataset "$dataset"  --model "$model" \
                           --len_seg "$len_seg"  --optimizer "$optimizer" --dim_feature "$hidden_sizes" \
                           --num_epoch $num_epoch --lr $lr  --last_time_step --alpha "${alphas[idx]}"

  done
  idx=$((idx+1))
done