datasets=("WN2" "WN11" "WN17" "WN21")
models=("RNN" "LSTM" "GRU")
len_segs=(300 400 500)
hidden_sizes=128
optimizer="SGD"
num_epoch=1000
lr=0.1
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for len_seg in "${len_segs[@]}"; do
      printf "\033[1;32mDataset:\t%s\nModel:\t%s\nLength of segments:\t%s\n\033[0m" \
             "$dataset" "$model" "$len_seg"
      python3 test.py --dataset "$dataset"  --model "$model" \
                      --len_seg "$len_seg"  --optimizer "$optimizer" --hidden_size "$hidden_sizes" \
                      --num_epoch $num_epoch --lr $lr  --last_time_step
    done
  done
done
