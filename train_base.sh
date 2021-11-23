#!/bin/bash
model="MLP"
len_segs=(300 400 500)
hidden_sizes=(128)
optimizer="SGD"
num_epoch=10000
if [ "$optimizer" == "SGD" ]; then
  lr=0.1
else
  lr=0.001
fi
for len_seg in "${len_segs[@]}"; do
  for hidden_size in "${hidden_sizes[@]}"; do
    printf "\033[1;32mModel:\t%s\nLength of segments:\t%s\nHidden size:\t%s\n\033[0m" \
           "$model" "$len_seg" "$hidden_size"
    python baseline.py --model "$model" --len_seg "$len_seg" --dim_feature "$hidden_size" \
                       --num_epoch $num_epoch --lr $lr  --last_time_step
  done
done
