for k in 2 4 8 16; do # vocab_size
for j in 512; do # batch_size
for l in 0.00001; do # learning rate
for i in 10 20 30; do # 3 seeds
    /root/.local/bin/poetry run python -m egg.zoo.compo_vs_generalization.train \
        --n_values=4 \
        --n_attributes=4 \
        --vocab_size=$k \
        --max_len=32 \
        --batch_size=$j \
        --n_epochs=4000 \
        --sender_cell=gru \
        --receiver_cell=gru \
        --sender_hidden 512 \
        --receiver_hidden 512 \
        --random_seed=$i \
        --exp_id=1 \
        --lr=$l
done
done
done
done