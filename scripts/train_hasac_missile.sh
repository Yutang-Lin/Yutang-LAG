#!/bin/sh

env="MultipleCombat"
scenario="2v2/ShootMissile/HierarchySelfplay"
algo="hasac"
exp="v1missile"
seed=0

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 32 --cuda --log-interval 1 --save-interval 1 \
    --buffer-size 3000 --num-env-steps 1e8 --num-mini-batch 5 \
    --lr 3e-4 --gamma 0.99 --tau 0.005 --alpha 0.2 --target-update-interval 1 \
    --automatic-entropy-tuning --max-grad-norm 2 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --use-selfplay --selfplay-algorithm "fsp" --n-choose-opponents 1 \
    --use-eval --n-eval-rollout-threads 1 --eval-interval 1 --eval-episodes 1 \
    --user-name "jyh" --use-wandb --wandb-name "thu_jsbsim" \
    --model-dir './results/MultipleCombat/2v2/ShootMissile/HierarchySelfplay/hasac/v1missile/wandb/latest-run/files'
