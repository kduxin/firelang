python -m scripts.train \
    --corpus_path=data/corpus/text8/text8.uncased.tokens \
    --sz_batch=8192 \
    --n_neg=1 \
    --lr=0.005 \
    --lr_scheduler=OneCycleLR \
    --dim=2 \
    --n_iters=100000 \
    --eval_interval=1000 \
    --savedir=results/ \
    --optimizer=adamw \
    --seed=0 \
    --accum_steps=10 \
    --func='MLPlanarDivFast(dim, 4)' \
    --measure='DiracMixture(dim, 10)' \
    --weight_decay=1e-6 \
    --use_wandb