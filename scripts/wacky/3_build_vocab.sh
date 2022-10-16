python -m scripts.build_vocab \
    --path_to_corpus='data/corpus/wacky/wacky.txt.uncased.tokens' \
    --min_count=100 \
    --infreq_replace="<unk>"