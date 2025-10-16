python -m scripts.build_vocab \
    --path_to_corpus='data/corpus/text8/text8.uncased.tokens' \
    --min_count=5 \
    --infreq_replace="<unk>"