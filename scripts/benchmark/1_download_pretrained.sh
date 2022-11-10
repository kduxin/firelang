BASE_URL="https://www.cl.rcast.u-tokyo.ac.jp/~duxin/firelang/pretrained/word/v1.0/"
MODEL_23="wacky_mlplanardiv_d2_l4_k1_polysemy.gz"
MODEL_50="wacky_mlplanardiv_d2_l4_k10.gz"
MODEL_100="wacky_mlplanardiv_d2_l8_k20.gz"

mkdir -p checkpoints
wget "$BASE_URL$MODEL_23" -O checkpoints/$MODEL_23
wget "$BASE_URL$MODEL_50" -O checkpoints/$MODEL_50
wget "$BASE_URL$MODEL_100" -O checkpoints/$MODEL_100

gzip -d checkpoints/*