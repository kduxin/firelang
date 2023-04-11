BASE_URL="https://www.l.sci.waseda.ac.jp/member/duxin/firelang/pretrained/word/"
VERSION="v1.1/"
MODEL_23="wacky_mlplanardiv_d2_l4_k1_polysemy.tar.gz"
MODEL_50="wacky_mlplanardiv_d2_l4_k10.tar.gz"
MODEL_100="wacky_mlplanardiv_d2_l8_k20.tar.gz"

mkdir -p checkpoints/$VERSION

wget "$BASE_URL$VERSION$MODEL_23" -O checkpoints/$VERSION$MODEL_23
tar zxvf checkpoints/$VERSION$MODEL_23 -C checkpoints/$VERSION
rm checkpoints/$VERSION$MODEL_23

wget "$BASE_URL$VERSION$MODEL_50" -O checkpoints/$VERSION$MODEL_50
tar zxvf checkpoints/$VERSION$MODEL_50 -C checkpoints/$VERSION
rm checkpoints/$VERSION$MODEL_50

wget "$BASE_URL$VERSION$MODEL_100" -O checkpoints/$VERSION$MODEL_100
tar zxvf checkpoints/$VERSION$MODEL_100 -C checkpoints/$VERSION
rm checkpoints/$VERSION$MODEL_100
