#CUDA_VISIBLE_DEVICES=0 python fenge.py --queue="queue1" --batch_size=6 &
#CUDA_VISIBLE_DEVICES=1 python fenge.py --queue="queue3" --batch_size=6 &
#CUDA_VISIBLE_DEVICES=2 python fenge.py --queue="queue3" --batch_size=6 &
#CUDA_VISIBLE_DEVICES=3 python fenge.py --queue="queue4" --batch_size=6 &
#CUDA_VISIBLE_DEVICES=4 python fenge.py --queue="queue5" --batch_size=6 &
#CUDA_VISIBLE_DEVICES=5 python fenge.py --queue="queue1" --batch_size=6 &
#CUDA_VISIBLE_DEVICES=2 python fenge.py --queue="queue7" --batch_size=1 &
#CUDA_VISIBLE_DEVICES=3 python fenge.py --queue="queue8" --batch_size=1 &
#CUDA_VISIBLE_DEVICES=0 python fenge.py --queue="queue9" --batch_size=1 &
#CUDA_VISIBLE_DEVICES=1 python fenge.py --queue="queue10" --batch_size=1 &
#CUDA_VISIBLE_DEVICES=2 python fenge.py --queue="queue11" --batch_size=1 &
#CUDA_VISIBLE_DEVICES=3 python fenge.py --queue="queue12" --batch_size=1 &
#CUDA_VISIBLE_DEVICES=0 python fenge.py --queue="queue13" --batch_size=1 &
#CUDA_VISIBLE_DEVICES=1 python fenge.py --queue="queue14" --batch_size=1 &
#CUDA_VISIBLE_DEVICES=2 python fenge.py --queue="queue15" --batch_size=1 &
#CUDA_VISIBLE_DEVICES=3 python fenge.py --queue="queue16" --batch_size=1 &

#python test.py &
CUDA_VISIBLE_DEVICES=6 python tts.py &
python server.py &
CUDA_VISIBLE_DEVICES=7 python asr.py &
CUDA_VISIBLE_DEVICES=6 python hey_tts.py &
python bqfk.py &
#CUDA_VISIBLE_DEVICES=4,5,6,7 python server.py &
