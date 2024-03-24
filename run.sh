
#python main.py --undirected --mapped 1 --epochs 500 --test_all_bits --num_hops 1 --num_layers 1 --lr 5e-4 --hidden_channels 256 --save_model --device 1 --bits 8 --dropout 0.5 --batch_size 1024 --lda1 5 --lda2 1 --heads 128
python main.py --undirected --mapped 1 --epochs 500 --test_all_bits --num_hops 8 --num_layers 1 --lr 5e-4 --hidden_channels 256 --save_model --device 1 --bits 8 --dropout 0.5 --batch_size 1024 --lda1 5 --lda2 1 --heads 32
