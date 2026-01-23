
echo "=====Cora====="
python main.py --dataset=Planetoid_Cora --ID_classes 0 1 2 3 --GAT_num_heads=2 --GAT_num_layers=2 --GAT_num_hidden=16 \
               --GAT_feat_drop=0.3 --GAT_attn_drop=0.3 --GAT_negative_slope=0.2 --tau=1 \
               --lr=0.01 --weight_decay=1e-3 --epochs=1000 --c_rate=0.1 \
               --w_s=1.0 --w_po=1.0 --w_pi=0.1 --w_os=0.1 --w_p2p=1.0 --w_n2p=1.0

wait
echo "=====Citeseer====="
python main.py --dataset=Planetoid_Citeseer --ID_classes 0 1 2 --GAT_num_heads=4 --GAT_num_layers=2 --GAT_num_hidden=16 \
               --GAT_feat_drop=0.3 --GAT_attn_drop=0.3 --GAT_negative_slope=0.2 --tau=1 \
               --lr=0.01 --weight_decay=1e-3 --epochs=1000 --c_rate=0.1 \
               --w_s=1.0 --w_po=1.0 --w_pi=0.1 --w_os=1.0 --w_p2p=10.0 --w_n2p=10.0

wait
echo "=====PubMed====="
python main.py --dataset=Planetoid_PubMed --ID_classes 0 1 --GAT_num_heads=4 --GAT_num_layers=2 --GAT_num_hidden=16 \
               --GAT_feat_drop=0.3 --GAT_attn_drop=0.3 --GAT_negative_slope=0.2 --tau=1 \
               --lr=0.01 --weight_decay=1e-3 --epochs=1000 --c_rate=0.1 \
               --w_s=1.0 --w_po=10.0 --w_pi=0.1 --w_os=0.1 --w_p2p=10.0 --w_n2p=10.0

wait
echo "=====Coauthor_CS====="
python main.py --dataset=Coauthor_CS --ID_classes 0 1 2 3 4 5 6 7 --GAT_num_heads=4 --GAT_num_layers=2 --GAT_num_hidden=16 \
               --GAT_feat_drop=0.3 --GAT_attn_drop=0.3 --GAT_negative_slope=0.2 --tau=1 \
               --lr=0.01 --weight_decay=1e-3 --epochs=1000 --c_rate=0.1 \
               --w_s=1.0 --w_po=10.0 --w_pi=0.1 --w_os=1.0 --w_p2p=10.0 --w_n2p=10.0

wait
echo "=====wiki-CS====="
python main.py --dataset=wiki-CS --ID_classes 0 1 2 3 4 --GAT_num_heads=2 --GAT_num_layers=2 --GAT_num_hidden=16 \
               --GAT_feat_drop=0.3 --GAT_attn_drop=0.3 --GAT_negative_slope=0.2 --tau=1 \
               --lr=0.01 --weight_decay=1e-3 --epochs=1000 --c_rate=0.1 \
               --w_s=1.0 --w_po=1.0 --w_pi=0.1 --w_os=1.0 --w_p2p=1.0 --w_n2p=1.0

wait
echo "=====Amazon_Computers====="
python main.py --dataset=Amazon_Computers --ID_classes 0 1 2 3 4 --GAT_num_heads=2 --GAT_num_layers=2 --GAT_num_hidden=16 \
               --GAT_feat_drop=0.3 --GAT_attn_drop=0.3 --GAT_negative_slope=0.2 --tau=1 \
               --lr=0.01 --weight_decay=1e-4 --epochs=1000 --c_rate=0.1 \
               --w_s=1.0 --w_po=10.0 --w_pi=1.0 --w_os=1.0 --w_p2p=10.0 --w_n2p=10.0

wait
echo "=====Amazon_Photo====="
python main.py --dataset=Amazon_Photo --ID_classes 0 1 2 3 --GAT_num_heads=2 --GAT_num_layers=2 --GAT_num_hidden=16 \
               --GAT_feat_drop=0.3 --GAT_attn_drop=0.3 --GAT_negative_slope=0.2 --tau=1 \
               --lr=0.01 --weight_decay=1e-3 --epochs=1000 --c_rate=0.1 \
               --w_s=1.0 --w_po=10.0 --w_pi=0.1 --w_os=0.1 --w_p2p=1.0 --w_n2p=1.0

wait
echo "=====arxiv-year====="
python main.py --dataset=arxiv-year --ID_classes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
               --GAT_num_heads=4 --GAT_num_layers=2 --GAT_num_hidden=16 \
               --GAT_feat_drop=0.3 --GAT_attn_drop=0.1 --GAT_negative_slope=0.2 --tau=0.1 \
               --lr=0.01 --weight_decay=1e-4 --epochs=1000 --c_rate=0.1 \
               --w_s=1.0 --w_po=1.0 --w_pi=1.0 --w_os=1.0 --w_p2p=0.1 --w_n2p=0.1