import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import NodePropPredDataset
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, auc, roc_curve
from sklearn.preprocessing import LabelEncoder
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, LastFMAsia, WikiCS
from torch_geometric.transforms import NormalizeFeatures, Compose
from torch_geometric.utils import homophily

from GAT_encoder import GAT
from data_process import data_process
from utils import acc

from torch_geometric.data import Data

def load_arxiv_year_dataset():
    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root="pyg_data/arxiv-year")
    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
    label = torch.as_tensor(ogb_dataset.labels).reshape(-1)

    le = LabelEncoder()
    label = torch.tensor(le.fit_transform(label.numpy()))

    data = Data(
        x=node_feat,
        edge_index=edge_index,
        y=label
    )
    return data


def generate_cl_prototype(gat_views, pseudo_labels):
    '''
    Generation of prototypes (class centres) for each category based on node embeddings and pseudo-labels in the GAT layer
    '''
    unique_labels = torch.unique(pseudo_labels)
    class_prototype = torch.zeros((gat_views.shape[0], len(unique_labels), gat_views.shape[-1]), dtype=gat_views.dtype,
                                  device=gat_views.device)

    for i, label in enumerate(unique_labels):
        mask = pseudo_labels == label
        class_features = gat_views[:, mask, :]
        class_prototype[:, i, :] = torch.mean(class_features, dim=1)

    return class_prototype


def calculate_unk_score(K_prob, K_1_prob, edge_index, neighbor_counts):
    '''
    Calculate the OOD score of nodes, based on entropy and unknown class probability, and perform OOD score neighbour aggregation
    '''
    ent = torch.sum(- K_prob * torch.log(K_prob + 1e-10), dim=1)
    ent_norm = (ent / np.log(K_prob.size(1)))

    self_unk_score = 1 * ent_norm + 0.1 * K_1_prob[:, -1]

    neighbor_scores = torch.zeros_like(self_unk_score).to(device)
    neighbor_scores.scatter_add_(0, edge_index[0], self_unk_score[edge_index[1]])
    neighbor_scores /= (neighbor_counts + 1e-10)

    unk_score = self_unk_score + neighbor_scores

    return unk_score


def multiview_p2p_loss(prototype_views):
    '''
    multi-view prototype-to-prototype contrastive loss
    '''
    num_views = prototype_views.shape[0]
    loss = torch.tensor(0, dtype=float, requires_grad=True).to(device)
    for i in range(1, num_views):
        loss += p2p_contrastive_loss(prototype_views[0], prototype_views[i])
    loss /= prototype_views.shape[0]
    return loss


def p2p_contrastive_loss(view_i, view_j):
    '''
    Compute the prototype-to-prototype contrastive loss between two views
    '''
    cat_view = torch.cat([view_i, view_j])
    norm_cat_view = F.normalize(cat_view, dim=-1)
    cosine_sim_matrix = torch.matmul(norm_cat_view, norm_cat_view.T)

    num_prototype = view_i.shape[0]

    prototype_index = torch.cat([torch.arange(num_prototype) for i in range(2)], dim=0).to(device)
    prototype_index = (prototype_index.unsqueeze(0) == prototype_index.unsqueeze(1)).float()

    mask = torch.eye(prototype_index.shape[0], dtype=torch.bool).to(device)
    prototype_index = prototype_index[~mask].view(prototype_index.shape[0], -1)
    sub_sim_matrix = cosine_sim_matrix[~mask].view(cosine_sim_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = sub_sim_matrix[prototype_index.bool()].view(prototype_index.shape[0], -1)
    # select only the negatives
    negatives = sub_sim_matrix[~prototype_index.bool()].view(sub_sim_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits = logits / args.tau
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    contrastive_loss = nn.CrossEntropyLoss()(logits, labels)
    return contrastive_loss


def multiview_n2p_loss(gat_views, total_prototype, labels):
    '''
    multi-view node-to-prototype contrastive loss
    '''
    num_views = gat_views.shape[0]
    loss = torch.tensor(0, dtype=float, requires_grad=True).to(device)
    for i in range(1, num_views):
        loss += n2p_contrastive_loss(0, i, gat_views, total_prototype, labels)
    loss /= num_views
    return loss


def n2p_contrastive_loss(view_i, view_j, gat_views, total_prototype, labels):
    '''
    Compute the node-to-prototype contrastive loss between two views
    '''
    unique_labels, label_indices = torch.unique(labels, return_inverse=True)

    norm_nodes_i = F.normalize(gat_views[view_i], p=2, dim=1)
    norm_nodes_j = F.normalize(gat_views[view_j], p=2, dim=1)
    norm_proto_i = F.normalize(total_prototype[view_i], p=2, dim=1)
    norm_proto_j = F.normalize(total_prototype[view_j], p=2, dim=1)

    nodei_protoi_sim = torch.mm(norm_nodes_i, norm_proto_i.t().contiguous())
    nodei_protoj_sim = torch.mm(norm_nodes_i, norm_proto_j.t().contiguous())
    nodej_protoi_sim = torch.mm(norm_nodes_j, norm_proto_i.t().contiguous())
    nodej_protoj_sim = torch.mm(norm_nodes_j, norm_proto_j.t().contiguous())

    f = lambda x: torch.exp(x / args.tau)
    nodei_protoi = f(nodei_protoi_sim)
    nodei_protoj = f(nodei_protoj_sim)
    nodej_protoi = f(nodej_protoi_sim)
    nodej_protoj = f(nodej_protoj_sim)

    loss_i = ((nodei_protoi[torch.arange(nodei_protoi.size(0)), label_indices] + nodei_protoj[
        torch.arange(nodei_protoj.size(0)), label_indices]) /
              (torch.sum(nodei_protoi, dim=1) + torch.sum(nodei_protoj, dim=1))) * 0.5
    loss_j = ((nodej_protoi[torch.arange(nodej_protoi.size(0)), label_indices] + nodej_protoj[
        torch.arange(nodej_protoj.size(0)), label_indices]) /
              (torch.sum(nodej_protoi, dim=1) + torch.sum(nodej_protoj, dim=1))) * 0.5

    loss_i = -torch.log(loss_i)
    loss_j = -torch.log(loss_j)

    contrastive_loss = torch.mean(torch.cat((loss_i, loss_j)))

    return contrastive_loss


def ood_score_regularization(train_score, po_score):
    '''
    OOD score regularization: encouraging low OOD scores for the training set and high OOD scores for the potential OOD nodes
    '''
    return torch.mean(train_score) - torch.mean(po_score)


def negtive_mix_up(ood_embs, sample_embs, ood_labels, sample_labels, num_classes):
    '''
    Negative Mixup for constructing mixed-up pseudo-OOD training samples
    '''
    # Get the size of ood_embs
    num_ood, emb_dim = ood_embs.shape
    num_samples = sample_embs.shape[0]

    # Randomly select num_ood points from sample_embs.
    random_indices = torch.randint(0, num_samples, (num_ood,))
    selected_embs = sample_embs[random_indices]
    selected_labels = sample_labels[random_indices]

    # Generate mixing coefficient λ
    beta_distribution = torch.distributions.Beta(torch.tensor(1.0), torch.tensor(1.0))
    lambdas = beta_distribution.sample((num_ood, 1)).to(device)

    # lambdas = torch.ones((num_ood, 1)).to(device)

    # mixup embs
    mixed_emb = lambdas * ood_embs + (1 - lambdas) * (-selected_embs)

    # MixUp label
    ood_labels_one_hot = F.one_hot(ood_labels, num_classes=num_classes).float()
    sample_labels_one_hot = F.one_hot(selected_labels, num_classes=num_classes).float()
    mixed_labels = lambdas * ood_labels_one_hot + (1 - lambdas) * (-sample_labels_one_hot)

    return mixed_emb, mixed_labels, ood_embs, selected_embs, selected_labels


def normal_mix_up(emb1, emb2, labels1, labels2, num_classes):
    '''
    Positive Mixup for constructing mixed-up pseudo-ID training samples
    '''
    # Get the size of emb1
    num1, emb_dim = emb1.shape
    num2 = emb2.shape[0]

    # Randomly select num1 points from emb2.
    random_indices = torch.randint(0, num2, (num1,))
    selected_emb2 = emb2[random_indices]
    selected_labels2 = labels2[random_indices]

    # Generate mixing coefficient λ
    beta_distribution = torch.distributions.Beta(torch.tensor(1.0), torch.tensor(1.0))
    lambdas = beta_distribution.sample((num1, 1)).to(device)  # 形状为 (num1, 1)

    # lambdas = torch.ones((num1, 1)).to(device)

    # mixup embs
    mixed_emb = lambdas * emb1 + (1 - lambdas) * selected_emb2

    # mixup label
    labels1_one_hot = F.one_hot(labels1, num_classes=num_classes).float()
    labels2_one_hot = F.one_hot(selected_labels2, num_classes=num_classes).float()
    mixed_labels = lambdas * labels1_one_hot + (1 - lambdas) * labels2_one_hot

    return mixed_emb, mixed_labels


def train(args, data):
    # Setting random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Data processing
    ID_mask, OOD_mask, train_mask, valid_mask, test_mask, test_seen_mask, test_unseen_mask, detection_y_test, joint_y_test, new_labels, new_unseen_label = data_process(
        data, args.ID_classes,
        args.train_rate, args.val_rate,
        args.seed)

    features = data.x.clone()
    edge_index = data.edge_index.clone()

    # Initialize the model and optimizer
    args.num_classes = torch.unique(new_labels).size(0)
    model = GAT(num_layers=args.GAT_num_layers,
                in_dim=features.shape[1],
                num_hidden=args.GAT_num_hidden,
                out_dim=args.num_classes,
                heads=args.GAT_num_heads,
                feat_drop=args.GAT_feat_drop,
                attn_drop=args.GAT_attn_drop,
                negative_slope=args.GAT_negative_slope)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model = model.to(device)
    features = features.to(device)
    new_labels = new_labels.to(device)
    edge_index = edge_index.to(device)
    best_t = args.epochs - 1
    min_val_loss = 100

    neighbor_counts = torch.zeros(features.shape[0]).to(device)
    neighbor_counts.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)).to(device))

    ood_number = []

    total_epoch_time = 0

    # GPU memory monitor initialization
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_peak_mem = 0

    for epoch in range(args.epochs):
        epoch_t = time.time()
        model.train()
        optimizer.zero_grad()
        loss = torch.tensor(0, dtype=float, requires_grad=True).to(device)

        gat_views, final_emb, output_logits, output_prob = model(features,
                                                                 edge_index)
        num_views = gat_views.shape[0]

        K_softmax_prob = F.softmax(output_logits[:, :-1], dim=1)

        unk_score = calculate_unk_score(K_softmax_prob, output_prob, edge_index, neighbor_counts)

        #clustering-then-ranking strategy to select high confidence potential OOD/ID nodes
        test_unk_scores = unk_score[test_mask].cpu().detach().numpy().reshape(-1, 1)
        ood_kmeans = KMeans(n_clusters=2).fit(test_unk_scores)
        ood_pred = ood_kmeans.predict(test_unk_scores)


        high_score_cluster = 0 if ood_kmeans.cluster_centers_[0] > ood_kmeans.cluster_centers_[1] else 1

        # Calculate the distance from each node to the cluster center in the high-score cluster.
        distances_to_high_center = np.abs(
            test_unk_scores[ood_pred == high_score_cluster] - ood_kmeans.cluster_centers_[high_score_cluster])

        # Find the indices of the top 10% closest nodes
        top_10_percent_idx = np.argsort(distances_to_high_center, axis=0)[
                             : int(len(distances_to_high_center) * args.c_rate)].flatten()

        pseudo_test_OOD_mask = np.zeros_like(ood_pred, dtype=bool)

        high_score_indices = np.where(ood_pred == high_score_cluster)[0]
        pseudo_test_OOD_mask[high_score_indices[top_10_percent_idx]] = True

        pseudo_test_OOD_mask = torch.tensor(pseudo_test_OOD_mask)

        pseudo_OOD_labels = torch.tensor([new_unseen_label] * torch.sum(pseudo_test_OOD_mask).item(),
                                         dtype=torch.int64).to(device)

        low_score_cluster = 1 - high_score_cluster

        # Calculate the distance from each node to the cluster center within the low-score clusters.
        distances_to_low_center = np.abs(
            test_unk_scores[ood_pred == low_score_cluster] - ood_kmeans.cluster_centers_[low_score_cluster])

        # Find the indices of the top 10% closest nodes
        top_10_percent_idx = np.argsort(distances_to_low_center, axis=0)[
                             : int(len(distances_to_low_center) * args.c_rate)].flatten()

        pseudo_test_ID_mask = np.zeros_like(ood_pred, dtype=bool)

        low_score_indices = np.where(ood_pred == low_score_cluster)[0]
        pseudo_test_ID_mask[low_score_indices[top_10_percent_idx]] = True
        pseudo_test_ID_mask = torch.tensor(pseudo_test_ID_mask)

        pseudo_ID_labels = torch.argmax(output_prob[:, :-1][test_mask][pseudo_test_ID_mask], dim=1).to(device)

        if epoch % 10 == 0:
            ood_number.append(torch.sum(pseudo_test_OOD_mask).item())

        print('epoch:{}'.format(epoch + 1))

        # Cross-entropy loss
        train_sup_loss = nn.CrossEntropyLoss()(output_logits[train_mask], new_labels[train_mask])
        loss += open_and_rate['train_sup_loss'][1] * train_sup_loss
        print('train_sup_loss: {:.4f}.'.format(train_sup_loss))

        # Positive and negative Learning Loss
        p_OOD_emb = final_emb[test_mask][pseudo_test_OOD_mask]
        train_emb = final_emb[train_mask]
        mixed_OOD_emb, mixed_OOD_labels, ood_embs, selected_embs, selected_labels = negtive_mix_up(p_OOD_emb, train_emb,
                                                                                                   pseudo_OOD_labels,
                                                                                                   new_labels[
                                                                                                       train_mask],
                                                                                                   output_prob.shape[1])

        mixed_OOD_logits = model.linear_layer(mixed_OOD_emb)
        mixed_OOD_probs = F.softmax(mixed_OOD_logits, dim=1)

        pseOOD_pos_loss = mixed_OOD_labels[:, -1] * (-torch.log(mixed_OOD_probs[:, -1] + 1e-10))
        pseOOD_neg_loss = torch.sum(
            torch.abs(mixed_OOD_labels[:, :-1]) * (-torch.log(1 - mixed_OOD_probs[:, :-1] + 1e-10)), dim=1)
        pseOOD_loss = torch.mean(pseOOD_pos_loss + pseOOD_neg_loss)

        loss += open_and_rate['pseOOD_loss'][1] * pseOOD_loss
        print('pseOOD_loss: {:.4f}.'.format(pseOOD_loss))

        # Mixup loss for training mixed-up pseudo-ID samples
        p_ID_emb = final_emb[test_mask][pseudo_test_ID_mask]
        train_emb = final_emb[train_mask]
        mixed_ID_emb, mixed_ID_labels = normal_mix_up(p_ID_emb, train_emb, pseudo_ID_labels, new_labels[train_mask],
                                                      output_prob.shape[1])

        mixed_ID_logits = model.linear_layer(mixed_ID_emb)
        mixed_ID_probs = F.log_softmax(mixed_ID_logits, dim=1)

        pseID_loss = torch.mean(torch.sum(mixed_ID_labels * (-mixed_ID_probs), dim=1))

        loss += open_and_rate['pseID_loss'][1] * pseID_loss
        print('pseID_loss: {:.4f}.'.format(pseID_loss))

        # OOD score regularization
        ood_score_loss = ood_score_regularization(unk_score[train_mask], unk_score[test_mask][pseudo_test_OOD_mask])
        loss += open_and_rate['ood_score_loss'][1] * ood_score_loss
        print('ood_score_loss: {:.4f}.'.format(ood_score_loss))

        # Cross-layer GCL module
        valid_test_mask = valid_mask | test_mask

        pseudo_labels = torch.zeros_like(new_labels).to(device)
        pseudo_labels[train_mask] = new_labels[train_mask]
        pseudo_labels[valid_test_mask] = torch.argmax(output_prob[valid_test_mask], dim=1)

        total_prototype = generate_cl_prototype(gat_views, pseudo_labels)

        if num_views > 1:
            # Prototype-to-prototype contrastive loss
            p2p_cl_loss = multiview_p2p_loss(total_prototype)
            loss += open_and_rate['multi_views_p2p_cl_loss_train_pseOOD'][1] * p2p_cl_loss
            print('multi_views_p2p_cl_loss_train_pseOOD: {:.4f}.'.format(p2p_cl_loss))

            # Node-to-prototype contrastive loss
            n2p_cl_loss = multiview_n2p_loss(gat_views, total_prototype, pseudo_labels)
            loss += open_and_rate['multi_views_n2p_cl_loss_train_pseOOD'][1] * n2p_cl_loss
            print('multi_views_n2p_cl_loss_train_pseOOD: {:.4f}.'.format(n2p_cl_loss))
        else:
            raise SystemExit('num_views<=1!')

        loss.backward()
        optimizer.step()

        peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        max_peak_mem = max(max_peak_mem, peak_mem)
        print(f"\n>>> Max GPU memory usage across all epochs: {max_peak_mem:.2f} MB")

        epoch_t = time.time() - epoch_t
        total_epoch_time += epoch_t
        print("epoch_time: {:.4f}s".format(epoch_t))

        # Validation
        def valid_loss():
            val_gat_views, val_final_emb, val_output_logits, val_output_prob = model(features, edge_index)
            K_softmax_prob = F.softmax(val_output_logits[:, :-1], dim=1)

            unk_score = calculate_unk_score(K_softmax_prob, val_output_prob, edge_index, neighbor_counts)

            test_unk_scores = unk_score[test_mask].cpu().detach().numpy().reshape(-1, 1)

            ood_kmeans = KMeans(n_clusters=2).fit(test_unk_scores)

            ood_pred = ood_kmeans.predict(test_unk_scores)

            high_score_cluster = 0 if ood_kmeans.cluster_centers_[0] > ood_kmeans.cluster_centers_[1] else 1

            distances_to_center = np.abs(
                test_unk_scores[ood_pred == high_score_cluster] - ood_kmeans.cluster_centers_[high_score_cluster])

            top_10_percent_idx = np.argsort(distances_to_center, axis=0)[
                                 : int(len(distances_to_center) * args.c_rate)].flatten()

            val_pseudo_test_OOD_mask = np.zeros_like(ood_pred, dtype=bool)

            high_score_indices = np.where(ood_pred == high_score_cluster)[0]
            val_pseudo_test_OOD_mask[high_score_indices[top_10_percent_idx]] = True

            val_pseudo_test_OOD_mask = torch.tensor(val_pseudo_test_OOD_mask)

            val_pseudo_OOD_labels = torch.tensor([new_unseen_label] * torch.sum(val_pseudo_test_OOD_mask).item(),
                                                 dtype=torch.int64).to(device)

            low_score_cluster = 1 - high_score_cluster

            distances_to_low_center = np.abs(
                test_unk_scores[ood_pred == low_score_cluster] - ood_kmeans.cluster_centers_[low_score_cluster])

            top_10_percent_idx = np.argsort(distances_to_low_center, axis=0)[
                                 :int(len(distances_to_low_center) * args.c_rate)].flatten()

            val_pseudo_test_ID_mask = np.zeros_like(ood_pred, dtype=bool)

            low_score_indices = np.where(ood_pred == low_score_cluster)[0]
            val_pseudo_test_ID_mask[low_score_indices[top_10_percent_idx]] = True

            val_pseudo_test_ID_mask = torch.tensor(val_pseudo_test_ID_mask)
            val_pseudo_ID_labels = torch.argmax(val_output_prob[:, :-1][test_mask][val_pseudo_test_ID_mask], dim=1).to(
                device)

            val_train_sup_loss = nn.CrossEntropyLoss()(val_output_logits[valid_mask], new_labels[valid_mask])

            p_OOD_emb = val_final_emb[test_mask][val_pseudo_test_OOD_mask]
            val_emb = val_final_emb[valid_mask]
            mixed_OOD_emb, mixed_OOD_labels, _, _, _ = negtive_mix_up(p_OOD_emb, val_emb, val_pseudo_OOD_labels,
                                                                      new_labels[valid_mask], val_output_prob.shape[1])
            mixed_OOD_logits = model.linear_layer(mixed_OOD_emb)
            mixed_OOD_probs = F.softmax(mixed_OOD_logits, dim=1)

            pseOOD_pos_loss = mixed_OOD_labels[:, -1] * (-torch.log(mixed_OOD_probs[:, -1] + 1e-10))
            pseOOD_neg_loss = torch.sum(
                torch.abs(mixed_OOD_labels[:, :-1]) * (-torch.log(1 - mixed_OOD_probs[:, :-1] + 1e-10)), dim=1)
            val_pseOOD_loss = torch.mean(pseOOD_pos_loss + pseOOD_neg_loss)
            # val_pseOOD_loss = torch.mean(pseOOD_pos_loss)
            # val_pseOOD_loss = torch.mean(pseOOD_neg_loss)

            p_ID_emb = val_final_emb[test_mask][val_pseudo_test_ID_mask]
            val_emb = val_final_emb[valid_mask]
            mixed_ID_emb, mixed_ID_labels = normal_mix_up(p_ID_emb, val_emb, val_pseudo_ID_labels,
                                                          new_labels[valid_mask], val_output_prob.shape[1])

            mixed_ID_logits = model.linear_layer(mixed_ID_emb)
            mixed_ID_probs = F.log_softmax(mixed_ID_logits, dim=1)

            val_pseID_loss = torch.mean(torch.sum(mixed_ID_labels * (-mixed_ID_probs), dim=1))

            val_ood_score_loss = ood_score_regularization(unk_score[valid_mask],
                                                          unk_score[test_mask][val_pseudo_test_OOD_mask])

            train_test_mask = train_mask | test_mask

            val_pseudo_labels = torch.zeros_like(new_labels).to(device)
            val_pseudo_labels[valid_mask] = new_labels[valid_mask]
            val_pseudo_labels[train_test_mask] = torch.argmax(val_output_prob[train_test_mask].detach(), dim=1)


            val_total_prototype = generate_cl_prototype(val_gat_views, val_pseudo_labels)

            if num_views > 1:
                val_p2p_cl_loss = multiview_p2p_loss(val_total_prototype)
                val_n2p_cl_loss = multiview_n2p_loss(val_gat_views, val_total_prototype, val_pseudo_labels)
            else:
                raise SystemExit('num_views<=1!')

            val_loss = open_and_rate['train_sup_loss'][1] * val_train_sup_loss + \
                       open_and_rate['pseOOD_loss'][1] * val_pseOOD_loss + \
                       open_and_rate['pseID_loss'][1] * val_pseID_loss + \
                       open_and_rate['multi_views_p2p_cl_loss_train_pseOOD'][1] * val_p2p_cl_loss + \
                       open_and_rate['multi_views_n2p_cl_loss_train_pseOOD'][1] * val_n2p_cl_loss + \
                       open_and_rate['ood_score_loss'][1] * val_ood_score_loss

            return val_loss

        model.eval()
        with torch.no_grad():
            val_loss = valid_loss()
            print('valid_loss: {:.4f}.'.format(val_loss))
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_t = epoch + 1
                torch.save(model.state_dict(), 'best_model.pkl')

    print("total_epoch_time: {:.4f}s".format(total_epoch_time))
    print("average_epoch_time: {:.4f}s".format(total_epoch_time / args.epochs))
    print("ood_number:", ood_number)

    # Testing
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_model.pkl'))
    model.eval()
    with torch.no_grad():
        _, test_emb, test_eval_logits, test_eval_prob = model(features, edge_index)
        pred = torch.argmax(test_eval_prob.detach().cpu(), dim=1).cpu()
        test_accuracy = acc(pred[test_mask].detach().cpu(), new_labels[test_mask].detach().cpu())
        test_seen_accuracy = acc(pred[test_seen_mask].detach().cpu(), new_labels[test_seen_mask].detach().cpu())
        test_unseen_accuracy = acc(pred[test_unseen_mask].detach().cpu(), new_labels[test_unseen_mask].detach().cpu())

        K_softmax_prob = F.softmax(test_eval_logits[:, :-1], dim=1)

        ood_scores = calculate_unk_score(K_softmax_prob, test_eval_prob, edge_index, neighbor_counts)

        auroc = roc_auc_score(detection_y_test, ood_scores[test_mask].cpu().detach())

        precision, recall, _ = precision_recall_curve(detection_y_test, ood_scores[test_mask].cpu().detach())
        aupr = auc(recall, precision)

        ap = average_precision_score(detection_y_test, ood_scores[test_mask].cpu().detach())

        macro_f1 = f1_score(new_labels[test_mask].cpu().detach(), pred[test_mask].cpu().detach(), average="macro")

        fpr, tpr, thresholds = roc_curve(detection_y_test, ood_scores[test_mask].cpu().detach(),
                                         drop_intermediate=False)
        f = fpr[abs((tpr - 0.95)) < 0.005].mean()
        if not np.isnan(f):
            fpr_95 = f
        else:
            fpr_95 = 0.0

    print('acc_test: {:.4f}'.format(test_accuracy),
          'acc_seen_test: {:.4f}'.format(test_seen_accuracy),
          'acc_unseen_test: {:.4f}'.format(test_unseen_accuracy),
          'auroc_test: {:.4f}'.format(auroc),
          'aupr_test: {:.4f}'.format(aupr),
          'ap_test: {:.4f}'.format(ap),
          'macro_f1:{:.4f}'.format(macro_f1),
          'fpr@95:{:.4f}'.format(fpr_95))
    print("======================================================================")

    return [test_accuracy, test_seen_accuracy, test_unseen_accuracy, auroc, aupr, macro_f1, fpr_95, ap]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='negMIX')

    parser.add_argument('--dataset', default="Planetoid_Cora", type=str,
                        help="which dataset to use.")

    parser.add_argument('--ID_classes', nargs='+', default=[0, 1, 2, 3], type=int)
    parser.add_argument('--train_rate', type=float, default=0.1)
    parser.add_argument('--val_rate', type=float, default=0.1)
    # parser.add_argument("--gpu", type=int, default=0,
    #                     help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument('--c_rate', type=float, default=0.1)

    parser.add_argument("--GAT_num_heads", type=int, default=2, help="number of hidden attention heads")
    parser.add_argument("--GAT_num_layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--GAT_num_hidden", type=int, default=16, help="number of hidden units")

    parser.add_argument("--GAT_feat_drop", type=float, default=0.3, help="input feature dropout")
    parser.add_argument("--GAT_attn_drop", type=float, default=0.3, help="attention dropout")
    parser.add_argument('--GAT_negative_slope', type=float, default=0.2, help="the negative slope of leaky relu")

    parser.add_argument("--tau", type=float, default=1, help="temperature_scales")
    parser.add_argument("--seed", type=int, default=123, help="random seed")

    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-3, help="weight decay")

    parser.add_argument('--w_s', type=float, default=1.0)
    parser.add_argument('--w_po', type=float, default=1.0)
    parser.add_argument('--w_pi', type=float, default=0.1)
    parser.add_argument('--w_os', type=float, default=0.1)
    parser.add_argument('--w_p2p', type=float, default=1.0)
    parser.add_argument('--w_n2p', type=float, default=1.0)

    args = parser.parse_args()
    print("args:", args)

    #get data
    dataset_str = args.dataset.split('_')[0]
    trans = []
    if args.dataset not in ['LastFMAsia', 'wiki-CS']:
        trans.append(NormalizeFeatures())
    trans = Compose(trans)

    if dataset_str == 'wiki-CS':
        dataset = WikiCS(root='pyg_data/wiki-CS', transform=trans)
    elif dataset_str == 'LastFMAsia':
        dataset = LastFMAsia(root='pyg_data/LastFMAsia', transform=trans)
    elif dataset_str == 'Amazon':
        dataset_name = args.dataset.split('_')[1]
        dataset = Amazon(root='pyg_data/Amazon', name=dataset_name, transform=trans)
    elif dataset_str == 'Coauthor':
        dataset_name = args.dataset.split('_')[1]
        dataset = Coauthor(root='pyg_data/Coauthor', name=dataset_name, transform=trans)
    elif dataset_str == 'Planetoid':
        dataset_name = args.dataset.split('_')[1]
        dataset = Planetoid(root='pyg_data/Planetoid', name=dataset_name, transform=trans)
    elif dataset_str == 'arxiv-year':
        dataset = load_arxiv_year_dataset()
    else:
        raise Exception('unknown dataset.')

    if dataset_str == 'arxiv-year':
        data = dataset
    else:
        data = dataset[0]

    print(f'Dataset: {dataset}:')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    # print(f'Number of classes: {dataset.num_classes}')
    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print(f'homophily: {homophily(data.edge_index, data.y)}')

    with open("ood_number.txt", "a") as file:
        file.write(f"dataset: {args.dataset}\n")
        file.write(f"c_rate: {args.c_rate}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f = open('results_10%/results_' + args.dataset + '.txt', 'a+')
    f.write('\n\n\n{}\n'.format(args))
    f.flush()

    open_and_rate = {
        "train_sup_loss": [1, args.w_s],
        "pseOOD_loss": [1, args.w_po],
        "pseID_loss": [1, args.w_pi],
        "multi_views_p2p_cl_loss_train_pseOOD": [1, args.w_p2p],
        "multi_views_n2p_cl_loss_train_pseOOD": [1, args.w_n2p],
        "ood_score_loss": [1, args.w_os]
    }

    all_seed = [123, 124, 125, 126, 127, 128, 129, 130, 131, 132]
    all_results = []
    selected_keys = "||".join([f"{value[1]}" for key, value in open_and_rate.items() if value[0] == 1])
    f.write("[" + selected_keys + "]" + "\n")

    seed_t = time.time()
    for i in range(len(all_seed)):
        print(f"Running random select {i + 1}: seed {all_seed[i]}")
        args.seed = all_seed[i]
        results = train(args, data)
        all_results.append(results)
    seed_t = time.time() - seed_t
    print("seed_time: {:.4f}s".format(seed_t))

    mean_all_results = np.mean(np.array(all_results), axis=0)
    std_all_results = np.std(np.array(all_results), axis=0)
    e_t = time.time()

    print("Overall:"
          "test_acc:{:.4f}+/-{:.4f},"
          "test_seen_acc:{:.4f}+/-{:.4f},"
          "test_unseen_acc:{:.4f}+/-{:.4f},"
          "auroc_test:{:.4f}+/-{:.4f},"
          "aupr_test:{:.4f}+/-{:.4f},"
          "macro_f1:{:.4f}+/-{:.4f},"
          "fpr@95:{:.4f}+/-{:.4f},"
          "ap:{:.4f}+/-{:.4f}\n".format(mean_all_results[0], std_all_results[0], mean_all_results[1],
                                        std_all_results[1], mean_all_results[2], std_all_results[2],
                                        mean_all_results[3], std_all_results[3], mean_all_results[4],
                                        std_all_results[4], mean_all_results[5], std_all_results[5],
                                        mean_all_results[6], std_all_results[6], mean_all_results[7],
                                        std_all_results[7]))
    f.write("test_acc:{:.4f}+/-{:.4f},"
            "test_seen_acc:{:.4f}+/-{:.4f},"
            "test_unseen_acc:{:.4f}+/-{:.4f},"
            "auroc_test:{:.4f}+/-{:.4f},"
            "aupr_test:{:.4f}+/-{:.4f},"
            "macro_f1:{:.4f}+/-{:.4f},"
            "fpr@95:{:.4f}+/-{:.4f},"
            "ap:{:.4f}+/-{:.4f}\n".format(mean_all_results[0], std_all_results[0], mean_all_results[1],
                                          std_all_results[1], mean_all_results[2], std_all_results[2],
                                          mean_all_results[3], std_all_results[3], mean_all_results[4],
                                          std_all_results[4], mean_all_results[5], std_all_results[5],
                                          mean_all_results[6], std_all_results[6], mean_all_results[7],
                                          std_all_results[7]))
