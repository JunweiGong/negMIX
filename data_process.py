from copy import deepcopy
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def data_process(data, ID_classes, train_rate, valid_rate, random_seed=123):
    seed = np.random.get_state()
    np.random.seed(random_seed)

    classes = set(data.y.tolist())
    n_classes = len(classes)
    n_nodes = len(data.y)

    seen_label = set(ID_classes)
    unseen_label = classes - seen_label
    new_unseen_label = len(seen_label)

    # re-map labels
    label_map = {y: i for i, y in enumerate(ID_classes)}

    new_labels = deepcopy(data.y)

    for i in range(n_nodes):
        new_labels[i] = label_map.get(data.y[i].item(), new_unseen_label)  # all labels for ID(0~ID_class_num-1) and OOD(ID_class_num)

    seen_indices = np.where(new_labels != new_unseen_label)[0]
    unseen_indices = np.where(new_labels == new_unseen_label)[0]

    seen_train_indices, seen_test_indices = train_test_split(seen_indices, test_size=1 - train_rate,
                                                             random_state=random_seed)
    train_indices = seen_train_indices

    test_seen_indices, valid_indices= train_test_split(seen_test_indices,
                                                        test_size=valid_rate / (1 - train_rate),
                                                        random_state=random_seed)

    test_indices = np.concatenate([test_seen_indices, unseen_indices], axis=0)


    expected_labels = torch.unique(new_labels)[:-1]
    assert all(label in new_labels[train_indices] for label in expected_labels), "new_labels[train_indices] does not contain all expected categories"

    train_mask, valid_mask, test_mask, test_seen_mask, test_unseen_mask = torch.zeros(n_nodes), torch.zeros(n_nodes), torch.zeros(n_nodes), torch.zeros(n_nodes), torch.zeros(n_nodes)
    ID_mask, OOD_mask = torch.zeros(n_nodes), torch.zeros(n_nodes)

    train_mask[train_indices] = 1
    valid_mask[valid_indices] = 1
    test_mask[test_indices] = 1
    test_seen_mask[test_seen_indices] = 1
    test_unseen_mask[unseen_indices] = 1
    ID_mask[seen_indices] = 1
    OOD_mask[unseen_indices] = 1

    train_mask, valid_mask, test_mask, test_seen_mask, test_unseen_mask = train_mask.bool(), valid_mask.bool(), test_mask.bool(), test_seen_mask.bool(), test_unseen_mask.bool()
    ID_mask, OOD_mask = ID_mask.bool(), OOD_mask.bool()

    detection_y_test = new_labels[test_mask]
    detection_y_test = [y == new_unseen_label for y in detection_y_test]
    # detection_y_test: detect the OOD nodes in test dataset

    joint_y_test = new_labels[test_mask]
    # joint_y_test: test dataset labels for ID(0~ID_class-1) and OOD(ID_classes)

    np.random.set_state(seed)

    return ID_mask, OOD_mask, train_mask, valid_mask, test_mask, test_seen_mask, test_unseen_mask, \
        detection_y_test, joint_y_test, new_labels, new_unseen_label