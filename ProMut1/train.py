import csv
import torch
import torch.nn as nn
import torch.optim as optim
from model_promut1 import ProMutBackbone
import numpy as np
from datasets import NpzDataset
import torch.utils.data as Data
import os
from torch.optim import lr_scheduler
from tqdm import tqdm
from path_config import PRETRAIN_TEST_DATA, PRETRAIN_TEST_LABEL, PRETRAIN_TRAIN_DATA_DIR, PROJECT_ROOT


def test_model(loader_, model_, device_):
    print("testing...")
    model_.eval()
    with torch.no_grad():
        correct_ = 0
        total_ = 0
        for data_ in tqdm(loader_):
            inputs_, labels_ = data_
            inputs_, labels_ = inputs_.to(device_), labels_.to(device_)
            labels_ = labels_.long()
            outputs_ = model_(inputs_.float())
            _, predicted_ = torch.max(outputs_.data, 1)
            total_ += labels_.size(0)
            correct_ += (predicted_ == labels_).sum().item()
    return 100 * correct_ / total_


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 150
    # 实例化模型和损失函数、优化器
    model = ProMutBackbone(in_chans=7, num_classes=20, use_roi_pooling=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.001)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    save_dir = os.path.join(PROJECT_ROOT, 'model_save', 'EMOCPD_VoxelRoI')
    os.makedirs(save_dir, exist_ok=True)

    epochs = 8
    iteration = 0
    data_root = PRETRAIN_TRAIN_DATA_DIR, PROJECT_ROOT

    test_data_file = PRETRAIN_TEST_DATA
    test_label_file = PRETRAIN_TEST_LABEL

    test_loader = None
    if os.path.exists(test_data_file) and os.path.exists(test_label_file):
        test_data_set = NpzDataset(test_data_file, test_label_file)

        test_loader = Data.DataLoader(
        # 从数据库中每次抽出batch size个样本
            dataset=test_data_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    best_acc = 0
    epoch = 0

    while epoch < epochs:
        print(f'training epoch {epoch + 1}' + '\n')
        k = 0
        while k < 20:

            running_loss = 0.0
            correct = 0
            total = 0
            patience = 0

            data_file = os.path.join(data_root, 'data_' + str(k) + '.npz')
            label_file = os.path.join(data_root, 'label_' + str(k) + '.npz')

            data_set = NpzDataset(data_file, label_file)

            loader = Data.DataLoader(
                dataset=data_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
            )
            for data in tqdm(loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.long()

                optimizer.zero_grad()
                outputs = model(inputs.float())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                iteration += 1
                if iteration % 100 == 0:
                    acc_temp = test_model(test_loader, model, device) if test_loader is not None else 0
                    model.train()
                    print("test acc: ", acc_temp)
                    with open(os.path.join(save_dir, 'train_logs.csv'), 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([iteration + 1, running_loss / total, correct / total, acc_temp])

                    if acc_temp > best_acc:
                        best_acc = acc_temp
                        state = {
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_acc': best_acc,
                            'k': k
                        }
                        torch.save(state, os.path.join(save_dir, 'best' + str(int(best_acc * 100)) + 'model.pth.tar'))
                        print('model saved' + ", acc = " + str(best_acc))
            del loader
            k += 1
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
            'k': k
        }
        torch.save(state, os.path.join(save_dir, 'train_epoch_' + str(int(epoch + 1)) + '_model.pth.tar'))
        epoch += 1
