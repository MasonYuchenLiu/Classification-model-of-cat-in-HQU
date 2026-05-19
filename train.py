# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
import os
from tqdm import tqdm
from data_loader import get_dataloaders
from model import CatCNN
from config import DATA_ROOT, DEVICE, NUM_EPOCHS, LR, MODEL_SAVE_PATH, HISTORY_SAVE_PATH

def train():
    # ===== 获取 DataLoader =====
    train_loader, val_loader, class_to_idx = get_dataloaders(DATA_ROOT)
    num_classes = len(class_to_idx)

    # ===== 构建模型 =====
    model = CatCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # ===== 自动检测已有模型 =====
    start_epoch = 0
    best_val_acc = 0.0
    patience = 3
    patience_counter = 0

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"检测到已有模型 {MODEL_SAVE_PATH}，加载并继续训练...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        if os.path.exists(HISTORY_SAVE_PATH):
            with open(HISTORY_SAVE_PATH, 'r') as f:
                history = json.load(f)
            start_epoch = len(history['train_loss'])
            if history['val_acc']:
                best_val_acc = max(history['val_acc'])
        print(f"从第 {start_epoch+1} 轮开始训练，当前最佳验证准确率: {best_val_acc:.4f}")
    else:
        print("未检测到已有模型，将从头训练")

    # ===== 开始训练 =====
    for epoch in range(start_epoch, NUM_EPOCHS):
        start_time = time.time()

        # ===== 训练 =====
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]", ncols=100)
        for images, labels in train_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_bar.set_postfix({
                'Loss': f"{running_loss/total:.4f}",
                'Acc': f"{correct/total:.4f}"
            })

        train_loss = running_loss / total
        train_acc = correct / total

        # ===== 验证 =====
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]", ncols=100)
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_bar.set_postfix({
                    'Loss': f"{val_loss_sum/val_total:.4f}",
                    'Acc': f"{val_correct/val_total:.4f}"
                })

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} "
              f"Time: {epoch_time:.2f}s")

        # ===== 保存历史 =====
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        with open(HISTORY_SAVE_PATH, 'w') as f:
            json.dump(history, f)

        # ===== 检查验证集准确率是否提升 =====
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"验证集准确率提升至 {best_val_acc:.4f}，已保存模型")
            patience_counter = 0  # 重置耐心计数器
        else:
            patience_counter += 1
            print(f"验证集准确率未提升，耐心计数器: {patience_counter}/{patience}")

        # ===== 提前停止判断 =====
        if patience_counter >= patience:
            print(f"连续 {patience} 轮验证准确率未提升，训练提前停止")
            break

    # 保存 class_to_idx
    with open("class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f)

    print("训练结束，最终模型和历史已保存。")

if __name__ == "__main__":
    train()