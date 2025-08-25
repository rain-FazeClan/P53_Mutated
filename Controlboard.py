import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
import matplotlib.pyplot as plt
import Data_Loader as data_loader
from EfficientNet import EfficientNet
from SwinTransformer import SwinTransformer
from ConvNext import ConvNeXt
from ResNet import ResNet
from Clinical import Clinical, CLINICAL_FEATURES, extract_clinical_features
import pandas as pd
from Augmentation import transforms

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced datasets"""

    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiModalModel(nn.Module):
    """融合影像和临床特征的多模态模型"""

    def __init__(self, image_model, clinical_features_dim, num_classes=1):
        super(MultiModalModel, self).__init__()
        self.image_model = image_model
        self.clinical_model = Clinical(clinical_features_dim, hidden_dims=[64, 32], num_classes=64)

        # 获取影像模型的特征维度
        image_features_dim = 512  # 默认值
        if hasattr(image_model, 'classifier'):
            # 移除原有分类器，获取特征
            if isinstance(image_model.classifier, nn.Sequential):
                for layer in reversed(image_model.classifier):
                    if isinstance(layer, nn.Linear):
                        image_features_dim = layer.in_features
                        break
            else:
                image_features_dim = image_model.classifier.in_features

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(image_features_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, image_data, clinical_data=None):
        # 提取影像特征
        if hasattr(self.image_model, 'backbone'):
            image_features = self.image_model.backbone(image_data)
        else:
            # 临时移除分类器来获取特征
            image_features = self.image_model(image_data)

        if clinical_data is not None:
            # 提取临床特征
            clinical_features = self.clinical_model(clinical_data)
            # 融合特征
            combined_features = torch.cat([image_features, clinical_features], dim=1)
        else:
            combined_features = image_features

        return self.fusion(combined_features)


def data_transforms(augment_mode):
    """获取数据变换"""

    # 使用基于torchvision的增强
    if augment_mode == 'full':
        train_transform = transforms('full',
                                   p_flip=0.5,
                                   degrees=15,
                                   p_rotate=0.5,
                                   translate=(0.1, 0.1),
                                   scale=(0.9, 1.1),
                                   p_affine=0.3,
                                   noise_std=0.05,
                                   p_noise=0.2,
                                   brightness_factor=0.2,
                                   p_brightness=0.3,
                                   contrast_factor=0.2,
                                   p_contrast=0.3)
    elif augment_mode == 'simple':
        train_transform = transforms('simple',
                                   p_flip=0.5,
                                   degrees=10,
                                   p_rotate=0.3,
                                   translate=(0.05, 0.05),
                                   scale=(0.95, 1.05),
                                   p_affine=0.2)
    else:
        train_transform = None


    # 验证集不使用增强
    val_transform = None

    return train_transform, val_transform


def create_model(model_type, model_name, num_classes, in_chans, pretrained, use_clinical, clinical_features_dim):
    """根据参数创建模型"""
    if model_type.lower() == 'efficientnet':
        base_model = EfficientNet(model_name=model_name, num_classes=num_classes,
                                  in_chans=in_chans, pretrained=pretrained)
    elif model_type.lower() == 'resnet':
        base_model = ResNet(model_name=model_name, num_classes=num_classes,
                            in_chans=in_chans, pretrained=pretrained)
    elif model_type.lower() == 'convnext':
        base_model = ConvNeXt(model_name=model_name, num_classes=num_classes,
                              in_chans=in_chans, pretrained=pretrained)
    elif model_type.lower() == 'swin':
        base_model = SwinTransformer(model_name=model_name, num_classes=num_classes,
                                     in_chans=in_chans, pretrained=pretrained)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    if use_clinical:
        model = MultiModalModel(base_model, clinical_features_dim, num_classes=1)
    else:
        model = base_model

    return model


def train(model, train_loader, val_loader, model_type, epochs=50, learning_rate=1e-4, weight_decay=1e-5,
          early_stopping_patience=8, resume=True, focal_alpha=0.75, focal_gamma=2.0,
          use_clinical=False, train_clinical_data=None, val_clinical_data=None, use_amp=True):
    model.to(device)
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 初始化混合精度训练
    scaler = GradScaler() if use_amp else None

    best_val_auc = 0.0
    start_epoch = 0

    # 根据模型类型定义保存文件名
    model_save_path = f"{model_type}_best_model.pth"
    epoch_file = f"{model_type}_last_epoch.txt"

    # 尝试加载已有权重和已训练轮数
    if resume and os.path.exists(model_save_path):
        print(f"Found existing {model_type} model weights. Loading...")
        model.load_state_dict(torch.load(model_save_path))
        if os.path.exists(epoch_file):
            with open(epoch_file, "r") as f:
                start_epoch = int(f.read().strip())
            print(f"Resuming from epoch {start_epoch + 1}")
        else:
            print(f"No {epoch_file} found, resuming from epoch 1")

    epochs_no_improve = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()

            # 混合精度训练
            if use_amp:
                with torch.amp.autocast('cuda'):
                    if use_clinical and train_clinical_data is not None:
                        # 获取当前batch对应的临床特征
                        batch_size = inputs.size(0)
                        start_idx = batch_idx * batch_size
                        end_idx = start_idx + batch_size
                        clinical_batch = torch.tensor(train_clinical_data[start_idx:end_idx]).to(device)
                        outputs = model(inputs, clinical_batch)
                    else:
                        outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if use_clinical and train_clinical_data is not None:
                    # 获取当前batch对应的临床特征
                    batch_size = inputs.size(0)
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_size
                    clinical_batch = torch.tensor(train_clinical_data[start_idx:end_idx]).to(device)
                    outputs = model(inputs, clinical_batch)
                else:
                    outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

        val_auc, _, _, _, _, _ = evaluate(model, val_loader, criterion, use_clinical, val_clinical_data, use_amp)
        print(f"Epoch [{epoch + 1}/{epochs}], Validation AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), model_save_path)
            print(f"New best {model_type} model saved with AUC: {best_val_auc:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")

        # 保存当前已训练轮数
        with open(epoch_file, "w") as f:
            f.write(str(epoch + 1))

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs. Best AUC: {best_val_auc:.4f}")
            break

    print("Training finished.")
    return model_save_path


def evaluate(model, data_loader, criterion=None, use_clinical=False, clinical_data=None, use_amp=True):
    model.to(device)
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    if use_clinical and clinical_data is not None:
                        # 获取当前batch对应的临床特征
                        batch_size = inputs.size(0)
                        start_idx = batch_idx * batch_size
                        end_idx = start_idx + batch_size
                        clinical_batch = torch.tensor(clinical_data[start_idx:end_idx]).to(device)
                        outputs = model(inputs, clinical_batch)
                    else:
                        outputs = model(inputs)

                    if criterion:
                        loss = criterion(outputs, labels)
                        total_loss += loss.item()
            else:
                if use_clinical and clinical_data is not None:
                    # 获取当前batch对应的临床特征
                    batch_size = inputs.size(0)
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_size
                    clinical_batch = torch.tensor(clinical_data[start_idx:end_idx]).to(device)
                    outputs = model(inputs, clinical_batch)
                else:
                    outputs = model(inputs)

                if criterion:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    preds_binary = (all_preds > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds_binary, average='binary',
                                                               zero_division=0)
    accuracy = np.mean(preds_binary == all_labels)
    avg_loss = total_loss / len(data_loader) if criterion else 0

    return roc_auc, accuracy, precision, recall, fpr, tpr


def plot_roc_curve(fpr, tpr, roc_auc, model_name):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f"roc_curve_{model_name.lower()}.png")
    print(f"ROC curve saved as roc_curve_{model_name.lower()}.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train deep learning models for HCC p53 prediction')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='efficientnet',
                        choices=['efficientnet', 'resnet', 'convnext', 'swin'],
                        help='Type of model to use')
    parser.add_argument('--model_name', type=str, default='tf_efficientnet_b0',
                        help='Specific model name')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--in_chans', type=int, default=6, help='Number of input channels')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights')

    # 临床特征融合
    parser.add_argument('--use_clinical', action='store_true', default=False,
                        help='Whether to use clinical features')

    # Focal Loss参数
    parser.add_argument('--focal_alpha', type=float, default=0.75, help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma parameter')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--early_stopping_patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--resume', type=bool, default=True, help='Resume training from checkpoint')

    # 数据增强参数
    parser.add_argument('--augment_mode', type=str, default='simple',
                        choices=['full', 'simple', 'none'],
                        help='3D augmentation mode')

    # 混合精度训练参数
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use Automatic Mixed Precision training')

    args = parser.parse_args()

    print(f"使用模型: {args.model_type} - {args.model_name}")
    print(f"融合临床特征: {args.use_clinical}")
    print(f"使用混合精度训练: {args.use_amp}")
    print(f"Focal Loss参数: alpha={args.focal_alpha}, gamma={args.focal_gamma}")

    # 获取数据增强变换
    train_transform, val_transform = data_transforms(args.augment_mode)

    print("Loading data manifest...")
    all_data = data_loader.load_data_manifest()

    print("Splitting data...")
    train_data, val_data = data_loader.split_data(all_data, test_size=0.3, random_state=42)

    # 准备临床特征数据
    train_clinical_data = None
    val_clinical_data = None
    clinical_features_dim = 0

    if args.use_clinical:
        print("Loading clinical features...")
        train_clinical_data = extract_clinical_features(train_data, CLINICAL_FEATURES)
        val_clinical_data = extract_clinical_features(val_data, CLINICAL_FEATURES)

        if train_clinical_data is not None:
            clinical_features_dim = train_clinical_data.shape[1]
            print(f"临床特征维度: {clinical_features_dim}")
        else:
            print("警告: 无法加载临床特征，将只使用影像数据")
            args.use_clinical = False

    print("Creating datasets and dataloaders...")
    target_channels = args.in_chans
    train_dataset = data_loader.PreprocessedDataset(train_data, target_channels=target_channels,
                                                    transform=train_transform)
    val_dataset = data_loader.PreprocessedDataset(val_data, target_channels=target_channels, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    print(f"Initializing {args.model_type} model...")
    model = create_model(args.model_type, args.model_name, args.num_classes, args.in_chans,
                         args.pretrained, args.use_clinical, clinical_features_dim)

    print("Starting training...")
    model_save_path = train(model, train_loader, val_loader, args.model_type, epochs=args.epochs,
                            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                            early_stopping_patience=args.early_stopping_patience, resume=args.resume,
                            focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
                            use_clinical=args.use_clinical, train_clinical_data=train_clinical_data,
                            val_clinical_data=val_clinical_data, use_amp=args.use_amp)

    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(model_save_path))
    roc_auc, accuracy, precision, recall, fpr, tpr = evaluate(model, val_loader,
                                                              use_clinical=args.use_clinical,
                                                              clinical_data=val_clinical_data,
                                                              use_amp=args.use_amp)

    print("\n--- Final Evaluation on Validation Set ---")
    print(f"Model: {args.model_type} - {args.model_name}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    plot_roc_curve(fpr, tpr, roc_auc, f"{args.model_type}_{args.model_name}")


if __name__ == "__main__":
    main()