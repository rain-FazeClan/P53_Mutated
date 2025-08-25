import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

# --- Configuration ---
PREPROCESSED_DIR = "preprocessed"
METADATA_FILE = "HCC_meta.xlsx"
MODALITIES = ["A", "DWI", "HBP", "P", "T1", "T2"]


class PreprocessedDataset(Dataset):
    """PyTorch Dataset for loading preprocessed medical images from .npy files."""

    def __init__(self, data_manifest, transform=None, target_channels=6):
        """
        Args:
            data_manifest (list): List of dictionaries containing patient data
            transform (callable, optional): Optional transform to be applied on the data
        """
        self.data_manifest = data_manifest
        self.transform = transform
        self.target_channels = target_channels

        if len(data_manifest) > 0:
            first_sample = np.load(data_manifest[0]['npy_path'])
            self.actual_channels = first_sample.shape[0]
            if self.target_channels is None:
                self.target_channels = self.actual_channels
                print(f"自动确定目标通道数: {self.target_channels}")
            elif self.actual_channels != self.target_channels:
                print(f"警告: 第一个样本的通道数 ({self.actual_channels}) 与目标通道数 ({self.target_channels}) 不一致")

    def __len__(self):
        return len(self.data_manifest)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (image, label) where image is preprocessed multi-channel 2D data
                  and label is the corresponding P53 status (0=Wild-type, 1=Mutated)
        """
        item = self.data_manifest[idx]
        npy_path = item['npy_path']
        img_data = np.load(npy_path)

        if img_data.shape[0] != self.target_channels:
            if img_data.shape[0] < self.target_channels:
                padded = np.zeros((self.target_channels, *img_data.shape[1:]), dtype=img_data.dtype)
                padded[:img_data.shape[0]] = img_data
                img_data = padded
            else:
                img_data = img_data[:self.target_channels]

        img_tensor = torch.from_numpy(img_data).float()

        if self.transform:
            img_tensor = self.transform(img_tensor)

        # 获取标签 (0=野生型, 1=突变型)
        label = torch.tensor(item['label'], dtype=torch.float32)

        return img_tensor, label


def find_preprocessed_file(patient_id, preprocessed_dir=PREPROCESSED_DIR):
    """查找预处理文件，支持单肿瘤和多肿瘤格式"""

    # 首先查找单肿瘤格式: {patient_id}.npy
    npy_path = os.path.join(preprocessed_dir, f"{patient_id}.npy")
    if os.path.exists(npy_path):
        return [npy_path]

    # 然后查找多肿瘤格式: {patient_id}_tumor{i}.npy
    tumor_files = []
    for i in range(10):  # 支持最多10个肿瘤
        tumor_path = os.path.join(preprocessed_dir, f"{patient_id}_tumor{i}.npy")
        if os.path.exists(tumor_path):
            tumor_files.append(tumor_path)

    return tumor_files if tumor_files else None


def load_data_manifest(metadata_path=METADATA_FILE, preprocessed_dir=PREPROCESSED_DIR, include_multiple_tumors=True):
    """加载数据清单"""

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"元数据文件未找到: {metadata_path}")
    if not os.path.exists(preprocessed_dir):
        raise FileNotFoundError(f"预处理数据目录未找到: {preprocessed_dir}")

    # 读取Excel文件
    metadata = pd.read_excel(metadata_path)

    all_patient_data = []
    missing_npy_files = []
    multiple_tumor_count = 0

    for _, row in metadata.iterrows():
        # 使用"住院号"列作为患者ID
        patient_id = str(row['住院号'])

        # 直接读取p53_status列
        p53_status = row.get('p53_status', None)
        if p53_status == "Mutated":
            label = 1
        elif p53_status == "Wild-type":
            label = 0
        else:
            label = 'nan'

        # 查找预处理的npy文件
        npy_files = find_preprocessed_file(patient_id, preprocessed_dir)

        if npy_files:
            if len(npy_files) == 1:
                patient_data = {
                    'id': patient_id,
                    'label': label,
                    'p53_status': p53_status,
                    'npy_path': npy_files[0],
                    'tumor_index': 0,  # 更新索引从0开始，与新命名格式一致
                    'total_tumors': 1,
                    'sample_id': patient_id
                }

                patient_data['sex'] = int(row['性别（男=1，女=2）']) if not pd.isna(row['性别（男=1，女=2）']) else None
                patient_data['tumor_size'] = row['MR最大径'] if not pd.isna(row['MR最大径']) else None
                patient_data['pathological_grade'] = row['病理分级【0:I-II级;1:III-IV级】'] if not pd.isna(
                    row['病理分级【0:I-II级;1:III-IV级】']) else None

                all_patient_data.append(patient_data)
                print(f"找到病例 {patient_id}: p53_status '{p53_status}' (label={label})")

            else:
                multiple_tumor_count += 1
                print(
                    f"找到多发肿瘤病例 {patient_id}: {len(npy_files)}个肿瘤, p53_status '{p53_status}' (label={label})")

                if include_multiple_tumors:
                    for tumor_idx, npy_path in enumerate(npy_files):
                        patient_data = {
                            'id': patient_id,
                            'label': label,
                            'p53_status': p53_status,
                            'npy_path': npy_path,
                            'tumor_index': tumor_idx,  # 索引从0开始
                            'total_tumors': len(npy_files),
                            'sample_id': f"{patient_id}_tumor{tumor_idx}"
                        }

                        patient_data['sex'] = int(row['性别（男=1，女=2）']) if not pd.isna(
                            row['性别（男=1，女=2）']) else None
                        patient_data['tumor_size'] = row['MR最大径'] if not pd.isna(row['MR最大径']) else None
                        patient_data['pathological_grade'] = row['病理分级【0:I-II级;1:III-IV级】'] if not pd.isna(
                            row['病理分级【0:I-II级;1:III-IV级】']) else None

                        all_patient_data.append(patient_data)
                        print(f"  - 肿瘤{tumor_idx}: {os.path.basename(npy_path)}")
                else:
                    patient_data = {
                        'id': patient_id,
                        'label': label,
                        'p53_status': p53_status,
                        'npy_path': npy_files[0],
                        'tumor_index': 0,
                        'total_tumors': len(npy_files),
                        'sample_id': f"{patient_id}_tumor0"
                    }

                    patient_data['sex'] = int(row['性别（男=1，女=2）']) if not pd.isna(row['性别（男=1，女=2）']) else None
                    patient_data['tumor_size'] = row['MR最大径'] if not pd.isna(row['MR最大径']) else None
                    patient_data['pathological_grade'] = row['病理分级【0:I-II级;1:III-IV级】'] if not pd.isna(
                        row['病理分级【0:I-II级;1:III-IV级】']) else None

                    all_patient_data.append(patient_data)
                    print(f"  - 仅使用肿瘤0: {os.path.basename(npy_files[0])}")
        else:
            missing_npy_files.append(patient_id)
            print(f"跳过病例 {patient_id}: 未找到对应的预处理npy文件")

    print(f"\n数据加载统计:")
    print(f"总样本数: {len(all_patient_data)}")
    print(f"多发肿瘤病例: {multiple_tumor_count}")

    mutated_count = sum(1 for p in all_patient_data if p['label'] == 1)
    wildtype_count = sum(1 for p in all_patient_data if p['label'] == 0)
    uncertain_count = sum(1 for p in all_patient_data if p['label'] == 'nan')

    total = len(all_patient_data)
    if total > 0:
        print(f"标签分布:")
        print(f"  突变型: {mutated_count} ({mutated_count / total * 100:.1f}%)")
        print(f"  野生型: {wildtype_count} ({wildtype_count / total * 100:.1f}%)")
        if uncertain_count > 0:
            print(f"  不确定: {uncertain_count} ({uncertain_count / total * 100:.1f}%)")

    # 输出缺失文件
    if missing_npy_files:
        print(f"\n缺失预处理文件的病例 ({len(missing_npy_files)}): {missing_npy_files[:10]}...")

    return all_patient_data


def split_data(all_patient_data, test_size=0.3, random_state=42, split_by_patient=True):
    """分割数据，支持按病例分割或按样本分割"""
    if not all_patient_data:
        raise ValueError("No patient data loaded to split.")

    # 过滤掉不确定标签的患者
    certain_patients = [p for p in all_patient_data if p['label'] != 'nan']
    uncertain_patients = [p for p in all_patient_data if p['label'] == 'nan']

    if uncertain_patients:
        print(f"注意: {len(uncertain_patients)} 个样本标签不确定，已从训练和验证集中排除")

    if not certain_patients:
        raise ValueError("没有找到有确定标签的患者数据，无法进行分割")

    if split_by_patient:
        # 按病例分割，避免同一病例的多个肿瘤分散到训练集和验证集
        unique_patient_ids = list(set(p['id'] for p in certain_patients))
        patient_labels = []

        for pid in unique_patient_ids:
            # 取该病例的第一个样本的标签（多发肿瘤共享同一标签）
            patient_label = next(p['label'] for p in certain_patients if p['id'] == pid)
            patient_labels.append(patient_label)

        train_patient_ids, val_patient_ids = train_test_split(
            unique_patient_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=patient_labels
        )

        train_data = [p for p in certain_patients if p['id'] in train_patient_ids]
        val_data = [p for p in certain_patients if p['id'] in val_patient_ids]

        print(f"按病例分割: {len(train_patient_ids)} 病例 → {len(train_data)} 训练样本")
        print(f"           {len(val_patient_ids)} 病例 → {len(val_data)} 验证样本")

        # 统计多发肿瘤分布
        train_multi_patients = len(set(p['id'] for p in train_data if p['total_tumors'] > 1))
        val_multi_patients = len(set(p['id'] for p in val_data if p['total_tumors'] > 1))
        if train_multi_patients > 0 or val_multi_patients > 0:
            print(f"多发肿瘤分布: 训练集{train_multi_patients}病例, 验证集{val_multi_patients}病例")

    else:
        # 按样本分割（原有逻辑）
        sample_ids = list(range(len(certain_patients)))
        labels = [p['label'] for p in certain_patients]

        train_indices, val_indices = train_test_split(
            sample_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )

        train_data = [certain_patients[i] for i in train_indices]
        val_data = [certain_patients[i] for i in val_indices]

        print(f"按样本分割: {len(train_data)} 训练样本, {len(val_data)} 验证样本")

        # 警告可能的数据泄露
        train_patient_ids = set(p['id'] for p in train_data)
        val_patient_ids = set(p['id'] for p in val_data)
        overlap = train_patient_ids & val_patient_ids
        if overlap:
            print(f"⚠️  警告: {len(overlap)} 个病例的样本同时出现在训练集和验证集中，可能存在数据泄露")
            print(f"重叠病例: {list(overlap)[:5]}...")

    # 统计标签分布
    train_labels = [p['label'] for p in train_data]
    val_labels = [p['label'] for p in val_data]
    print(
        f"训练集标签: {sum(train_labels)} 突变 / {len(train_labels)} 总数 ({sum(train_labels) / len(train_labels) * 100:.1f}%)")
    print(
        f"验证集标签: {sum(val_labels)} 突变 / {len(val_labels)} 总数 ({sum(val_labels) / len(val_labels) * 100:.1f}%)")

    return train_data, val_data


def get_data_statistics(all_patient_data):
    """获取数据集统计信息"""
    if not all_patient_data:
        return

    print(f"\n=== 数据集统计信息 ===")

    # 基本统计
    total_samples = len(all_patient_data)
    unique_patients = len(set(p['id'] for p in all_patient_data))

    print(f"总样本数: {total_samples}")
    print(f"总病例数: {unique_patients}")
    print(f"平均每病例样本数: {total_samples / unique_patients:.2f}")

    # 肿瘤数量分布
    tumor_counts = {}
    for p in all_patient_data:
        patient_id = p['id']
        if patient_id not in tumor_counts:
            tumor_counts[patient_id] = p['total_tumors']

    single_tumor_patients = sum(1 for count in tumor_counts.values() if count == 1)
    multi_tumor_patients = sum(1 for count in tumor_counts.values() if count > 1)

    print(f"\n肿瘤分布:")
    print(f"  单发肿瘤病例: {single_tumor_patients} ({single_tumor_patients / unique_patients * 100:.1f}%)")
    print(f"  多发肿瘤病例: {multi_tumor_patients} ({multi_tumor_patients / unique_patients * 100:.1f}%)")

    if multi_tumor_patients > 0:
        max_tumors = max(tumor_counts.values())
        print(f"  最大肿瘤数: {max_tumors}")

        # 详细分布
        for tumor_num in range(2, max_tumors + 1):
            count = sum(1 for c in tumor_counts.values() if c == tumor_num)
            if count > 0:
                print(f"  {tumor_num}个肿瘤: {count}病例")

    # 标签分布
    mutated = sum(1 for p in all_patient_data if p['label'] == 1)
    wildtype = sum(1 for p in all_patient_data if p['label'] == 0)
    uncertain = sum(1 for p in all_patient_data if p['label'] == 'nan')

    print(f"\n标签分布:")
    print(f"  突变型样本: {mutated} ({mutated / total_samples * 100:.1f}%)")
    print(f"  野生型样本: {wildtype} ({wildtype / total_samples * 100:.1f}%)")
    if uncertain > 0:
        print(f"  不确定样本: {uncertain} ({uncertain / total_samples * 100:.1f}%)")


# 测试函数
if __name__ == "__main__":
    print("\n=== 测试1: 包含所有肿瘤样本 ===")
    all_data = load_data_manifest(include_multiple_tumors=True)
    get_data_statistics(all_data)

    if all_data:
        train_data, val_data = split_data(all_data, split_by_patient=True)

        # 创建数据集
        train_dataset = PreprocessedDataset(train_data)
        val_dataset = PreprocessedDataset(val_data)

        print(f"\n训练数据集: {len(train_dataset)} 样本")
        print(f"验证数据集: {len(val_dataset)} 样本")

        # 测试加载第一个样本
        if len(train_dataset) > 0:
            img, label = train_dataset[0]
            print(f"样本形状: {img.shape}, 标签: {label}")

    print("\n" + "=" * 50)