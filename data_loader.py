import os
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import random

# --- Configuration ---
# 注释掉TCIA相关配置
# TCIA_DATA_DIR = "path/to/your/tcia_data"
LOCAL_DATA_DIR = "F:/GXMU_HCC/HCC_1"
METADATA_FILE = "F:/小肝癌初筛_BH列"  # 元数据文件路径

# 更新为6种影像模态
MODALITIES = ["A", "DWI", "HBP", "P", "T1", "T2"]
# 更新分割文件后缀
SEG_SUFFIX = "_seg.nii.gz"  # 每个模态对应的分割标注都是 模态名_seg.nii.gz
# --- ---

def find_patient_files(patient_id, data_source="Local"):
    """Finds all required MRI and segmentation files for a patient."""
    # 只使用本地数据路径
    base_dir = LOCAL_DATA_DIR
    patient_files = {'id': patient_id, 'source': data_source}
    valid = True
    
    # 病人文件夹路径
    patient_dir = os.path.join(base_dir, patient_id)
    if not os.path.exists(patient_dir):
        print(f"Warning: Patient directory not found for {patient_id}")
        return None
        
    for mod in MODALITIES:
        # 查找影像文件
        mod_file = os.path.join(patient_dir, f"{mod}.nii.gz")
        if os.path.exists(mod_file):
            patient_files[mod] = mod_file
        else:
            print(f"Warning: Missing {mod} file for patient {patient_id}")
            valid = False
            break  # 如果缺少模态则跳过该患者
            
        # 查找对应的分割文件
        seg_file = os.path.join(patient_dir, f"{mod}{SEG_SUFFIX}")
        if os.path.exists(seg_file):
            patient_files[f'{mod}_seg'] = seg_file
        else:
            print(f"Warning: Missing {mod} segmentation for patient {patient_id}")
            patient_files[f'{mod}_seg'] = None  # 标记为缺失
    
    return patient_files if valid else None

def load_data_manifest(metadata_path=METADATA_FILE):
    """Loads patient metadata and finds corresponding files."""
    if not os.path.exists(metadata_path + ".xlsx"):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}.xlsx")

    # 读取Excel文件，指定文件后缀
    metadata = pd.read_excel(metadata_path + ".xlsx")
    
    all_patient_data = []
    for _, row in metadata.iterrows():
        # 使用"住院号"列作为患者ID（第2列）
        patient_id = str(row['住院号'])
        
        # 使用"P53"列作为标签（第60列）
        p53_value = row['P53']
        
        # 检查P53值是否有效
        if pd.isna(p53_value):
            print(f"Skipping patient {patient_id} due to missing P53 value.")
            continue
            
        # 转换P53值为整数标签 (0 表示野生型, 1 表示突变型)
        # 假设P53列使用0和1表示状态，如果使用其他值需调整
        try:
            label = int(p53_value)
        except (ValueError, TypeError):
            # 如果不是直接的0/1值，可能需要根据实际情况转换
            print(f"Warning: Invalid P53 value for patient {patient_id}: {p53_value}")
            continue
            
        # 查找患者文件
        files = find_patient_files(patient_id, "Local")
        if files:
            files['label'] = label
            # 可以添加其他感兴趣的元数据
            # 例如：性别、年龄、临床指标等
            files['sex'] = int(row['性别（男=1，女=2）']) if not pd.isna(row['性别（男=1，女=2）']) else None
            files['tumor_size'] = row['MR最大径'] if not pd.isna(row['MR最大径']) else None
            files['pathological_grade'] = row['病理分级【0:I-II级;1:III-IV级】'] if not pd.isna(row['病理分级【0:I-II级;1:III-IV级】']) else None
            
            all_patient_data.append(files)
        else:
            print(f"Skipping patient {patient_id} due to missing image files.")

    print(f"Total patients with valid data: {len(all_patient_data)}")
    print(f"P53 mutated: {sum(p['label'] for p in all_patient_data)}, "
          f"Wild-type: {len(all_patient_data) - sum(p['label'] for p in all_patient_data)}")
          
    return all_patient_data

def split_data(all_patient_data, test_size=0.3, random_state=42):
    """Splits the list of patient data dictionaries into training and validation sets."""
    # Paper split: 170 training, 74 validation (~30.3% validation)
    if not all_patient_data:
        raise ValueError("No patient data loaded to split.")

    patient_ids = [p['id'] for p in all_patient_data]
    labels = [p['label'] for p in all_patient_data]

    train_ids, val_ids = train_test_split(
        patient_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=labels # Stratify to keep label distribution similar
    )

    train_data = [p for p in all_patient_data if p['id'] in train_ids]
    val_data = [p for p in all_patient_data if p['id'] in val_ids]

    print(f"Data split: {len(train_data)} training, {len(val_data)} validation samples.")
    # Optional: Print label distribution in splits
    train_labels = [p['label'] for p in train_data]
    val_labels = [p['label'] for p in val_data]
    print(f"Training labels: {sum(train_labels)} mutated / {len(train_labels)}")
    print(f"Validation labels: {sum(val_labels)} mutated / {len(val_labels)}")


    return train_data, val_data

# 添加测试代码
if __name__ == "__main__":
    print("开始测试数据加载功能...")
    try:
        print("1. 测试加载元数据...")
        all_data = load_data_manifest()
        
        if len(all_data) > 0:
            print(f"成功加载 {len(all_data)} 个患者数据")
            
            # 输出第一个患者的数据结构示例
            print("\n示例患者数据:")
            sample_patient = all_data[0]
            print(f"患者ID: {sample_patient['id']}")
            print(f"P53状态: {'突变' if sample_patient['label'] == 1 else '野生型'}")
            print(f"影像文件路径示例 (A): {sample_patient['A']}")
            print(f"分割文件路径示例 (A_seg): {sample_patient['A_seg']}")
            
            # 测试数据分割
            print("\n2. 测试数据分割...")
            train_data, val_data = split_data(all_data)
            
            print("\n测试完成!")
        else:
            print("没有找到符合条件的患者数据")
    
    except Exception as e:
        print(f"测试过程中出现错误: {e}")