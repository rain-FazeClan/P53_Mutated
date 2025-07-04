import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import glob
import random

# --- Configuration ---
# 注释掉TCIA相关配置
# TCIA_DATA_DIR = "path/to/your/tcia_data"
# 更新为包含全部4个数据文件夹的列表
LOCAL_DATA_DIRS = [
    "D:/GXMU_HCC/HCC_1",
    "D:/GXMU_HCC/HCC_2",
    "D:/GXMU_HCC/HCC_3",
    "D:/GXMU_HCC/HCC_4"
]
METADATA_FILE = "HCC_meta.xlsx"  # 元数据文件路径

# 更新为6种影像模态
MODALITIES = ["A", "DWI", "HBP", "P", "T1", "T2"]
# 更新分割文件后缀
SEG_SUFFIX = "_seg.nii.gz"  # 每个模态对应的分割标注都是 模态名_seg.nii.gz
# --- ---

def classify_p53(value):
    """
    根据 P53 免疫组化结果字符串判断突变型 (Mutated)、野生型 (Wild-type) 或不确定 (Uncertain)
    
    参数:
        value (str): P53 免疫组化结果字符串（如 "强阳，90%", "弱+，5%", "nan"）
        
    返回:
        str: "Mutated", "Wild-type" 或 "Uncertain"
    """
    # 调试输出，便于跟踪问题
    original_value = value
    
    # 处理非字符串输入和None值
    if value is None:
        print(f"DEBUG: 输入为None，归类为Uncertain")
        return "Uncertain"
    
    if not isinstance(value, str):
        if pd.isna(value):
            print(f"DEBUG: 输入为NaN，归类为Uncertain")
            return "Uncertain"
        value = str(value)
    
    # 预处理：去除空格、统一符号
    value = value.strip().replace(" ", "").replace("，", ",").replace("＞", ">")
    
    # 明确的缺失值处理
    if value.lower() in ["nan", "na", "", "null", "none"] or pd.isna(value):
        print(f"DEBUG: '{original_value}' 处理为空值，归类为Uncertain")
        return "Uncertain"
    
    # 完全阴性 (这是突变型)
    if value == "-" or value == "－":
        print(f"DEBUG: '{original_value}' 识别为完全阴性，归类为Mutated")
        return "Mutated"
    
    # 突变型关键词判断
    mutation_keywords = ["突变", "无意义突变", "错义突变", "突变型", "突变表达"]
    if any(kw in value for kw in mutation_keywords):
        print(f"DEBUG: '{original_value}' 包含突变关键词，归类为Mutated")
        return "Mutated"
    
    # 强阳性描述
    strong_pos_keywords = ["强阳", "强+", "+++", "3+", "弥漫强+"]
    if any(kw in value for kw in strong_pos_keywords):
        print(f"DEBUG: '{original_value}' 包含强阳性关键词，归类为Mutated")
        return "Mutated"
    
    # 高比例阳性判断
    percent_match = re.search(r"(\d+)%", value)
    if percent_match:
        percent = int(percent_match.group(1))
        if percent >= 80:
            print(f"DEBUG: '{original_value}' 阳性比例≥80%，归类为Mutated")
            return "Mutated"
    
    # >80%或>90%表达
    if ">" in value and ("80%" in value or "90%" in value):
        print(f"DEBUG: '{original_value}' 表达>80%或>90%，归类为Mutated")
        return "Mutated"
    
    # 野生型关键词判断
    if "野生型" in value:
        print(f"DEBUG: '{original_value}' 包含野生型关键词，归类为Wild-type")
        return "Wild-type"
    
    # 弱阳性描述
    weak_pos_keywords = ["弱+", "弱阳", "弱阳性", "1+"]
    if any(kw in value for kw in weak_pos_keywords):
        print(f"DEBUG: '{original_value}' 包含弱阳性关键词，归类为Wild-type")
        return "Wild-type"
    
    # 局灶性描述
    focal_keywords = ["散在", "灶", "部分", "少数"]
    if any(kw in value for kw in focal_keywords):
        print(f"DEBUG: '{original_value}' 包含局灶性关键词，归类为Wild-type")
        return "Wild-type"
    
    # 中等阳性且比例<80%
    medium_pos_keywords = ["中+", "中阳", "2+", "中等+"]
    if any(kw in value for kw in medium_pos_keywords):
        if percent_match and int(percent_match.group(1)) < 80:
            print(f"DEBUG: '{original_value}' 中等阳性且比例<80%，归类为Wild-type")
            return "Wild-type"
        elif not percent_match:  # 若无比例，默认野生型
            print(f"DEBUG: '{original_value}' 中等阳性无比例信息，归类为Wild-type")
            return "Wild-type"
    
    # 默认处理
    print(f"DEBUG: '{original_value}' 未匹配任何规则，默认归类为Wild-type")
    return "Wild-type"

def find_patient_files(patient_id, data_source="Local"):
    """Finds all required MRI and segmentation files for a patient."""
    # 在所有数据目录中查找患者数据
    patient_files = None
    
    for base_dir in LOCAL_DATA_DIRS:
        # 病人文件夹路径
        patient_dir = os.path.join(base_dir, patient_id)
        if not os.path.exists(patient_dir):
            continue  # 如果在当前目录中找不到，尝试下一个目录
            
        # 找到有效目录后，开始检查文件
        patient_files = {'id': patient_id, 'source': data_source, 'data_dir': base_dir}
        valid = True
        
        for mod in MODALITIES:
            # 查找影像文件
            mod_file = os.path.join(patient_dir, f"{mod}.nii.gz")
            if os.path.exists(mod_file):
                patient_files[mod] = mod_file
            else:
                print(f"Warning: Missing {mod} file for patient {patient_id} in {base_dir}")
                valid = False
                break  # 如果缺少模态则跳过该患者
                
            # 查找对应的分割文件
            seg_file = os.path.join(patient_dir, f"{mod}{SEG_SUFFIX}")
            if os.path.exists(seg_file):
                patient_files[f'{mod}_seg'] = seg_file
            else:
                print(f"Warning: Missing {mod} segmentation for patient {patient_id} in {base_dir}")
                patient_files[f'{mod}_seg'] = None  # 标记为缺失
        
        # 如果在当前目录找到有效数据，则返回
        if valid:
            return patient_files
        else:
            patient_files = None  # 重置为None，继续查找下一个目录
    
    # 所有目录都没有找到完整数据
    if patient_files is None:
        print(f"Warning: Patient {patient_id} not found in any data directory or has incomplete data")
    
    return patient_files

def load_data_manifest(metadata_path=METADATA_FILE):
    """Loads patient metadata and finds corresponding files."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # 读取Excel文件
    metadata = pd.read_excel(metadata_path)
    
    all_patient_data = []
    classification_issues = []  # 记录分类可能有问题的病例
    
    for _, row in metadata.iterrows():
        # 使用"住院号"列作为患者ID（第2列）
        patient_id = str(row['住院号'])
        
        # 使用"P53"列作为免疫组化结果
        p53_value = row['P53']
        
        # 使用classify_p53函数判断P53状态
        p53_status = classify_p53(p53_value)
        
        # 记录特殊情况
        if p53_value == '-' and p53_status != 'Mutated':
            classification_issues.append({
                'id': patient_id, 
                'value': p53_value, 
                'classified_as': p53_status
            })
        
        # 无论是否确定，都处理该样本
        # 转换P53状态为标签值 (0 表示野生型, 1 表示突变型, 'nan' 表示不确定)
        if p53_status == "Mutated":
            label = 1
        elif p53_status == "Wild-type":
            label = 0
        else:  # Uncertain
            label = 'nan'  # 使用'nan'字符串表示不确定
            
        print(f"Patient {patient_id}: P53 value '{p53_value}' classified as '{p53_status}' (label={label})")
            
        # 查找患者文件（在所有目录中）
        files = find_patient_files(patient_id, "Local")
        if files:
            files['label'] = label
            files['p53_status'] = p53_status  # 保存原始P53状态信息
            # 可以添加其他感兴趣的元数据
            files['sex'] = int(row['性别（男=1，女=2）']) if not pd.isna(row['性别（男=1，女=2）']) else None
            files['tumor_size'] = row['MR最大径'] if not pd.isna(row['MR最大径']) else None
            files['pathological_grade'] = row['病理分级【0:I-II级;1:III-IV级】'] if not pd.isna(row['病理分级【0:I-II级;1:III-IV级】']) else None
            
            all_patient_data.append(files)
        else:
            print(f"Skipping patient {patient_id} due to missing image files.")

    print(f"Total patients with valid data: {len(all_patient_data)}")
    # 统计各类别数量
    mutated_count = sum(1 for p in all_patient_data if p['label'] == 1)
    wildtype_count = sum(1 for p in all_patient_data if p['label'] == 0)
    uncertain_count = sum(1 for p in all_patient_data if p['label'] == 'nan')
    
    total = len(all_patient_data)
    print(f"P53分类统计: 突变({mutated_count}人, {mutated_count/total:.1%}), "
          f"野生型({wildtype_count}人, {wildtype_count/total:.1%}), "
          f"不确定({uncertain_count}人, {uncertain_count/total:.1%})")
    
    # 按数据来源统计患者数量
    data_dir_stats = {}
    for patient in all_patient_data:
        data_dir = patient['data_dir']
        if data_dir not in data_dir_stats:
            data_dir_stats[data_dir] = 0
        data_dir_stats[data_dir] += 1
    
    print("患者数据来源统计:")
    for data_dir, count in data_dir_stats.items():
        print(f"  {os.path.basename(data_dir)}: {count}人")
    
    # 在函数结束前输出可能的分类问题
    if classification_issues:
        print("\n可能的分类问题:")
        for issue in classification_issues:
            print(f"  患者{issue['id']}: 值为'{issue['value']}' 但被分类为'{issue['classified_as']}'")
    
    return all_patient_data

def split_data(all_patient_data, test_size=0.3, random_state=42):
    """Splits the list of patient data dictionaries into training and validation sets."""
    # Paper split: 170 training, 74 validation (~30.3% validation)
    if not all_patient_data:
        raise ValueError("No patient data loaded to split.")

    # 过滤掉不确定标签的患者
    certain_patients = [p for p in all_patient_data if p['label'] != 'nan']
    uncertain_patients = [p for p in all_patient_data if p['label'] == 'nan']
    
    if uncertain_patients:
        print(f"注意: {len(uncertain_patients)} 名患者标签不确定，已从训练和验证集中排除")
    
    if not certain_patients:
        raise ValueError("没有找到有确定标签的患者数据，无法进行分割")

    patient_ids = [p['id'] for p in certain_patients]
    labels = [p['label'] for p in certain_patients]

    train_ids, val_ids = train_test_split(
        patient_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=labels # Stratify to keep label distribution similar
    )

    train_data = [p for p in certain_patients if p['id'] in train_ids]
    val_data = [p for p in certain_patients if p['id'] in val_ids]

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
            print(f"数据来源: {os.path.basename(sample_patient['data_dir'])}")
            print(f"P53状态: {sample_patient['p53_status']} (标签值: {sample_patient['label']})")
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