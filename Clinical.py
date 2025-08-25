import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class Clinical(nn.Module):
    """临床特征处理模型"""

    def __init__(self, input_dim, hidden_dims=[64, 32], num_classes=64):
        super(Clinical, self).__init__()

        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            last_dim = h
        layers.append(nn.Linear(last_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 临床特征列表
CLINICAL_FEATURES = [
    '结节大小分组【0：≤2cm；1：＞2cm】',
    '边缘【肝胆期肿瘤边缘光整：0；边缘不光整：1】',
    '肝包膜【门静脉期无包膜：0；有包膜：1】',
    'T1信号【0：低信号；1：等/高及混杂信号】',
    'T2信号【0：高信号；1：等/低信号】',
    'T2信号【0：均匀；1：不均匀】',
    '瘤周强化【动脉期瘤周肝实质存在异常强化区域，在门静脉期及平衡期呈等信号】0：无；1：有',
    '瘤内血管无：0；有：1',
    '瘤周低信号【肝胆期肿瘤周围是否存在低信号，定义为肿瘤周围低于正常肝实质信号、高于肿瘤本身信号的区域；少数 ＨＣＣ 病灶肝胆期表现为等或高信号，其瘤周异常信号区信号强度低于病灶而高于其低信号环】',
    '非富血供肝胆期低信号结节（ＮＨＨＮ）肝胆期低摄取而动脉期无明显强化的结节【无：0；1：有】',
    '瘤内出血（0无，1有）',
    'T1WI肿瘤',
    'HBP肿瘤',
    'HBP RER',
    'T1WI肝脏',
    'HBP肝脏',
    'T2WI肿瘤',
    'T2WI肝脏',
    'DWI肿瘤',
    'DWI肝脏',
    '动脉期强化【0无强化；1均匀强化；2不均匀强化】',
    '动脉期强化（等/低0，非环形1，环形2）',
    '门脉期washout（0无，1有）',
    '延迟强化（0无1有）',
    '强化类型分组【0：快进快出；1：非快进快出】',
    'DWI信号 0：高信号；1：等信号',
    '弥散受限【0：无；1：有】',
    '癌栓【0：无；1：有】',
    '腹水【0：无；1：有】'
]


def extract_clinical_features(data_manifest, clinical_columns=None):
    """提取临床特征数据"""
    if clinical_columns is None:
        clinical_columns = CLINICAL_FEATURES

    try:
        metadata = pd.read_excel("HCC_meta.xlsx")
        clinical_data = []

        for item in data_manifest:
            patient_id = item['id']
            patient_row = metadata[metadata['住院号'].astype(str) == str(patient_id)]

            if len(patient_row) > 0:
                row = patient_row.iloc[0]
                features = []
                for col in clinical_columns:
                    if col in metadata.columns:
                        value = row[col]
                        if pd.isna(value):
                            features.append(0.0)  # 用0填充缺失值
                        else:
                            features.append(float(value))
                    else:
                        features.append(0.0)
                clinical_data.append(features)
            else:
                clinical_data.append([0.0] * len(clinical_columns))

        return np.array(clinical_data, dtype=np.float32)
    except Exception as e:
        print(f"加载临床特征时出错: {e}")
        return None
