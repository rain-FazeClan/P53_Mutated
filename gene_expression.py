import pandas as pd

try:
    # 修改读取方式：设置分隔符为分号
    print("Step 1: 读取 metadata.csv 文件...")
    metadata = pd.read_csv("TCGA-LIHC/metadata.csv", sep=";", header=0)  # 设置分隔符为 ";"
    print("文件前几行内容：")
    print(metadata.head())  # 打印前几行内容
    print("\n文件列名：")
    print(metadata.columns)  # 打印列名
except Exception as e:
    print("Error in Step 1:", e)
    exit()

try:
    # Step 2: 按 Subject 分组并合并 Series Description
    print("\nStep 2: 按 Subject 分组并合并 Series Description...")
    metadata_grouped = metadata.groupby("Subject")["Series Description"].apply(lambda x: ','.join(x.unique())).reset_index()
    print(metadata_grouped.head())
except Exception as e:
    print("Error in Step 2:", e)
    exit()

try:
    # Step 3: 读取 clinical.tsv 文件
    print("\nStep 3: 读取 clinical.tsv 文件...")
    clinical = pd.read_csv("TCGA-LIHC/TCGA-LIHC.clinical.tsv", sep="\t")
    print(clinical.head())
except Exception as e:
    print("Error in Step 3:", e)
    exit()

try:
    # Step 4: 创建 Subject-ID 和 Sample 的映射关系
    print("\nStep 4: 创建 Subject-ID 和 Sample 的映射关系...")
    subject_sample_mapping = clinical.set_index("submitter_id")["sample"].to_dict()
    print("Mapping example:", list(subject_sample_mapping.items())[:5])
except Exception as e:
    print("Error in Step 4:", e)
    exit()

try:
    # Step 5: 将 Sample 信息补充到 metadata 中
    print("\nStep 5: 将 Sample 信息补充到 metadata 中...")
    metadata_grouped["Sample"] = metadata_grouped["Subject"].map(subject_sample_mapping)
    print(metadata_grouped.head())
except Exception as e:
    print("Error in Step 5:", e)
    exit()

try:
    # Step 6: 整合 metadata 和 clinical 数据
    print("\nStep 6: 整合 metadata 和 clinical 数据...")
    metadata_with_clinical = metadata_grouped.merge(clinical, left_on="Sample", right_on="sample", how="left")
    print(metadata_with_clinical.head())
except Exception as e:
    print("Error in Step 6:", e)
    exit()

try:
    # Step 7: 读取 somaticmutation.tsv 文件
    print("\nStep 7: 读取 somaticmutation.tsv 文件...")
    mutation = pd.read_csv("TCGA-LIHC/TCGA-LIHC.somaticmutation.tsv", sep="\t")
    print(mutation.head())
except Exception as e:
    print("Error in Step 7:", e)
    exit()

try:
    # Step 8: 提取基因为 p53 的所有样本编号
    print("\nStep 8: 提取基因为 p53 的所有样本编号...")
    p53_subjects = mutation[mutation["gene"].str.upper() == "TP53"]["Tumor_Sample_Barcode"].str[:12].unique()
    print("Subjects with p53 mutations:", p53_subjects[:5])
except Exception as e:
    print("Error in Step 8:", e)
    exit()

try:
    # Step 9: 标记是否有 p53 突变
    print("\nStep 9: 标记是否有 p53 突变...")
    metadata_with_clinical["p53_status"] = metadata_with_clinical["Subject"].apply(lambda x: "阳性" if x in p53_subjects else "阴性")
    print(metadata_with_clinical[["Subject", "p53_status"]].head())
except Exception as e:
    print("Error in Step 9:", e)
    exit()

try:
    # Step 10: 保存结果为新的 CSV 文件
    print("\nStep 10: 保存结果为新的 CSV 文件...")
    metadata_with_clinical.to_csv("integrated_patient_data.csv", index=False)
    print("整合完成，结果已保存为 integrated_patient_data.csv 文件。")
except Exception as e:
    print("Error in Step 10:", e)
    exit()