import pandas as pd

try:
    # Step 1: 读取 metadata.csv 文件（并筛选 Modality 为 MR 的记录）
    print("Step 1: 读取 metadata.csv 文件并筛选 Modality 为 MR...")
    metadata = pd.read_csv("TCGA-LIHC/metadata.csv", sep=";", header=0)
    print("原始文件列名：", metadata.columns)
    print("原始文件前几行：\n", metadata.head())

    # 筛选 Modality 为 MR
    metadata_filtered = metadata[metadata["Modality"] == "MR"]
    print("筛选后的文件前几行：\n", metadata_filtered.head())

    # 按 Subject 分组并合并 Series Description
    metadata_grouped = metadata_filtered.groupby("Subject")["Series Description"].apply(
        lambda x: ','.join(x.unique())).reset_index()
    print("按 Subject 分组并合并后的结果：\n", metadata_grouped.head())
except Exception as e:
    print("Error in Step 1:", e)
    exit()

try:
    # Step 2: 读取 clinical.tsv 文件
    print("\nStep 2: 读取 clinical.tsv 文件...")
    clinical = pd.read_csv("TCGA-LIHC/TCGA-LIHC.clinical.tsv", sep="\t")
    print("clinical.tsv 文件前几行：\n", clinical.head())

    # 创建 Subject-ID 和 Sample 的映射关系
    subject_sample_mapping = clinical.set_index("submitter_id")["sample"].to_dict()
    print("映射关系示例：", list(subject_sample_mapping.items())[:5])

    # 将 Sample 信息补充到 metadata 中
    metadata_grouped["Sample"] = metadata_grouped["Subject"].map(subject_sample_mapping)
    print("补充 Sample 信息后的结果：\n", metadata_grouped.head())

    # 整合 metadata 和 clinical 数据
    metadata_with_clinical = metadata_grouped.merge(clinical, left_on="Sample", right_on="sample", how="left")
    print("整合后的数据前几行：\n", metadata_with_clinical.head())
except Exception as e:
    print("Error in Step 2:", e)
    exit()

try:
    # Step 3: 读取 somaticmutation.tsv 文件
    print("\nStep 3: 读取 somaticmutation.tsv 文件...")
    mutation = pd.read_csv("TCGA-LIHC/TCGA-LIHC.somaticmutation.tsv", sep="\t")
    print("somaticmutation.tsv 文件前几行：\n", mutation.head())

    # 提取基因为 p53 的所有样本编号
    p53_subjects = mutation[mutation["gene"].str.upper() == "P53"]["Tumor_Sample_Barcode"].str[:12].unique()
    print("有 p53 突变的 Subject ID：", p53_subjects[:5])
except Exception as e:
    print("Error in Step 3:", e)
    exit()

try:
    # Step 4: 标记是否有 p53 突变
    print("\nStep 4: 标记是否有 p53 突变...")
    metadata_with_clinical["p53_status"] = metadata_with_clinical["Subject"].apply(
        lambda x: "阳性" if x in p53_subjects else "阴性")
    print("新增 p53_status 列后的数据：\n", metadata_with_clinical[["Subject", "p53_status"]].head())
except Exception as e:
    print("Error in Step 4:", e)
    exit()

try:
    # Step 5: 保存结果为新的 CSV 文件
    print("\nStep 5: 保存结果为新的 CSV 文件...")
    metadata_with_clinical.to_csv("integrated_patient_data.csv", index=False)
    print("整合完成，结果已保存为 integrated_patient_data.csv 文件。")
except Exception as e:
    print("Error in Step 5:", e)
    exit()