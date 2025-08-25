import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom, label

# --- 配置 ---
folders = ["HCC_1", "HCC_2", "HCC_3", "HCC_4"]
modalities = ["A", "DWI", "HBP", "P", "T1", "T2"]
target_shape_2d = (384, 384)
output_dir = "preprocessed"
os.makedirs(output_dir, exist_ok=True)


# --- 配置结束 ---

def find_modality_file(patient_dir, modality, is_seg=False):

    suffix = "_seg" if is_seg else ""
    modality_prefix = modality.lower()
    for file in os.listdir(patient_dir):
        file_lower = file.lower()
        if file_lower.startswith(modality_prefix) and \
                file_lower.split('_')[0] == modality_prefix and \
                suffix in file_lower and \
                (file.endswith('.nii') or file.endswith('.nii.gz')):
            return os.path.join(patient_dir, file)
    return None


def resample_image_to_reference(image, reference_image, default_pixel_value=0.0):

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetOutputPixelType(image.GetPixelID())

    return resampler.Execute(image)


def find_largest_tumor_slices(seg_arr):

    if seg_arr is None:
        return []
    # 识别独立的肿瘤区域
    labeled_seg, num_features = label(seg_arr > 0)
    if num_features == 0:
        return []

    largest_slices = []
    for i in range(1, num_features + 1):
        tumor_mask = (labeled_seg == i)

        areas = np.sum(tumor_mask, axis=(1, 2))
        if np.max(areas) > 0:
            slice_idx = np.argmax(areas)
            largest_slices.append(int(slice_idx))

    return sorted(list(set(largest_slices)))


def process_slice(slice_2d, target_shape):
    slice_2d = slice_2d.astype(np.float32)

    mean = np.mean(slice_2d)
    std = np.std(slice_2d)
    if std > 0:
        slice_2d = (slice_2d - mean) / (std + 1e-8)
    slice_2d = np.clip(slice_2d, -5, 5)

    zoom_factors = [target_shape[0] / slice_2d.shape[0], target_shape[1] / slice_2d.shape[1]]
    resampled_slice = zoom(slice_2d, zoom_factors, order=3)  # 使用三次样条插值

    return resampled_slice


def main():
    all_patients = set()
    for folder in folders:
        if not os.path.exists(folder): continue
        for pid in os.listdir(folder):
            if os.path.isdir(os.path.join(folder, pid)):
                all_patients.add(pid)
    all_patients = sorted(list(all_patients))

    print(f"开始预处理，将使用物理坐标对齐。目标shape: {target_shape_2d}，保存到: {output_dir}")
    total_files_saved = 0

    for pid in all_patients:
        print(f"\n处理病例: {pid}")

        reference_seg_path = None
        patient_base_dir = None
        for folder in folders:
            patient_dir = os.path.join(folder, pid)
            if os.path.exists(patient_dir):
                seg_path = find_modality_file(patient_dir, "HBP", is_seg=True)
                if seg_path:
                    reference_seg_path = seg_path
                    patient_base_dir = patient_dir
                    print(f"  找到参考分割文件: {reference_seg_path}")
                    break

        if not reference_seg_path:
            for mod_ref in modalities:
                if reference_seg_path: break
                for folder in folders:
                    patient_dir = os.path.join(folder, pid)
                    if os.path.exists(patient_dir):
                        seg_path = find_modality_file(patient_dir, mod_ref, is_seg=True)
                        if seg_path:
                            reference_seg_path = seg_path
                            patient_base_dir = patient_dir
                            print(f"  警告: 未找到HBP分割，使用 {mod_ref} 分割作为参考: {reference_seg_path}")
                            break

        if not reference_seg_path:
            print(f"  病例 {pid} 未找到任何分割文件，跳过。")
            continue

        try:
            reference_seg_image = sitk.ReadImage(reference_seg_path, sitk.sitkUInt8)
            reference_seg_arr = sitk.GetArrayFromImage(reference_seg_image)
        except Exception as e:
            print(f"  加载参考文件 {reference_seg_path} 失败: {e}，跳过病例。")
            continue

        slice_indices = find_largest_tumor_slices(reference_seg_arr)
        if not slice_indices:
            print(f"  在 {reference_seg_path} 中未找到肿瘤，跳过病例。")
            continue

        print(f"  找到 {len(slice_indices)} 个肿瘤最大面积切片，索引为: {slice_indices}")

        for i, slice_idx in enumerate(slice_indices):
            stacked_modalities = []

            for mod in modalities:
                mod_slice = np.zeros(target_shape_2d, dtype=np.float32)
                try:
                    nii_path = None
                    for folder in folders:
                        patient_dir = os.path.join(folder, pid)
                        if os.path.exists(patient_dir):
                            path = find_modality_file(patient_dir, mod)
                            if path:
                                nii_path = path
                                break

                    if nii_path:
                        image = sitk.ReadImage(nii_path)
                        resampled_image = resample_image_to_reference(image, reference_seg_image)
                        resampled_arr = sitk.GetArrayFromImage(resampled_image)

                        slice_2d = resampled_arr[slice_idx, :, :]

                        mod_slice = process_slice(slice_2d, target_shape_2d)
                    else:
                        print(f"    缺失 {mod} 模态，将补零。")

                except Exception as e:
                    print(f"    处理 {mod} 模态时出错: {e}，将补零。")

                stacked_modalities.append(mod_slice)

            all_data = np.stack(stacked_modalities, axis=0)  # Shape: (6, H, W)

            out_filename = f"{pid}_tumor{i}.npy"
            out_path = os.path.join(output_dir, out_filename)
            np.save(out_path, all_data)
            print(f"  已保存: {out_path}, shape: {all_data.shape}")
            total_files_saved += 1

    print(f"\n预处理完成! 共为 {len(all_patients)} 个病例生成了 {total_files_saved} 个文件。")


if __name__ == "__main__":
    main()