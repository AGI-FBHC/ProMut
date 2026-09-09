import torch
import os
import numpy as np
for _alias, _target in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
}.items():
    if _alias not in np.__dict__:
        setattr(np, _alias, _target)
import csv
import time
import argparse
from path_config import PRETRAINED_MODEL, PROMUT2_CSV_DIR, PREDICTION_OUTPUT_DIR

# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Device selection respects the caller environment.


def find_default_protein_csv():
    candidates = [
        os.path.join(PROMUT2_CSV_DIR, "csv", "00000004_a_3.csv"),
        os.path.join(PROMUT2_CSV_DIR, "csv"),
        os.path.join(os.path.dirname(PROMUT2_CSV_DIR), "data_csv"),
    ]

    if os.path.isfile(candidates[0]):
        return candidates[0]

    for directory in candidates[1:]:
        if os.path.isdir(directory):
            csv_files = sorted(file for file in os.listdir(directory) if file.lower().endswith(".csv"))
            if csv_files:
                return os.path.join(directory, csv_files[0])

    raise FileNotFoundError("No protein CSV file found. Run data_utils.py first or check data_csv/.")



def predict(
    model_path,
    protein_file,
    out_file,
    chain='A',
    k=10,
    effect_out_file=None,
    protein_name=None,
    position_offset=0,
):
    from model_promut1 import ProMutBackbone
    from data_utils import prepareBox
    from generate_full_sidechain_box_20A import grab_PDB_csv
    from atom_res_dict import label_res_dict, abrev

    total_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Device selection respects the caller environment.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model_load_start = time.perf_counter()

    model = ProMutBackbone().to(device)  # 确保模型在设备上
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_time = time.perf_counter() - model_load_start

    preprocessing_start = time.perf_counter()
    PROTEIN = grab_PDB_csv(protein_file)
    ID_dict = PROTEIN[0]

    all_resids = [k[1] for k in ID_dict.keys() if k[0] == chain]  # 直接获取字符串形式的resid

    # 创建positions列表
    positions = [(chain, str(resid)) for resid in all_resids]
    boxes, labels, valid_aa, _ = prepareBox(PROTEIN, atom_density=0.0, v=False, check=False, positions=positions)

    # 检查 boxes 是否为空
    if not boxes:
        raise ValueError("No boxes generated for the given positions.", protein_file)

    input = torch.tensor(np.vstack(boxes), dtype=torch.float32).to(device)  # 确保输入也在设备上
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    preprocessing_time = time.perf_counter() - preprocessing_start
    preprocessing_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    preprocessing_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0
    print(input.shape)

    # 使用模型进行预测
    with torch.no_grad():
        warmup_batch = input[:min(8, input.shape[0])]
        for _ in range(3):
            _ = model(warmup_batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_start_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    inference_start_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0

    inference_start = time.perf_counter()
    with torch.no_grad():
        output = model(input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - inference_start
    inference_end_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    inference_end_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0
    peak_allocated_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0

    postprocess_start = time.perf_counter()
    logits_cpu = output.cpu()
    output = torch.nn.Softmax(dim=1)(output)
    output_cpu = output.cpu()
    values, indices = torch.topk(output_cpu.data, k=k, dim=1, largest=True, sorted=True)
    postprocess_time = time.perf_counter() - postprocess_start
    #values是置信度，indices是对应的氨基酸编号，1，2，3这种
    print("values:",values)
    print("indices:",indices)

    # 结果保存到CSV文件
    save_start = time.perf_counter()
    effect_rows = []
    with open(out_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        headers = ['Position', 'Actual AA']
        for i in range(k):
            headers.extend([f'Pred AA {i + 1}', f'Confidence {i + 1}'])
        writer.writerow(headers)  # 在这里写入表头

        predictions = []
        for i, (chain_id, resid) in enumerate(valid_aa):
            # 调整位置编号使其从 1 开始
            adjusted_resid = int(float(resid))
            effect_resid = adjusted_resid + int(position_offset)
            # 获取实际的氨基酸
            actual_aa = abrev[label_res_dict[labels[i]]]

            # 获取预测的氨基酸和对应的置信度
            predicted_aas = [abrev[label_res_dict[int(j)]] for j in indices[i]]
            confidences = values[i].tolist()
            full_probs = output_cpu[i].tolist()
            actual_idx = int(labels[i])
            wt_prob = max(float(full_probs[actual_idx]), 1e-12)

            # 将氨基酸和置信度配对
            paired_results = list(zip(predicted_aas, confidences))

            # 确保预测的氨基酸和置信度列表长度相同
            assert len(predicted_aas) == len(confidences), "Length of predicted AAs and confidences do not match."

            # 按照置信度排序
            paired_results.sort(key=lambda x: x[1], reverse=True)

            # 将排序后的配对结果展平
            flat_results = sum(paired_results, ())

            # 写入位置、实际氨基酸和预测结果
            row = [adjusted_resid, actual_aa] + list(flat_results)
            writer.writerow(row)

            if effect_out_file:
                for aa_idx, mut_prob_raw in enumerate(full_probs):
                    mut_aa = abrev[label_res_dict[int(aa_idx)]]
                    mut_prob = max(float(mut_prob_raw), 1e-12)
                    log_effect = float(np.log(mut_prob) - np.log(wt_prob))
                    prob_delta = float(mut_prob - wt_prob)
                    effect_rows.append({
                        "protein_name": protein_name or os.path.splitext(os.path.basename(protein_file))[0],
                        "mut_info": f"{actual_aa}{effect_resid}{mut_aa}",
                        "position": effect_resid,
                        "raw_position": adjusted_resid,
                        "wildtype": actual_aa,
                        "mutant": mut_aa,
                        "wt_probability": wt_prob,
                        "mutant_probability": mut_prob,
                        "probability_delta": prob_delta,
                        "score": log_effect,
                    })

            # 添加预测结果到列表
            predictions.append({
                'position': adjusted_resid,
                'Actual AA': actual_aa,
                'Predictions': paired_results
            })
    save_time = time.perf_counter() - save_start
    if effect_out_file:
        with open(effect_out_file, "w", newline="") as effect_csv:
            fieldnames = [
                "protein_name",
                "mut_info",
                "position",
                "raw_position",
                "wildtype",
                "mutant",
                "wt_probability",
                "mutant_probability",
                "probability_delta",
                "score",
            ]
            writer = csv.DictWriter(effect_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(effect_rows)
        print("Mutation-effect scores have been saved to", effect_out_file)
    end_to_end_time = time.perf_counter() - total_start
    num_samples = int(input.shape[0])
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = os.path.getsize(model_path) / 1024 / 1024 if os.path.exists(model_path) else 0
    time_file = out_file.replace(".csv", "_time.csv")
    with open(time_file, "w", newline="") as time_csv:
        writer = csv.writer(time_csv)
        writer.writerow([
            "model_name",
            "model_path",
            "protein_file",
            "device",
            "num_samples",
            "model_size_mb",
            "parameter_count",
            "trainable_parameter_count",
            "model_load_time_sec",
            "preprocessing_time_sec",
            "inference_time_sec",
            "avg_inference_time_per_sample_sec",
            "preprocessing_gpu_allocated_mb",
            "preprocessing_gpu_reserved_mb",
            "inference_start_gpu_allocated_mb",
            "inference_start_gpu_reserved_mb",
            "inference_end_gpu_allocated_mb",
            "inference_end_gpu_reserved_mb",
            "peak_gpu_allocated_mb",
            "peak_gpu_reserved_mb",
            "postprocess_time_sec",
            "result_save_time_sec",
            "end_to_end_time_sec",
        ])
        writer.writerow([
            model.__class__.__name__,
            model_path,
            protein_file,
            str(device),
            num_samples,
            f"{model_size_mb:.6f}",
            parameter_count,
            trainable_parameter_count,
            f"{model_load_time:.6f}",
            f"{preprocessing_time:.6f}",
            f"{inference_time:.6f}",
            f"{inference_time / max(num_samples, 1):.9f}",
            f"{preprocessing_allocated_mb:.6f}",
            f"{preprocessing_reserved_mb:.6f}",
            f"{inference_start_allocated_mb:.6f}",
            f"{inference_start_reserved_mb:.6f}",
            f"{inference_end_allocated_mb:.6f}",
            f"{inference_end_reserved_mb:.6f}",
            f"{peak_allocated_mb:.6f}",
            f"{peak_reserved_mb:.6f}",
            f"{postprocess_time:.6f}",
            f"{save_time:.6f}",
            f"{end_to_end_time:.6f}",
        ])
    print("Prediction results have been saved to", out_file)
    print("Timing results have been saved to", time_file)
    return predictions  # 返回预测结果




# 示例调用
def parse_args():
    parser = argparse.ArgumentParser(description="Generate ProMut top-k predictions for one processed protein CSV.")
    parser.add_argument(
        "--protein-csv",
        default=None,
        help="Processed protein CSV with columns atom,serial,name,resname,resid,x,y,z,charges,radius,sasa.",
    )
    parser.add_argument("--out", default=None, help="Output prediction CSV.")
    parser.add_argument("--model", default=PRETRAINED_MODEL, help="Path to ProMut checkpoint.")
    parser.add_argument("--chain", default="A", help="Chain ID used by processed CSV. Default: A.")
    parser.add_argument("--topk", type=int, default=20, help="Number of amino-acid predictions per residue.")
    parser.add_argument(
        "--effect-out",
        default=None,
        help="Optional long-format mutation-effect CSV using score = log P(mutant) - log P(wildtype).",
    )
    parser.add_argument("--protein-name", default=None, help="Protein name written to --effect-out.")
    parser.add_argument(
        "--position-offset",
        type=int,
        default=0,
        help="Offset added to raw residue numbers when writing --effect-out mut_info/position.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)
    model_path = args.model
    protein_file = args.protein_csv or find_default_protein_csv()
    protein_id = os.path.splitext(os.path.basename(protein_file))[0]
    out_file = args.out or os.path.join(PREDICTION_OUTPUT_DIR, protein_id + "_predictions.csv")
    predict(
        model_path,
        protein_file,
        out_file,
        args.chain,
        args.topk,
        effect_out_file=args.effect_out,
        protein_name=args.protein_name,
        position_offset=args.position_offset,
    )
