#从PDB文件中分离链、提取原子数据、计算属性并将其转换为适合机器学习分析的数据格式。
#pdb->csv
import os
import csv
import shutil
import sys
import pandas as pd
from tqdm import tqdm
from atom_res_dict import *
from generate_full_sidechain_box_20A import get_position_dict, pts_to_Xsmooth, grab_PDB_csv
import subprocess
try:
    import freesasa
except ModuleNotFoundError:
    freesasa = None


def get_pdb2pqr_bin():
    candidates = [
        os.environ.get("PDB2PQR_BIN"),
        shutil.which("pdb2pqr"),
        shutil.which("pdb2pqr30"),
        os.path.join(sys.prefix, "Scripts", "pdb2pqr30.exe"),
        os.path.join(sys.prefix, "Scripts", "pdb2pqr30-script.exe"),
    ]

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Cannot find pdb2pqr. Set PDB2PQR_BIN to the full path of pdb2pqr30.exe."
    )


def seperate_chains(_file, _target_path):  #读取pdb文件将其分离成单个链文件
    pdb_id = _file[-8:-4]
    # print(pdb_id)
    with open(_file, 'r') as _f:
        line = " "
        chain_id = ""
        while line:
            line = _f.readline()
            if line.startswith('ATOM'):
                chain_now = line.split()[4][0]
                if chain_id != chain_now:
                    if chain_id == "":
                        _t = open(os.path.join(_target_path, pdb_id + chain_now + ".pdb"), 'w')
                    else:
                        _t.close()
                    chain_id = chain_now
                    _t = open(os.path.join(_target_path, pdb_id + chain_now + ".pdb"), 'w')

                _t.writelines(line)
        _t.close()


def get_data_from_file(_file, start=0):  #读取pdb文件并提取以ATOM开头的行，数据存储在列表中并返回
    _data = []
    with open(_file, "r") as _f:
        line = " "
        while line:
            line = _f.readline()
            if line.startswith('ATOM'):
                if start != -1:
                    data_row = line.split()[start:]
                else:
                    data_row = line.split()[start]
                _data.append(data_row)

    return _data

#处理pdb到csv 调用pdb2pqr和freesasa工具分别计算原子的电荷和溶剂可及表面积，结果保存为csv格式
def process_pdb2csv(_target_path, _output_path):
    if freesasa is None:
        raise ModuleNotFoundError(
            "freesasa is required when converting PDB files to ProMut CSV. "
            "It is not required when predicting from existing processed CSV files."
        )
    file_list1 = os.listdir(_target_path)

    error_file = []
    head = ["atom", "serial", "name", "resname", "resid", "x", "y", "z", "charges", "radius", "sasa"]
    csv_output_dir = os.path.join(_output_path, "csv")
    os.makedirs(csv_output_dir, exist_ok=True)
    pdb2pqr_bin = get_pdb2pqr_bin()

    for file in file_list1:
        if not file.endswith(".pdb"):
            continue
        pdb_chain_id = file[:-4]
        csv_file = os.path.join(csv_output_dir, pdb_chain_id + ".csv")
        if os.path.exists(csv_file):
            continue

        pqr_output_file = os.path.join(_output_path, pdb_chain_id + "_pqr.pdb")
        protein_file = os.path.join(_target_path, file)

        # 使用 pdb2pqr
        try:
            subprocess.run([pdb2pqr_bin, '--ff', 'CHARMM', protein_file, pqr_output_file], check=True)
        except subprocess.CalledProcessError:
            error_file.append(pdb_chain_id)
            print("pdb2pqr fails on", pdb_chain_id)
            continue

        # 使用 freesasa 的 Python API
        structure = freesasa.Structure(pqr_output_file)
        result = freesasa.calc(structure)

        # 读取 PQR 文件的数据
        base_rows = get_data_from_file(pqr_output_file, start=0)

        # 提取 SASA 数据
        sasa_list = [result.atomArea(i) for i in range(len(base_rows))]

        # 写入 CSV 文件
        with open(csv_file, 'w', newline='') as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow(head)

            if len(base_rows) == len(sasa_list):
                for i, row in enumerate(base_rows):
                    row.append(sasa_list[i])
                    writer.writerow(row)
            else:
                error_file.append(pdb_chain_id)

    print("error files:")
    print(error_file)
    return csv_file

def prepareBox(PROTEIN, channels=7, atom_density=0.01, box_size=20, pixel_size=1, v=True, check=True, positions=None):
    ''' #准备机器学习模型的数据盒子，根据原子密度、通道数、盒子尺寸和分辨率来生成数据，返回的盒子包含平滑处理后的原子数据以及相应的标签。
        PROTEIN:蛋白质数据文件路径
        channels:通道数，默认为7维（C、N、O、S、H、partial_charge、sasa）
        atom_density: 盒子里的原子密度，控制一个盒子中最少有多少原子, num_of_atom / box volume
        box_size: 盒子尺寸
        pixel_size: 分辨率，默认为1埃
    '''
    ID_dict, _, _, _, _, _, _ = PROTEIN
    boxes = []
    labels = []
    valid_aa = []
    skip = []
    keys = list(ID_dict.keys())

    if positions is not None:
        positions = [(chain, str(resid)) for chain, resid in positions]

        keys = positions

    for chain_ID in keys:
        res_atoms = ID_dict[chain_ID]
        res = res_atoms[0].res
        if res in list(res_label_dict.keys()):
            label = res_label_dict[res]
            get_position = get_position_dict(res_atoms)
            if "CA" in get_position.keys():
                ctr = get_position["CA"]
                pts = [ctr, chain_ID, label]
                X_smooth, label, _, _, _, valid_box = pts_to_Xsmooth(PROTEIN, pts, atom_density,
                                                                     channels, pixel_size, box_size, check=check)
                if valid_box:
                    boxes.append([X_smooth])
                    labels.append(label)
                    valid_aa.append(chain_ID)
                else:
                    if v:
                        print(f'{chain_ID[0]}链的第{int(float(chain_ID[1]))}号氨基酸采样的原子过少，被忽略')
                    skip.append(chain_ID)
            else:
                if v:
                    print(f'{chain_ID[0]}链的第{int(float(chain_ID[1]))}号氨基酸没有CA原子，被忽略')
                skip.append(chain_ID)
    return boxes, labels, valid_aa, skip


if __name__ == '__main__':
 project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
 target_path = os.path.join(project_root, "promut2_dataset")
 output_path = os.path.join(project_root, "promut2_csv")
 os.makedirs(output_path, exist_ok=True)

 # 调用process_pdb2csv函数
 pdbcsv_file = process_pdb2csv(target_path, output_path)

 print(f"Conversion completed. CSV file saved at: {pdbcsv_file}")
