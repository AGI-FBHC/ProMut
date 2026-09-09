# csv->npz
import numpy
import os
import csv
from scipy import ndimage
import json
import collections
import random
from tqdm import tqdm
from atom_res_dict import *
from path_config import DATA_CSV_DIR, PRETRAIN_NPZ_DIR, PROJECT_ROOT
# from scipy import spatial
GLY = []
CYS = []
ARG = []
SER = []
THR = []
LYS = []
MET = []
ALA = []
LEU = []
ILE = []
VAL = []
ASP = []
GLU = []
HIS = []
ASN = []
PRO = []
GLN = []
PHE = []
TRP = []
TYR = []

res_container_dict = {0: HIS, 1: LYS, 2: ARG, 3: ASP, 4: GLU, 5: SER, 6: THR, 7: ASN, 8: GLN, 9: ALA, 10: VAL, 11: LEU,
                      12: ILE, 13: MET, 14: PHE, 15: TYR, 16: TRP, 17: PRO, 18: GLY, 19: CYS}


class PDB_atom:
    '''
        定义一个蛋白质类，包含10个属性
        atom: ...
    '''
    def __init__(self, atom_type, res, chain_ID, x, y, z, index, value, charge=0., sasa=0.):
        self.atom = atom_type
        self.res = res
        self.chain_ID = chain_ID
        self.x = x
        self.y = y
        self.z = z
        self.index = index
        self.value = value
        self.charge = charge
        self.sasa = sasa

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def center_and_transform(label, get_position):
    reference = get_position["CA"]
    axis_x = numpy.array(get_position["N"]) - numpy.array(get_position["CA"])
    pseudo_axis_y = numpy.array(get_position["C"]) - numpy.array(get_position["CA"])
    axis_z = numpy.cross(axis_x, pseudo_axis_y)
    if not label == 18:
        direction = numpy.array(get_position["CB"]) - numpy.array(get_position["CA"])
        axis_z *= numpy.sign(direction.dot(axis_z))
    axis_y = numpy.cross(axis_z, axis_x)

    axis_x /= numpy.sqrt(sum(axis_x ** 2))
    axis_y /= numpy.sqrt(sum(axis_y ** 2))
    axis_z /= numpy.sqrt(sum(axis_z ** 2))

    transform = numpy.array([axis_x, axis_y, axis_z], 'float16').T
    return reference, transform


def get_position_dict(all_PDB_atoms):
    get_position = {}
    for a in all_PDB_atoms:
        get_position[a.atom] = (a.x, a.y, a.z)
    return get_position


def load_pdb_set(d_name):
    PDB_list_file = open(os.path.join(PROJECT_ROOT, 'PDB_' + d_name + '.txt'))
    PDB_Set = set()
    for line in PDB_list_file:
        PDB_ID = line.split()[0]
        PDB_Set.add(PDB_ID.lower())
    return PDB_Set


def grab_PDB_csv(path):
    ID_dict = collections.OrderedDict()
    # ID_dict = {'A': []}  # 暂时只有一条A链
    all_pos = []
    all_atom_type = []
    PDB_entries = []
    atom_index = 0
    all_x = []
    all_y = []
    all_z = []
    with open(path, 'r') as f:

        reader = csv.DictReader(f, fieldnames=['atom', 'serial', 'name', 'resname', 'resid', 'x', 'y', 'z', 'charges',
                                               'radius', 'sasa'])
        next(reader)
        for row in reader:
            if row['atom'] != 'ATOM':
                break
            try:
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])
                # charge = float(row['charges'])
                # sasa = float(row['sasa'])
                # 检查charges是否为空或只包含空白字符
                charges_str = row['charges'].strip()
                charge = float(charges_str) if charges_str else 0.0  # 或者使用默认值，例如0.0

                # 检查sasa是否为空或只包含空白字符
                sasa_str = row['sasa'].strip()
                sasa = float(sasa_str) if sasa_str else 0.0  # 或者使用默认值，例如0.0

                resid = row['resid']

                all_pos.append([x, y, z])  # 原子名称
                all_x.append(x)
                all_y.append(y)
                all_z.append(z)
                atom_name = str(row['name'])
                all_atom_type.append(atom_name[0])  # 原子名称
                res = str(row['resname'])  # 残基名称

                chain_ID = ('A', str(resid))  # 链名称暂定为A
                if chain_ID not in ID_dict:
                    ID_dict[chain_ID] = [PDB_atom(atom_name, res, chain_ID, x, y, z,
                                                index=atom_index, value=1, charge=charge, sasa=sasa)]
                else:
                    ID_dict[chain_ID].append(
                        PDB_atom(atom_name, res, chain_ID, x, y, z,
                                index=atom_index, value=1, charge=charge, sasa=sasa)
                    )
                PDB_entries.append(
                    PDB_atom(atom_name, res, chain_ID, x, y, z,
                            index=atom_index, value=1, charge=charge, sasa=sasa))
                atom_index += 1
            except ValueError as e:
                print(f"警告: 文件 {file} 中的数据格式异常，跳过该行: {e}")
                continue 

    # 创建PROTEIN_csv列表
    PROTEIN_csv = [ID_dict, all_pos, all_atom_type, PDB_entries, all_x, all_y, all_z]

    return PROTEIN_csv


def pts_to_Xsmooth(PROTEIN, pts, atom_density, num_of_channels, pixel_size, box_size, check=True):
    num_3d_pixel = int(box_size / pixel_size)
    ID_dict, all_pos, all_atom_type, PDB_entries, _, _, _ = PROTEIN
    [pos, chain_ID, label] = pts
    backbone = ID_dict[chain_ID][0:4]
    deleted_res = ID_dict[chain_ID][4:]
    deleted_res_index = [atom.index for atom in deleted_res]
    box_ori = []
    X_smooth = []
    reference = []
    sample = []
    new_pos_in_box = []
    valid_box = False
    box_x_min = -box_size / 2
    box_x_max = +box_size / 2
    box_y_min = -box_size / 2
    box_y_max = +box_size / 2
    box_z_min = -box_size / 2
    box_z_max = +box_size / 2

    get_position = get_position_dict(ID_dict[chain_ID])

    # print("if:", set(get_position.keys()) == label_atom_type_dict[label])
    if check and set(get_position.keys()) != label_atom_type_dict[label]:
        pass
    else:
        reference, transform = center_and_transform(label, get_position)
        all_pos = numpy.array(all_pos)
        centered_pos = all_pos - reference
        transformed_pos = numpy.empty_like(centered_pos)
        transformed_pos[:, 0] = (
            centered_pos[:, 0] * transform[0, 0]
            + centered_pos[:, 1] * transform[1, 0]
            + centered_pos[:, 2] * transform[2, 0]
        )
        transformed_pos[:, 1] = (
            centered_pos[:, 0] * transform[0, 1]
            + centered_pos[:, 1] * transform[1, 1]
            + centered_pos[:, 2] * transform[2, 1]
        )
        transformed_pos[:, 2] = (
            centered_pos[:, 0] * transform[0, 2]
            + centered_pos[:, 1] * transform[1, 2]
            + centered_pos[:, 2] * transform[2, 2]
        )
        transformed_pos = transformed_pos - bias
        x_index = numpy.intersect1d(numpy.where(transformed_pos[:, 0] > box_x_min),
                                    numpy.where(transformed_pos[:, 0] < box_x_max))
        y_index = numpy.intersect1d(numpy.where(transformed_pos[:, 1] > box_y_min),
                                    numpy.where(transformed_pos[:, 1] < box_y_max))
        z_index = numpy.intersect1d(numpy.where(transformed_pos[:, 2] > box_z_min),
                                    numpy.where(transformed_pos[:, 2] < box_z_max))

        final_index = numpy.intersect1d(x_index, y_index)
        final_index = numpy.intersect1d(final_index, z_index)
        final_index = final_index.tolist()
        final_index = [ind for ind in final_index if ind not in deleted_res_index]
        final_index = [ind for ind in final_index if (
                    all_atom_type[ind] == 'C' or all_atom_type[ind] == 'O' or all_atom_type[ind] == 'S' or
                    all_atom_type[ind] == 'N' or all_atom_type[ind] == 'H')]

        box_ori = [PDB_entries[i] for i in final_index]
        new_pos_in_box = transformed_pos[final_index]
        atom_count = len(box_ori)
        threshold = (box_size ** 3) * atom_density
        # print("atom_count: ", atom_count)
        # print("threshold: ", threshold)

        if atom_count > threshold:
            valid_box = True
            sample = numpy.zeros((num_of_channels, num_3d_pixel, num_3d_pixel, num_3d_pixel))

            for i in range(0, len(box_ori)):
                atoms = box_ori[i]
                x = new_pos_in_box[i][0]
                y = new_pos_in_box[i][1]
                z = new_pos_in_box[i][2]

                x_new = x - box_x_min
                y_new = y - box_y_min
                z_new = z - box_z_min
                bin_x = int(numpy.floor(x_new / pixel_size))
                bin_y = int(numpy.floor(y_new / pixel_size))
                bin_z = int(numpy.floor(z_new / pixel_size))

                if (bin_x == num_3d_pixel):
                    bin_x = num_3d_pixel - 1

                if (bin_y == num_3d_pixel):
                    bin_y = num_3d_pixel - 1

                if (bin_z == num_3d_pixel):
                    bin_z = num_3d_pixel - 1

                if atoms.atom[0] == 'O':
                    sample[0, bin_x, bin_y, bin_z] = sample[0, bin_x, bin_y, bin_z] + atoms.value
                elif atoms.atom[0] == 'C':
                    sample[1, bin_x, bin_y, bin_z] = sample[1, bin_x, bin_y, bin_z] + atoms.value
                elif atoms.atom[0] == 'N':
                    sample[2, bin_x, bin_y, bin_z] = sample[2, bin_x, bin_y, bin_z] + atoms.value
                elif atoms.atom[0] == 'S':
                    sample[3, bin_x, bin_y, bin_z] = sample[3, bin_x, bin_y, bin_z] + atoms.value
                elif atoms.atom[0] == 'H':
                    sample[4, bin_x, bin_y, bin_z] = sample[4, bin_x, bin_y, bin_z] + atoms.value
                sample[5, bin_x, bin_y, bin_z] = sample[5, bin_x, bin_y, bin_z] + atoms.charge
                sample[6, bin_x, bin_y, bin_z] = sample[6, bin_x, bin_y, bin_z] + atoms.sasa

    return sample, label, reference, box_ori, new_pos_in_box, valid_box

if __name__ == '__main__':

    # d_name = sys.argv[1]  # d_name can be 'train' or 'test'
    d_name = 'homelog_merge'
    num_of_channels = 7
    atom_density = 0.01  # defalut = 0.01, desired threshold of atom density of boxes defined by num_of_atom / box volume
    box_size = 20
    pixel_size = 1
    threshold = 0.5
    test = True

    PDB_DIR = DATA_CSV_DIR
    dat_dir = PRETRAIN_NPZ_DIR
    dict_name = d_name + '_20AA_boxes.json'
    # sample_block = 1000 if d_name == 'train' else 100
    sample_block = 100
    samples = []

    dict_dir = os.path.join(PROJECT_ROOT, "dict")
    if not os.path.exists(dict_dir):
        os.makedirs(dict_dir)

    if not os.path.exists(dat_dir):
        os.makedirs(dat_dir)

    PDBs = [os.path.splitext(file)[0] for file in os.listdir(PDB_DIR) if file.lower().endswith(".csv")]
    res_count_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0,
                      15: 0, 16: 0, 17: 0, 18: 0, 19: 0}

    for PDB_ID in tqdm(PDBs):

        file = os.path.join(PDB_DIR, PDB_ID + '.csv')
        #print(file)
        #file = os.path.join(PDB_DIR, PDB_ID + '.csv')
        if os.path.isfile(file):
            PROTEIN = grab_PDB_csv(file)
            ID_dict, all_pos, all_atom_type, PDB_entries, all_x, all_y, all_z = PROTEIN
            ID_dict_keys = list(ID_dict.keys())
            aa_num = len(ID_dict_keys)

            ctr_pos = []
            visited = set()

            if not test:
                sample_num = min(int(threshold * aa_num), 100)
                random.shuffle(ID_dict_keys)
                sample_aa_chain_ids = ID_dict_keys[:sample_num]
            else:
                sample_aa_chain_ids = ID_dict_keys

            for chain_ID in sample_aa_chain_ids:
                res_atoms = ID_dict[chain_ID]
                res = res_atoms[0].res
                if res in list(res_label_dict.keys()):
                    label = res_label_dict[res]
                    get_position = get_position_dict(res_atoms)
                    if "CA" in get_position.keys():
                        ctr = get_position["CA"]
                        if ctr not in visited:
                            visited.add(ctr)
                            ctr_pos.append([ctr, chain_ID, label])

            for pts in ctr_pos:

                Xsmooth, label, _, _, _, valid_box = pts_to_Xsmooth(PROTEIN, pts, atom_density,
                                                                    num_of_channels, pixel_size,
                                                                    box_size)
                if valid_box:

                    res_container_dict[label].append(Xsmooth)
                    if (len(res_container_dict[label]) == sample_block):
                        sample_time_t = numpy.array(res_container_dict[label], dtype=numpy.float64)
                        res_container_dict[label] = []
                        save_file = dat_dir + '/' + label_res_dict[label] + "_" + str(res_count_dict[label]) + '.npz'
                        numpy.savez_compressed(save_file, arr=sample_time_t)
                        # sample_time_t.dump(
                        #     dat_dir + '/' + label_res_dict[label] + "_" + str(res_count_dict[label]) + '.dat')
                        res_count_dict[label] += 1
                        with open(os.path.join(dict_dir, dict_name), 'w') as f:
                            json.dump(res_count_dict, f)
                            # print("dump dictionary...")

    print("done generating 20 amino acid boxes, storing dictionary..")
    for label, samples in res_container_dict.items():
        if samples:
            sample_time_t = numpy.array(samples, dtype=numpy.float64)
            save_file = os.path.join(dat_dir, label_res_dict[label] + "_" + str(res_count_dict[label]) + ".npz")
            numpy.savez_compressed(save_file, arr=sample_time_t)
            res_count_dict[label] += 1

    with open(os.path.join(dict_dir, dict_name), 'w') as f:
        json.dump(res_count_dict, f)


