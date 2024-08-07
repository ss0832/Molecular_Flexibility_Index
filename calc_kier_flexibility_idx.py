import numpy as np
import itertools

# Kier flexibility index
# ref.1: Quant. Struct.‐Act. Relat. 1989, 8 (3), 221–224.
# ref.2: J. Chem. Inf. Comput. Sci. 1996, 36, 711-716


covalent_radii = {"H": 0.32, "He": 0.46, 
           "Li": 1.33, "Be": 1.02, "B": 0.85, "C": 0.75, "N": 0.71, "O": 0.63, "F": 0.64, "Ne": 0.67, 
           "Na": 1.55, "Mg": 1.39, "Al":1.26, "Si": 1.16, "P": 1.11, "S": 1.03, "Cl": 0.99, "Ar": 0.96, 
           "K": 1.96, "Ca": 1.71, "Sc": 1.48, "Ti": 1.36, "V": 1.34, "Cr": 1.22, "Mn": 1.19, "Fe": 1.16, "Co": 1.11, "Ni": 1.10, "Cu": 1.12, "Zn": 1.18, "Ga": 1.24, "Ge": 1.24, "As": 1.21, "Se": 1.16, "Br": 1.14, "Kr": 1.17, 
           "Rb": 2.10, "Sr": 1.85, "Y": 1.63, "Zr": 1.54,"Nb": 1.47,"Mo": 1.38,"Tc": 1.28,"Ru": 1.25,"Rh": 1.25,"Pd": 1.20,"Ag": 1.28,"Cd": 1.36,"In": 1.42,"Sn": 1.40,"Sb": 1.40,"Te": 1.36,"I": 1.33,"Xe": 1.31,
           "Cs": 2.32,"Ba": 1.96,"La":1.80,"Ce": 1.63,"Pr": 1.76,"Nd": 1.74,"Pm": 1.73,"Sm": 1.72,"Eu": 1.68,"Gd": 1.69 ,"Tb": 1.68,"Dy": 1.67,"Ho": 1.66,"Er": 1.65,"Tm": 1.64,"Yb": 1.70,"Lu": 1.62,"Hf": 1.52,"Ta": 1.46,"W": 1.37,"Re": 1.31,"Os": 1.29,"Ir": 1.22,"Pt": 1.23,"Au": 1.24,"Hg": 1.33,"Tl": 1.44,"Pb":1.44,"Bi":1.51,"Po":1.45,"At":1.47,"Rn":1.42}#ang. (H...Rn) single_bond
           # ref. Pekka Pyykkö; Michiko Atsumi (2009) Chemistry: A European Journal 15: 186–197. doi:10.1002/chem.200800987. 
            
Csp3_covalent_radius = 0.75 # ref. Pekka Pyykkö; Michiko Atsumi (2009) Chemistry: A European Journal 15: 186–197. doi:10.1002/chem.200800987. 



def read_xyz(file_path):
    xyz = []
    element_list = []
    
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    
    for i in range(2, len(lines)):
        if lines[i] == '':
            break
        element_list.append(lines[i].split()[0])
        xyz.append([float(x) for x in lines[i].split()[1:4]])    
    
    xyz = np.array(xyz, dtype="float64")
    
    return element_list, xyz

def extract_molecule_skeleton_from_xyz(element_list, xyz):
    skeleton_element_list = []
    skeleton_xyz = []
    
    for i in range(len(element_list)):
        if element_list[i] == "H":
            continue
        else:
            skeleton_element_list.append(element_list[i])
            skeleton_xyz.append(xyz[i])
    skeleton_xyz = np.array(skeleton_xyz, dtype="float64")
    return skeleton_element_list, skeleton_xyz

def calc_a_value(skeleton_element_list, skeleton_xyz, list_of_bonds, list_of_bond_lengths):
    # Calculate covalent radii based on the skeleton
    # There are some modifications for the molecular structure optimized by a initio calculation
    covalent_radii_list_based_on_sk_xyz = [] 
    
    for i in range(len(skeleton_element_list)):
        
        cov_radii_list_i = []
        
        for j in range(len(list_of_bonds)):
            if i in list_of_bonds[j]:
                cov_radii_list_i.append(list_of_bond_lengths[j] / 2.0)
        cov_radii_list_i = np.array(cov_radii_list_i)
        #print("cov_radii_list_i: \n", cov_radii_list_i)
        mean_cov_radii_i = np.mean(cov_radii_list_i)
        #print("mean_cov_radii_i: ", mean_cov_radii_i)
        covalent_radii_list_based_on_sk_xyz.append(mean_cov_radii_i)
    
    
    print("covalent_radii_list_based_on_sk_xyz: \n", covalent_radii_list_based_on_sk_xyz)
    a_value = 0.0
    if len(skeleton_element_list) == 1:
        return a_value
    
    for i in range(len(skeleton_element_list)):
        covalent_radius_i = covalent_radii_list_based_on_sk_xyz[i]
        a_value += (covalent_radius_i / Csp3_covalent_radius) - 1.0
    print("a_value: ", a_value)
    return a_value

def calc_kier_flexibility_idx(element_list, xyz, covalent_radius_scaling_factor=1.2):

    skeleton_element_list, skeleton_xyz = extract_molecule_skeleton_from_xyz(element_list, xyz)
    print("skeleton_element_list: \n", skeleton_element_list)
    print("skeleton_xyz: \n", skeleton_xyz)
    # Number of atoms
    n = len(skeleton_element_list)
    sk_atom_idx_list = list(range(n))
    
    # Calculate the number of bonds
    num_bonds = 0
    list_of_bonds = []
    list_of_bond_lengths = []
    for i ,j in itertools.combinations(sk_atom_idx_list, 2):
        covalent_length_ij = (covalent_radii[skeleton_element_list[i]] + covalent_radii[skeleton_element_list[j]]) * covalent_radius_scaling_factor
        length_ij = np.linalg.norm(skeleton_xyz[i] - skeleton_xyz[j])
        if length_ij < covalent_length_ij:
            num_bonds += 1
            list_of_bonds.append([i, j])
            list_of_bond_lengths.append(length_ij)
                
    print("list_of_bonds: \n", list_of_bonds)
    print("list_of_bond_lengths: \n", list_of_bond_lengths)
    
    
    # Calculate the number of bond angles
    num_bond_angles = 0
    list_of_bond_angles = []
    
    for i ,j, k in itertools.combinations(sk_atom_idx_list, 3):
        
        covalent_length_ij = (covalent_radii[skeleton_element_list[i]] + covalent_radii[skeleton_element_list[j]]) * covalent_radius_scaling_factor
        covalent_length_jk = (covalent_radii[skeleton_element_list[j]] + covalent_radii[skeleton_element_list[k]]) * covalent_radius_scaling_factor
        covalent_length_ki = (covalent_radii[skeleton_element_list[k]] + covalent_radii[skeleton_element_list[i]]) * covalent_radius_scaling_factor
        
        if np.linalg.norm(skeleton_xyz[i] - skeleton_xyz[j]) < covalent_length_ij and np.linalg.norm(skeleton_xyz[j] - skeleton_xyz[k]) < covalent_length_jk:
            num_bond_angles += 1
            list_of_bond_angles.append([i, j, k])
        
        if np.linalg.norm(skeleton_xyz[j] - skeleton_xyz[k]) < covalent_length_jk and np.linalg.norm(skeleton_xyz[k] - skeleton_xyz[i]) < covalent_length_ki:
            num_bond_angles += 1
            list_of_bond_angles.append([j, k, i])
        
        if np.linalg.norm(skeleton_xyz[k] - skeleton_xyz[i]) < covalent_length_ki and np.linalg.norm(skeleton_xyz[i] - skeleton_xyz[j]) < covalent_length_ij:
            num_bond_angles += 1
            list_of_bond_angles.append([k, i, j])
            
    print("list_of_bond_angles: \n", list_of_bond_angles)
    """
    # Calculate the number of torsion angles
    num_torsion_angles = 0
    list_of_torsion_angles = []
    for i, j ,k ,l in itertools.combinations(sk_atom_idx_list, 4):
        covalent_length_ij = (covalent_radii[skeleton_element_list[i]] + covalent_radii[skeleton_element_list[j]]) * covalent_radius_scaling_factor
        covalent_length_ik = (covalent_radii[skeleton_element_list[i]] + covalent_radii[skeleton_element_list[k]]) * covalent_radius_scaling_factor
        covalent_length_il = (covalent_radii[skeleton_element_list[i]] + covalent_radii[skeleton_element_list[l]]) * covalent_radius_scaling_factor
        covalent_length_jk = (covalent_radii[skeleton_element_list[j]] + covalent_radii[skeleton_element_list[k]]) * covalent_radius_scaling_factor
        covalent_length_jl = (covalent_radii[skeleton_element_list[j]] + covalent_radii[skeleton_element_list[l]]) * covalent_radius_scaling_factor
        covalent_length_kl = (covalent_radii[skeleton_element_list[k]] + covalent_radii[skeleton_element_list[l]]) * covalent_radius_scaling_factor
        
        # ...
        
    print("list_of_torsion_angles: \n", list_of_torsion_angles)
    """
    # Calculate the Kier flexibility index

    a = calc_a_value(skeleton_element_list, skeleton_xyz, list_of_bonds, list_of_bond_lengths)

    kappa_1 = (n + a) * (n + a - 1) ** 2 / (num_bonds + a + 1e-8) ** 2 
    kappa_2 = (n + a - 1) * (n + a - 2) ** 2 / (num_bond_angles + a + 1e-8) ** 2
    
    kier_flexibility_idx_phi = (kappa_1 * kappa_2) / n
    
    
    return kier_flexibility_idx_phi


if __name__ == "__main__":
    import sys
    file_path = sys.argv[1]
    element_list, xyz = read_xyz(file_path)
    print("element_list: \n", element_list)
    print("xyz: \n", xyz)
    
    kier_flexibility_idx = calc_kier_flexibility_idx(element_list, xyz)
    print(file_path," Kier flexibility index (phi): ", kier_flexibility_idx)
    