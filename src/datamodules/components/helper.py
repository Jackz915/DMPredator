import os
from shutil import rmtree, copyfile
import subprocess
import math
import numpy as np
import sys
import pandas as pd
import requests
from tempfile import mkdtemp
from itertools import combinations_with_replacement
import prody
import signal
from Bio.PDB import PDBParser, Chain, Residue, Atom, is_aa, PDBIO
from Bio import pairwise2
from Bio.SeqUtils import seq1
import src.utils.static as STATIC
from src.utils.AAWaveEncoding import AAWaveEncoding


def normalize(mydict):
    """Normalize a dictionary of values to the range [0, 1]."""
    min_val = min(mydict.values())
    max_val = max(mydict.values())
    return {key: (val - min_val) / (max_val - min_val) for key, val in mydict.items()}

def get_static_features():
    """Retrieve and normalize static features from STATIC module."""
    AA_FORMAL_CHARGE = normalize(STATIC.AA_FORMAL_CHARGE)
    NORMALIZED_VAN_DER_WAALS_VOL = normalize(STATIC.NORMALIZED_VAN_DER_WAALS_VOL)
    KYTE_HYDROPATHY_INDEX = normalize(STATIC.KYTE_HYDROPATHY_INDEX)
    STERIC_PARAMETER = normalize(STATIC.STERIC_PARAMETER)
    POLARITY = normalize(STATIC.POLARITY)
    RASA_TRIPEPTIDE = normalize(STATIC.RASA_TRIPEPTIDE)
    RESIDUE_VOLUME = normalize(STATIC.RESIDUE_VOLUME)
    return {
        'AA_FORMAL_CHARGE': AA_FORMAL_CHARGE,
        'NORMALIZED_VAN_DER_WAALS_VOL': NORMALIZED_VAN_DER_WAALS_VOL,
        'KYTE_HYDROPATHY_INDEX': KYTE_HYDROPATHY_INDEX,
        'STERIC_PARAMETER': STERIC_PARAMETER,
        'POLARITY': POLARITY,
        'RASA_TRIPEPTIDE': RASA_TRIPEPTIDE,
        'RESIDUE_VOLUME': RESIDUE_VOLUME
    }

def aa_wave_encoding():
    """Retrieve amino acid wave encodings."""
    return AAWaveEncoding().AA_ENCODED_MEMORY

def one_hot_encode(aa_letter, smooth=False):
    """One-hot encode a single amino acid."""
    aa_to_index = {aa: idx for idx, aa in enumerate(STATIC.AA_1_LETTER)}
    onehot = np.zeros(len(STATIC.AA_1_LETTER), dtype=np.float32)
    idx = aa_to_index.get(aa_letter, -1)
    if idx != -1:
        onehot[idx] = 1.0
    if smooth:
        onehot = onehot * 0.9 + 0.1 / len(STATIC.AA_1_LETTER)
    return onehot

def get_node_features(aa_letter, aa_encodings, static_features):
    """Extract features for a single amino acid."""
    # Amino acid wave encoding
    aa_enc = aa_encodings.get(aa_letter, np.zeros(5, dtype=np.float32))

    # One-hot encoding
    onehot_enc = one_hot_encode(aa_letter, smooth=False)

    # Static features
    normalized_vdw_vol = static_features['NORMALIZED_VAN_DER_WAALS_VOL'].get(aa_letter, 0)
    hydropathy = static_features['KYTE_HYDROPATHY_INDEX'].get(aa_letter, 0)
    steric = static_features['STERIC_PARAMETER'].get(aa_letter, 0)
    polarity = static_features['POLARITY'].get(aa_letter, 0)
    rasa_tripeptide = static_features['RASA_TRIPEPTIDE'].get(aa_letter, 0)
    vol = static_features['RESIDUE_VOLUME'].get(aa_letter, 0)

    features = np.array([
        normalized_vdw_vol, hydropathy, steric,
        polarity, rasa_tripeptide, vol
    ], dtype=np.float32)

    # Concatenate all features
    features = np.hstack([features, aa_enc, onehot_enc])
    return features

def get_all_node_features(fasta_sequence):
    """Extract features for an entire protein sequence."""
    static_features = get_static_features()
    aa_encodings = aa_wave_encoding()
    features = []
    
    for aa in fasta_sequence:
        if aa not in aa_encodings:
            # Use a default feature vector
            default_feature = np.hstack([
                np.zeros(49, dtype=np.float32) # Static features + aa_encodings + one-hot
            ])
            features.append(default_feature)
        else:
            features.append(get_node_features(aa, aa_encodings, static_features))
    
    all_features = np.array(features, dtype=np.float32)
    seq = all_features[:, -23:]
    static = all_features[:, :-23]
    return seq, static

def get_fasta_from_uniprot(uniprot_id):
    """
    Get FASTA sequence for a given UniProt ID.
    :param uniprot_id: UniProt ID.
    :return: FASTA sequence without the header line or an empty string if not found.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        fasta_data = response.text
        if fasta_data:
            fasta_lines = fasta_data.splitlines()
            if len(fasta_lines) > 1:
                # Remove header and join the sequence lines into one string without line breaks
                return "".join(fasta_lines[1:])
        return ""
    except requests.RequestException as e:
        return ""

def get_fasta_from_structure(structure):
    fasta_sequence = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True): 
                    try:
                        # Convert 3-letter residue name to 1-letter code
                        fasta_sequence.append(seq1(residue.get_resname()))
                    except KeyError:
                        pass

    return ''.join(fasta_sequence)
    
def repair_pdb_file(input_pdb):
    """Repair PDB file using ProDy by parsing it and saving the corrected version."""
    try:
        prody.confProDy(verbosity='none')
        structure = prody.parsePDB(input_pdb)
        prody.writePDB(input_pdb, structure)
    except Exception as e:
        pass
        # print(f"Error repairing PDB file: {e}")

def combind_dock(receptor_dock, ligand_dock, output_pdb):
    """
    Combines receptor and ligand PDB files into a single structure.
    """

    parser = PDBParser(QUIET=True)
    io = PDBIO()

    receptor_structure = parser.get_structure("receptor", receptor_dock)
    ligand_structure = parser.get_structure("ligand", ligand_dock)

    # Change receptor chain IDs to 'R'
    for model in receptor_structure:
        for chain in model:
            chain.id = 'R'

    # Change ligand chain IDs to 'L'
    for model in ligand_structure:
        for chain in model:
            chain.id = 'L'

    # Merge structures
    merged_structure = receptor_structure
    merged_structure[0].add(ligand_structure[0]['L'])

    io.set_structure(merged_structure)
    io.save(output_pdb)

def singleProtein_distMat(residues, repr_atom='CA', contact_thresh=8.0, symmetric=True):
    """Generate the distance matrix and contact map for a list of residues."""
    seq_len = len(residues)
    dist_map = np.full((seq_len, seq_len), float('inf'), dtype="float64")
    
    for i in range(seq_len):
        for j in range(i, seq_len):
            try:
                res_a = residues[i]
                res_b = residues[j]
                dist = np.linalg.norm(res_a[repr_atom].get_coord() - res_b[repr_atom].get_coord())
                dist_map[i, j] = dist
                if symmetric:
                    dist_map[j, i] = dist
            except KeyError:
                pass

    # dist_map = np.ma.masked_array(dist_map, np.isnan(dist_map), fill_value=np.nan)
    contact_map = (dist_map < contact_thresh).astype(float)
    return dist_map, contact_map

def run_diffdock(diffdock_loc, 
                 receptor,
                 ligand,
                 pdb_path,
                 cache_path, 
                 fix=True, 
                 conda_env_name='DMPredator', 
                 config_name='config/single_pair_inference.yaml',
                 run_name='temp', 
                 batch_size=1,  
                 num_folds=1, 
                 num_gpu=1, 
                 gpu=0, 
                 num_samples=10, 
                 seed=0, 
                 logger='wandb', 
                 project="DiffDock_Tuning", 
                 score_model_path='checkpoints/large_model_dips/fold_0/', 
                 visualize_n_val_graphs=25, 
                 filtering_model_path='checkpoints/confidence_model_dips/fold_0/'):

    dock_basename = f"{receptor}_{ligand}"
    
    try:
        # Create temporary working directory
        scratch_dir = mkdtemp(prefix="diffdock_")
        struct_dir = os.path.join(diffdock_loc, 'datasets', 'single_pair_dataset', 'structures')
        os.makedirs(os.path.join(scratch_dir, 'visualization'), exist_ok=True)
        os.makedirs(os.path.join(scratch_dir, 'storage'), exist_ok=True)

        # Construct receptor and ligand paths (replace with actual paths if needed)
        receptor_path = os.path.join(pdb_path, f'{receptor}.pdb')
        ligand_path = os.path.join(pdb_path, f'{ligand}.pdb')

        # Copy receptor and ligand files to the required location
        receptor_backup_path = os.path.join(struct_dir, f'{dock_basename}_r_b.pdb')
        ligand_backup_path = os.path.join(struct_dir, f'{dock_basename}_l_b.pdb')
        copyfile(receptor_path, receptor_backup_path)
        copyfile(ligand_path, ligand_backup_path)
        
        # Remove old cache files
        cache_filenames = ['splits_test_cache_v2_b.pkl', 'splits_test_esm_b.pkl', 'splits_test.csv']
        for cache_filename in cache_filenames:
            cache_file_path = os.path.join(diffdock_loc, 'datasets', 'single_pair_dataset', cache_filename)
            if os.path.exists(cache_file_path):
                os.remove(cache_file_path)

        # Create a test split file for the current run
        split_test_file = os.path.join(diffdock_loc, 'datasets', 'single_pair_dataset', 'splits_test.csv')
        with open(split_test_file, mode='w') as file:
            file.write("path,split\n")
            file.write(f"{dock_basename},test\n")

        # Run DiffDock inference for docking
        cmd = (
            f'conda run -n DMPredator python {diffdock_loc}/src/main_inf.py '
            f'--mode "test" '
            f'--config_file {diffdock_loc}/{config_name} '
            f'--run_name {run_name} '
            f'--save_path {scratch_dir} '
            f'--batch_size {batch_size} '
            f'--num_folds {num_folds} '
            f'--num_gpu {num_gpu} '
            f'--gpu {gpu} '
            f'--num_samples {num_samples} '
            f'--seed {seed} '
            f'--logger {logger} '
            f'--project {project} '
            f'--visualization_path {scratch_dir}/visualization/{run_name} '
            f'--visualize_n_val_graphs {visualize_n_val_graphs} '
            f'--score_model_path {diffdock_loc}/{score_model_path} '
            f'--filtering_model_path {diffdock_loc}/{filtering_model_path} '
            f'--prediction_storage {scratch_dir}/storage/{run_name}.pkl '
        )
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) #, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #os.system(cmd)
        
        # Post-process docking results
        receptor_dock = os.path.join(scratch_dir, 'visualization', run_name, 'epoch-0', dock_basename, f'{dock_basename}-receptor.pdb')
        ligand_dock = os.path.join(scratch_dir, 'visualization', run_name, 'epoch-0', dock_basename, f'{dock_basename}-ligand-{num_samples}.pdb')

        repair_pdb_file(receptor_dock)
        repair_pdb_file(ligand_dock)
        output_pdb = os.path.join(cache_path, f'{dock_basename}.pdb')
        combind_dock(receptor_dock, ligand_dock, output_pdb)
        
    except KeyboardInterrupt:
        print("Process interrupted by user (Ctrl+C). Cleaning up...")
    except Exception as e:
        raise e
        # print(f"Error docking {dock_basename}: {e}")
    finally:
        # Clean up temporary files and directories
        if scratch_dir and os.path.exists(scratch_dir):
            rmtree(scratch_dir)

def get_struct_feature(receptor, ligand=None, cache_path=None, return_fasta=True):
    parser = PDBParser(QUIET=True)
    
    if ligand is None:
        structure = parser.get_structure("receptor", receptor)
        distance_map, contact_map = singleProtein_distMat(list(structure.get_residues()))
    else:
        dock_path = os.path.join(cache_path, f'{os.path.splitext(os.path.basename(receptor))[0]}_{os.path.splitext(os.path.basename(ligand))[0]}.pdb')
        structure = parser.get_structure("dock", dock_path)
        distance_map, contact_map = singleProtein_distMat(list(structure.get_residues()))

    fasta = get_fasta_from_structure(structure) if return_fasta else ""
    return distance_map, contact_map, fasta

def align_sequences(seq1, seq2):
    """Align two sequences, padding the shorter one and return the alignment."""
    alignments = pairwise2.align.globalxx(seq1, seq2)
    best_alignment = alignments[0]
    return best_alignment
