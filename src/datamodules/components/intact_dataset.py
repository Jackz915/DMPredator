import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
from typing import *
from src.datamodules.components.helper import *


class IntactDataset(Dataset):
    def __init__(self,
                 path: Path = None,
                 entries: List[Dict] = None,
                 mask_prob: float = 0.1,
                 cache_path: Path = None,
                 cache_processed_data: bool = True,
                 force_process_data: bool = False) -> None:
        super().__init__()

        self.path = path
        self.entries = entries
        self.mask_prob = mask_prob
        self.cache_path = cache_path
        self.cache_processed_data = cache_processed_data
        self.force_process_data = force_process_data

        # Initialize label categories and AA_dict once
        self.AA_dict = AAWaveEncoding().AA_dict
        self.label_categories = {
            'mutation(MI:0118)': 0, 'mutation causing(MI:2227)': 1, 
            'mutation increasing(MI:0382)': 1, 'mutation increasing rate(MI:1131)': 1, 
            'mutation increasing strength(MI:1132)': 1, 'mutation with no effect(MI:2226)': 2, 
            'mutation disrupting strength(MI:1128)': 3, 'mutation disrupting(MI:0573)': 3, 
            'mutation decreasing(MI:0119)': 3, 'mutation decreasing strength(MI:1133)': 3, 
            'mutation disrupting rate(MI:1129)': 3, 'mutation decreasing rate(MI:1130)': 3
        }

        self.epsilon = 1e-6 
        self.bins = [0, 4, 8, 12, 16, float('inf')]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Data:
        entry = self.entries[index]
        return self.transform(entry)

    def transform(self, entry) -> Data:
        single_cache_path = self.get_single_cache(entry)
        protein_data = torch.load(single_cache_path) if os.path.exists(str(single_cache_path)) else None

        data = self.prot_to_data(entry, protein_data, single_cache_path)
        data = self.get_mask_with_prob(data, self.mask_prob)
        data = self.get_mut_seq(data, entry['Mutation'])
        data = self.get_label(data, entry['Mutation_Effect_Label'])
        return data

    def get_single_cache(self, entry) -> Path:
        return Path(self.cache_path, f"{entry['Target_ID']}_{entry['Partner_ID']}.pt")

    def prot_to_data(self, entry, protein_data=None, cache_path=None) -> Data:
        # If protein_data is not provided or reprocessing is forced
        if protein_data is None or self.force_process_data:
            # Retrieve features like distance_map, contact_map, align_mask, etc.
            distance_map, contact_map, align_mask, combined_fasta = self.get_features_by_type(entry)
            seq, static_feature = get_all_node_features(combined_fasta)

            num_nodes = seq.shape[0]
            # Initialize node features for protein_data
            node_features = {
                "seq": torch.from_numpy(seq).int(),
                "target_static": torch.from_numpy(static_feature).float(),
            }
    
            # Handle seq-seq type, which does not require edge information
            if entry['Pair_type'] == 'seq-seq':
                protein_data = Data(**node_features, num_nodes=num_nodes)
    
            # Handle cases where edge information is needed
            else:
                # Preprocess distance_map to avoid zero distances and discretize it
                distance_map += self.epsilon  # Avoid zero distances
                distance_map = np.digitize(distance_map, self.bins) - 1  # Discretize distances

                # Convert contact_map to torch.Tensor
                contact_map = torch.from_numpy(contact_map)
                distance_map = torch.from_numpy(distance_map)
            
                # Create edge index and edge attributes
                edge_index = torch.nonzero(contact_map).t().contiguous()  # Edge indices [2, num_edges]
                edge_attr = distance_map[edge_index[0], edge_index[1]].unsqueeze(1)  # Edge attributes (distances)
    
                # Create protein_data containing both node and edge information
                protein_data = Data(
                    **node_features,
                    edge_index=edge_index,  # Edge indices
                    edge_attr=edge_attr,    # Edge attributes (distance values)
                    align_mask=torch.from_numpy(align_mask).float()  # Additional feature
                )
    
            # Ensure no NaN values in the processed data
            protein_data = protein_data.apply(lambda x: torch.nan_to_num(x) if isinstance(x, torch.Tensor) else x)
    
            # Save the processed data if caching is enabled
            if self.cache_processed_data:
                torch.save(protein_data, str(cache_path))
    
        return protein_data


    def get_features_by_type(self, entry):
        pair_type = entry['Pair_type']
        target_pdb_chain = entry['target_pdb_chain']
        partner_pdb_chain = entry['partner_pdb_chain']

        receptor_uniport_fasta = get_fasta_from_uniprot(entry['Target_ID'])
        ligand_uniport_fasta = get_fasta_from_uniprot(entry['Partner_ID'])
        combined_uniport_fasta = receptor_uniport_fasta + ligand_uniport_fasta

        def process_pdb_file(pdb_file, uniport_fasta):
            """Helper function to process PDB file and return padded distance/contact maps."""
            if isinstance(pdb_file, tuple):
                distance_map, contact_map, pdb_fasta = get_struct_feature(
                    receptor=pdb_file[0], ligand=pdb_file[1], 
                    cache_path=self.cache_path,
                    return_fasta=True)
            else:
                distance_map, contact_map, pdb_fasta = get_struct_feature(
                    receptor=pdb_file, 
                    cache_path=self.cache_path,
                    return_fasta=True)
                
            align_mask = self.get_align_mask(uniport_fasta, pdb_fasta, 
                                             align_sequences(uniport_fasta, pdb_fasta))
            
            pad_distance_map, pad_contact_map = self.pad_maps_with_mask(distance_map, contact_map, align_mask)
            return pad_distance_map, pad_contact_map, align_mask
        
        # Handle 'str-str' and 'str-seq' cases
        if pair_type in ['str-str', 'str-seq']:
            r_pdb_file = Path(self.path, f"{target_pdb_chain}.pdb")
            
            # 'str-str' with different chains
            if pair_type == 'str-str' and target_pdb_chain != partner_pdb_chain:
                l_pdb_file = Path(self.path, f"{partner_pdb_chain}.pdb")
                distance_map, contact_map, align_mask = process_pdb_file(
                    (r_pdb_file, l_pdb_file), combined_uniport_fasta)
                return distance_map, contact_map, align_mask, combined_uniport_fasta
            
            # 'str-seq' or 'str-str' with same chain
            distance_map, contact_map, align_mask = process_pdb_file(r_pdb_file, combined_uniport_fasta)
            return distance_map, contact_map, align_mask, combined_uniport_fasta
    
        # Handle 'seq-str' case
        elif pair_type == 'seq-str':
            l_pdb_file = Path(self.path, f"{partner_pdb_chain}.pdb")
            distance_map, contact_map, align_mask = process_pdb_file(l_pdb_file, combined_uniport_fasta)
            return distance_map, contact_map, align_mask, combined_uniport_fasta
    
        # Handle 'seq-seq' case
        elif pair_type == 'seq-seq':
            return [], [], [], combined_uniport_fasta

        raise ValueError(f"Invalid pair type: {pair_type}")

    def get_mask_with_prob(self, data, prob):
        seq_len = data.seq.shape[0]
        max_masked = math.ceil(prob * seq_len)
        rand = torch.rand(seq_len)
        sampled_indices = rand.topk(max_masked, dim=-1).indices
        mask = torch.zeros(seq_len)
        mask.scatter_(0, sampled_indices, 1.0)
        data.mask = mask
        return data

    def get_align_mask(self, uniprot_sequence, pdb_sequence, alignment):
        """Generate a mask to indicate which residues in UniProt sequence are in the PDB sequence."""
        mask = np.zeros(len(uniprot_sequence), dtype=np.float32)

        if not alignment:
            return mask
            
        uniprot_aligned, pdb_aligned, score, begin, end = alignment
        
        pdb_idx = 0  
        seq_idx = 0 
    
        for uniprot_char in uniprot_aligned:
            if seq_idx >= len(uniprot_sequence):
                break  
            if uniprot_char != '-': 
                if pdb_idx < len(pdb_aligned) and pdb_aligned[pdb_idx] != '-':  
                    mask[seq_idx] = 1.0
                if pdb_idx < len(pdb_aligned): 
                    pdb_idx += 1
            seq_idx += 1 
    
        return mask


    def pad_maps_with_mask(self, dist_map, contact_map, mask):
        """Pad dist_map and contact_map to match the length of mask."""
        seq_len = len(mask)  
    
        padded_dist_map = np.full((seq_len, seq_len), float('inf'), dtype=np.float64)
        padded_contact_map = np.full((seq_len, seq_len), 0.0, dtype=np.float64)
        
        original_idx = np.where(mask == 1.0)[0]
        
        for i, original_i in enumerate(original_idx):
            for j, original_j in enumerate(original_idx):
                padded_dist_map[original_i, original_j] = dist_map[i, j]
                padded_contact_map[original_i, original_j] = contact_map[i, j]
        
        return padded_dist_map, padded_contact_map


    def get_mut_seq(self, data, mutation):
        original_aa, position, mutated_aa = mutation[0], int(mutation[1:-1]), mutation[-1]
    
        seq_one_hot = data.seq.clone()  
        original_aa_code = self.AA_dict[original_aa]

        if position > seq_one_hot.shape[0]:
            data.mut_seq = seq_one_hot
            data.mut_index = 0
            return data
            
        if seq_one_hot[position - 1].argmax().item() != original_aa_code:
            data.mut_seq = seq_one_hot
            data.mut_index = 0
            return data
            #raise ValueError(f"Original amino acid at position {position} does not match the mutation {original_aa}")
    
        mut_seq = seq_one_hot.clone()  
        mutated_aa_code = self.AA_dict[mutated_aa]
        mut_seq[position - 1] = torch.zeros_like(mut_seq[position - 1]) 
        mut_seq[position - 1][mutated_aa_code] = 1 
    
        data.mut_seq = mut_seq
        data.mut_index = position - 1
        return data

    def get_label(self, data, label):
        data.target_label = self.label_categories.get(label, 1)  # Default to 1 if label not found
        return data
