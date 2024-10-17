import random
from pytorch_lightning import LightningDataModule
# from pytorch_lightning.utilities import CombinedLoader
import pickle
from src.datamodules.components.intact_dataset import IntactDataset
from src.datamodules.components.helper import run_diffdock
import torch.nn.functional as F
import os
import torch
from tqdm import tqdm
from pathlib import Path

from typing import *
from torch.utils.data import DataLoader
from torch_geometric.data import Data


class IntactDataModule(LightningDataModule):
    def __init__(
        self,
        mode: str = None,
        data_dir: Path = None,
        intact_filtered_entries: str = None,
        pdb_filename: str = None,
        cache_filename: str = "dataset_cache",
        cache_processed_data: bool = True,
        force_process_data: bool = False,

        mask_prob: float = 0.1,
        data_split: Tuple[int, int, int] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.mode = self.hparams.mode
        self.cache_path = Path(self.hparams.data_dir, self.hparams.cache_filename)
        self.pdb_path = Path(self.hparams.data_dir, self.hparams.pdb_filename)
        
        self.split_file = Path(self.hparams.data_dir, "intact_train_data.pkl")
        self.data_splits = self.prcoess()
        
    def prcoess(self) -> Dict:
        entries = self.load_intact_entries()
        
        if not os.path.exists(self.split_file):
            data_splits = self.random_split(entries)
        else:
            with open(self.split_file, 'rb') as f:
                data_splits = pickle.load(f)
        return data_splits

    def load_intact_entries(self) -> List[Dict]:
        if self.hparams.intact_filtered_entries is None:
            raise ValueError("intact_filtered_entries path must be provided")
    
        with open(Path(self.hparams.data_dir, self.hparams.intact_filtered_entries), 'rb') as f:
            entries = pickle.load(f)
        return entries

    def random_split(self, entries: List[Dict]) -> Dict:
        grouped_entries = {}
        for entry in entries:
            pair_type = entry['Pair_type']
            if pair_type not in grouped_entries:
                grouped_entries[pair_type] = []
            grouped_entries[pair_type].append(entry)
        
        data_splits = {'train': [], 'valid': [], 'test': []}

        for pair_type, group in grouped_entries.items():
            random.shuffle(group)
            total_count = len(group)
            test_count = int(self.hparams.data_split[2] * total_count)
            valid_count = int(self.hparams.data_split[1] * total_count)

            data_splits['test'].extend(group[:test_count])
            data_splits['valid'].extend(group[test_count:(test_count + valid_count)])
            data_splits['train'].extend(group[(test_count + valid_count):])

        for key in data_splits:
            random.shuffle(data_splits[key])

        with open(self.split_file, 'wb') as f:
            pickle.dump(data_splits, f)
        
        return data_splits

    def setup(self, stage: Optional[str] = None):
        train_entries = self.data_splits['train']
        valid_entries = self.data_splits['valid']
        test_entries = self.data_splits['test']

        self.train_set = IntactDataset(
            path=self.pdb_path,
            entries=[entry for entry in train_entries if entry['Pair_type'] == self.mode],
            mask_prob=self.hparams.mask_prob,
            cache_path=self.cache_path,
            cache_processed_data=self.hparams.cache_processed_data,
            force_process_data=self.hparams.force_process_data
        ) 

        self.val_set = IntactDataset(
            path=self.pdb_path,
            entries=[entry for entry in valid_entries if entry['Pair_type'] == self.mode],
            mask_prob=self.hparams.mask_prob,
            cache_path=self.cache_path,
            cache_processed_data=self.hparams.cache_processed_data,
            force_process_data=self.hparams.force_process_data
        ) 

        self.test_set = IntactDataset(
            path=self.pdb_path,
            entries=[entry for entry in test_entries if entry['Pair_type'] == self.mode],
            mask_prob=self.hparams.mask_prob,
            cache_path=self.cache_path,
            cache_processed_data=self.hparams.cache_processed_data,
            force_process_data=self.hparams.force_process_data
        ) 

    def get_dataloader(
            self,
            dataset: IntactDataset,
            batch_size: int,
            pin_memory: bool,
            shuffle: bool,
            drop_last: bool
    ) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=self.hparams.num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=self.collate_fn
        )

    def train_dataloader(self):
        dataloaders = self.get_dataloader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True
        ) 
        return dataloaders

    def val_dataloader(self):
        dataloaders = self.get_dataloader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True
        ) 
        return dataloaders

    def test_dataloader(self):
        dataloaders = self.get_dataloader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True
        ) 
        return dataloaders

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    def collate_fn(self, protein_batch):
        max_size = max([protein.seq.shape[0] for protein in protein_batch])
    
        def _maybe_pad(tensor, max_size, dim):
            if dim == 1:  # For [seq, seq] tensors (2D padding)
                padding_size = [0, max_size - tensor.shape[1], 0, max_size - tensor.shape[0]]
            else:  # For [seq] tensors (1D padding)
                padding_size = [0] * (len(tensor.shape) * 2)
                padding_size[-1] = max_size - tensor.shape[0]
    
            if tensor.shape[0] < max_size or (dim == 1 and tensor.shape[1] < max_size):  # If padding is needed
                return F.pad(tensor, padding_size)
            return tensor
    
        if self.mode == 'seq-seq':
            batch = Data(
                seq=torch.stack([_maybe_pad(protein.seq, max_size, 0) for protein in protein_batch]),
                mut_seq=torch.stack([_maybe_pad(protein.mut_seq, max_size, 0) for protein in protein_batch]),
                mask=torch.stack([_maybe_pad(protein.mask, max_size, 0) for protein in protein_batch]),
                target_static=torch.stack([_maybe_pad(protein.target_static, max_size, 0) for protein in protein_batch]),
                target_label=torch.tensor([protein.target_label for protein in protein_batch])
            )
        else:
            edge_indices = []
            edge_attrs = []
            batch_edge_labels = []  
    
            for i, protein in enumerate(protein_batch):
                edge_indices.append(protein.edge_index)
                edge_attrs.append(protein.edge_attr)
                
                # For each edge, append the batch index i
                num_edges = protein.edge_index.shape[1]
                batch_edge_labels.append(torch.full((num_edges,), i, dtype=torch.long))
    
            # Concatenate edge indices, edge attributes, and batch edge labels
            edge_index = torch.cat(edge_indices, dim=1)  # Stack without offset
            edge_attr = torch.cat(edge_attrs, dim=0)     # Stack without padding
            batch_edge_labels = torch.cat(batch_edge_labels, dim=0)  # Combine all batch labels for edges
    
            batch = Data(
                seq=torch.stack([_maybe_pad(protein.seq, max_size, 0) for protein in protein_batch]),
                mut_seq=torch.stack([_maybe_pad(protein.mut_seq, max_size, 0) for protein in protein_batch]),
                mask=torch.stack([_maybe_pad(protein.mask, max_size, 0) for protein in protein_batch]),
                align_mask=torch.stack([_maybe_pad(protein.align_mask, max_size, 0) for protein in protein_batch]),
                edge_index=edge_index,  # No offset
                edge_attr=edge_attr,    # No padding
                batch_edge_labels=batch_edge_labels,  # Edge-level batch labels
                target_static=torch.stack([_maybe_pad(protein.target_static, max_size, 0) for protein in protein_batch]),
                target_label=torch.tensor([protein.target_label for protein in protein_batch])
            )
    
        return batch
            
