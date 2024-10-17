import os
import pickle
import logging
from tqdm import tqdm
from pathlib import Path
import argparse
import sys

sys.path.append('../')
from src.datamodules.components.helper import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()

def get_chain_length(pdb_file):
    """Helper function to get the number of residues in a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    chain_length = sum(1 for _ in structure.get_residues())
    return chain_length

def process_docking_tasks(entries, diffdock_loc, pdb_path, cache_path, length_threshold):
    """Process docking tasks using DiffDock."""
    to_remove = []  # Store entries to remove on error

    for entry in tqdm(entries, desc="Processing docking tasks", unit="entry"):
        if entry['Pair_type'] != 'str-str':
            continue

        if entry in to_remove:
            continue

        receptor = entry['target_pdb_chain']
        ligand = entry['partner_pdb_chain']
        dock_basename = f"{receptor}_{ligand}"

        
        # Check if receptor and ligand files exist before proceeding
        receptor_pdb = Path(pdb_path, f"{receptor}.pdb")
        ligand_pdb = Path(pdb_path, f"{ligand}.pdb")
        if not receptor_pdb.exists() or not ligand_pdb.exists():
            log.error(f"Receptor or ligand PDB file missing: {receptor_pdb} or {ligand_pdb}")
            to_remove.append(entry)
            continue

        # Calculate the sum of the receptor and ligand lengths
        receptor_length = get_chain_length(receptor_pdb)
        ligand_length = get_chain_length(ligand_pdb)

        if receptor_length + ligand_length > length_threshold:
            log.error(f"Skipping docking {dock_basename} due to length constraint: receptor + ligand length > {length_threshold}")
            to_remove.append(entry)
            continue

        # Skip if the output file already exists
        output_pdb_path = Path(cache_path, f'{dock_basename}.pdb')
        if output_pdb_path.exists():
            try:
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure("test", output_pdb_path)
            except Exception as e:
                log.error(f"Invalid docking file {dock_basename}: {str(e)}")
                to_remove.append(entry)  # Mark entry for removal
            continue

        try:
            # Call run_diffdock with the provided parameters
            result = run_diffdock(
                diffdock_loc=diffdock_loc,
                receptor=receptor_pdb,
                ligand=ligand_pdb,
                pdb_path=pdb_path,
                cache_path=cache_path
            )
                        
        except Exception as e:
            log.error(f"Skipping docking {dock_basename} due to error: {str(e)}")
            to_remove.append(entry)  # Mark entry for removal

    # Remove the entries that caused errors or length exceeded
    for entry in to_remove:
        entries.remove(entry)

    return entries

def main(args):
    # Load and filter entries
    with open(args.pickle_file, 'rb') as f:
        entries = pickle.load(f)
            
    # Process docking tasks
    filtered_entries = process_docking_tasks(
        entries,
        diffdock_loc=args.diffdock_loc,
        pdb_path=args.pdb_path,
        cache_path=args.cache_path,
        length_threshold=args.length_threshold  # Pass the threshold from args
    )

    # Save the filtered entries back
    with open(args.pickle_file, 'wb') as f:
        pickle.dump(filtered_entries, f)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run DiffDock docking tasks on filtered entries.")
    
    parser.add_argument('--pickle_file', type=str, required=True, 
                        help="Path to the pickle file containing the entries.")
    parser.add_argument('--diffdock_loc', type=str, required=True, 
                        help="Location of the DiffDock installation.")
    parser.add_argument('--pdb_path', type=str, required=True, 
                        help="Directory containing PDB files.")
    parser.add_argument('--cache_path', type=str, required=True, 
                        help="Directory to store output cached PDB files.")
    parser.add_argument('--length_threshold', type=int, required=False, default=3000,
                        help="Threshold for the sum of receptor and ligand lengths.")

    # Parse arguments
    args = parser.parse_args()

    # Run the main function
    main(args)
