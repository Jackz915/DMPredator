import requests
import os
from io import StringIO
import re
import logging
import argparse
import pickle
from tqdm import tqdm
from Bio.PDB import PDBParser, PDBIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

intact_dataset_url = "https://ftp.ebi.ac.uk/pub/databases/intact/current/various/mutations.tsv"


def download_and_filter(intact_dataset_url):
    filtered_data = []
    seen_entries = set() 

    logging.info(f"Downloading IntAct mutations dataset from {intact_dataset_url}")
    
    try:
        response = requests.get(intact_dataset_url)
        response.raise_for_status()
        lines = response.text.splitlines()
    except requests.RequestException as e:
        logging.error(f"Failed to download the file: {e}")
        raise
    
    logging.info(f"Filtering...")
    
    # Define a regular expression pattern to extract uniprotkb IDs
    uniprotkb_pattern = re.compile(r'uniprotkb:([A-Z0-9]+)')

    for line in lines[1:]:
        columns = line.split("\t")
        if len(columns) < 15:
            continue

        # Extract relevant information
        target_info = columns[7].strip()
        partner_info = columns[11].strip()
        feature_range = columns[2].strip()
        original_seq = columns[3].strip()
        resulting_seq = columns[4].strip()
        organism = columns[10].strip()
            
        if (organism == "9606 - Homo sapiens" and 
            original_seq != '.' and resulting_seq != '.' and 
            len(original_seq) == 1 and len(resulting_seq) == 1):

            if '-' in feature_range:
                mut_location = feature_range.split('-')[0]
            else:
                continue
            
            # Find all uniprotkb IDs in the column
            target_ids = uniprotkb_pattern.findall(target_info)
            partner_ids = uniprotkb_pattern.findall(partner_info)
            
            if not target_ids or not partner_ids:
                continue  # Skip rows with no uniprotkb IDs

            # Append the filtered and expanded data
            for target_id in target_ids:
                for partner_id in partner_ids:
                    if target_id == partner_id:
                        continue 
                        
                    entry = {
                        'Feature_AC': columns[0].strip(),
                        'Target_ID': target_id,
                        'Partner_ID': partner_id,
                        'Mutation': original_seq + mut_location + resulting_seq,
                        'Mutation_Effect_Label': columns[5].strip(),
                        'Affected_protein_symbol': columns[8].strip()
                    }

                    entry_tuple = tuple(entry.items())
                    if entry_tuple not in seen_entries:
                        seen_entries.add(entry_tuple)
                        filtered_data.append(entry)

    return filtered_data

def get_pdb_ids_from_uniprot(uniprot_id):
    """
    Get detailed PDB information associated with a given UniProt ID.
    :param uniprot_id: UniProt ID.
    :return: List containing a dictionary with PDB information or an empty list if no details are found.
    """
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uniprot_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if uniprot_id in data:
            entry = data[uniprot_id][0]
            pdb_info = {
                        'pdb_id': entry.get('pdb_id'),
                        'chain_id': entry.get('chain_id'),
                        # 'experimental_method': entry.get('experimental_method'),
                        # 'resolution': entry.get('resolution')
                    }
            return pdb_info
        return []

    except requests.Timeout:
        # logging.warning(f"Timeout occurred for UniProt ID: {uniprot_id}. Skipping...")
        return []
        
    except requests.RequestException as e:
        # logging.error(f"Request failed: {e}")
        return []

def datahandler(intact_dataset_url, save_dir="pdbs", output_file="intact_filtered_entries.pkl"):
    """
    Download PDB files for all filtered UniProt ID pairs and associated data.
    :param save_dir: Directory where PDB files will be saved.
    :param output_file: PKL file where filtered data and PDB information will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    downloaded_pdbs = set()
    results = []

    # Download and filter the dataset
    filtered_data = download_and_filter(intact_dataset_url)

    logging.info(f"Downloading pdbs...")
    
    def download_pdb(pdb_info, save_dir):
        """Helper function to download and save the PDB chain"""
        pdb_id = pdb_info.get('pdb_id')
        chain_id = pdb_info.get('chain_id')
        pdb_id_chain = pdb_id + '_' + chain_id
        pdb_file_path = os.path.join(save_dir, f"{pdb_id_chain}.pdb")

        if pdb_id_chain in downloaded_pdbs or os.path.exists(pdb_file_path):
            return True  # Skip if already downloaded

        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            response = requests.get(pdb_url, timeout=10)

            # Handle 404 errors for missing PDB files
            if response.status_code == 404:
                logging.warning(f"PDB {pdb_id} not found (404). Skipping {pdb_id_chain}.")
                return False

            response.raise_for_status()  # Raise an exception for other HTTP errors

            # Parse and save the specific chain
            pdb_parser = PDBParser(QUIET=True)
            pdb_content = StringIO(response.text)
            structure = pdb_parser.get_structure(pdb_id, pdb_content)

            chain = structure[0][chain_id]
            io = PDBIO()
            io.set_structure(chain)
            io.save(pdb_file_path)

            downloaded_pdbs.add(pdb_id_chain)
            return True  # Successfully downloaded and saved

        except requests.Timeout:
            logging.warning(f"Timeout while downloading {pdb_id_chain}.")
            return False
        
        except requests.RequestException as e:
            logging.error(f"Failed to download {pdb_id_chain}: {e}")
            return False

    for entry in tqdm(filtered_data, desc="Processing Entries"):
        target_id = entry['Target_ID']
        partner_id = entry['Partner_ID']

        target_info = get_pdb_ids_from_uniprot(target_id)
        partner_info = get_pdb_ids_from_uniprot(partner_id)

        # Try downloading PDBs for both target and partner
        target_downloaded = download_pdb(target_info, save_dir) if target_info else False
        partner_downloaded = download_pdb(partner_info, save_dir) if partner_info else False

        # Skip this entry if either the target or partner PDB couldn't be downloaded
        if target_info and not target_downloaded:
            logging.info(f"Skipping entry with Target ID {target_id} and Partner ID {partner_id} due to missing PDB.")
            continue

        if partner_info and not partner_downloaded:
            logging.info(f"Skipping entry with Target ID {target_id} and Partner ID {partner_id} due to missing PDB.")
            continue

        # Determine Pair_type
        pair_type = (
            'str-str' if target_info and partner_info else
            'seq-seq' if not target_info and not partner_info else
            'seq-str' if not target_info else
            'str-seq'
        )

        # Combine filtered data with PDB details, Pair_type
        combined_data = {
            **entry,
            'target_pdb_chain': f"{target_info['pdb_id']}_{target_info['chain_id']}" if target_info else '',
            'partner_pdb_chain': f"{partner_info['pdb_id']}_{partner_info['chain_id']}" if partner_info else '',
            'Pair_type': pair_type,
        }
        results.append(combined_data)

    # Save the combined data to a PKL file
    with open(output_file, mode='wb') as file:
        pickle.dump(results, file)

    logging.info(f"Filtered data with PDB information saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PDB files and save filtered data.")
    parser.add_argument('--save_dir', type=str, default="pdbs", help="Directory where PDB files will be saved.")
    parser.add_argument('--output_file', type=str, default="intact_filtered_entries.pkl", help="PKL file where filtered data and PDB information will be saved.")
    args = parser.parse_args()

    datahandler(intact_dataset_url=intact_dataset_url, save_dir=args.save_dir, output_file=args.output_file)
