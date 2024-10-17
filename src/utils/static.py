import sys

# FORMAT
# ARNDCQEGHILKMFPSTWYVUXO
# {"A": , "R": , "N": , "D": , "C": , "Q": , "E": , "G": , "H": , "I": ,
#  "L": , "K": , "M": , "F": , "P": , "S": , "T": , "W": , "Y": , "V": ,
#  "U": , "X": , "O": }

AA_1_LETTER = "ARNDCQEGHILKMFPSTWYVUXO"
AA_NAMES = {
    "A": "Alanine", "R": "Arginine", "N": "Asparagine", "D": "Aspartic acid", "C": "Cysteine", 
    "Q": "Glutamine", "E": "Glutamic acid", "G": "Glycine", "H": "Histidine", "I": "Isoleucine",
    "L": "Leucine", "K": "Lysine", "M": "Methionine", "F": "Phenylalanine", "P": "Proline", 
    "S": "Serine", "T": "Threonine", "W": "Tryptophan", "Y": "Tyrosine", "V": "Valine",
    "U": "Selenocysteine", "X": "Unknown", "O": "Pyrrolysine"
}

# source: taken from SCONES supplementary table s1: https://pubs.acs.org/doi/suppl/10.1021/acs.jpcb.1c04913/suppl_file/jp1c04913_si_001.pdf
AA_FORMAL_CHARGE = {
    "A": 0, "C": 0, "D": -1, "E": -1, "F": 0, "G": 0, "H": 1, "I": 0, "K": 1, "L": 0, 
    "M": 0, "N": 0, "P": 0, "Q": 1, "R": 1, "S": 0, "T": 0, "V": 0, "W": 0, "Y": 0,
    "U": 0, "X": 0, "O": 0
}

# source: https://www.genome.jp/entry/aaindex:FAUJ880103
NORMALIZED_VAN_DER_WAALS_VOL = {
    "A": 1.00, "R": 6.13, "N": 2.95, "D": 2.78, "C": 2.43, "Q": 3.95, "E": 3.78, "G": 0.00, 
    "H": 4.66, "I": 4.00, "L": 4.00, "K": 4.77, "M": 4.43, "F": 5.89, "P": 2.72, "S": 1.60, 
    "T": 2.60, "W": 8.08, "Y": 6.47, "V": 3.00, "U": 0.00, "X": 0.00, "O": 0.00
}

# source: https://www.genome.jp/entry/aaindex:KYTJ820101
KYTE_HYDROPATHY_INDEX = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, 
    "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, 
    "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2, "U": 0.0, "X": 0.0, "O": 0.0
}

# source: https://www.genome.jp/entry/aaindex:CHAM810101
STERIC_PARAMETER = {
    "A": 0.52, "R": 0.68, "N": 0.76, "D": 0.76, "C": 0.62, "Q": 0.68, "E": 0.68, "G": 0.00, 
    "H": 0.70, "I": 1.02, "L": 0.98, "K": 0.68, "M": 0.78, "F": 0.70, "P": 0.36, "S": 0.53, 
    "T": 0.50, "W": 0.70, "Y": 0.70, "V": 0.76, "U": 0.50, "X": 0.50, "O": 0.50
}

# source: https://www.genome.jp/entry/aaindex:GRAR740102
POLARITY = {
    "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5, "Q": 10.5, "E": 12.3, "G": 9.0, 
    "H": 10.4, "I": 5.2, "L": 4.9, "K": 11.3, "M": 5.7, "F": 5.2, "P": 8.0, "S": 9.2, 
    "T": 8.6, "W": 5.4, "Y": 6.2, "V": 5.9, "U": 5.0, "X": 5.0, "O": 5.0
}

# Residue-accessible surface area in tripeptide. source: https://www.genome.jp/entry/aaindex:CHOC760101
RASA_TRIPEPTIDE = {
    "A": 115., "R": 225., "N": 160., "D": 150., "C": 135., "Q": 180., "E": 190., "G": 75., 
    "H": 195., "I": 175., "L": 170., "K": 200., "M": 185., "F": 210., "P": 145., "S": 115., 
    "T": 140., "W": 255., "Y": 230., "V": 155., "U": 115., "X": 115., "O": 115.
}

# source: https://www.genome.jp/entry/aaindex:GOLD730102
RESIDUE_VOLUME = {
    "A": 88.3, "R": 181.2, "N": 125.1, "D": 110.8, "C": 112.4, "Q": 148.7, "E": 140.5, "G": 60.0, 
    "H": 152.6, "I": 168.5, "L": 168.5, "K": 175.6, "M": 162.2, "F": 189.0, "P": 122.2, "S": 88.7, 
    "T": 118.2, "W": 227.0, "Y": 193.0, "V": 141.4, "U": 120.0, "X": 120.0, "O": 120.0
}
