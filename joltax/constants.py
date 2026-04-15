"""
joltax/constants.py
Taxonomic constants and rank mappings for joltax.
"""

# Standard canonical ranks in order (highest to lowest)
# Including both superkingdom and domain for compatibility with pre/post-2025 taxonomies
CANONICAL_RANKS = [
    'superkingdom', 'domain', 'kingdom', 'phylum', 
    'class', 'order', 'family', 'genus', 'species'
]

# Mapping rank names to standard Kraken-style codes
RANK_TO_CODE = {
    'superkingdom': 'D',
    'domain': 'D',
    'kingdom': 'K',
    'phylum': 'P',
    'class': 'C',
    'order': 'O',
    'family': 'F',
    'genus': 'G',
    'species': 'S'
}

# Macro Groups for coarse-grained classification
# tax_id: The primary NCBI TaxID for the group
# name: The expected scientific name for verification
MACRO_GROUPS = {
    'Bacteria': {'tax_id': 2, 'name': 'Bacteria'},
    'Archaea': {'tax_id': 2157, 'name': 'Archaea'},
    'Viruses': {'tax_id': 10239, 'name': 'Viruses'},
    'Eukaryota': {'tax_id': 2759, 'name': 'Eukaryota'}, # Intermediate group for Protists
    'Fungi': {'tax_id': 4751, 'name': 'Fungi'},
    'Metazoa': {'tax_id': 33208, 'name': 'Metazoa'},
    'Viridiplantae': {'tax_id': 33090, 'name': 'Viridiplantae'}
}

# Ordered list of macro group labels for vectorized lookup
MACRO_GROUP_LABELS = [
    'Other', 
    'Bacteria', 
    'Archaea', 
    'Viruses', 
    'Fungi', 
    'Metazoa', 
    'Viridiplantae', 
    'Protists'
]
