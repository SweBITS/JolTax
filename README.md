# joltax

**High-performance, vectorized NCBI taxonomy library for large-scale bioinformatics research.**

`joltax` is a Python library designed to handle the massive NCBI taxonomy (and derivatives like GTDB) with extreme efficiency. By replacing traditional object-oriented trees with contiguous NumPy arrays and leveraging Polars for mass data handling, it achieves lightning-fast traversals, constant-time clade queries, and rapid mass annotation of large datasets.

## Key Features

- **Vectorized Performance:** Uses hardware-accelerated NumPy operations for O(1) property lookups.
- **2025 Taxonomy Ready:** Native support for the recent NCBI/GTDB shift from `superkingdom` to `domain`.
- **Fuzzy Name Search:** Rapid, rank-aware fuzzy matching using RapidFuzz to find TaxIDs from names.
- **Euler Tour Indexing:** Instant clade range queries (even for millions of nodes).
- **Binary Lifting (Skip Tables):** Logarithmic-time Lowest Common Ancestor (LCA) and distance calculations.
- **Mass Annotation:** Annotate tables with 200,000+ rows in under a second using Polars.
- **Full Provenance:** Binary caches store build timestamps, version validation, and source file paths.

## Quick Start

```python
from joltax import TaxonomyTree

# Build and process the NCBI taxonomy
tree = TaxonomyTree(nodes_file='nodes.dmp', names_file='names.dmp')

# Save for instant loading next time
tree.save('my_taxonomy_cache')

# Re-load in seconds (with version validation)
tree = TaxonomyTree.load('my_taxonomy_cache')

# Fuzzy search for a name (returns a Polars DataFrame)
results = tree.search_name('Escherchia', fuzzy=True)
print(results)

# Get all genera in the Bacteria clade
bacteria_genera = tree.get_clade_at_rank(2, 'genus')
```

## Installation

```bash
cd joltax
pip install .
```

Requires: `numpy`, `polars`, `pandas`, `rapidfuzz`.

## Documentation

For a detailed API reference and a comprehensive "How-To" guide with example workflows, please see [USAGE.md](./USAGE.md).
