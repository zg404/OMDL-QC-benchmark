# Nanopore Consensus Sequence Benchmark (Dorado vs. Guppy)

## Project Goal

This project provides a Python-based workflow and interactive Jupyter notebook (`analysis_interface.ipynb`) to facilitate a comparative analysis of Nanopore consensus sequences generated from the same raw sequencing data but processed using two different basecalling pipelines: **Dorado** (newer) and **Guppy** (older standard). The objective is to quantitatively assess differences and potential improvements offered by the Dorado pipeline by comparing key sequence metrics and characteristics.

## Background

The analysis focuses on fungal Internal Transcribed Spacer (ITS) amplicon sequencing data from multiplexed Nanopore runs. Raw signal data for each run (`OMDL*`) was independently processed through both Guppy and Dorado basecallers, followed by demultiplexing, clustering, and consensus generation. This project aims to compare the outputs (FASTA consensus sequences) from these parallel pipelines.

## Features

* **Run Discovery:** Automatically identifies sequencing runs with paired Dorado and Guppy FASTA files in the `seqs/` directory.
* **Sequence Loading:** Parses FASTA files, extracting header information like Sample ID and Reads in Consensus (RiC).
* **Sequence Matching:** Implements logic to pair corresponding consensus sequences from Dorado and Guppy datasets for the same biological sample, handling 1:1 and complex N:M scenarios using k-mer similarity and pairwise alignment.
* **Metric Calculation:** Computes various metrics for comparison, including:
    * Reads in Consensus (RiC) 
    * Sequence Length 
    * GC Content 
    * Alignment Identity, Mismatches, Insertions, Deletions 
    * Homopolymer run characteristics 
    * Ambiguity code counts/frequency 
* **Statistical Analysis:** Performs paired non-parametric tests (Wilcoxon signed-rank test) to evaluate the significance of differences between matched pairs for key metrics.
* **Interactive Visualization:** Generates plots within the Jupyter notebook to compare metrics (scatter plots, difference histograms) for interactively selected runs.
* **Alignment Viewer:** Provides an interactive tool within the notebook to view highlighted pairwise alignments of matched sequences.
* **Data Export:** Saves detailed run-specific comparison data and an overall summary across all runs to TSV files in the `results/` directory.

## Project Structure

```
nanopore-consensus-benchmark/
├── README.md                 # This file
├── analysis_interface.ipynb  # Main Jupyter notebook for running analysis
├── data_functions.py         # Core Python module for data loading, matching, analysis
├── viz_handler.py            # Python module for plotting and visualization functions
├── requirements.txt          # Python package dependencies
├── test_book.ipynb           # Notebook for testing functions (optional)
├── .dev.blueprint.md         # Development plan (internal use)
├── .dev.outline.md           # Project outline and context (internal use)
├── results/                  # Directory for output TSV files
│   ├── {run_id}_comparison_data.tsv
│   └── overall_comparison_summary.tsv
└── seqs/                     # Directory for input FASTA sequence files
    ├── OMDL*_seqs_dorado.fasta
    └── OMDL*_seqs_guppy.fasta
```

## Setup

1.  **Clone Repository:**
    ```bash
    git clone <your-repo-url>
    cd nanopore-consensus-benchmark
    ```
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Prepare Input Data:** Place your paired Nanopore consensus sequence FASTA files into the `seqs/` directory. Ensure filenames follow the pattern `OMDL{number}_seqs_{basecaller}.fasta` (e.g., `OMDL1_seqs_dorado.fasta`, `OMDL1_seqs_guppy.fasta`).
2.  **Launch Jupyter:** Start Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```
3.  **Run Notebook:** Open and run the `analysis_interface.ipynb` notebook. The notebook will:
    * Discover available runs.
    * Process each run with paired data: load sequences, match pairs, calculate metrics, perform statistics, and save run-specific results.
    * Generate and save an overall summary table.
    * Provide interactive widgets to select runs and view detailed analysis plots and alignments.

## Outputs

Analysis results are saved in the `results/` directory:

1.  **`{run_id}_comparison_data.tsv`:** A tab-separated file generated for each processed run, containing detailed metrics for every matched sequence pair. Includes columns for Sample ID, headers, RiC, length, GC content, alignment identity, mismatches, indels, homopolymer counts, ambiguity counts, and the calculated differences between Dorado and Guppy for relevant metrics.
2.  **`overall_comparison_summary.tsv`:** A single tab-separated file summarizing key statistics across all processed runs. Includes columns for Run ID, counts of matched/unmatched sequences, and aggregate statistics (median differences, p-values) for major metrics like RiC and Length.

## Dependencies

Key Python packages used (see `requirements.txt` for specific versions):

* pandas
* numpy
* biopython (>=1.80 recommended)
* scipy
* matplotlib
* seaborn
* ipywidgets
* IPython
* natsort