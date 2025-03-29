import os
import re
import glob
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from Bio import SeqIO
# from Bio.SeqRecord import SeqRecord
from Bio import Align
from Bio.SeqUtils import gc_fraction
from scipy import stats
import numpy as np
from natsort import natsorted # For natural sorting of run IDs


# Define standard IUPAC ambiguity codes (excluding A, C, G, T)
IUPAC_AMBIGUITY_CODES = "RYMKWSBVDHN"

def extract_run_info(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract run ID (e.g., "OMDL1") and basecaller (e.g., "dorado") from filename.
    Assumes filename format like 'OMDL{number}_seqs_{basecaller}.fasta'.
    """
    # Use regex to capture the number and the basecaller name
    # Pattern: OMDL followed by digits, then _seqs_, then the basecaller name, ending with .fasta
    pattern = r"OMDL(\d+)_seqs_(\w+)\.fasta"
    match = re.search(pattern, filename, re.IGNORECASE) # Use IGNORECASE if dorado/guppy might vary in case
    if match:
        run_number = match.group(1)
        basecaller = match.group(2).lower() # Normalize to lowercase
        run_id = f"OMDL{run_number}"
        # Ensure only expected basecallers are recognized
        if basecaller in ['dorado', 'guppy']:
            return run_id, basecaller
    return None, None

def discover_runs(seqs_dir: str) -> Tuple[pd.DataFrame, dict]:
    """
    Discover all available runs in the seqs directory and check for paired Dorado/Guppy data.

    Args:
        seqs_dir: The path to the directory containing sequence FASTA files.

    Returns:
        A tuple containing:
        - A pandas DataFrame summarizing runs and basecaller availability.
        - A dictionary mapping run IDs to their basecaller status.
    """
    # Define the pattern to search for FASTA files
    pattern = os.path.join(seqs_dir, "OMDL*_seqs_*.fasta")
    seq_files = glob.glob(pattern)

    # Use a dictionary to store run status: {run_id: {'dorado': False, 'guppy': False}}
    all_runs_status = {}

    # Extract run IDs and basecallers from filenames
    for file_path in seq_files:
        filename = os.path.basename(file_path)
        run_id, basecaller = extract_run_info(filename)

        if run_id and basecaller:  # Ensure both were successfully extracted
            # Initialize run_id entry if it's the first time seeing it
            if run_id not in all_runs_status:
                all_runs_status[run_id] = {'dorado': False, 'guppy': False}
            # Update the status for the detected basecaller
            all_runs_status[run_id][basecaller] = True

    # Prepare data for DataFrame conversion
    if all_runs_status:
        # Sort the dictionary by run_id using the natural sort key
        sorted_run_ids = natsorted(all_runs_status.keys())
        sorted_runs_dict = {run_id: all_runs_status[run_id] for run_id in sorted_run_ids}

        # Convert the sorted dictionary to a DataFrame
        runs_df = pd.DataFrame.from_dict(sorted_runs_dict, orient='index')
        runs_df.index.name = 'Run ID'
        # Add the 'Both Available' column
        runs_df['Both Available'] = runs_df['dorado'] & runs_df['guppy']
    else:
        # Create an empty DataFrame with the expected columns if no runs were found
        runs_df = pd.DataFrame(columns=['dorado', 'guppy', 'Both Available'])
        runs_df.index.name = 'Run ID'

    return runs_df, all_runs_status # Return both the DF and the dict

def extract_sample_id(header: str) -> Optional[str]:
    """
    Extract the unique sample identifier (e.g., "OMDL00009") from a FASTA header.
    Example Header: >ONT01.09-A02-OMDL00009-iNat169115711-1 ric=388
    """
    # Regex to find the OMDL part specifically
    # Looks for '-OMDL' followed by digits, capturing the 'OMDL' + digits part
    pattern = r"-(OMDL\d+)"
    match = re.search(pattern, header)
    if match:
        return match.group(1) # Returns "OMDLxxxxx"
    return None

def parse_ric_value(header: str) -> Optional[int]:
    """
    Extract RiC (Reads in Consensus) value from sequence header. eg ric=388
    """
    pattern = r"ric=(\d+)" # Looks for 'ric=' followed by digits
    match = re.search(pattern, header)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            # Handle case where digits might be invalid, though unlikely with \d+
            return None
    return None

def load_sequences(run_id: str, basecaller: str, seqs_dir: str) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    Load sequences from a FASTA file for a specific run and basecaller,
    organizing them by sample ID.

    Args:
        run_id: The run identifier (e.g., "OMDL1").
        basecaller: The basecaller name ("dorado" or "guppy").
        seqs_dir: The path to the directory containing sequence FASTA files.

    Returns:
        A dictionary mapping sample IDs (e.g., "OMDL00009") to lists of
        sequence record dictionaries, or None if the file doesn't exist.
        Each sequence dict contains: 'header', 'sequence', 'length', 'ric', 'seq_object'.
    """
    # Construct the full path to the FASTA file
    filename = f"{run_id}_seqs_{basecaller}.fasta"
    filepath = os.path.join(seqs_dir, filename)

    # Check if the file exists before attempting to open
    if not os.path.exists(filepath):
        print(f"Warning: File not found - {filepath}")
        return None

    # Use defaultdict for convenient appending to lists for each sample_id
    sequences_by_sample = defaultdict(list)

    try:
        # Parse the FASTA file using Biopython
        for record in SeqIO.parse(filepath, "fasta"):
            # Extract the unique sample ID (e.g., "OMDLxxxxx")
            sample_id = extract_sample_id(record.description)

            # If a valid sample ID is found, process the record
            if sample_id:
                ric_value = parse_ric_value(record.description)
                sequence_str = str(record.seq)
                sequence_len = len(sequence_str)

                # Store the relevant information in a dictionary
                sequence_data = {
                    'header': record.description,
                    'sequence': sequence_str,
                    'length': sequence_len,
                    'ric': ric_value,
                    # store the full SeqRecord object if needed for complex BioPython tasks later
                    'seq_object': record
                }
                # Append this sequence's data to the list for its sample ID
                sequences_by_sample[sample_id].append(sequence_data)

    except FileNotFoundError:
        # This case is technically handled by os.path.exists, but good practice
        print(f"Error: File not found during parsing - {filepath}")
        return None
    except Exception as e:
        # Catch other potential errors during file parsing
        print(f"Error parsing FASTA file {filepath}: {e}")
        return None # Or raise the exception depending on desired error handling

    # Return the dictionary (convert defaultdict to dict if preferred, though not necessary)
    return dict(sequences_by_sample)

def calculate_kmer_similarity(seq1: str, seq2: str, k: int = 5) -> float:
    """
    Calculate similarity between two sequences based on shared k-mers.
    Uses counts of k-mers to provide a similarity score.

    Args:
        seq1: The first sequence string.
        seq2: The second sequence string.
        k: The k-mer size (default: 7).

    Returns:
        A similarity score between 0.0 and 100.0, representing the percentage
        of k-mers in seq2 that are also found in seq1 (considering counts).
        Returns 0.0 if either sequence is too short for k-mers or if seq2 has no k-mers.
    """
    len1 = len(seq1)
    len2 = len(seq2)

    # --- Edge Case Handling ---
    # If either sequence is shorter than k, no k-mers can be generated.
    if len1 < k or len2 < k:
        return 0.0

    # --- Generate k-mer counts for seq1 ---
    seq1_kmers = {} # Dictionary to store k-mer counts for seq1
    for i in range(len1 - k + 1):
        kmer = seq1[i:i+k]
        seq1_kmers[kmer] = seq1_kmers.get(kmer, 0) + 1

    # --- Compare k-mers in seq2 ---
    shared_kmers_count = 0
    total_kmers_in_seq2 = len2 - k + 1 # Total k-mers possible in seq2

    # Keep track of k-mers already counted from seq2 to respect counts in seq1
    seq2_kmers_counted = {}

    for i in range(total_kmers_in_seq2):
        kmer = seq2[i:i+k]

        # Check if this k-mer exists in seq1
        if kmer in seq1_kmers:
            # Check how many times we've seen this k-mer in seq2 so far
            current_count_in_seq2 = seq2_kmers_counted.get(kmer, 0)
            # If we haven't counted this k-mer from seq2 more times than it appears in seq1,
            # increment shared count and update counted dictionary for seq2.
            if current_count_in_seq2 < seq1_kmers[kmer]:
                shared_kmers_count += 1
                seq2_kmers_counted[kmer] = current_count_in_seq2 + 1

    # --- Calculate Similarity Score ---
    if total_kmers_in_seq2 == 0:
        return 0.0 # Avoid division by zero if seq2 has no k-mers (though handled by length check)

    similarity = (shared_kmers_count / total_kmers_in_seq2) * 100.0
    return similarity

def align_sequences(seq1: str, seq2: str) -> Optional[Dict[str, Any]]:
    """
    Performs global pairwise alignment of two sequences using Bio.Align.PairwiseAligner
    and calculates alignment metrics.

    Args:
        seq1: The first sequence string.
        seq2: The second sequence string.

    Returns:
        A dictionary containing alignment metrics:
        {'identity': float, 'mismatches': int, 'insertions': int, 'deletions': int,
         'alignment_length': int, 'score': float, 'alignment_obj': Bio.Align.Alignment object}
        or None if alignment fails or sequences are empty.
        'insertions' are gaps in seq1 relative to seq2.
        'deletions' are gaps in seq2 relative to seq1.
    """
    # Handle empty sequences
    if not seq1 or not seq2:
        return None

    # --- Configure the Aligner ---
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global' # Global alignment (Needleman-Wunsch)

    # --- !!! CHANGE SCORING HERE !!! ---
    # Penalize mismatches and gaps in GLOBAL alignment
    # The default scoring is match=1, mismatch=-1, open_gap=-2, extend_gap=-1
    # Example scores (these can be tuned):
    aligner.match_score = 1.0   # Score for a match
    aligner.mismatch_score = -1.0 # usually negative of match_score, possibly x2
    aligner.open_gap_score = -2 # Penalty for opening a gap; should be larger than extend_gap_score
    aligner.extend_gap_score = -1 # Penalty for extending a gap


    try:
        # --- Perform Alignment ---
        # aligner.align returns an iterator; get the best one (or first if scores are simple)
        # Using next() is efficient to get just the first/best result
        alignment = next(aligner.align(seq1, seq2), None)

    except OverflowError:
        # Handle cases where alignment complexity is too high
        print(f"Warning: Alignment OverflowError for sequences of length {len(seq1)} and {len(seq2)}. Skipping alignment.")
        return None # Indicate failure
    except Exception as e:
        # Catch other potential alignment errors
        print(f"Warning: Alignment failed for sequences of length {len(seq1)} and {len(seq2)}: {e}")
        return None # Indicate failure

    # Check if an alignment was found
    if alignment is None:
        # This might happen if sequences are extremely dissimilar with heavy penalties
        return None

    # --- Calculate Metrics Directly from Alignment Object ---
    # Biopython's alignment object allows direct comparison of aligned sequences
    aligned_seq1, aligned_seq2 = alignment[0], alignment[1]
    alignment_length = len(aligned_seq1)

    if alignment_length == 0: # Should not happen if sequences were not empty, but check.
         return None

    matches = 0
    mismatches = 0
    insertions = 0 # Gaps in seq1 ('-')
    deletions = 0  # Gaps in seq2 ('-')

    for char1, char2 in zip(aligned_seq1, aligned_seq2):
        if char1 == char2:
            matches += 1
        elif char1 == '-':
            insertions += 1
        elif char2 == '-':
            deletions += 1
        else:
            mismatches += 1

    # Calculate percentage identity
    identity_percent = (matches / alignment_length) * 100.0

    # Store results in a dictionary
    results = {
        'identity': identity_percent,
        'mismatches': mismatches,
        'insertions': insertions, # gaps in seq1
        'deletions': deletions,   # gaps in seq2
        'alignment_length': alignment_length,
        'score': alignment.score,
        'alignment_obj': alignment # Store the object if needed later (e.g., for visualization)
    }
    return results

def _prefilter_and_align_pairs(
    dorado_records: List[Dict[str, Any]],
    guppy_records: List[Dict[str, Any]],
    kmer_similarity_threshold: float,
    length_ratio_threshold: float,
    max_alignments: int
) -> Tuple[List[Tuple[int, int, Dict[str, Any]]], Dict[Tuple[int, int], Optional[Dict[str, Any]]]]:
    """
    Prefilters sequence pairs based on length and k-mer similarity,
    then performs pairwise alignment on the most promising pairs.

    Args:
        dorado_records: List of Dorado sequence record dictionaries for the sample.
        guppy_records: List of Guppy sequence record dictionaries for the sample.
        kmer_similarity_threshold: Minimum k-mer similarity score (%).
        length_ratio_threshold: Minimum length ratio (shorter/longer).
        max_alignments: Maximum number of alignments to perform.

    Returns:
        A tuple containing:
        - aligned_pairs: List of tuples [(d_idx, g_idx, alignment_results), ...],
                         sorted by identity (desc), for successfully aligned pairs.
        - alignment_cache: Dictionary caching alignment results {(d_idx, g_idx): alignment_results}.
    """
    potential_pair_scores = [] # Store tuples: (d_idx, g_idx, kmer_score)
    # 1. Pre-filter pairs using k-mer similarity and length ratio
    for d_idx, d_record in enumerate(dorado_records):
        for g_idx, g_record in enumerate(guppy_records):
            len1, len2 = d_record['length'], g_record['length']
            if min(len1, len2) <= 0: continue # Skip empty sequences
            length_ratio = min(len1, len2) / max(len1, len2)

            if length_ratio >= length_ratio_threshold:
                kmer_sim = calculate_kmer_similarity(d_record['sequence'], g_record['sequence'])

                if kmer_sim >= kmer_similarity_threshold:
                    potential_pair_scores.append((d_idx, g_idx, kmer_sim))

    if not potential_pair_scores:
            # No pairs passed pre-filtering, all sequences are unmatched for this sample
            return [], {} # They will be added to _only lists later

    else:
        # 2. Perform full alignment on promising pairs
        potential_pair_scores.sort(key=lambda x: x[2], reverse=True) # Sort by k-mer score DESC
        aligned_pairs = [] # Store tuples: (d_idx, g_idx, alignment_results)
        alignment_cache = {} # Cache results: {(d_idx, g_idx): alignment_results}

        pairs_to_align = potential_pair_scores[:max_alignments] # Limit alignments


        for d_idx, g_idx, kmer_score in pairs_to_align:
            if (d_idx, g_idx) not in alignment_cache:
                    alignment_results = align_sequences(dorado_records[d_idx]['sequence'], guppy_records[g_idx]['sequence'])
                    alignment_cache[(d_idx, g_idx)] = alignment_results # Cache even if None

            alignment_results = alignment_cache[(d_idx, g_idx)]
            if alignment_results: # Only proceed if alignment was successful
                aligned_pairs.append((d_idx, g_idx, alignment_results))

        # 3. Assign matches based on alignment identity
        if aligned_pairs:
            aligned_pairs.sort(key=lambda x: x[2]['identity'], reverse=True) # Sort by identity DESC
        return aligned_pairs, alignment_cache

def _assign_matches(
    aligned_pairs: List[Tuple[int, int, Dict[str, Any]]],
    alignment_cache: Dict[Tuple[int, int], Optional[Dict[str, Any]]],
    dorado_records: List[Dict[str, Any]],
    guppy_records: List[Dict[str, Any]],
    sample_id: str,
    high_identity_threshold: float,
    multiple_match_identity_diff: float
) -> Tuple[List[Dict[str, Any]], Set[int], Set[int]]:
    """
    Assigns matches based on pre-computed alignment results, handling
    unique high-confidence matches and ambiguous cases.

    Args:
        aligned_pairs: List of tuples [(d_idx, g_idx, alignment_results), ...]
                       sorted by identity (desc) from _prefilter_and_align_pairs.
        alignment_cache: Dictionary caching alignment results.
        dorado_records: List of Dorado sequence record dictionaries for the sample.
        guppy_records: List of Guppy sequence record dictionaries for the sample.
        sample_id: The sample identifier.
        high_identity_threshold: Identity (%) for high-confidence 1:1 match.
        multiple_match_identity_diff: Max identity % difference for considering ambiguous matches.

    Returns:
        A tuple containing:
        - assigned_matched_pairs: List of dictionaries for assigned matches in this sample.
        - used_dorado_indices: Set of indices for used Dorado records.
        - used_guppy_indices: Set of indices for used Guppy records.
    """
    # ... implementation to be moved here ...
    assigned_matched_pairs = []
    used_dorado_indices = set()
    used_guppy_indices = set()


    # First pass: Assign high-confidence unique matches
    for d_idx, g_idx, align_res in aligned_pairs:
        if d_idx not in used_dorado_indices and g_idx not in used_guppy_indices:
            if align_res['identity'] >= high_identity_threshold:
                assigned_matched_pairs.append({
                    'sample_id': sample_id,
                    'dorado': dorado_records[d_idx],
                    'guppy': guppy_records[g_idx],
                    'alignment': align_res,
                    'multiple_matches': False,
                    'match_confidence': 'high'
                })
                used_dorado_indices.add(d_idx)
                used_guppy_indices.add(g_idx)

    # Second pass: Handle remaining sequences and potential ambiguities
    # Group remaining possible matches by dorado index
    remaining_potentials = defaultdict(list)
    for d_idx, g_idx, align_res in aligned_pairs:
            if d_idx not in used_dorado_indices and g_idx not in used_guppy_indices:
                remaining_potentials[d_idx].append({'g_idx': g_idx, 'identity': align_res['identity']})

    for d_idx, possible_matches in remaining_potentials.items():
            if not possible_matches: continue # Should not happen based on logic, but safe check

            # Sort this dorado seq's possible guppy matches by identity
            possible_matches.sort(key=lambda x: x['identity'], reverse=True)
            best_match = possible_matches[0]
            best_g_idx = best_match['g_idx']
            best_identity = best_match['identity']

            # Find other matches within the identity difference threshold
            ambiguous_matches = [best_match]
            for match in possible_matches[1:]:
                if best_identity - match['identity'] <= multiple_match_identity_diff:
                    ambiguous_matches.append(match)
                else:
                    break # Since they are sorted

            # Assign match(es)
            if len(ambiguous_matches) == 1:
                # Single clear best match for this dorado seq among remaining
                g_idx = best_g_idx
                align_res = alignment_cache.get((d_idx, g_idx))
                if align_res: # Should exist
                    assigned_matched_pairs.append({
                        'sample_id': sample_id,
                        'dorado': dorado_records[d_idx],
                        'guppy': guppy_records[g_idx],
                        'alignment': align_res,
                        'multiple_matches': False,
                        'match_confidence': 'medium' if best_identity >= 80 else 'low'
                    })
                    used_dorado_indices.add(d_idx)
                    used_guppy_indices.add(g_idx)
            else:
                # Ambiguous case: Multiple guppy seqs match this dorado seq similarly well
                for match in ambiguous_matches:
                    g_idx = match['g_idx']
                    # Check again if guppy index was used by another ambiguous match in this loop
                    if g_idx not in used_guppy_indices:
                        align_res = alignment_cache.get((d_idx, g_idx))
                        if align_res:
                            assigned_matched_pairs.append({
                                'sample_id': sample_id,
                                'dorado': dorado_records[d_idx],
                                'guppy': guppy_records[g_idx],
                                'alignment': align_res,
                                'multiple_matches': True, # Flag as ambiguous
                                'match_confidence': 'ambiguous',
                                'similarity_to_best': (match['identity'] / best_identity * 100.0) if best_identity > 0 else 0
                            })
                            used_guppy_indices.add(g_idx) # Mark guppy seq as used
                # Mark dorado seq as used after processing all its ambiguous matches
                used_dorado_indices.add(d_idx)
    return assigned_matched_pairs, used_dorado_indices, used_guppy_indices


def match_sequences(
    dorado_seqs: Dict[str, List[Dict[str, Any]]],
    guppy_seqs: Dict[str, List[Dict[str, Any]]]
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Matches sequences between Dorado and Guppy datasets for each sample,
    using k-mer similarity and pairwise alignment.

    Args:
        dorado_seqs: Dictionary mapping sample IDs to lists of Dorado sequence records.
                     (Output of load_sequences).
        guppy_seqs: Dictionary mapping sample IDs to lists of Guppy sequence records.
                    (Output of load_sequences).

    Returns:
        A tuple containing three lists:
        - matched_pairs: List of dictionaries, each representing a matched pair.
                         Includes sample_id, dorado record, guppy record, alignment results,
                         multiple_matches flag, and match_confidence.
        - dorado_only: List of dictionaries for Dorado sequences with no match found.
                       Includes sample_id and the dorado record.
        - guppy_only: List of dictionaries for Guppy sequences with no match found.
                      Includes sample_id and the guppy record.
    """
    matched_pairs = []
    dorado_only = []
    guppy_only = []

    # --- !!! CHANGE K-mer SCORING HERE !!! ---
    # --- Configuration Thresholds ---
    KMER_SIMILARITY_THRESHOLD = 50.0  # Min k-mer similarity (%) for considering alignment
    LENGTH_RATIO_THRESHOLD = 0.5     # Min length ratio (shorter/longer) to consider match
    HIGH_IDENTITY_THRESHOLD = 95.0    # Identity (%) for high-confidence 1:1 match
    MULTIPLE_MATCH_IDENTITY_DIFF = 5.0 # Max identity % difference for considering ambiguous matches
    # Maximum number of alignments to perform per sample in complex cases to limit computation
    MAX_ALIGNMENTS_PER_SAMPLE = 20

    # Get sets of sample IDs present in each dataset
    dorado_sample_ids = set(dorado_seqs.keys())
    guppy_sample_ids = set(guppy_seqs.keys())

    # Identify common samples, and samples unique to each dataset
    common_samples = dorado_sample_ids.intersection(guppy_sample_ids)
    dorado_unique_samples = dorado_sample_ids - guppy_sample_ids
    guppy_unique_samples = guppy_sample_ids - dorado_sample_ids

    print(f"Processing {len(common_samples)} samples common to both Dorado and Guppy...")

    # --- Process Common Samples ---
    for sample_id in common_samples:
        dorado_records = dorado_seqs[sample_id]
        guppy_records = guppy_seqs[sample_id]
        num_dorado = len(dorado_records)
        num_guppy = len(guppy_records)

        # Keep track of used sequence indices within this sample
        used_dorado_indices = set()
        used_guppy_indices = set()

        # === Case 1: Simple 1-to-1 Match ===
        if num_dorado == 1 and num_guppy == 1:
            d_record = dorado_records[0]
            g_record = guppy_records[0]
            alignment_results = align_sequences(d_record['sequence'], g_record['sequence']) # Step 2.2
            MIN_IDENTITY_THRESHOLD_1_TO_1 = 70.0 # Example threshold, adjust if needed
            if alignment_results and alignment_results['identity'] >= MIN_IDENTITY_THRESHOLD_1_TO_1:
                matched_pairs.append({
                    'sample_id': sample_id,
                    'dorado': d_record,
                    'guppy': g_record,
                    'alignment': alignment_results,
                    'multiple_matches': False,
                    'match_confidence': 'high' if alignment_results['identity'] >= HIGH_IDENTITY_THRESHOLD else ('medium' if alignment_results['identity'] >= 80 else 'low')
                })
                used_dorado_indices.add(0)
                used_guppy_indices.add(0)
            else:
                # Alignment failed, or identity too low. Treat as unmatched.
                pass # They will be added to _only lists later

        # === Case 2: Complex Match (Multiple Sequences in Dorado or Guppy or Both) ===
        elif num_dorado > 0 and num_guppy > 0:
            # Prefilter and align promising pairs
            aligned_pairs, alignment_cache = _prefilter_and_align_pairs(
                dorado_records,
                guppy_records,
                KMER_SIMILARITY_THRESHOLD,
                LENGTH_RATIO_THRESHOLD,
                MAX_ALIGNMENTS_PER_SAMPLE
            )

            if aligned_pairs:
                sample_matched_pairs, sample_used_dorado, sample_used_guppy = _assign_matches(
                    aligned_pairs,
                    alignment_cache,
                    dorado_records,
                    guppy_records,
                    sample_id,
                    HIGH_IDENTITY_THRESHOLD, # Use the constants defined in match_sequences
                    MULTIPLE_MATCH_IDENTITY_DIFF
                )
                matched_pairs.extend(sample_matched_pairs) # Add results to the main list

                # CRITICAL: Update the 'used' sets defined *within* the sample loop
                # These sets were defined just before the 'elif' block
                used_dorado_indices.update(sample_used_dorado)
                used_guppy_indices.update(sample_used_guppy)
            # The 'else' case (no aligned pairs found) requires no action here,
            # as the used_dorado_indices/used_guppy_indices sets will remain empty
            # for this sample, and the subsequent code will correctly handle it.


        # --- Add any remaining unused sequences for this common sample to _only lists ---
        for d_idx, d_record in enumerate(dorado_records):
            if d_idx not in used_dorado_indices:
                dorado_only.append({'sample_id': sample_id, 'record': d_record})
        for g_idx, g_record in enumerate(guppy_records):
            if g_idx not in used_guppy_indices:
                guppy_only.append({'sample_id': sample_id, 'record': g_record})

    # --- Add sequences from samples unique to one dataset ---
    print(f"Processing {len(dorado_unique_samples)} samples unique to Dorado...")
    for sample_id in dorado_unique_samples:
        for record in dorado_seqs[sample_id]:
            dorado_only.append({'sample_id': sample_id, 'record': record})

    print(f"Processing {len(guppy_unique_samples)} samples unique to Guppy...")
    for sample_id in guppy_unique_samples:
        for record in guppy_seqs[sample_id]:
            guppy_only.append({'sample_id': sample_id, 'record': record})

    print(f"Matching complete. Found {len(matched_pairs)} matched pairs, "
          f"{len(dorado_only)} Dorado-only sequences, {len(guppy_only)} Guppy-only sequences.")

    return matched_pairs, dorado_only, guppy_only



def calculate_gc_content(sequence_str: str) -> Optional[float]:
    """
    Calculates the GC content (fraction) of a DNA sequence string.

    Args:
        sequence_str: The DNA sequence as a string.

    Returns:
        The GC content as a float (fraction between 0.0 and 1.0),
        or None if the input is invalid or calculation fails.
    """
    if not isinstance(sequence_str, str) or not sequence_str:
        # Handle non-string or empty input
        return None
    try:
        # Calculate GC fraction. Biopython's gc_fraction handles 'N' and other non-ATGC chars appropriately.
        # It returns a value between 0 and 1.
        return gc_fraction(sequence_str)
    except Exception as e:
        # Catch potential errors during calculation
        print(f"Warning: Could not calculate GC content for sequence snippet '{sequence_str[:20]}...': {e}")
        return None

def analyze_homopolymers(sequence_str: str, min_len: int = 5) -> Optional[Dict[str, Any]]:
    """
    Analyzes a sequence for homopolymer runs of a minimum length.

    Args:
        sequence_str: The DNA sequence as a string.
        min_len: The minimum length of a homopolymer run to report (default: 5).

    Returns:
        A dictionary summarizing homopolymer runs:
        {
            'A': [list of lengths of A runs >= min_len],
            'C': [list of lengths of C runs >= min_len],
            'G': [list of lengths of G runs >= min_len],
            'T': [list of lengths of T runs >= min_len],
            'total_count': Total number of homopolymer runs found,
            'max_len': Maximum length of any homopolymer run found
        }
        or None if the input is invalid.
    """
    if not isinstance(sequence_str, str) or not sequence_str or not isinstance(min_len, int) or min_len < 1:
        return None # Invalid input

    results = {'A': [], 'C': [], 'G': [], 'T': [], 'total_count': 0, 'max_len': 0}
    bases = ['A', 'T', 'C', 'G'] # Case-insensitive matching usually desired

    for base in bases:
        # Regex to find runs of 'base' with length >= min_len (case-insensitive)
        pattern = re.compile(f"({base}{{{min_len},}})", re.IGNORECASE)
        for match in pattern.finditer(sequence_str):
            run_len = len(match.group(1))
            # Store based on the actual base found (upper case)
            actual_base = match.group(1)[0].upper()
            if actual_base in results: # Should always be true for ATCG
                 results[actual_base].append(run_len)
                 results['total_count'] += 1
                 if run_len > results['max_len']:
                      results['max_len'] = run_len

    return results

def analyze_ambiguity(sequence_str: str) -> Optional[Dict[str, Any]]:
    """
    Analyzes a sequence for the count and frequency of IUPAC ambiguity codes.

    Args:
        sequence_str: The DNA sequence as a string.

    Returns:
        A dictionary summarizing ambiguity codes:
        {
            'total_count': Total number of ambiguity characters found,
            'frequency': Total frequency (total_count / sequence_length),
            'counts_per_code': {code: count for each code found}
        }
        or None if the input is invalid or sequence is empty. Returns 0 counts/frequency
        if no ambiguity codes are found.
    """
    if not isinstance(sequence_str, str):
        return None # Invalid input type

    seq_len = len(sequence_str)
    if seq_len == 0:
        return None # Cannot calculate frequency for empty sequence

    total_ambiguity_count = 0
    counts_per_code = {}

    # Use regex for efficient counting (case-insensitive)
    # Creates a pattern like [RYMKWSBVDHN]
    pattern = re.compile(f"([{IUPAC_AMBIGUITY_CODES}])", re.IGNORECASE)

    for match in pattern.finditer(sequence_str):
        code = match.group(1).upper() # Get the matched code, ensure uppercase
        counts_per_code[code] = counts_per_code.get(code, 0) + 1
        total_ambiguity_count += 1

    frequency = total_ambiguity_count / seq_len

    return {
        'total_count': total_ambiguity_count,
        'frequency': frequency,
        'counts_per_code': counts_per_code
    }

def generate_comparison_dataframe(matched_pairs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Generates a pandas DataFrame containing consolidated metrics for matched sequence pairs.

    Args:
        matched_pairs: A list of dictionaries, where each dictionary represents
                       a matched pair from the match_sequences function.

    Returns:
        A pandas DataFrame with columns for sample ID, headers, RiC, length, GC content,
        alignment metrics, homopolymer stats, ambiguity stats, and differences.
    """
    comparison_data = []

    if not matched_pairs:
        # Return an empty DataFrame with expected columns if no pairs are provided
        # Define expected columns structure here if needed, or return empty DF
        return pd.DataFrame()

    for pair in matched_pairs:
        row_data = {}

        # --- Basic Info ---
        row_data['Sample_ID'] = pair.get('sample_id', None)
        dorado_rec = pair.get('dorado', {})
        guppy_rec = pair.get('guppy', {})
        row_data['Dorado_Header'] = dorado_rec.get('header', None)
        row_data['Guppy_Header'] = guppy_rec.get('header', None)
        row_data['Dorado_RiC'] = dorado_rec.get('ric', None)
        row_data['Guppy_RiC'] = guppy_rec.get('ric', None)
        row_data['Dorado_Length'] = dorado_rec.get('length', 0) # Use 0 for calculations if missing
        row_data['Guppy_Length'] = guppy_rec.get('length', 0)

        # --- Match Quality ---
        row_data['Multiple_Matches'] = pair.get('multiple_matches', False)
        row_data['Match_Confidence'] = pair.get('match_confidence', None)

        # --- Alignment Metrics (from pair['alignment']) ---
        alignment_res = pair.get('alignment', {})
        row_data['Identity_Percent'] = alignment_res.get('identity', None)
        row_data['Mismatches'] = alignment_res.get('mismatches', None)
        row_data['Insertions_vs_Guppy'] = alignment_res.get('insertions', None) # Gaps in Dorado seq
        row_data['Deletions_vs_Guppy'] = alignment_res.get('deletions', None)   # Gaps in Guppy seq
        row_data['Alignment_Length'] = alignment_res.get('alignment_length', None)
        row_data['Alignment_Score'] = alignment_res.get('score', None)


        # --- Calculated Metrics ---
        dorado_seq = dorado_rec.get('sequence', "")
        guppy_seq = guppy_rec.get('sequence', "")

        # GC Content (handle None return)
        row_data['Dorado_GC'] = calculate_gc_content(dorado_seq)
        row_data['Guppy_GC'] = calculate_gc_content(guppy_seq)

        # Homopolymers (store key results, handle None return)
        dorad_homop = analyze_homopolymers(dorado_seq)
        guppy_homop = analyze_homopolymers(guppy_seq)
        row_data['dorad_homop_Count'] = dorad_homop['total_count'] if dorad_homop else None
        row_data['guppy_homop_Count'] = guppy_homop['total_count'] if guppy_homop else None
        row_data['dorad_homop_MaxLen'] = dorad_homop['max_len'] if dorad_homop else None
        row_data['guppy_homop_MaxLen'] = guppy_homop['max_len'] if guppy_homop else None
        # Optionally store the full dicts for debugging:
        # row_data['dorad_homop_Details'] = dorad_homop
        # row_data['guppy_homop_Details'] = guppy_homop

        # Ambiguity (store key results, handle None return)
        dorado_ambig = analyze_ambiguity(dorado_seq)
        guppy_ambig = analyze_ambiguity(guppy_seq)
        row_data['Dorado_Ambig_Count'] = dorado_ambig['total_count'] if dorado_ambig else None
        row_data['Guppy_Ambig_Count'] = guppy_ambig['total_count'] if guppy_ambig else None
        row_data['Dorado_Ambig_Freq'] = dorado_ambig['frequency'] if dorado_ambig else None
        row_data['Guppy_Ambig_Freq'] = guppy_ambig['frequency'] if guppy_ambig else None
        # Optionally store the full dicts for debugging:
        # row_data['Dorado_Ambig_Details'] = dorado_ambig
        # row_data['Guppy_Ambig_Details'] = guppy_ambig

        # --- Difference Metrics (handle None values carefully) ---
        try:
            row_data['RiC_Difference'] = (row_data['Dorado_RiC'] - row_data['Guppy_RiC']) if row_data['Dorado_RiC'] is not None and row_data['Guppy_RiC'] is not None else None
        except TypeError:
            row_data['RiC_Difference'] = None
        try:
            row_data['Length_Difference'] = row_data['Dorado_Length'] - row_data['Guppy_Length']
        except TypeError:
             row_data['Length_Difference'] = None
        try:
            row_data['GC_Difference'] = (row_data['Dorado_GC'] - row_data['Guppy_GC']) if row_data['Dorado_GC'] is not None and row_data['Guppy_GC'] is not None else None
        except TypeError:
             row_data['GC_Difference'] = None

        comparison_data.append(row_data)

    # Create DataFrame
    run_comparison_df = pd.DataFrame(comparison_data)

    return run_comparison_df

def perform_paired_nonparametric_test(
    data1: Union[List[float], np.ndarray],
    data2: Union[List[float], np.ndarray],
    test_type: str = 'wilcoxon'
) -> Optional[Tuple[float, float]]:
    """
    Performs a paired non-parametric statistical test between two datasets.

    Args:
        data1: First list or array of paired numerical data.
        data2: Second list or array of paired numerical data. Must be the same length as data1.
        test_type: The type of non-parametric test to perform. Currently supports 'wilcoxon'.

    Returns:
        A tuple containing the test statistic and the p-value,
        or None if the test cannot be performed (e.g., insufficient data, inputs invalid).
    """
    # Basic validation: Check if inputs are lists or numpy arrays
    if not isinstance(data1, (list, np.ndarray)) or not isinstance(data2, (list, np.ndarray)):
        print("Warning: Inputs must be lists or numpy arrays.")
        return None

    # Convert to numpy arrays for easier handling
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    # Check for equal length
    if len(data1) != len(data2):
        print("Warning: Input data lists must have the same length.")
        return None

    # Check for sufficient data points (Wilcoxon needs at least a few pairs)
    # Wilcoxon handles zero differences, but raises ValueError if data is identical
    # or length is very small so we need a basic length check.
    min_required_pairs = 1 # Wilcoxon technically runs with 1, but warns for small N.
    if len(data1) < min_required_pairs:
         print(f"Warning: Insufficient data for {test_type} test (requires at least {min_required_pairs} pairs).")
         return None

    # Check if data is numeric (redundant if type hints are enforced, but good practice)
    if not np.issubdtype(data1.dtype, np.number) or not np.issubdtype(data2.dtype, np.number):
         print("Warning: Input data must be numeric.")
         return None

    # Check for NaN values - Wilcoxon might handle them depending on version/options,
    # but it's often better to remove pairs with NaN explicitly
    combined_data = np.vstack((data1, data2)).T
    combined_data = combined_data[~np.isnan(combined_data).any(axis=1)]
    if len(combined_data) < min_required_pairs: return None
    data1, data2 = combined_data[:, 0], combined_data[:, 1]

    if test_type.lower() == 'wilcoxon':
        try:
            # Calculate differences, ignoring pairs where difference is zero for Wilcoxon
            diff = data1 - data2
            diff_nonzero = diff[diff != 0]

            # If all differences are zero, the test is not applicable / p-value is 1
            if len(diff_nonzero) == 0:
                 print("Warning: All differences are zero, Wilcoxon test not applicable (p=1.0 assumed).")
                 return 0.0, 1.0 # Or return None, depending on desired handling

            # Perform the Wilcoxon signed-rank test
            # Use alternative='two-sided' for standard comparison
            # zero_method='wilcox' is default, handles zeros appropriately
            # correction=False is default, set True for continuity correction if needed
            statistic, p_value = stats.wilcoxon(data1, data2, zero_method='wilcox', alternative='two-sided')

            return statistic, p_value

        except ValueError as ve:
            # Handle specific errors, e.g., insufficient data after removing zeros
            print(f"Error during Wilcoxon test: {ve}")
            return None
        except Exception as e:
            # Catch any other unexpected errors
            print(f"An unexpected error occurred during the {test_type} test: {e}")
            return None
    else:
        print(f"Error: Unsupported test type '{test_type}'. Only 'wilcoxon' is implemented.")
        return None

def calculate_run_statistics(run_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Calculates descriptive statistics and performs paired non-parametric tests
    on the comparison DataFrame for a single run.

    Args:
        run_df: The DataFrame containing paired comparison data for one run
                (output of generate_comparison_dataframe).

    Returns:
        A dictionary containing statistics (median differences, p-values, etc.)
        for key metrics, or None if input is invalid or empty.
    """
    # Inside calculate_run_statistics function:

    if not isinstance(run_df, pd.DataFrame) or run_df.empty:
        print("Warning: Input must be a non-empty pandas DataFrame.")
        return None

    # Define the key metric pairs to analyze
    metric_pairs = {
        'RiC': ('Dorado_RiC', 'Guppy_RiC'),
        'Length': ('Dorado_Length', 'Guppy_Length'),
        'GC': ('Dorado_GC', 'Guppy_GC'),
        'Homop_Count': ('dorad_homop_Count', 'guppy_homop_Count'),
        'Homop_MaxLen': ('dorad_homop_MaxLen', 'guppy_homop_MaxLen'),
        'Ambig_Count': ('Dorado_Ambig_Count', 'Guppy_Ambig_Count'),
        'Ambig_Freq': ('Dorado_Ambig_Freq', 'Guppy_Ambig_Freq')
    }

    results = {} # Dictionary to store all results

    # Check if required columns exist
    required_cols = [col for pair in metric_pairs.values() for col in pair]
    if not all(col in run_df.columns for col in required_cols):
         missing = [col for col in required_cols if col not in run_df.columns]
         print(f"Warning: DataFrame missing required columns: {missing}. Cannot perform all statistics.")
    # Inside calculate_run_statistics function (after validation):
    for metric_name, (col1, col2) in metric_pairs.items():
        results[metric_name] = {} # Sub-dictionary for each metric

        # Check if both columns for this metric exist
        if col1 not in run_df.columns or col2 not in run_df.columns:
            print(f"Skipping statistics for {metric_name} due to missing columns.")
            results[metric_name]['error'] = "Missing columns"
            continue

        # Extract data, dropping rows where either value is NaN for this pair
        valid_data = run_df[[col1, col2]].dropna()

        if valid_data.empty:
             print(f"No valid data pairs for {metric_name} after dropping NaN.")
             results[metric_name]['error'] = "No valid data"
             continue

        data1 = valid_data[col1]
        data2 = valid_data[col2]

        # Calculate Differences (Data1 - Data2, e.g., Dorado - Guppy)
        differences = data1 - data2

        # --- Descriptive Statistics for Differences ---
        results[metric_name]['median_diff'] = np.median(differences)
        results[metric_name]['mean_diff'] = np.mean(differences)
        # Calculate IQR (Interquartile Range)
        q1 = np.percentile(differences, 25)
        q3 = np.percentile(differences, 75)
        results[metric_name]['iqr_diff'] = q3 - q1
        results[metric_name]['n_pairs'] = len(valid_data) # Number of pairs used

        # --- Perform Paired Non-parametric Test ---
        test_result = perform_paired_nonparametric_test(data1.to_list(), data2.to_list(), test_type='wilcoxon')

        if test_result:
            statistic, p_value = test_result
            results[metric_name]['test_statistic'] = statistic
            results[metric_name]['p_value'] = p_value
        else:
            results[metric_name]['test_statistic'] = None
            results[metric_name]['p_value'] = None
            results[metric_name]['error'] = results[metric_name].get('error', 'Test failed or insufficient data') # Keep previous error if exists

    return results

def save_run_comparison(
    run_df: pd.DataFrame,
    run_id: str,
    output_dir: str,
    format: str = 'tsv'
) -> Optional[str]:
    """
    Saves the run-specific comparison DataFrame to a file (TSV or CSV).

    Args:
        run_df: The DataFrame containing detailed comparison data for matched pairs.
        run_id: The run identifier (e.g., "OMDL1").
        output_dir: The path to the directory where the file will be saved.
        format: The output file format ('tsv' or 'csv'). Defaults to 'tsv'.

    Returns:
        The full path to the saved file, or None if saving fails.
    """
    if not isinstance(run_df, pd.DataFrame):
        print("Error: Input 'run_df' must be a pandas DataFrame.")
        return None
    if run_df.empty:
        print("Warning: Input DataFrame is empty. No file will be saved.")
        return None
    if not run_id or not isinstance(run_id, str):
        print("Error: Invalid 'run_id'.")
        return None
    if not output_dir or not isinstance(output_dir, str):
         print("Error: Invalid 'output_dir'.")
         return None

    # Determine separator and file extension based on format
    if format.lower() == 'csv':
        separator = ','
        file_extension = 'csv'
    elif format.lower() == 'tsv':
        separator = '\t'
        file_extension = 'tsv'
    else:
        print(f"Error: Unsupported format '{format}'. Use 'tsv' or 'csv'.")
        return None

    # Construct filename and full path
    filename = f"{run_id}_comparison_data.{file_extension}"
    filepath = os.path.join(output_dir, filename)

    # Ensure output directory exists (optional, could rely on notebook setup in future)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
         print(f"Error creating output directory {output_dir}: {e}")
         return None

    try:
        run_df.to_csv(filepath, sep=separator, index=False, encoding='utf-8')
        print(f"Run comparison data saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving DataFrame to {filepath}: {e}")
        return None

def generate_overall_summary(
    all_runs_results: Dict[str, Dict[str, Any]],
    output_dir: str,
    format: str = 'tsv'
) -> Optional[str]:
    """
    Generates an overall summary DataFrame aggregating key statistics and counts
    across all processed runs and saves it to a file.

    Args:
        all_runs_results: A dictionary where keys are run IDs and values are
                          dictionaries containing processed data for each run.
                          Expected structure per run_id:
                          {
                              'stats': dict_output_from_calculate_run_statistics,
                              'counts': {'matched': int, 'dorado_only': int, 'guppy_only': int}
                              # Add other relevant top-level counts if needed, e.g., total sequences
                          }
        output_dir: The path to the directory where the summary file will be saved.
        format: The output file format ('tsv' or 'csv'). Defaults to 'tsv'.

    Returns:
        The full path to the saved summary file, or None if saving fails or input is invalid.
    """
    if not isinstance(all_runs_results, dict) or not all_runs_results:
        print("Error: Input 'all_runs_results' must be a non-empty dictionary.")
        return None
    if not output_dir or not isinstance(output_dir, str):
         print("Error: Invalid 'output_dir'.")
         return None

    summary_data_list = []
    processed_run_ids = natsorted(all_runs_results.keys()) # Use natural sort for Run IDs

    for run_id in processed_run_ids:
        run_result = all_runs_results[run_id]
        row_data = {'Run_ID': run_id}

        # --- Extract Counts ---
        counts = run_result.get('counts', {})
        row_data['Matched_Pairs'] = counts.get('matched', 0)
        row_data['Dorado_Only_Seqs'] = counts.get('dorado_only', 0)
        row_data['Guppy_Only_Seqs'] = counts.get('guppy_only', 0)
        row_data['Dorado_Total_Seqs'] = counts.get('dorado_total', 0)
        row_data['Guppy_Total_Seqs'] = counts.get('guppy_total', 0)


        # --- Extract Key Statistics (Median Diffs, P-values) ---
        stats = run_result.get('stats', {}) # Get the stats dict

        # Define metrics to extract from the stats dict
        metrics_to_summarize = ['RiC', 'Length', 'GC', 'Homop_Count', 'Homop_MaxLen', 'Ambig_Count']

        for metric in metrics_to_summarize:
            metric_stats = stats.get(metric, {}) # Get sub-dict for this metric
            row_data[f'{metric}_Median_Diff'] = metric_stats.get('median_diff') # Use .get() for safety
            row_data[f'{metric}_p_value'] = metric_stats.get('p_value')
            row_data[f'{metric}_Mean_Diff'] = metric_stats.get('mean_diff')
            # row_data[f'{metric}_IQR_Diff'] = metric_stats.get('iqr_diff')
            row_data[f'{metric}_N_Pairs'] = metric_stats.get('n_pairs', 0) # Include number of pairs used for test


        summary_data_list.append(row_data)
    if not summary_data_list:
        print("Warning: No data aggregated for the overall summary. No file saved.")
        return None

    # Create DataFrame
    overall_summary_df = pd.DataFrame(summary_data_list)

    # --- Save the DataFrame ---
    # Determine separator and file extension
    if format.lower() == 'csv':
        separator = ','
        file_extension = 'csv'
    elif format.lower() == 'tsv':
        separator = '\t'
        file_extension = 'tsv'
    else:
        print(f"Error: Unsupported format '{format}'. Use 'tsv' or 'csv'.")
        return None

    # Construct filename and full path
    filename = f"overall_comparison_summary.{file_extension}"
    filepath = os.path.join(output_dir, filename)

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
         print(f"Error creating output directory {output_dir}: {e}")
         return None

    # Save the file
    try:
        overall_summary_df.to_csv(filepath, sep=separator, index=False, encoding='utf-8', float_format='%.4f') # Format floats nicely
        print(f"Overall summary data saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving overall summary DataFrame to {filepath}: {e}")
        return None