import os
import re
import glob
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio import Align


def natural_sort_key(s):
    """Helper function for natural sorting of strings containing numbers with prefixes"""
    # Extract just the number from OMDL prefix
    match = re.search(r'OMDL(\d+)', s)
    if match:
        # Return the number as an integer for proper numerical sorting
        return int(match.group(1))
    # Fallback for strings without the expected format or if sorting non-OMDL strings
    # Returning a large number ensures non-matching formats sort last,
    # or return 0/s depending on desired behavior for malformed names.
    return float('inf') # Or return 0, or s


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
        sorted_run_ids = sorted(all_runs_status.keys(), key=natural_sort_key)
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
    # Fallback or logging if pattern not found, depending on how strict you need to be
    # print(f"Warning: Could not extract OMDL sample ID from header: {header}")
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
                    # Optionally store the full SeqRecord object if needed for complex BioPython tasks later
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
    # Penalize mismatches and gaps
    # Example scores (these can be tuned):
    aligner.match_score = 1.0   # Score for a match (default is 1.0)
    aligner.mismatch_score = -1.0 # usually negative of match_score, possibly x2
    aligner.open_gap_score = -2 # Penalty for opening a gap (default is -2.0); should be larger than extend_gap_score
    aligner.extend_gap_score = -1 # Penalty for extending a gap (default is -1.0)


    try:
        # --- Perform Alignment ---
        # aligner.align returns an iterator; get the best one (or first if scores are simple)
        # Using next() is efficient to get just the first/best result
        alignment = next(aligner.align(seq1, seq2), None)

    except OverflowError:
        # Handle cases where alignment complexity is too high [cite: 24]
        print(f"Warning: Alignment OverflowError for sequences of length {len(seq1)} and {len(seq2)}. Skipping alignment.")
        return None # Indicate failure
    except Exception as e:
        # Catch other potential alignment errors
        print(f"Warning: Alignment failed for sequences of length {len(seq1)} and {len(seq2)}: {e}")
        return None # Indicate failure

    # Check if an alignment was found
    if alignment is None:
        # This might happen if sequences are extremely dissimilar with heavy penalties,
        # though unlikely with 0 penalties.
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
            potential_pair_scores = [] # Store tuples: (d_idx, g_idx, kmer_score)

            # 1. Pre-filter pairs using k-mer similarity and length ratio
            for d_idx, d_record in enumerate(dorado_records):
                for g_idx, g_record in enumerate(guppy_records):
                    len1, len2 = d_record['length'], g_record['length']
                    if min(len1, len2) <= 0: continue # Skip empty sequences
                    length_ratio = min(len1, len2) / max(len1, len2)

                    if length_ratio >= LENGTH_RATIO_THRESHOLD:
                        kmer_sim = calculate_kmer_similarity(d_record['sequence'], g_record['sequence']) # Step 2.1

                        if kmer_sim >= KMER_SIMILARITY_THRESHOLD:
                            potential_pair_scores.append((d_idx, g_idx, kmer_sim))

            if not potential_pair_scores:
                 # No pairs passed pre-filtering, all sequences are unmatched for this sample
                 pass # They will be added to _only lists later

            else:
                # 2. Perform full alignment on promising pairs
                potential_pair_scores.sort(key=lambda x: x[2], reverse=True) # Sort by k-mer score DESC
                aligned_pairs = [] # Store tuples: (d_idx, g_idx, alignment_results)
                alignment_cache = {} # Cache results: {(d_idx, g_idx): alignment_results}

                pairs_to_align = potential_pair_scores[:MAX_ALIGNMENTS_PER_SAMPLE] # Limit alignments


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

                    # First pass: Assign high-confidence unique matches
                    for d_idx, g_idx, align_res in aligned_pairs:
                        if d_idx not in used_dorado_indices and g_idx not in used_guppy_indices:
                            if align_res['identity'] >= HIGH_IDENTITY_THRESHOLD:
                                matched_pairs.append({
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
                             if best_identity - match['identity'] <= MULTIPLE_MATCH_IDENTITY_DIFF:
                                 ambiguous_matches.append(match)
                             else:
                                 break # Since they are sorted

                         # Assign match(es)
                         if len(ambiguous_matches) == 1:
                             # Single clear best match for this dorado seq among remaining
                             g_idx = best_g_idx
                             align_res = alignment_cache.get((d_idx, g_idx))
                             if align_res: # Should exist
                                 matched_pairs.append({
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
                                         matched_pairs.append({
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



# Currently unused, but keeping for potential future use
def load_summary(run_id: str, basecaller: str, summary_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load summary data from the TSV-like .txt file for a specific run and basecaller.
    Parses the main data table and extracts summary statistics from the end of the file.

    Args:
        run_id: The run identifier (e.g., "OMDL1").
        basecaller: The basecaller name ("dorado" or "guppy").
        summary_dir: The path to the directory containing summary .txt files.

    Returns:
        A dictionary containing the summary data DataFrame ('data') and
        a dictionary of summary statistics ('stats'), or None if the file doesn't exist.
        Stats dict includes: 'unique_samples', 'consensus_sequences', 'total_ric'.
    """
    # Construct the full path to the summary file (note the .txt extension)
    filename = f"{run_id}_summary_{basecaller}.txt"
    filepath = os.path.join(summary_dir, filename)

    # Check if the file exists before attempting to open
    if not os.path.exists(filepath):
        print(f"Warning: Summary file not found - {filepath}")
        return None

    summary_df = None
    summary_stats = {
        'unique_samples': None,
        'consensus_sequences': None,
        'total_ric': None
    }

    try:
        # Step 1: Read the main data table using pandas
        # Assuming the table starts from the first line (header=0)
        # and ends before the summary lines. pandas might stop reading
        # automatically if the summary lines have a different number of columns,
        # but explicitly handling might be safer if format varies.
        # We'll read the whole file first, then extract summary lines separately.
        summary_df = pd.read_csv(filepath, sep='\t', header=0)

        # Remove potential summary lines that might have been read into the DataFrame
        # Identify rows where the first column doesn't look like a filename (e.g., doesn't start with 'ONT')
        # This assumes filenames always start with 'ONT', adjust if needed based on actual data.
        if not summary_df.empty and summary_df.columns[0] == 'Filename': # Check if first column is 'Filename'
             summary_df = summary_df[summary_df['Filename'].str.startswith('ONT', na=False)]

        # Step 2: Read the file again to reliably extract summary statistics from the end
        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line: # Skip empty lines
                continue

            parts = line.split('\t')
            if len(parts) >= 2:
                key = parts[0].strip()
                value_str = parts[-1].strip() # Take the last part as value

                try:
                    value_int = int(value_str)
                    if "Total Unique Samples" in key:
                        summary_stats['unique_samples'] = value_int
                    elif "Total Consensus Sequences" in key:
                        summary_stats['consensus_sequences'] = value_int
                    elif "Total Reads in Consensus Sequences" in key:
                        summary_stats['total_ric'] = value_int
                except ValueError:
                    # Ignore lines where the value isn't an integer
                    continue

    except FileNotFoundError:
        print(f"Error: Summary file not found during processing - {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Warning: Summary file is empty - {filepath}")
        # Return dictionary with empty DataFrame and None stats
        return {'data': pd.DataFrame(), 'stats': summary_stats}
    except Exception as e:
        print(f"Error processing summary file {filepath}: {e}")
        return None # Or return partial data if appropriate

    # Check if DataFrame was successfully loaded
    if summary_df is None:
         summary_df = pd.DataFrame() # Ensure a DataFrame is always returned

    return {'data': summary_df, 'stats': summary_stats}