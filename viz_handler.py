import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import display, Markdown, clear_output, HTML
import ipywidgets as widgets
from natsort import natsorted

import re 


def plot_metric_comparison(run_df: pd.DataFrame,
                           dorado_col: str,
                           guppy_col: str,
                           title: str = None,
                           xlabel: str = None,
                           ylabel: str = None,
                           figsize: tuple = (8, 8)) -> tuple:
    """
    Creates a scatter plot comparing a metric between Dorado and Guppy.

    Args:
        run_df: DataFrame containing the run comparison data.
        dorado_col: Column name for the Dorado metric (y-axis).
        guppy_col: Column name for the Guppy metric (x-axis).
        title: Plot title.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        figsize: Figure size.

    Returns:
        Tuple (matplotlib Figure, matplotlib Axes).
    """
    if run_df is None or run_df.empty:
        print(f"Warning: DataFrame is empty, cannot plot {dorado_col} vs {guppy_col}.")
        return None, None
    if dorado_col not in run_df.columns or guppy_col not in run_df.columns:
        print(f"Warning: Columns '{dorado_col}' or '{guppy_col}' not found in DataFrame.")
        return None, None

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(run_df[guppy_col], run_df[dorado_col], alpha=0.6, label=f'{run_df.shape[0]} pairs')

    # Default labels and title
    xl = xlabel if xlabel else guppy_col.replace('_', ' ')
    yl = ylabel if ylabel else dorado_col.replace('_', ' ')
    t = title if title else f"{yl} vs {xl}"

    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(t)

    # Add diagonal line (y=x) for reference
    min_val = min(run_df[guppy_col].min(), run_df[dorado_col].min())
    max_val = max(run_df[guppy_col].max(), run_df[dorado_col].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x')

    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig, ax

    # Add to data_functions.py (or define in notebook)
def plot_histogram(run_df: pd.DataFrame,
                   metric_col: str,
                   title: str = None,
                   xlabel: str = None,
                   ylabel: str = 'Frequency',
                   bins: int = 30,
                   reference_line: float = None,
                   figsize: tuple = (10, 6)) -> tuple:
    """
    Creates a histogram for a specific metric.

    Args:
        run_df: DataFrame containing the run comparison data.
        metric_col: Column name for the metric to plot.
        title: Plot title.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        bins: Number of histogram bins.
        reference_line: Value for a vertical reference line (e.g., 0 for difference plots).
        figsize: Figure size.

    Returns:
        Tuple (matplotlib Figure, matplotlib Axes).
    """
    if run_df is None or run_df.empty:
        print(f"Warning: DataFrame is empty, cannot plot histogram for {metric_col}.")
        return None, None
    if metric_col not in run_df.columns:
        print(f"Warning: Column '{metric_col}' not found in DataFrame.")
        return None, None

    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram
    ax.hist(run_df[metric_col].dropna(), bins=bins, label=f'{run_df[metric_col].notna().sum()} values') # Drop NaN for plotting

    # Default labels and title
    xl = xlabel if xlabel else metric_col.replace('_', ' ')
    t = title if title else f"Distribution of {xl}"

    ax.set_xlabel(xl)
    ax.set_ylabel(ylabel)
    ax.set_title(t)

    # Add reference line if specified
    if reference_line is not None:
        ax.axvline(x=reference_line, color='r', linestyle='--', alpha=0.7, label=f'x={reference_line}')

    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    return fig, ax

    # Add to data_functions.py (or define in notebook)
def plot_comparison_with_difference(run_df: pd.DataFrame,
                                    dorado_col: str,
                                    guppy_col: str,
                                    diff_col: str,
                                    figure_title: str = None,
                                    figsize: tuple = (18, 7)) -> tuple:
    """
    Creates a combined figure with a scatter plot (Dorado vs Guppy)
    and a histogram of the differences.

    Args:
        run_df: DataFrame containing the run comparison data.
        dorado_col: Column name for the Dorado metric (scatter y-axis).
        guppy_col: Column name for the Guppy metric (scatter x-axis).
        diff_col: Column name for the difference metric (histogram).
        figure_title: Overall title for the combined figure.
        figsize: Figure size for the entire figure.

    Returns:
        Tuple (matplotlib Figure, array of matplotlib Axes).
    """
    if run_df is None or run_df.empty:
        print(f"Warning: DataFrame is empty, cannot create combined plot.")
        return None, None
    required_cols = [dorado_col, guppy_col, diff_col]
    if not all(col in run_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in run_df.columns]
        print(f"Warning: Missing columns for combined plot: {missing}")
        return None, None

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Scatter Plot (Left) ---
    scatter_ax = axes[0]
    scatter_ax.scatter(run_df[guppy_col], run_df[dorado_col], alpha=0.6, label=f'{run_df.shape[0]} pairs')
    xl_scatter = guppy_col.replace('_', ' ')
    yl_scatter = dorado_col.replace('_', ' ')
    scatter_ax.set_xlabel(f"Guppy {xl_scatter.split(' ')[-1]}") # Extract metric name
    scatter_ax.set_ylabel(f"Dorado {yl_scatter.split(' ')[-1]}")
    scatter_ax.set_title(f"Dorado vs Guppy: {yl_scatter.split(' ')[-1]}")

    # Diagonal line
    min_val = min(run_df[guppy_col].min(), run_df[dorado_col].min())
    max_val = max(run_df[guppy_col].max(), run_df[dorado_col].max())
    scatter_ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x')
    scatter_ax.legend()
    scatter_ax.grid(True)

    # --- Histogram Plot (Right) ---
    hist_ax = axes[1]
    hist_data = run_df[diff_col].dropna()
    hist_ax.hist(hist_data, bins=30, label=f'{len(hist_data)} values')
    # Reference line at 0
    hist_ax.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='x=0')

    xl_hist = diff_col.replace('_', ' ')
    hist_ax.set_xlabel(f"Difference ({xl_hist})")
    hist_ax.set_ylabel("Frequency")
    hist_ax.set_title(f"Distribution of Differences")
    hist_ax.legend()
    hist_ax.grid(True, axis='y')

    if figure_title:
        fig.suptitle(figure_title, fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95] if figure_title else None) # Adjust layout for suptitle
    return fig, axes

def display_run_analysis(run_id, all_results_data):
    """Retrieves data and displays statistics and plots for the selected run_id."""

    print(f"Generating analysis display for Run ID: {run_id}")

    # --- Retrieve Data ---
    run_data = all_results_data.get(run_id)
    if not run_data:
        display(Markdown(f"**Error:** No processed data found for run `{run_id}`."))
        return

    run_df = run_data.get('comparison_df')
    run_stats = run_data.get('stats') # Dictionary of calculated statistics
    run_counts = run_data.get('counts', {})

    if run_df is None or run_df.empty:
        display(Markdown(f"**Warning:** No matched pairs found or comparison DataFrame is empty for run `{run_id}`. Cannot generate detailed plots."))
        # Display basic counts if available
        display(Markdown(f"### Basic Counts for {run_id}"))
        display(Markdown(f"- Matched Pairs: {run_counts.get('matched', 'N/A')}"))
        display(Markdown(f"- Dorado-Only Sequences: {run_counts.get('dorado_only', 'N/A')}"))
        display(Markdown(f"- Guppy-Only Sequences: {run_counts.get('guppy_only', 'N/A')}"))
        return

    # --- Display Summary Statistics --- [cite: 345]
    display(Markdown(f"### Summary Statistics for {run_id} ({run_df.shape[0]} matched pairs)"))
    if run_stats:
        # Format and display key stats nicely
        stats_display = []
        for metric, values in run_stats.items():
            if 'median_diff' in values and 'p_value' in values:
                p_val_str = f"{values['p_value']:.4f}" if values['p_value'] is not None else 'N/A'
                significance = "**(Significant)**" if values['p_value'] is not None and values['p_value'] < 0.05 else ""
                stats_display.append(f"- **{metric}**: Median Diff = {values.get('median_diff', 'N/A'):.2f}, p-value = {p_val_str} {significance} (n={values.get('n_pairs', run_df.shape[0])})")
        display(Markdown("\n".join(stats_display)))
    else:
        display(Markdown("*Statistics calculation skipped or failed for this run.*"))

    # --- Generate Plots --- [cite: 346, 347, 348]
    display(Markdown("---")) # Separator

    # Plot 1: RiC Comparison [cite: 346]
    try:
        display(Markdown(f"#### Reads in Consensus (RiC) Comparison"))
        fig_ric, _ = plot_comparison_with_difference(
            run_df,
            dorado_col='Dorado_RiC',
            guppy_col='Guppy_RiC',
            diff_col='RiC_Difference',
            figure_title=f'{run_id} - RiC Comparison'
        )
        if fig_ric: plt.show(fig_ric)
    except Exception as e:
        print(f"Error plotting RiC: {e}")

    # Plot 2: Length Comparison [cite: 347]
    try:
        display(Markdown(f"#### Sequence Length Comparison"))
        fig_len, _ = plot_comparison_with_difference(
            run_df,
            dorado_col='Dorado_Length',
            guppy_col='Guppy_Length',
            diff_col='Length_Difference',
            figure_title=f'{run_id} - Length Comparison'
        )
        if fig_len: plt.show(fig_len)
    except Exception as e:
        print(f"Error plotting Length: {e}")

    # Plot 3: GC Content Comparison [cite: 347]
    try:
        # Check if GC columns exist and have data before plotting
        if 'Dorado_GC' in run_df.columns and 'Guppy_GC' in run_df.columns and run_df[['Dorado_GC', 'Guppy_GC']].notna().all(axis=1).any():
             display(Markdown(f"#### GC Content Comparison"))
             fig_gc, _ = plot_comparison_with_difference(
                 run_df,
                 dorado_col='Dorado_GC',
                 guppy_col='Guppy_GC',
                 diff_col='GC_Difference', # Assumes this column exists from Step 3.4
                 figure_title=f'{run_id} - GC Content Comparison'
             )
             if fig_gc: plt.show(fig_gc)
        else:
             display(Markdown(f"*(Skipping GC Content plot: GC columns missing or contain only NaN values)*"))
    except Exception as e:
        print(f"Error plotting GC Content: {e}")

    # Plot 4: Identity Distribution [cite: 348]
    try:
        display(Markdown(f"#### Sequence Identity Distribution"))
        fig_identity, _ = plot_histogram(
             run_df,
             metric_col='Identity_Percent',
             title=f'{run_id} - Sequence Identity Distribution',
             xlabel='Sequence Identity (%)',
             bins=25 # More bins might be useful here
        )
        if fig_identity: plt.show(fig_identity)
    except Exception as e:
        print(f"Error plotting Identity: {e}")

    # Plot 5 & 6: Homopolymer Count/MaxLen Difference (Optional) [cite: 348]
    # Example for Homo_Count difference
    try:
        if 'Dorado_Homo_Count' in run_df.columns and 'Guppy_Homo_Count' in run_df.columns:
             run_df['Homo_Count_Difference'] = run_df['Dorado_Homo_Count'] - run_df['Guppy_Homo_Count']
             if run_df['Homo_Count_Difference'].notna().any():
                 display(Markdown(f"#### Homopolymer Count Difference (Min Length 5)"))
                 fig_homo_c, _ = plot_histogram(
                      run_df,
                      metric_col='Homo_Count_Difference',
                      title=f'{run_id} - Homopolymer Count Difference Distribution',
                      xlabel='Difference in Homopolymer Runs (Dorado - Guppy)',
                      reference_line=0
                 )
                 if fig_homo_c: plt.show(fig_homo_c)
             else:
                 display(Markdown(f"*(Skipping Homopolymer Count plot: No non-NaN difference values)*"))

    except Exception as e:
        print(f"Error plotting Homopolymer Count Difference: {e}")

def create_sequence_alignment_viewer(run_id, all_results_data):
    """Creates widgets to select a sample and view its alignment."""

    print(f"Initializing alignment viewer for Run ID: {run_id}")

    # --- Get Data Needed ---
    run_data = all_results_data.get(run_id)
    if not run_data:
        display(Markdown("**Error:** Run data not found."))
        return

    matched_pairs = run_data.get('matched_pairs') # Get the list of matched pairs
    run_df = run_data.get('comparison_df') # Get the comparison DataFrame

    if not matched_pairs:
         display(Markdown("**Warning:** No matched pairs found for this run. Cannot create viewer."))
         return
    if run_df is None or run_df.empty:
         display(Markdown("**Warning:** Comparison DataFrame not found or empty. Cannot populate sample list."))
         return

    # --- Create Widgets ---
    # Dropdown for Sample ID
    sample_options = natsorted(run_df['Sample_ID'].unique().tolist())
    if not sample_options:
         display(Markdown("**Warning:** No unique Sample IDs found in comparison data."))
         return

    sample_dropdown = widgets.Dropdown(
        options=sample_options,
        description='Select Sample:',
        style={'description_width': 'initial'}
    )

    # Slider for window size (optional)
    window_slider = widgets.IntSlider(
        value=100, min=50, max=200, step=10,
        description='Window Size:', style={'description_width': 'initial'}
    )

    # Button to trigger display
    view_button = widgets.Button(description="View Alignment", button_style='info')

    # Output area for the alignment HTML
    alignment_output = widgets.Output()

    # --- Button Click Handler ---
    def on_view_button_click(button):
        selected_sample_id = sample_dropdown.value
        selected_window_size = window_slider.value

        with alignment_output:
            clear_output(wait=True)
            print(f"Searching for alignment data for Sample ID: {selected_sample_id}...")

            # Find the correct matched_pair dictionary for the selected Sample_ID
            # This assumes Sample_ID is unique enough in the context of matched pairs for this run.
            # If multiple matches exist for one Sample_ID (ambiguous case), this might just pick the first.
            target_pair = None
            for pair in matched_pairs:
                # Use Sample_ID from the run_df as the key to find the pair
                # We might need a more robust way if Sample_ID isn't unique in run_df
                # or if multiple matches for a sample exist.
                # Let's assume run_df has unique Sample_IDs for now, or we take the first match.
                 if pair.get('sample_id') == selected_sample_id:
                     # Additional check: Match headers if possible to be more specific,
                     # requires headers to be present in the run_df row.
                     # row = run_df[run_df['Sample_ID'] == selected_sample_id].iloc[0]
                     # if pair['dorado']['header'] == row['Dorado_Header']: # Requires these cols in run_df
                          target_pair = pair
                          break # Found the first match for this Sample ID

            if target_pair and 'alignment' in target_pair:
                display(Markdown(f"**Displaying Alignment for Sample:** {selected_sample_id}"))
                # Display some basic info about the pair
                display(Markdown(f"- Dorado Header: `{target_pair['dorado'].get('header', 'N/A')}`"))
                display(Markdown(f"- Guppy Header: `{target_pair['guppy'].get('header', 'N/A')}`"))
                display(Markdown(f"- Identity: {target_pair['alignment'].get('identity', 'N/A'):.2f}% | Mismatches: {target_pair['alignment'].get('mismatches', 'N/A')} | Insertions: {target_pair['alignment'].get('insertions', 'N/A')} | Deletions: {target_pair['alignment'].get('deletions', 'N/A')}"))
                display(Markdown("---"))

                # Call the HTML formatting function
                html_display = format_alignment_html(target_pair, selected_window_size)
                display(html_display)
            else:
                print(f"Could not find alignment data for Sample ID: {selected_sample_id}")

    # Link button event
    view_button.on_click(on_view_button_click)

    # --- Display Widgets ---
    controls = widgets.VBox([
        widgets.HTML("Select a matched sample pair and click 'View Alignment':"), # Instructions
        sample_dropdown,
        window_slider,
        view_button
    ])
    display(widgets.VBox([controls, alignment_output]))

def format_alignment_html(alignment_dict: dict, window_size: int = 100) -> HTML:
    """
    Formats a Biopython pairwise alignment into HTML with highlighting.

    Args:
        alignment_dict: The dictionary for a matched pair, MUST contain
                        the 'alignment' sub-dictionary which includes the
                        'alignment_obj' (a Bio.Align.Alignment object).
        window_size: Number of bases to show per line chunk.

    Returns:
        IPython.display.HTML object containing the formatted alignment.
    """
    if not alignment_dict or 'alignment' not in alignment_dict or 'alignment_obj' not in alignment_dict['alignment']:
        return HTML("<p>Error: Alignment data is missing or invalid.</p>")

    alignment = alignment_dict['alignment']['alignment_obj']
    identity_percent = alignment_dict['alignment'].get('identity', 0) # Get identity from dict

    if alignment is None:
         return HTML("<p>Error: Alignment object not found in provided data.</p>")

    try:
        # Biopython's Alignment object can be indexed to get aligned sequences
        # The structure might vary slightly based on Biopython version.
        # Often it's alignment[0] for target (e.g., Dorado) and alignment[1] for query (e.g., Guppy)

        # Safely access aligned sequences
        if hasattr(alignment, 'target') and hasattr(alignment, 'query'):
             aligned_dorado = str(alignment.target) # Adapt if keys are different
             aligned_guppy = str(alignment.query)
        elif isinstance(alignment, (list, tuple)) and len(alignment) >= 2 and hasattr(alignment[0], 'seq') and hasattr(alignment[1], 'seq'):
             # Handle older or different alignment object structures if needed
             aligned_dorado = str(alignment[0].seq)
             aligned_guppy = str(alignment[1].seq)
        else:
            aligned_seqs_str = str(alignment) # Get the full string representation
            lines = aligned_seqs_str.strip().split('\n')
            aligned_dorado = ""
            aligned_guppy = ""
            # Counter to track lines within a potential alignment block (target, match, query)
            # We assume target is line 0, query is line 2 of these blocks
            line_counter_in_block = 0

            for line in lines:
                line_strip = line.strip()
                if not line_strip: continue # Skip empty lines
                # Find the sequence part (bases and dashes) at the end of the line
                seq_part_match = re.search(r'[ACGTN-]+$', line)

                if seq_part_match:
                    seq_part = seq_part_match.group(0) # Extract the sequence part
                    # Assuming Target (Dorado) is the first sequence line in a block
                    if line_counter_in_block == 0:
                        aligned_dorado += seq_part
                    # Assuming Query (Guppy) is the third sequence line in a block
                    elif line_counter_in_block == 2:
                        aligned_guppy += seq_part
                # Increment and wrap the counter (assumes blocks of 3 lines: target, match, query)
                # Only increment if we likely processed a line within a block (heuristic: it contained sequence)
                # A more robust parser might be needed if format isn't strictly 3 lines always
                if seq_part_match:
                    # This simple modulo assumes strict 3-line blocks; adjust if format varies
                    line_counter_in_block = (line_counter_in_block + 1) % 3
                else:
                    # Reset if we encounter a line without sequence (e.g., header/footer)
                    line_counter_in_block = 0
            else:
                 raise ValueError("Could not parse aligned sequences from alignment object string representation.")

        # --- Debug Prints (Keep these) ---
        print("\n" + "=" * 20 + " DEBUG Alignment Data " + "=" * 20)
        print(f"Sample ID (from dict): {alignment_dict.get('sample_id', 'N/A')}") # Make sure sample_id is passed if you want it here
        print(f"Raw Alignment Object (`alignment` variable):\n{alignment}")
        print("-" * 62)
        print(f"Extracted Dorado String (len={len(aligned_dorado)}):")
        print(aligned_dorado)
        print("-" * 62)
        print(f"Extracted Guppy String (len={len(aligned_guppy)}):")
        print(aligned_guppy)
        print("=" * 62 + "\n")
        # --- End Debug Prints ---
        seq_length = len(aligned_dorado)

        if seq_length != len(aligned_guppy):
            print(f"ERROR in format_alignment_html: Aligned sequence lengths still differ after parsing!")
            print(f"  Dorado Length: {seq_length}, Guppy Length: {len(aligned_guppy)}")
            return HTML("<p><b>Error: Cannot display alignment. Internal inconsistency detected (aligned sequence lengths differ AFTER parsing). Check parsing logic.</b></p>")

        if seq_length == 0:
             return HTML("<p>Error: Alignment resulted in empty sequences.</p>")


        # Start HTML generation
        html_parts = []
        html_parts.append(f"<h4>Sequence Alignment (Identity: {identity_percent:.2f}%)</h4>")
        html_parts.append("<div style='font-family: monospace; line-height: 1.6;'>") # Monospace font

        # Legend
        html_parts.append("<div style='margin-bottom:10px;'>")
        html_parts.append("<span style='background-color:#c8e6c9; padding: 2px 5px; border-radius: 3px; margin-right: 5px;'>Match</span>")
        html_parts.append("<span style='background-color:#f8bbd0; padding: 2px 5px; border-radius: 3px; margin-right: 5px;'>Mismatch</span>")
        html_parts.append("<span style='background-color:#bbdefb; padding: 2px 5px; border-radius: 3px;'>Gap</span>")
        html_parts.append("</div>")

        # Process alignment in chunks
        for i in range(0, seq_length, window_size):
            chunk_dorado = aligned_dorado[i:min(i + window_size, seq_length)]
            chunk_guppy = aligned_guppy[i:min(i + window_size, seq_length)]

            html_d = ""
            html_g = ""
            match_line = ""

            for j in range(len(chunk_dorado)):
                d_char = chunk_dorado[j]
                g_char = chunk_guppy[j]

                if d_char == g_char: # Match
                    style = 'background-color:#c8e6c9;'
                    match_line += "|"
                elif d_char == '-' or g_char == '-': # Gap
                    style = 'background-color:#bbdefb;'
                    match_line += " "
                else: # Mismatch
                    style = 'background-color:#f8bbd0;'
                    match_line += "."

                html_d += f"<span style='{style}'>{d_char}</span>"
                html_g += f"<span style='{style}'>{g_char}</span>"

            # Add chunk to HTML
            html_parts.append(f"<div style='margin-bottom: 15px;'>")
            html_parts.append(f"<div>Position {i+1} - {min(i+window_size, seq_length)}</div>")
            html_parts.append(f"<div>Dorado: {html_d}</div>")
            html_parts.append(f"<div style='color: #777;'>Match: &nbsp;{match_line}</div>") # Match line
            html_parts.append(f"<div>Guppy: &nbsp;{html_g}</div>") # Add space for alignment
            html_parts.append("</div>")

        html_parts.append("</div>") # Close monospace div
        return HTML(''.join(html_parts))

    except Exception as e:
        # Catch any error during formatting
        return HTML(f"<p>Error generating alignment display: {str(e)}</p>")