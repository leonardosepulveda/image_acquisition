import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from typing import Optional, List


def get_color_to_channel_dict(microscope='MF3'):
    if microscope=='MF3' or \
       microscope=='MF5':
        d = dict(zip([np.nan, 405,488,560,650,750],[np.nan,4,3,2,1,0]))
    return d

def get_frame_table(bead_z , bead_seq, color_seq , end_seq, z_pos, microscope='MF3'):
    """
    For simplicity, the shutter files have the following features:
        - fixed number of frames per z position (to have a nZ x m frames)
        - first z position is z = 0
        - last z position is z = 0
        - blank frames have np.nan color
    
    """
   
    color_2_channel_dict = get_color_to_channel_dict(microscope=microscope)

    data = []

    # add frames for the imaging of beads at z == bead_z
    for i , color in enumerate(bead_seq):
        data.append([color, color_2_channel_dict[color], bead_z])

    # add frames for the imaging of sample
    for z in z_pos:
        for c in color_seq:
            data.append([c, color_2_channel_dict[c], z])

    # add frames for the imaging of beads at z == bead_z
    for i , color in enumerate(end_seq):
        data.append([color, color_2_channel_dict[color], bead_z])

    frame_df = pd.DataFrame(columns=['color', 'channel', 'z'], data=data)
            
    return frame_df

def print_frame_table(df):
    """
    To easily visualize the structure of the shutter file
    """
    
    # extract the number of frames per z
    counts = df['z'].value_counts()
    frames_per_z = min(counts)
    
    col_width = 6
    col_sep = ' '*6
    
    # Header
    header_width = col_width * frames_per_z
    print(f'{"frames":{header_width}s}{col_sep}'
          f'{"color":{header_width}s}{col_sep}'
          f'{"channel":{header_width}s}{col_sep}'
          f'{"z":{header_width}s}{col_sep}')
    print()  # blank line

    n = len(df)

    for start in range(0, n, frames_per_z):
        end = start + frames_per_z
        group = df.iloc[start:end]

        # Skip incomplete group if desired; remove this if you want to print it
        if len(group) < frames_per_z:
            break

        # Frames (index)
        frames_str = ''.join(f'{int(idx):{col_width}d}' for idx in group.index)

        # Helper for NaNs in int-like columns
        def fmt_int_like(val):
            if pd.isna(val):
                return f'{"nan":>{col_width}}'
            return f'{int(val):{col_width}d}'

        # Colors
        colors_str = ''.join(fmt_int_like(v) for v in group['color'])

        # Channels
        channels_str = ''.join(fmt_int_like(v) for v in group['channel'])

        # z as float with 2 decimals
        z_str = ''.join(f'{float(v):{col_width}.2f}' for v in group['z'])

        print(f'{frames_str}{col_sep}{colors_str}{col_sep}{channels_str}{col_sep}{z_str}')
        
def get_color_sequence_name(df):
    """
    Given a DataFrame with a 'color' column, return a succinct name like:
    'blkf8_405f2_650f4_750f4'
    where:
      - 'blk' corresponds to NaN (blank) entries
      - other entries are color values (as ints)
      - 'fN' is the total count of frames with that color
    """
    col = df['color']

    # Count NaNs separately
    n_blk = col.isna().sum()

    # Count non-NaN colors
    counts = col.value_counts(dropna=True)

    # Build parts of the name
    parts = []

    if n_blk > 0:
        parts.append(f"blkf{int(n_blk)}")

    # Sort colors numerically, if they are numeric
    # (convert index to float, then int for printing)
    for color in sorted(counts.index.astype(float)):
        count = int(counts.loc[color])
        parts.append(f"{int(color)}f{count}")

    # Join with underscores
    return "_".join(parts)

def create_shutter_file(df, filename, oversampling=1, default_power=1.0):
    """
    Convert a DataFrame with columns: ['color', 'channel', 'z']
    index = frame number (on-time)
    to an XML file of events, pretty-printed with CRLF line endings.

    Additionally:
      - Embed the full frame table (as CSV) into the first XML comment so
        that the original z (and full frame info) can be reconstructed
        later from the shutter file alone.
    """
    # Ensure index is numeric and sorted
    df = df.sort_index()

    # Root element
    root = ET.Element("repeat")

#     # --- Add frame table as CSV in a comment at the beginning ---
#     # We include the index as "frame"
#     csv_buf = io.StringIO()
#     df.to_csv(csv_buf, index=True)   # index name will become first column header
#     csv_text = csv_buf.getvalue()

#     frame_table_comment_text = (
#         "FRAME_TABLE_CSV_START\n"
#         + csv_text +
#         "FRAME_TABLE_CSV_END"
#     )

#     comment_el = ET.Comment(frame_table_comment_text)
#     root.append(comment_el)
#     # Two empty lines after the comment, before <oversampling>...
#     # (minidom/toprettyxml will preserve this as blank lines)
#     comment_el.tail = "\n\n"

    # <oversampling>
    overs_el = ET.SubElement(root, "oversampling")
    overs_el.text = str(oversampling)

    # <frames>
    frames_el = ET.SubElement(root, "frames")
    frames_el.text = str(len(df))

    last_z = None  # to track when z changes and add a comment

    for frame, row in df.iterrows():
        channel = row["channel"]
        z = row["z"]
        color = row.get("color", np.nan)  # in case 'color' column might be missing

        # Skip non-event frames (NaN channel)
        if pd.isna(channel):
            continue

        # Add comment when z changes (or for the first event)
        if (last_z is None) or (z != last_z):
            if z == 0 and not pd.isna(color) and int(color) == 405:
                comment_text = f" z = {int(z)} um, 405 beads"
            else:
                if float(z).is_integer():
                    comment_text = f" z = {int(z)} um"
                else:
                    comment_text = f" z = {z} um"
            root.append(ET.Comment(comment_text))
            last_z = z

        # <event>
        event_el = ET.SubElement(root, "event")

        ch_el = ET.SubElement(event_el, "channel")
        ch_el.text = str(int(channel))

        pw_el = ET.SubElement(event_el, "power")
        pw_el.text = f"{default_power:.1f}"

        on_el = ET.SubElement(event_el, "on")
        on_el.text = f"{float(frame):.1f}"

        off_el = ET.SubElement(event_el, "off")
        off_el.text = f"{float(frame + 1):.1f}"

    # ---- Pretty-print + CRLF writing ----
    rough_bytes = ET.tostring(root, encoding="utf-8")
    dom = minidom.parseString(rough_bytes)
    pretty_xml = dom.toprettyxml(indent="  ", encoding="ISO-8859-1")  # bytes

    # Decode to text (LF newlines for now)
    pretty_text = pretty_xml.decode("ISO-8859-1")

    # Add an empty line before every comment line for legibility
    pretty_text = pretty_text.replace("\n  <!--", "\n\n  <!--")

    # Normalize to CRLF for Windows
    pretty_text = pretty_text.replace("\n", "\r\n")

    with open(filename, "w", encoding="ISO-8859-1", newline="") as f:
        f.write(pretty_text)
        
def format_z_offsets_from_frame_table(df: pd.DataFrame) -> str:
    """
    Build the text for <z_offsets> from the 'z' column of frame_table.

    Logic (matches your example):
      - Determine group size as the length of a contiguous run of identical z
        (assumes all runs have same length).
      - Take one z per run (i.e. 0.0, 1.0, 1.5, 2.0, 2.5, 0.0 in your example).
      - For each such z, repeat it group_size times.
      - Arrange values in rows of 'group_size' values.
      - Add a comma after every value except the very last overall value.
    """
    z = df['z'].astype(float).to_list()

    # Compute run lengths to infer group size
    run_lengths = []
    last = None
    count = 0
    for val in z:
        if last is None or val == last:
            count += 1
        else:
            run_lengths.append(count)
            count = 1
        last = val
    if count:
        run_lengths.append(count)

    group_size = run_lengths[0]
    # (Optional sanity check)
    if not all(rl == group_size for rl in run_lengths):
        raise ValueError(f"Inconsistent run lengths in z column: {run_lengths}")

    # Unique z per run (in order)
    unique_z_per_run = []
    last = object()
    for val in z:
        if val != last:
            unique_z_per_run.append(val)
            last = val

    # Build flat list: each run's z repeated group_size times
    values = []
    for val in unique_z_per_run:
        values.extend([val] * group_size)

    # Now format as text for the XML, with commas as specified
    lines = []
    total = len(values)

    for i, val in enumerate(values):
        is_last = (i == total - 1)
        suffix = '' if is_last else ','  # no comma after very last value
        token = f"{val:.1f}{suffix}"

        row_idx = i // group_size
        if len(lines) <= row_idx:
            lines.append([])
        lines[row_idx].append(token)

    # Indentation is mainly cosmetic; important part is comma placement
    # We’ll match your style roughly:
    #   0.0,  0.0,  0.0,
    # etc., with two spaces between tokens.
    indent = "         "  # 9 spaces, as in your example
    inner_lines = []
    for row in lines:
        inner_lines.append(indent + "  ".join(row))

    # Surround with newlines so ElementTree pretty-prints reasonably.
    text = "\n" + "\n".join(inner_lines) + "\n      "
    return text

def create_hal_config(prefix: str,
                      frame_table: pd.DataFrame,
                      default_power: Optional[List[float]] = None,
                      xml_dir: str or Path = ".",
                      output_dir: str or Path = ".") -> Path:
    """
    Read '{prefix}.xml' as plain text from xml_dir, update:
      - <frames>            -> len(frame_table)
      - <default_power>     -> comma-joined default_power (if provided)
      - <shutters>          -> '{color_sequence_name}_shutter.xml'
      - <z_offsets>         -> formatted from frame_table['z']

    Then write '{output_dir}/{prefix}-{color_sequence_name}.xml',
    preserving original comments and overall layout as much as possible,
    and adding exactly one empty line before each comment for legibility.
    """
    xml_dir = Path(xml_dir)
    output_dir = Path(output_dir)

    in_path = xml_dir / f"{prefix}.xml"
    with open(in_path, "rb") as f:
        raw = f.read()

    # Decode and normalize to '\n' internally
    text = raw.decode("ISO-8859-1").replace("\r\n", "\n")

    # --- 1. frames ---
    def repl_frames(m):
        open_tag, _, close_tag = m.groups()
        return f"{open_tag}{len(frame_table)}{close_tag}"

    text = re.sub(
        r"(<frames[^>]*>)(.*?)(</frames>)",
        repl_frames,
        text,
        flags=re.DOTALL,
    )

    # --- 2. default_power (optional) ---
    if default_power is not None:
        dp_str = ",".join(str(v) for v in default_power)

        def repl_default_power(m):
            open_tag, _, close_tag = m.groups()
            return f"{open_tag}{dp_str}{close_tag}"

        text = re.sub(
            r"(<default_power[^>]*>)(.*?)(</default_power>)",
            repl_default_power,
            text,
            flags=re.DOTALL,
        )

    # --- 3. shutters ---
    seq_name = get_color_sequence_name(frame_table)
    shutters_value = f"{seq_name}_shutter.xml"

    def repl_shutters(m):
        open_tag, _, close_tag = m.groups()
        return f"{open_tag}{shutters_value}{close_tag}"

    text = re.sub(
        r"(<shutters[^>]*>)(.*?)(</shutters>)",
        repl_shutters,
        text,
        flags=re.DOTALL,
    )

    # --- 4. z_offsets from frame_table['z'] ---
    new_z_block = format_z_offsets_from_frame_table(frame_table)
    # new_z_block should already contain leading/trailing newlines and indentation

    def repl_z_offsets(m):
        open_tag, _, close_tag = m.groups()
        return f"{open_tag}{new_z_block}{close_tag}"

    text = re.sub(
        r"(<z_offsets[^>]*>)(.*?)(</z_offsets>)",
        repl_z_offsets,
        text,
        flags=re.DOTALL,
    )

    # --- 5. Ensure exactly one empty line before each comment ---
    # Only add an extra blank line if there isn't already one just before the comment.
    # Preserve indentation before <!-- by capturing spaces/tabs only.
    text = re.sub(
        r"(?<!\n\n)\n([ \t]*)<!--",   # a newline, optional spaces/tabs, then <!--
        r"\n\n\1<!--",
        text,
    )

    # --- 6. Write out with CRLF line endings ---
    out_path = output_dir / f"{prefix}-{seq_name}.xml"
    with open(out_path, "w", encoding="ISO-8859-1", newline="") as f:
        f.write(text.replace("\n", "\r\n"))

    return out_path

def _strip_whitespace_etree(elem):
    """
    Remove whitespace-only .text and .tail from an ElementTree tree.
    This prevents minidom.toprettyxml from inserting extra blank lines.
    """
    # Clean children first
    for child in list(elem):
        _strip_whitespace_etree(child)
        if child.tail is not None and not child.tail.strip():
            child.tail = None

    # Clean this element's text
    if elem.text is not None and not elem.text.strip():
        elem.text = None
        
    
def print_xml_raw(path, encoding="ISO-8859-1"):
    """
    Print the XML file exactly as it is on disk.
    - Comments are preserved.
    - Existing indentation and spacing are preserved.
    - CRLF line endings are preserved (or optionally normalized).
    """
    with open(path, "rb") as f:
        data = f.read()

    # Option A: preserve original CRLF exactly
    text = data.decode(encoding)
    print(text, end="")  # no extra newline at end

    # If you prefer normalized LF for your console instead, use:
    # text = data.decode(encoding).replace("\r\n", "\n")
    # print(text, end="")
    
def get_channel_to_color_dict(microscope='MF3'):
    """
    Invert get_color_to_channel_dict: channel -> color.
    """
    color_to_channel = get_color_to_channel_dict(microscope=microscope)
    ch2col = {}
    for col, ch in color_to_channel.items():
        if pd.isna(ch):
            continue
        ch2col[int(ch)] = col
    return ch2col

def read_shutter_file_to_frame_table(filename, microscope='MF3'):
    """
    Recreate a frame table from a shutter XML file created by `create_shutter_file`.

    Logic:
      - The first XML comment contains the full frame table as CSV, between
        the markers 'FRAME_TABLE_CSV_START' and 'FRAME_TABLE_CSV_END'.
      - We parse that CSV to recover z for every frame (and can also see
        color/channel, if desired).
      - We then parse <event> elements to rebuild the actual color/channel
        activity per frame (overwriting any color/channel from the CSV
        with what the events say).
      - Result:
          * z comes from the first comment's CSV.
          * color & channel come from the events (non-event frames get NaN).

    Returns
    -------
    frame_df : pandas.DataFrame
        Columns: ['color', 'channel', 'z'], indexed by frame number.
    """
    # --- 1. Parse XML with comments preserved ---
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(filename, parser=parser)
    root = tree.getroot()   # <repeat>

    # --- 2. Read total number of frames (for sanity) ---
    frames_el = root.find("frames")
    if frames_el is None or frames_el.text is None:
        raise ValueError("Could not find <frames> element in shutter file.")
    n_frames = int(float(frames_el.text))

    # --- 3. Extract frame table CSV from the first comment ---
    csv_text = None
    for child in root:
        if child.tag is Comment and child.text:
            txt = child.text
            if "FRAME_TABLE_CSV_START" in txt and "FRAME_TABLE_CSV_END" in txt:
                # Extract between the markers
                start = txt.index("FRAME_TABLE_CSV_START") + len("FRAME_TABLE_CSV_START")
                end = txt.index("FRAME_TABLE_CSV_END")
                csv_body = txt[start:end].strip("\n")
                csv_text = csv_body
                break

    if csv_text is None:
        raise ValueError("Could not find FRAME_TABLE_CSV comment in shutter file.")

    # Parse CSV into DataFrame
    csv_buf = io.StringIO(csv_text)
    # The first column should be "frame" (index); we force that to be index_col=0
    df_csv = pd.read_csv(csv_buf, index_col=0)
    df_csv.index.name = "frame"

    # Sanity check length
    if len(df_csv) != n_frames:
        # Not fatal, but warn user
        print(f"Warning: frames in CSV ({len(df_csv)}) != <frames> ({n_frames})")

    # We will take z from the CSV, but color/channel from events
    z_series = df_csv["z"].astype(float)

    # --- 4. Build inverse color<->channel mapping (in case needed) ---
    color_to_channel = get_color_to_channel_dict(microscope=microscope)
    channel_to_color = {}
    for col, ch in color_to_channel.items():
        if pd.isna(col) or pd.isna(ch):
            continue
        channel_to_color[int(ch)] = float(col)

    # --- 5. Walk children again, reading <event> elements ---
    events = {}

    for child in root:
        if child.tag != "event":
            continue

        on_el = child.find("on")
        ch_el = child.find("channel")
        if on_el is None or ch_el is None:
            continue

        frame = int(float(on_el.text))
        channel = int(ch_el.text)

        # Color from channel
        color = channel_to_color.get(channel, np.nan)

        events[frame] = (color, float(channel))

    # --- 6. Build full frame table using:
    #       - z from comment-CSV
    #       - color/channel from events
    #       - NaNs where there is no event
    data = []
    max_frame = max(n_frames, len(z_series))
    for f in range(max_frame):
        z = z_series.loc[f] if f in z_series.index else np.nan

        if f in events:
            color, channel = events[f]
        else:
            color = np.nan
            channel = np.nan

        data.append([color, channel, z])

    frame_df = pd.DataFrame(data, columns=["color", "channel", "z"])
    frame_df.index.name = "frame"
    return frame_df

def visualize_shutter_sequence(frame_df, title=None, savepath=None):
    """
    Visualize a shutter/frame sequence from a frame table DataFrame, vertically.

    Parameters
    ----------
    frame_df : pandas.DataFrame
        Must have columns ['color', 'channel', 'z'] and integer index = frame.
        Typically produced by get_frame_table(...) or
        read_shutter_file_to_frame_table(...).

    title : str, optional
        Title for the plot.
    """
    df = frame_df.copy()
    df = df.reset_index().rename(columns={'index': 'frame'})

    if df.empty:
        print("Frame table is empty.")
        return

    # Map wavelengths to colors for plotting
    wavelength_to_color = {
        405.0: '#9467bd',   # purple
        488.0: '#1f77b4',   # blue
        560.0: '#ff7f0e',   # orange
        650.0: '#2ca02c',   # green
        750.0: '#d62728',   # red
    }
    default_col = '#7f7f7f'  # for unknown wavelengths

    fig, ax = plt.subplots(figsize=(5, 8))

    # Determine x-levels (channels on x, frames on y)
    active_channels = sorted(df['channel'].dropna().unique())
    blank_x = -1  # column where we draw blank frames
    all_x = [blank_x] + active_channels

    # Ensure sorted by frame so separators and annotations line up
    df = df.sort_values('frame')

    # Plot one horizontal bar per frame
    for _, row in df.iterrows():
        frame = int(row['frame'])
        ch = row['channel']
        wav = row['color']

        if pd.isna(ch):
            # Blank frame -> black box at "blank" column
            x_center = blank_x
            facecolor = 'black'
        else:
            x_center = int(ch)
            facecolor = wavelength_to_color.get(float(wav), default_col)

        ax.barh(
            y=frame,
            width=0.8,             # bar width in x-direction
            left=x_center - 0.4,
            height=1.0,            # bar height in y-direction
            align='center',
            color=facecolor,
            edgecolor='k',
            linewidth=0.2,
        )

    # Add horizontal separators between different z positions
    # Draw line between frame i and i+1 if z changes
    df_sorted = df.sort_values('frame')
    prev_frame = None
    prev_z = None
    for _, row in df_sorted.iterrows():
        f = int(row['frame'])
        z = row['z']
        if prev_frame is not None and z != prev_z:
            y_sep = (prev_frame + f) / 2.0
            ax.axhline(
                y=y_sep,
                color='0.7',
                linestyle='--',
                linewidth=0.5,
                zorder=0,
            )
        prev_frame = f
        prev_z = z

    ax.set_ylabel("Frame")
    ax.set_xlabel("Channel / blank")

    if title is None:
        title = "Shutter sequence (vertical)"
    ax.set_title(title)

    # x-ticks: "blank" column plus actual channels
    xticks = all_x
    xtick_labels = ['blank'] + [str(int(c)) for c in active_channels]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    # Annotate first frame of each z to the right of the top-most channel
    if active_channels or True:
        # choose a right margin a bit beyond the max x position
        right_x = (max(all_x) + 1.0) if all_x else 0.5
        by_z = df.groupby('z', sort=True)['frame'].min()
        for z_val, f0 in by_z.items():
            if pd.isna(z_val):
                continue
            ax.text(
                right_x,
                f0,
                f"z={z_val:g}",
                fontsize=8,
                va='center',
                ha='left',
            )

    # Limits
    if all_x:
        ax.set_xlim(min(all_x) - 1, max(all_x) + 2)
    else:
        ax.set_xlim(-1, 1)

    ax.set_ylim(df['frame'].min() - 1, df['frame'].max() + 1)

    plt.tight_layout()
    
    # NEW: save figure if requested
    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    
    plt.show()