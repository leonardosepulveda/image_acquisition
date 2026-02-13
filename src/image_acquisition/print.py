import pandas as pd

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