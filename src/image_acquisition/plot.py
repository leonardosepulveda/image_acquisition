import matplotlib.pyplot as plt
import pandas as pd

def visualize_shutter_sequence(frame_table, title=None, savepath=None):
    """
    Visualize a shutter/frame sequence from a frame table DataFrame, vertically.

    Parameters
    ----------
    frame_table : pandas.DataFrame
        Must have columns ['color', 'channel', 'z'] and integer index = frame.
        Typically produced by get_frame_table(...) or
        read_shutter_file_to_frame_table(...).

    title : str, optional
        Title for the plot.
    """
    df = frame_table.copy()
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