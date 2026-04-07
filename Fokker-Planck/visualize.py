"""
open up multiple tabs in the browser
 - each one showing a 3d interative view of the molecule with 7 atoms
"""

import utility_fp


def show(points, name=""):
    import plotly.graph_objects as go
    import numpy as np
    points_d = np.array(points)

    lines = []
    for i in range(points_d.shape[0]):
        for j in range(i + 1, points_d.shape[0]):
            lines.append(go.Scatter3d(
                x=[points_d[i, 0], points_d[j, 0]],
                y=[points_d[i, 1], points_d[j, 1]],
                z=[points_d[i, 2], points_d[j, 2]],
                mode='lines',
                line=dict(color='blue', width=5),
                showlegend=False
            ))
    scatter_plot = go.Scatter3d(
        x=points_d[:, 0],
        y=points_d[:, 1],
        z=points_d[:, 2],
        mode='markers',
        marker=dict(size=15, color='red', opacity=0.8),
        showlegend=True,
        name=name
    )
    fig = go.Figure(data=[scatter_plot] + lines)

    fig.update_layout(scene=dict(
      xaxis_title='X',
      yaxis_title='Y',
      zaxis_title='Z',
      aspectmode='cube'
    ))
    print("show")
    fig.show()



if __name__=="__main__":

    minima = utility_fp.read_molecules_from_dir("molecules/mol.")
    for i in range(len(minima)):
        cluster = minima[i]
        show(cluster, f"minimum {i+1}")

    #saddles = read_mols_in_dir("./LJ7_ts/points.")
    #for i in range(1): #range(len(saddles)):
    #    cluster = saddles[i]
    #    show(cluster, f"saddle {i+1}")
