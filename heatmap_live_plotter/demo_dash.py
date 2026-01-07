import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State

# --- toy "time-dependent" field ---------------------------------
nx, ny = 100, 100         # use 200x200 in your real case
nt = 200                # number of time steps
t_max = 10.0

x = np.linspace(-2, 2, nx)
y = np.linspace(-2, 2, ny)

def field(t_index):
    """Simple example: a Gaussian bump moving in x and oscillating in amplitude."""
    t = t_max * t_index / nt
    X, Y = np.meshgrid(x, y, indexing="ij")
    cx = np.sin(t)      # center x
    cy = 0.0            # center y
    r2 = (X - cx)**2 + (Y - cy)**2
    return np.exp(-r2 * 4.0) * np.cos(2.0 * t)

# --- initial figure ----------------------------------------------
z0 = field(0)

fig = go.Figure(
    data=go.Heatmap(
        z=z0,
        x=x,
        y=y,
        colorscale="Viridis",
        zmin=-1,
        zmax=1,
        colorbar=dict(title="u(x,y,t)"),
    )
)
fig.update_layout(
    margin=dict(l=40, r=40, t=40, b=40),
    xaxis_title="x",
    yaxis_title="y",
)

# --- Dash app ----------------------------------------------------
app = Dash(__name__)

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div([
                    html.Div("time", style={"width": "10%"}),
                    html.Div(dcc.Slider(id="slider_t", min=0, max=nt, value=nt/4), style={"width": "40%"}),
                    html.Div(html.Button("Play", id="play-button", n_clicks=0), style={"width": "10%"}),
                    html.Div("", style={"width": "40%"}),
                ], style={"display": "flex", "alignItems": "center"}),

                dcc.Interval(
                    id="interval",
                    interval=50, # ms between frames (20 FPS)
                    n_intervals=0,
                    disabled=True, # start paused
                ),
                dcc.Store(id="t-index", data=0),
                dcc.Store(id="is-playing", data=False),
            ]
        ),

        html.Div([
            html.Div(
                html.Div([
                    html.Div("x-axis", style={"width": "20%"}),
                    html.Div(
                        dcc.Dropdown([*range(10)], 2, id="x_axis_di"),
                        style={"width": "80%"}
                    )],
                    style={"display": "flex", "alignItems": "center"},
                ), style={"width": "45%"}
            ),
            html.Div(
                html.Div([
                    html.Div("y-axis", style={"width": "20%"}),
                    html.Div(
                        dcc.Dropdown([*range(10)], 2, id="y_axis_di"),
                        style={"width": "80%"}
                    )],
                    style={"display": "flex", "alignItems": "center"},
                ), style={"width": "45%"}
            ),
            ], style={
                "display": "flex",
                "justifyContent": "space-between"
            }
        ),


        #dcc.Dropdown([*range(10)], 4, id="z_axis_di"),
        dcc.Graph(id="heatmap", figure=fig, style={"height": "80vh"}),
        html.Div([
            html.Div(
                [html.Label("x1"), dcc.Slider(id="slider_x1", min=0, max=10, value=3)],
                style={"width": "45%"}
            ),
            html.Div(
                [html.Label("x2"), dcc.Slider(id="slider_x2", min=0, max=10, value=2)],
                style={"width": "45%"}
            ),
            ], style={
                "display": "flex",
                "justifyContent": "space-between"
            }
        ),
        html.Div([
            html.Div(
                [html.Label("x3"), dcc.Slider(id="slider_x3", min=0, max=10, value=3)],
                style={"width": "45%"}
            ),
            html.Div(
                [html.Label("x4"), dcc.Slider(id="slider_x4", min=0, max=10, value=2)],
                style={"width": "45%"}
            ),
            ], style={
                "display": "flex",
                "justifyContent": "space-between"
            }
        ),
    ]
)

# Toggle play / pause
@app.callback(
    Output("is-playing", "data"),
    Output("interval", "disabled"),
    Output("play-button", "children"),
    Input("play-button", "n_clicks"),
    State("is-playing", "data"),
)
def toggle_play(n_clicks, is_playing):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    new_state = not is_playing
    return new_state, (not new_state), ("Pause" if new_state else "Play")


# Increase slider value while playing
@app.callback(
    Output("slider_t", "value"),
    Input("interval", "n_intervals"),
    State("slider_t", "value"),
)
def update_slider(n, value):
    if value is None:
        value = 0
    #return min(value + 1, 100)
    return (value + 1) % nt

# Advance time index when playing
@app.callback(
    Output("t-index", "data"),
    Input("interval", "n_intervals"),
    State("t-index", "data"),
    prevent_initial_call=True,
)
def advance_time(n_intervals, t_index):
    # simple wrap-around
    return (t_index + 1) % nt

# Update heatmap when time index changes
@app.callback(
    Output("heatmap", "figure"),
    Input("t-index", "data"),
    State("heatmap", "figure"),
)
def update_heatmap(t_index, current_fig):
    z = field(t_index).tolist()
    current_fig["data"][0]["z"] = z
    return current_fig

if __name__ == "__main__":
    app.run(debug=True)
