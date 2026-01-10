import numpy as np
import torch
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State


# read from metadata
t_max = 10.0
d = 8
def u_analytic(X):
    # X.shape = (batch size, spatial+time dims)
    bs, D = X.shape
    d = D-1
    alpha = 0.01
    u_space = torch.prod(torch.sin(torch.pi * X[:,:-1]), dim=1) # .shape = (bs,)
    u_time = torch.exp(- d * alpha * torch.pi**2 * X[:,-1]) # .shape = (bs,)
    # return shape = (batch size, 1)
    return (u_space * u_time).unsqueeze(dim=1)
    #return torch.array([u_space * u_time])


def get_coord_names(d):
    return [f"x{i+1}" for i in range(d)]


# --- create grid data ---------------------------------
nx = 100
nt = 200

#xi = torch.linspace(0, 1, nx)
#xj = torch.linspace(0, 1, nx)
xi = torch.linspace(-2.0, 2.0, nx)
xj = torch.linspace(-2.0, 2.0, nx)
Xi_grid, Xj_grid = torch.meshgrid(xi, xj, indexing='ij')
xi_flat = Xi_grid.reshape(-1, 1)
xj_flat = Xj_grid.reshape(-1, 1)

# define domain
def define_domain(plot_dims, x_vals):
    x_flat_list = []
    for di in range(d):
        if di == plot_dims[0]:
            x_flat_list.append(xi_flat)
        elif di == plot_dims[1]:
            x_flat_list.append(xj_flat)
        else:
            fixed_flat = torch.ones_like(xi_flat) * x_vals[di]
            x_flat_list.append(fixed_flat)
    return x_flat_list


# initialize
t_val_init = 0.0
x_vals_init = 0.5*torch.ones(d)
plot_dims = [0,1]
x_flat_list = define_domain(plot_dims, x_vals_init)
t_flat = torch.ones_like(xi_flat) * t_val_init
X = torch.cat([*x_flat_list, t_flat], dim=1)
Y = u_analytic(X)
Y_grid = Y.reshape(nx, nx)
print("sessin saved")


"""
animation
- just keep increating the value of one coordinate

slider
- set val of one coord

selecting
- keep X the same, just redraw
"""



# --- initial figure ----------------------------------------------
fig = go.Figure(
    data=go.Heatmap(
        z=Y_grid.numpy(),
        x=xi.numpy(),
        y=xj.numpy(),
        colorscale="Viridis",
        zmin=-1,
        zmax=1,
        #colorbar=dict(title="u(x1,..,xd,t)", len=0.75, thickness=7, title_side="right"),
        colorbar=dict(len=0.75, thickness=7),
    )
)
fig.update_layout(
    #title="model",
    title=dict(text='Heatmap 1', y=0.98, yanchor='top'),
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis_title="x1",
    yaxis_title="x2",
    # --- enforce equal aspect ratio ---
    xaxis=dict(scaleanchor="y", constrain="domain"),
    yaxis=dict(constrain="domain")
)


#tick_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
#fig.update_xaxes(
#    tickmode="array",
#    tickvals=tick_vals,
#    ticktext=[str(v) for v in tick_vals]
#)
#fig.update_yaxes(
#    tickmode="array",
#    tickvals=tick_vals,
#    ticktext=[str(v) for v in tick_vals]
#)


# --- Dash app ----------------------------------------------------
app = Dash(__name__)

# As many sliders as there are spatial dimensions
N_SLIDERS_PER_ROW = 3
# number of full sliders
n_full_slider_rows = d//N_SLIDERS_PER_ROW
# number of sliders in the last unfilled row
n_sliders_last_row = d - N_SLIDERS_PER_ROW*n_full_slider_rows
assert N_SLIDERS_PER_ROW*n_full_slider_rows + n_sliders_last_row == d

spatial_sliders = []
for ri in range(n_full_slider_rows+1):
    if ri != n_full_slider_rows:
        n_sliders = N_SLIDERS_PER_ROW
    else:
        n_sliders = n_sliders_last_row
    if n_sliders != 0:
        spatial_sliders.append(
            html.Div(
                [html.Div(
                    [html.Label(f"x{s+1}"), dcc.Slider(id=f"slider_x{s+1}", min=0, max=1, value=0.5)],
                    style={"width": "30%"}
                ) for s in range(N_SLIDERS_PER_ROW*ri, N_SLIDERS_PER_ROW*ri+n_sliders)],
                style={
                    "display": "flex",
                    "justifyContent": "space-between"
                }
            )
        )



app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=True),
        html.Div(
            [
                html.Div([
                    html.Div("", style={"width": "10%"}),
                    html.Div("animate", style={"width": "10%"}),
                    html.Div(
                        dcc.Dropdown(["t"] + get_coord_names(d), "t", id="animate_di"),
                        style={"width": "10%"}
                    ),
                    html.Div(dcc.Slider(id="slider_t", min=0, max=t_max, value=0.0), style={"width": "30%"}),
                    html.Div(html.Button("Play", id="play-button", n_clicks=0), style={"width": "10%"}),
                    #html.Div("", style={"width": "40%"}),
                ], style={"display": "flex", "alignItems": "center"}),

                dcc.Interval(
                    id="interval",
                    interval=50, # ms between frames (20 FPS)
                    n_intervals=0,
                    disabled=True, # start paused
                ),
                dcc.Store(id="is-playing", data=False),
            ]
        ),

        html.Div([
            html.Div(
                html.Div([
                    html.Div("", style={"width": "10%"}),
                    html.Div("xi-axis", style={"width": "10%"}),
                    html.Div(
                        dcc.Dropdown(get_coord_names(d), "x1", id="xi-axis"),
                        style={"width": "10%"}
                    )],
                    style={"display": "flex", "alignItems": "center"},
                ), #style={"width": "50%"}
            ),
            dcc.Store(id="xi-axis-prev", data="x1"),
            html.Div(
                html.Div([
                    html.Div("", style={"width": "10%"}),
                    html.Div("xj-axis", style={"width": "10%"}),
                    html.Div(
                        dcc.Dropdown(get_coord_names(d), "x2", id="xj-axis"),
                        style={"width": "10%"}
                    )],
                    style={"display": "flex", "alignItems": "center"},
                ), #style={"width": "50%"}
            ),
            dcc.Store(id="xj-axis-prev", data="x2"),
            ]
            #], style={
            #    "display": "flex",
            #    "justifyContent": "space-between"
            #}
        ),

        #dcc.Graph(id="heatmap", figure=fig, style={"height": "80vh"}),
        #dcc.Graph(id="heatmap", figure=fig),
        html.Div([
            html.Div([
                dcc.Graph(figure=fig, id="heatmap")
            ], style={'width': '33%', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(figure=fig)
            ], style={'width': '33%', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(figure=fig)
            ], style={'width': '33%', 'display': 'inline-block'})
        ]),


        *spatial_sliders,
    ]
)

#@app.callback(
#    Output("heatmap", "figure"),
#    Input('url', 'href'), # Triggers on page load or refresh
#    State("heatmap", "figure"),
#)
#def reset_heatmap(href, current_fig):
#    print("sessin reloaded")
#    t_flat = torch.ones_like(xi_flat) * t_val
#    X = torch.concatenate([*x_flat_list, t_flat], dim=1)
#    Y = u_analytic(X)
#    Y_grid = Y.reshape(nx, nx)
#    current_fig["data"][0]["z"] = Y_grid
#    return current_fig




# Toggle play / pause
@app.callback(
    Output("is-playing", "data"),
    Output("interval", "disabled"),
    Output("play-button", "children"),
    Input("play-button", "n_clicks"),
    State("is-playing", "data"),
    prevent_initial_call=True,
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
def update_slider(n_intervals, value):
    if value is None:
        value = 0.0
    #return (value + 1) % nt
    value += 10*1/nt
    if value > t_max:
        return 0.0
    else:
        return value

from dash import callback_context
# Update heatmap when
# - choose different axes
# - move the time or space sliders
@app.callback(
    # outputs
    Output("heatmap", "figure"),
    [Output(f"slider_x{i+1}", "disabled") for i in range(d)],
    [Output("xi-axis-prev", "data"), Output("xj-axis-prev", "data")],
    # inputs
    Input('url', 'href'), # Triggers on page load or refresh
    [Input("xi-axis", "value"), Input("xj-axis", "value")],
    [Input("slider_t", "value")] + [Input(f"slider_x{i+1}", "value") for i in range(d)],
    # states
    [State("xi-axis-prev", "data"), State("xj-axis-prev", "data")],
    State("heatmap", "figure"),
)
def update_heatmap(*args):
    trigger = callback_context.triggered_id
    # inputs
    href = args[0]
    xi_axis = args[1]
    xj_axis = args[2]
    t_value = args[3]
    x_values = args[4:-3]
    # states
    xi_axis_prev = args[-3]
    xj_axis_prev = args[-2]
    current_fig = args[-1]
    #### disable sliders ####
    disabed_list = []
    if xi_axis == xj_axis:
        print("Illegal to set same xi xj")
    for i in range(d):
        if f"x{i+1}" == xi_axis or f"x{i+1}" == xj_axis:
            disabed_list.append(True)
        else:
            disabed_list.append(False)
    #### update X ####
    global X
    ###### initial heat map ######
    if trigger == "url":
        t_flat = torch.ones_like(xi_flat) * t_val_init
        print("session refreshed")
        X = torch.cat([*x_flat_list, t_flat], dim=1)
    ###### choosing different axes ######
    elif trigger == "xi-axis" or trigger == "xj-axis":
        prev_axes = [int(xi_axis_prev[1:])-1, int(xj_axis_prev[1:])-1]
        X[:, prev_axes[0]] = torch.ones(nx**2) * x_values[prev_axes[0]]
        X[:, prev_axes[1]] = torch.ones(nx**2) * x_values[prev_axes[1]]
        plot_dims = [int(xi_axis[1:])-1, int(xj_axis[1:])-1]
        X[:, plot_dims[0]] = xi_flat[:,0]
        X[:, plot_dims[1]] = xj_flat[:,0]
    ###### setting values via slider ######
    else:
        coord_name = trigger[len("slider_"):]
        if coord_name == "t":
            X[:, -1:] = t_value * torch.ones_like(xi_flat)
        elif coord_name[0] == "x":
            di = int(coord_name[1:])-1
            X[:, di:di+1] = x_values[di] * torch.ones_like(xi_flat)
    # update figure
    Y = u_analytic(X)
    Y_grid = Y.reshape(nx, nx)
    current_fig["data"][0]["z"] = Y_grid.numpy()
    current_fig["layout"]["xaxis"]["title"]["text"] = xi_axis
    current_fig["layout"]["yaxis"]["title"]["text"] = xj_axis
    return current_fig, *disabed_list, xi_axis, xj_axis


if __name__ == "__main__":
    app.run(debug=True)
