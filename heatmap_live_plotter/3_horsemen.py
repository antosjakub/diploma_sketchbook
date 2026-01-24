import dash
from dash import dcc, html
import plotly.graph_objects as go
import numpy as np

# Create sample data for three heatmaps on [0,1]x[0,1] discretized as 300x300
np.random.seed(42)
x = np.linspace(0, 1, 300)
y = np.linspace(0, 1, 300)
X, Y = np.meshgrid(x, y)

# Generate sample data (you can replace these with your own functions)
data1 = np.sin(5 * np.pi * X) * np.cos(5 * np.pi * Y)
data2 = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.1)
data3 = X * Y

# Create three heatmap figures with minimal margins
fig1 = go.Figure(data=go.Heatmap(
    z=data1,
    x=x,
    y=y,
    colorscale='Viridis',
    showscale=True,
    colorbar=dict(len=0.9, thickness=15)
))
fig1.update_layout(
    title='Heatmap 1',
    height=400,
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(title='x', constrain="domain"),
    yaxis=dict(title='y', scaleanchor='x', scaleratio=1, constrain="domain")
)

fig2 = go.Figure(data=go.Heatmap(
    z=data2,
    x=x,
    y=y,
    colorscale='Plasma',
    showscale=True,
    colorbar=dict(len=0.9, thickness=15)
))
fig2.update_layout(
    title='Heatmap 2',
    height=500,
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(title='x'),
    yaxis=dict(title='y', scaleanchor='x', scaleratio=1)
)

fig3 = go.Figure(data=go.Heatmap(
    z=data3,
    x=x,
    y=y,
    colorscale='Hot',
    showscale=True,
    colorbar=dict(len=0.9, thickness=15)
))
fig3.update_layout(
    title='Heatmap 3',
    height=500,
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(title='x'),
    yaxis=dict(title='y', scaleanchor='x', scaleratio=1)
)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Graph(figure=fig1, config={'displayModeBar': False})
        ], style={'width': '33.33%', 'display': 'inline-block', 'padding': '0px', 'margin': '0px'}),
        html.Div([
            dcc.Graph(figure=fig2, config={'displayModeBar': False})
        ], style={'width': '33.33%', 'display': 'inline-block', 'padding': '0px', 'margin': '0px'}),
        html.Div([
            dcc.Graph(figure=fig3, config={'displayModeBar': False})
        ], style={'width': '33.33%', 'display': 'inline-block', 'padding': '0px', 'margin': '0px'})
    ], style={'whiteSpace': 'nowrap', 'fontSize': '0'})
], style={'margin': '0px', 'padding': '0px'})

if __name__ == '__main__':
    app.run(debug=True, port=9090)