import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.graph_objs as go
import plotly.express as px
import plotly.colors as pc
from dash import Dash, dcc, html, Input, Output
import os

app = Dash(__name__)

df = None
group_color_map = {}

file_options = {
    "sampled_all_250_cleanforPCA.csv": "All groups",
    "sampled_GS_df_250_cleanforPCA.csv": "Gesturer vs Signer",
    "sampled_NN_df_250_cleanforPCA.csv": "L2 Signer vs Native Signer"
}

def get_scree_plot(df_features):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df_features)
    pca = PCA()
    pca.fit(standardized_data)
    explained = pca.explained_variance_ratio_
    cumulative = explained.cumsum()

    n_80 = np.argmax(cumulative >= 0.8) + 1
    n_95 = np.argmax(cumulative >= 0.95) + 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=cumulative, mode='lines+markers', name='Cumulative Variance'))
    fig.add_vline(x=n_80-1, line=dict(color='red', dash='dash'), annotation_text=f'80% at {n_80}')
    fig.add_vline(x=n_95-1, line=dict(color='green', dash='dash'), annotation_text=f'95% at {n_95}')
    fig.update_layout(title="Cumulative Explained Variance", xaxis_title="Principal Component", yaxis_title="Cumulative Variance")
    return fig, pca, standardized_data

def get_top_features_text(pca, feature_names):
    lines = ["Top 5 Highest Feature Loadings per 3 PCs:\n"]
    for i in range(3):
        component = pca.components_[i]
        top_indices = np.argsort(np.abs(component))[-5:][::-1]
        top_features = [feature_names[idx] for idx in top_indices]
        lines.append(f"PC{i+1}: {', '.join(top_features)}")
    return lines

app.layout = html.Div([
    # Header section — blue background (#1b29a8)
    html.Div([
    html.H1("PCA Visualizer Dashboard", style={'textAlign': 'center', 'marginBottom': '0px', 'color': 'white'}),
    html.P([
        "Authors: ",
        "Sarka Kadava", html.Sup("1"), ", ",
        "Door Spruijt", html.Sup("2")
    ], style={'textAlign': 'center', 'marginTop': '2px', 'marginBottom': '0px', 'color': 'white'}),
    html.P([
        html.Sup("1"), "Leibniz Center General Linguistics ",  html.Sup("2"), "University of Cologne"
    ], style={'textAlign': 'center', 'marginTop': '0px', 'marginBottom': '0px', 'fontStyle': 'italic', 'fontSize': '14px', 'color': 'white'}),
], style={'backgroundColor': '#1b29a8', 'padding': '15px 0'}),

    # Body section — orange background (#e6927b) with top padding to create the gap
    html.Div([
        html.Div(
    dcc.Dropdown(
        id='file-selector',
        options=[{'label': name, 'value': fname} for fname, name in file_options.items()],
        placeholder='Select groups to compare',
        style={'width': '300px'}  # set your preferred width
    ),
    style={'display': 'flex', 'justifyContent': 'center', 'marginTop': '15px', 'marginBottom': '15px', 'width': '90%'}
),

        html.Div([
            dcc.Graph(id='pca-3d', style={'width': '65%', 'height': '800px', 'display': 'inline-block'}),
            dcc.Graph(id='scree-plot', style={'width': '35%', 'height': '600px', 'display': 'inline-block'}),
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'flex-start', 'marginTop': '20px'}),

        html.Div(id='top-features-text', style={'whiteSpace': 'pre-line', 'padding': '10px', 'textAlign': 'center', 'fontWeight': 'bold'}),

        html.Div([
            html.H2("Choose a feature to plot", style={'color': 'white', 'marginBottom': '10px'}),
            dcc.Dropdown(id='feature1-dropdown', style={'width': '60%', 'margin': '0 auto'}),
            dcc.Graph(id='boxplot1', style={'width': '90%', 'margin': '20px auto'})
        ], style={
    'backgroundColor': '#1b29a8',
    'padding': '20px',
    'marginTop': '20px',
    'textAlign': 'center',
    'width': '65%',
    'margin': '0 auto',
})
    ], style={'backgroundColor': '#e6927b', 'minHeight': '100vh', 'paddingTop': '30px', 'paddingBottom': '30px'})
])


@app.callback(
    Output('pca-3d', 'figure'),
    Output('scree-plot', 'figure'),
    Output('feature1-dropdown', 'options'),
    Output('top-features-text', 'children'),
    Input('file-selector', 'value')
)
def load_selected_file(filename):
    global df, group_color_map

    if filename is None:
        return go.Figure(), go.Figure(), [], ""

    df = pd.read_csv(f'assets/{filename}')
    feature_cols = df.columns[:-1]
    label_col = df.columns[-1]

    scree_fig, pca, scaled_data = get_scree_plot(df[feature_cols])
    X_pca = pca.transform(scaled_data)
    coeff = pca.components_.T

    top_n = 5
    top_vars = set()
    for i in range(3):
        top_vars.update(np.argsort(np.abs(pca.components_[i]))[-top_n:])
    selected_vars = sorted(list(top_vars))

    y = df[label_col].values
    unique_labels = np.unique(y)
    color_map = {label: pc.qualitative.Set1[i % len(pc.qualitative.Set1)] for i, label in enumerate(unique_labels)}

   # Filter points where PC3 <= 15
    valid_idx = X_pca[:, 2] <= 15
    X_pca_filtered = X_pca[valid_idx]
    y_filtered = y[valid_idx]

    fig = go.Figure()

    # Scatter points with consistent colors (filtered)
    for label in unique_labels:
        indices = np.where(y_filtered == label)
        fig.add_trace(go.Scatter3d(
            x=X_pca_filtered[indices, 0].flatten(),
            y=X_pca_filtered[indices, 1].flatten(),
            z=X_pca_filtered[indices, 2].flatten(),
            mode='markers',
            marker=dict(size=5, color=color_map[label], opacity=0.7),
            name=str(label),
            text=[f"Sample {i}" for i in indices[0]],
            showlegend=True
        ))

    # Centroids based on filtered data
    for label in unique_labels:
        indices = np.where(y_filtered == label)
        centroid = X_pca_filtered[indices].mean(axis=0)
        fig.add_trace(go.Scatter3d(
            x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
            mode='markers+text',
            marker=dict(size=6, color=color_map[label], symbol='x', line=dict(color='black', width=8)),
            showlegend=False
        ))

    # Feature vectors (PCA loadings) — always full scale, no filtering here
    scale = 10
    for i in selected_vars:
        fig.add_trace(go.Scatter3d(
            x=[0, coeff[i, 0] * scale],
            y=[0, coeff[i, 1] * scale],
            z=[0, coeff[i, 2] * scale],
            mode='lines',
            line=dict(color='black', width=4),
            hovertext=df.columns[i],
            name='',
            showlegend=False
        ))

    # Layout with fixed z-axis limit
    fig.update_layout(
        title="3D PCA ~ plots first three principal components and feature vectors",
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            zaxis=dict(range=[None, 15])  # restrict z-axis visible range
        ),
        height=800,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing='constant')
    )



    feature_options = [{'label': col, 'value': col} for col in feature_cols]
    top_features_text = "\n".join(get_top_features_text(pca, df.columns[:-1]))

    return fig, scree_fig, feature_options, top_features_text

@app.callback(
    Output('boxplot1', 'figure'),
    Input('file-selector', 'value'),
    Input('feature1-dropdown', 'value')
)
def update_violin_plot(filename, feature):
    if filename is None or feature is None:
        return go.Figure()

    df = pd.read_csv(f'assets/{filename}')
    group_col = df.columns[-1]

    # Tukey filtering
    cleaned_df = pd.DataFrame()
    for group, subset in df.groupby(group_col):
        Q1 = subset[feature].quantile(0.25)
        Q3 = subset[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        filtered = subset[(subset[feature] >= lower) & (subset[feature] <= upper)]
        cleaned_df = pd.concat([cleaned_df, filtered], ignore_index=True)

    fig = px.violin(cleaned_df, y=feature, x=group_col, color=group_col, box=True, points="all",
                    title=f'{feature} by Group')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
