import base64
import io
import dash
import numpy as np
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Initialize app
app = dash.Dash(__name__)
server = app.server
app.title = "Gesture to Sign Trajectory - Variance Explorer"

df = pd.DataFrame()

# Layout
app.layout = html.Div([
    html.H1("Gesture to Sign Trajectory - Variance Explorer", style={'textAlign': 'center', 'marginBottom': '30px', 'fontFamily': 'Segoe UI, sans-serif'}),
    # add subtitle with authors
    html.H2("Authors: Sarka Kadava and Door Spruijt", style={'textAlign': 'center', 'marginBottom': '30px', 'fontFamily': 'Segoe UI, sans-serif'}),
    html.Div([
        # LEFT COLUMN
        html.Div([
            html.Label("Select groups to compare:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='file-selector',
                options=[
                    {'label': 'All groups', 'value': 'all_data.csv'},
                    {'label': 'Gesture vs sign', 'value': 'gesture_sign.csv'},
                    {'label': 'Native signs vs L2 signs', 'value': 'native_l2.csv'}
                ],
                placeholder='Select a file',
                style={'marginBottom': '20px'}
            ),
            html.Div(id='file-info', style={'marginBottom': '20px', 'fontStyle': 'italic'}),
            dcc.Graph(id='scree-plot', style={'height': '500px'}),
            dcc.Graph(id='pca-plot', style={'height': '500px'}),
        ], style={'flex': '1', 'padding': '8px'}),

        # RIGHT COLUMN
        html.Div([
            html.Label("Select Feature 1:"),
            dcc.Dropdown(id='feature1-dropdown', style={'marginBottom': '10px'}),
            html.Label("Select Feature 2:"),
            dcc.Dropdown(id='feature2-dropdown', style={'marginBottom': '20px'}),

            dcc.Graph(id='correlation-plot', style={'height': '450px'}),
            dcc.Graph(id='boxplot1', style={'height': '400px'}),
            dcc.Graph(id='boxplot2', style={'height': '400px'}),
        ], style={'flex': '1', 'padding': '8px', 'minWidth': '300px'}),
    ], style={'display': 'flex', 'flexDirection': 'row', 'flexWrap': 'wrap', 'fontFamily': 'Segoe UI, sans-serif', 'padding': '20px'})
])

# Helper to remove outliers
def remove_outliers(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[feature] >= lower) & (df[feature] <= upper)]

# Scree plot and PCA
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

# Handle file upload
@app.callback(
    [Output('file-info', 'children'),
     Output('feature1-dropdown', 'options'),
     Output('feature2-dropdown', 'options'),
     Output('scree-plot', 'figure'),
     Output('pca-plot', 'figure')],
    Input('file-selector', 'value')
)

def load_selected_file(filename):
    global df
    global group_color_map

    if filename is None:
        return "Select a CSV file from the dropdown", [], [], go.Figure(), go.Figure()

    df = pd.read_csv(f'assets/{filename}')

    # colors
    unique_groups = df[df.columns[-1]].unique()
    color_palette = px.colors.qualitative.Plotly
    group_color_map = {
        group: color_palette[i % len(color_palette)]
        for i, group in enumerate(sorted(unique_groups))
    }

    feature_cols = df.columns[:-1]
    label_col = df.columns[-1]

    scree_fig, pca, scaled_data = get_scree_plot(df[feature_cols])

    X_pca = pca.transform(scaled_data)
    coeff = pca.components_.T
    top_n = 10
    top_vars = set()
    for i in range(3):
        top_vars.update(np.argsort(np.abs(pca.components_[i]))[-top_n:])
    selected_vars = sorted(list(top_vars))

    if df[label_col].dtype == 'object' or df[label_col].dtype.name == 'category':
        y = LabelEncoder().fit_transform(df[label_col])
    else:
        y = df[label_col]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
        mode='markers',
        marker=dict(size=5, color=y, colorscale='Viridis'),
        name='Samples',
        text=[f"Sample {i}" for i in range(len(X_pca))],
        showlegend=False
    ))

    scale = 100
    for i in selected_vars:
        fig.add_trace(go.Scatter3d(
            x=[0, coeff[i, 0] * scale],
            y=[0, coeff[i, 1] * scale],
            z=[0, coeff[i, 2] * scale],
            mode='lines',
            line=dict(color='red', width=4),
            hovertext=df.columns[i],
            name='',
            showlegend=False
        ))

    fig.update_layout(title="3D PCA Biplot",
                      scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
                      height=800, margin=dict(l=0, r=0, b=0, t=40))

    options = [{'label': col, 'value': col} for col in feature_cols]
    return f'Selected: {filename} ({df.shape[0]} rows, {df.shape[1]} columns)', options, options, scree_fig, fig

# Correlation plot
@app.callback(
    Output('correlation-plot', 'figure'),
    Input('feature1-dropdown', 'value'),
    Input('feature2-dropdown', 'value')
)
def update_corr(feature1, feature2):
    if df.empty or not feature1 or not feature2:
        return go.Figure()
    dff = remove_outliers(df, feature1)
    dff = remove_outliers(dff, feature2)
    corr = dff[feature1].corr(dff[feature2])
    fig = px.scatter(dff, x=feature1, y=feature2, trendline="ols",
                     title=f"Correlation: {corr:.2f}")
    return fig

# Boxplot 1
@app.callback(
    Output('boxplot1', 'figure'),
    Input('feature1-dropdown', 'value')
)
def box1(feature):
    if df.empty or not feature or df.columns[-1] not in df.columns:
        return go.Figure()
    dff = remove_outliers(df, feature)
    return px.violin(
        dff,
        x=df.columns[-1],
        y=feature,
        points="all",
        box=True,
        title=f'{feature} by {df.columns[-1]}',
        color=df.columns[-1],
        color_discrete_map=group_color_map
    )
# Boxplot 2
@app.callback(
    Output('boxplot2', 'figure'),
    Input('feature2-dropdown', 'value')
)

def box2(feature):
    if df.empty or not feature or df.columns[-1] not in df.columns:
        return go.Figure()
    dff = remove_outliers(df, feature)
    return px.violin(
        dff,
        x=df.columns[-1],
        y=feature,
        points="all",
        box=True,
        title=f'{feature} by {df.columns[-1]}',
        color=df.columns[-1],
        color_discrete_map=group_color_map
    )

if __name__ == '__main__':
    app.run_server(debug=True)
