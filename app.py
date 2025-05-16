#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import nibabel as nib
import os
import networkx as nx
import time
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
from nilearn import plotting
from scipy.ndimage import map_coordinates
from dash import Dash, html,dash_table,dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage import measure
from scipy.ndimage import zoom



mask_path = r"C:\Users\zmaso\OneDrive\data science\phd\phdproject\data\2mm data\P1\GM\GM_mask_2mm.nii"
msk = nib.load(mask_path)
msk_data = msk.get_fdata()


N_slice_x, N_slice_y, N_slice_z = msk_data.shape
mid_z = N_slice_z//2
mid_y = N_slice_y//2
mid_x = N_slice_x//2

#non zero masked features of 4 mri scans:CBF, CMRO2, CVR, OEF, GM
non_zero_masked_features = pd.read_csv(r"C:\Users\zmaso\OneDrive\data science\phd\phdproject\data\2mm data\P1\p1_masked_data_nonlinear_nonzero_2mm_.csv")
#all data from mri scans with zero values
im_data =  pd.read_csv(r"C:\Users\zmaso\OneDrive\data science\phd\phdproject\data\2mm data\P1\p1_feature_matrix_nonlinear_2mm.csv")
g1 = nx.read_graphml(r"C:\Users\zmaso\OneDrive\data science\phd\phdproject\data\2mm data\P1\subgraphs\ego_patch_node0_r2.graphml")

#Fetch patch info 
nodes = g1.nodes
edges = g1.edges(data=True)
nodes = [int(node) for node in g1.nodes]

#Normalizing data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(non_zero_masked_features.iloc[:,[0,1,2,3,4]])
scaled_features = pd.DataFrame(scaled_features, columns=non_zero_masked_features.columns[:5])
scaled_data = pd.concat([scaled_features, non_zero_masked_features.iloc[:,[6,7,8]]], axis=1)

scaled_raw_data = scaler.fit_transform(im_data)
scaled_raw_data = pd.DataFrame(scaled_raw_data, columns=im_data.columns)

#each feature into Separate daatframe  for normalized data
mask_gm = np.array(scaled_raw_data.iloc[:,4]).reshape(N_slice_x, N_slice_y, N_slice_z)
cbf =  np.array(scaled_raw_data.iloc[:,0]).reshape(N_slice_x, N_slice_y, N_slice_z)
cmro2 = np.array(scaled_raw_data.iloc[:,1]).reshape(N_slice_x, N_slice_y, N_slice_z)
cvr = np.array(scaled_raw_data.iloc[:,2]).reshape(N_slice_x, N_slice_y, N_slice_z)
oef = np.array(scaled_raw_data.iloc[:,3]).reshape(N_slice_x, N_slice_y, N_slice_z)

#Each feature corresponding to the nodes of patch
CBF_g = [scaled_data.iloc[int(node),0] for node in nodes]
CMRO2_g = [scaled_data.iloc[int(node),1] for node in nodes]
CVR_g = [scaled_data.iloc[int(node),2] for node in nodes]
OEF_g = [scaled_data.iloc[int(node),3] for node in nodes]
GM_g = [scaled_data.iloc[int(node),4] for node in nodes]

#graph data
degrees = [d for n,d in g1.degree]
betweenness = [b for n,b in nx.betweenness_centrality(g1,weight='weight').items()]



#sizing nodes of patch based on their properties
sizes_degree = [3 * d for d in degrees]
sizes_betweenness = [100 * b for b in betweenness]  # scale if values are small

voxel_coords = scaled_data.iloc[nodes, [5,6,7]].values  # X,Y,Z in voxel

# Step 1: Extract the surface mesh using marching cubes
#For implimentation on documentation on:https://scikit-image.org/docs/stable/auto_examples/edges/plot_marching_cubes.html
verts, faces, normals, values = measure.marching_cubes(mask_gm)

#embed mask_gm in verts space which in mni space. 
GM_vertex_values = map_coordinates(mask_gm, verts.T, order=3, mode='nearest') #order will give more precise value in mapping to the nes space than order=1,2
CBF_vertex_values = map_coordinates(cbf, verts.T, order=3, mode='nearest')
CMRO2_vertex_values = map_coordinates(cmro2, verts.T, order=3, mode='nearest')
CVR_vertex_values = map_coordinates(cvr, verts.T, order=3, mode='nearest')
OEF_vertex_values = map_coordinates(oef, verts.T, order=3, mode='nearest')


patch_x, patch_y, patch_z = voxel_coords.T#voxel_coordsis in shape(20*3) but we want 3*20
# Step 2: Prepare data for Plotly mesh
x, y, z = verts.T##verts in shape(N*3) but we want 3*N
i, j, k = faces.T  # triangles

text_labels_g = [f"CBF: {cbf:.2f}<br>CMRO2: {cmro2:.2f}<br>CVR: {cvr:.2f}<br>OEF: {oef:.2f}<br>GM: {gm:.2f}<br>Degree:{degrees}<br>Betweenness:{betweenness}"
                for  cbf, cmro2, cvr, oef, gm, degrees,betweenness in zip( CBF_g, CMRO2_g,CVR_g, OEF_g, GM_g, degrees,betweenness)]
# === Step 2: Create patch node scatter ===
# Assume voxel_coords are the 3D points of  patch
# If in MNI space, they need to be in the same voxel space as msk_data
histogram_trace = go.Histogram(x=degrees,
                               nbinsx=15,marker_color='light green',name='Degree Distribution',showlegend=False,
                              )

patch_nodes = go.Scatter3d(x=patch_x, y=patch_y, z=patch_z,
                           mode='markers', name='Nodes', text=text_labels_g, hoverinfo='text',
                           marker=dict(size=[3 * d for d in degrees],color=CBF_g,colorscale='Viridis',showscale=True,
                                       colorbar=dict(title='Node color',thickness=10,x=-0.03,xanchor='left',y=0.5,yanchor='middle')
                                      )
                           )

# Downsample the 3D mask before extracting surface (e.g., 2x smaller)
# msk_downsampled = zoom(mask_gm, zoom=0.8, order=0)
# === Step 3: Combine and plot ===
gray_mesh = go.Mesh3d(x=x, y=y, z=z,i=i, j=j, k=k,
                      intensity = GM_vertex_values,color='lightgray',opacity=0.2,colorscale='gray',name='Gray Matter',showscale=True,visible=True,
                      colorbar=dict(title='Gray Matter',thickness=10,x=0.05,xanchor='left',y=0.5,yanchor='middle')
                      )

# colored mesh
cbf_mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, 
                    intensity=CBF_vertex_values, colorscale='hot',opacity=0.08, name='CBF', showscale=True, visible=False,
                    colorbar=dict(title='CBF', thickness=10, x=0.05, xanchor='left', y=0.5,yanchor='middle')  # move colorbar farther right 
                    )


cvr_mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                     intensity=CVR_vertex_values, colorscale='hot', opacity=0.08, name='CVR',showscale=True, visible=False,
                     colorbar=dict(title='CVR', thickness=10, x=0.05, xanchor='left', y=0.5, yanchor='middle')
                    )

oef_mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                     intensity=CVR_vertex_values, colorscale='hot', opacity=0.08, name='OEF',showscale=True, visible=False,
                     colorbar=dict(title='CVR', thickness=10, x=0.05, xanchor='left', y=0.5, yanchor='middle')
    
                    )

cmro2_mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k,
                     intensity=CVR_vertex_values, colorscale='hot', opacity=0.08, name='CMRO2',showscale=True, visible=False,
                     colorbar=dict(title='CMRO2', thickness=10, x=0.05, xanchor='left', y=0.5, yanchor='middle')
                      )

# Add placeholder for animated slice (trace index 7)
sliced_mesh_placeholder = go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[],
                                    name='Animated Slice', opacity=0.2, color='lightgray',showscale=True, visible=True,
                                    colorbar=dict(title='Gray Matter', thickness=10, x=0.05, xanchor='left', y=0.5, yanchor='middle'),
                                    )
# Create subplot layout
fig = make_subplots(rows=1, cols=2,column_widths=[0.85, 0.15],
                    specs=[[{'type': 'scene'}, {'type': 'xy'}]],horizontal_spacing=0.01, subplot_titles=('',  'Degree Distribution'))

# Add static traces (mesh types and nodes)
fig.add_trace(gray_mesh, row=1, col=1)
fig.add_trace(cbf_mesh, row=1, col=1)
fig.add_trace(cvr_mesh, row=1, col=1)
fig.add_trace(oef_mesh, row=1, col=1)
fig.add_trace(cmro2_mesh, row=1, col=1)
fig.add_trace(patch_nodes, row=1, col=1)
fig.add_trace(sliced_mesh_placeholder, row=1, col=1)
fig.add_trace(histogram_trace, row=1, col=2)


# Add frames to the figure
frames = []
z_thresholds = np.arange(10, 80, 2)
for z_thresh in z_thresholds:
    mask = verts[:, 2] > z_thresh
    if mask.sum() == 0:
        continue

    old_to_new = -1 * np.ones(len(verts), dtype=int)
    old_to_new[mask] = np.arange(np.sum(mask))
    valid_faces = np.all(mask[faces], axis=1)
    new_faces = old_to_new[faces[valid_faces]]
    new_verts = verts[mask]

    if len(new_faces) == 0:
        continue

    x_, y_, z_ = new_verts.T
    i_, j_, k_ = new_faces.T

    frame = go.Frame(
        data=[go.Mesh3d(
            x=x_, y=y_, z=z_,
            i=i_, j=j_, k=k_,
            opacity=0.3,
            color='skyblue',
            showscale=False,
            name='Sliced Mesh'
        )],
        name=str(z_thresh)
    )

    frames.append(frame)

fig.frames = frames


# Create slider steps
slider_steps = [dict(method='animate',
                     args=[[str(z_thresh)], dict(mode='immediate',frame=dict(duration=100, redraw=True), transition=dict(duration=0))],
                     label=f'{z_thresh}') for z_thresh in z_thresholds
               ]

sliders = [dict(steps=slider_steps,transition=dict(duration=0),x=0.1,xanchor="left", y=-0.2, yanchor="top",
          currentvalue=dict(font=dict(size=14), prefix="Z slices: ", visible=True), len=0.9)]

# Add dropdowns and play/pause controls
fig.update_layout(
    sliders=sliders,

    updatemenus=[
        # Play/Pause buttons
        dict(type='buttons',showactive=False, y=-0.4, x=0,xanchor='left', yanchor='top', 
              buttons=[
                dict(label='Play', method='animate',args=[None, dict(frame=dict(duration=100, redraw=True), transition=dict(duration=0), fromcurrent=True, mode='immediate')]),
                dict(label='Pause', method='animate', args=[[None],  dict(frame=dict(duration=0, redraw=False), mode='immediate')])
                      ]),

        # Mesh toggle
        dict(active=1, direction='down', showactive=True,  x=0.5, xanchor='left', y=1.15, yanchor='top',
            buttons=[
                dict(label='Choose Mesh Type', method='skip'),
                dict(label='Gray Matter', method='update',args=[{'visible': [True, False, False, False, False, True, True, True]}]),
                dict(label='CBF', method='update',args=[{'visible': [False, True, False, False, False, True, True, True]}]),
                dict(label='CVR', method='update',args=[{'visible': [False, False, True, False, False, True, True, True]}]),
                dict(label='OEF', method='update',args=[{'visible': [False, False, False, True, False, True, True, True]}]),
                dict(label='CMRO2', method='update',args=[{'visible': [False, False, False, False, True, True, True, True]}])
                     ]),
            
        # Node size
        dict(active=1, direction='down',showactive=True,x=0.6,xanchor='left',y=1.15,yanchor='top',
            buttons=[
                dict(label='Node Size', method='skip'),
                dict(label='Degree',method='restyle',args=[{'marker.size': [sizes_degree]}, [5]]),
                dict(label='Betweenness',method='restyle',args=[{'marker.size': [sizes_betweenness]}, [5]])
                     ]),

        # Node color
        dict(active=1, direction='down',showactive=True,x=0.7,xanchor='left',y=1.15,yanchor='top',
            buttons=[
                dict(label='Node color', method='skip'),
                dict(label='CBF',method='restyle',args=[{'marker.color': [CBF_g],'marker.colorbar.title.text': ['CBF']}, [5]]),
                dict(label='CVR',method='restyle',args=[{'marker.color': [CVR_g],'marker.colorbar.title.text': ['CVR']}, [5]]),
                dict(label='OEF',method='restyle',args=[{'marker.color': [OEF_g],'marker.colorbar.title.text': ['OEF']}, [5]]),
                dict(label='CMRO2',method='restyle',args=[{'marker.color': [CMRO2_g],'marker.colorbar.title.text': ['CMRO2']}, [5]])
                     ]),
    ],
    
    scene=dict(xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False)),
    title='Gray Matter Mesh with Optional Physiological Coloring + Slicing',
    margin=dict(l=0, r=0, b=0, t=50),
    bargap=0.2,#bandgap for bins in histogram
    plot_bgcolor='white' 
)
fig.update_xaxes(title_text='Degree',showline=True,tickmode='linear',tick0=0,dtick=1, linecolor='black', showgrid=False, row=1, col=2)
fig.update_yaxes(title_text='Node Count',showline=True,tickmode='linear',tick0=0,dtick=1,linecolor='black', showgrid=False, row=1, col=2)


# fig.show()  # Call this when you run the script locally



# Initialize Dash app
app = Dash(__name__)

# Define layout to render your full Plotly figure
app.layout = html.Div([
    html.H2("Brain Graph Mesh Viewer"),
    dcc.Graph(id='main-figure', figure=fig, style={'height': '90vh', 'width': '100%'})],
    style={'margin': '0px', 'padding': '0px'})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)