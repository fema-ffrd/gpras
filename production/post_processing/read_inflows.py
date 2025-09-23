"""Reads inflows from hdf files."""

# %% import modules
import os

import h5py
import numpy as np
import pandas as pd

# %% write paths
folder_hdf_lf = r"/workspaces/gpras/data/hdfs/hdf_lf"
path_hdf_data = r"/workspaces/gpras/production/post_processing/data/hdf_lf_data"
file_hdf_p36 = "bridgeport.p36.hdf"
path_hdf_p36 = os.path.join(folder_hdf_lf, file_hdf_p36)

loc_hdf_cell_area = "/Geometry/2D Flow Areas/bridgeport_1/Cells Surface Area"
loc_hdf_cell_coordinates = "/Geometry/2D Flow Areas/bridgeport_1/Cells Center Coordinate"
loc_hdf_aoi_inflow = (
    "/Event Conditions/Unsteady/Boundary Conditions/Flow Hydrographs/2D: bridgeport_1 BCLine: bc_west-fork_s340_base"
)

# %% define parameters
list_event_ids = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]


# %% make folders for data, if they don't already exist
for i in list_event_ids:
    os.makedirs(os.path.join(path_hdf_data, f"p{i}"), exist_ok=True)


# %% read and save cell araa df

for i in list_event_ids:
    with h5py.File(os.path.join(folder_hdf_lf, f"bridgeport.p{i}.hdf")) as f:
        arr_cell_area = f[loc_hdf_cell_area][()]
        pd_cell_area = pd.DataFrame(
            index=np.arange(len(arr_cell_area)),
            data={"hdf_cell_id": np.arange(len(arr_cell_area)), "cell_area_ft2": arr_cell_area},
        )
        pd_cell_area.index.name = "idx_hdf_cell_id"
        pd_cell_area.to_csv(os.path.join(path_hdf_data, f"p{i}", f"p{i}_cell_area.csv"))

# %% read and save cell center coordinate

for i in list_event_ids:
    with h5py.File(os.path.join(folder_hdf_lf, f"bridgeport.p{i}.hdf")) as f:
        arr_cell_coords = f[loc_hdf_cell_coordinates][()]
        pd_cell_coords = pd.DataFrame(
            index=np.arange(len(arr_cell_coords)),
            data={
                "hdf_cell_id": np.arange(len(arr_cell_coords)),
                "cell_coord_x": arr_cell_coords[:, 0],
                "cell_coord_y": arr_cell_coords[:, 1],
            },
        )
        pd_cell_coords.index.name = "idx_hdf_cell_id"
        pd_cell_coords.to_csv(os.path.join(path_hdf_data, f"p{i}", f"p{i}_cell_coords.csv"))

# %% read and save cell inflows west-fork_s340
for i in list_event_ids:
    with h5py.File(os.path.join(folder_hdf_lf, f"bridgeport.p{i}.hdf")) as f:
        arr_inflows = f[loc_hdf_aoi_inflow][()]
        pd_inflows = pd.DataFrame(
            index=np.arange(len(arr_inflows)),
            data={
                "hdf_cell_id": np.arange(len(arr_inflows)),
                "time_hr": arr_inflows[:, 0],
                "flow_cfs": arr_inflows[:, 1],
            },
        )
        pd_inflows.index.name = "idx_hdf_cell_id"
        pd_inflows.to_csv(os.path.join(path_hdf_data, f"p{i}", f"p{i}_aoi_inflow.csv"))


# %% plot inflow data and compare with given values
