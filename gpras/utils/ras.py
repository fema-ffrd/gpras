"""Functions to import and export HEC-RAS .hdf data and execute rasmapper."""

import errno
import os
from functools import cached_property
from typing import cast

import numpy as np
from hecstac.ras.assets import GeometryHdfAsset, PlanHdfAsset
from hecstac.ras.item import RASModelItem
from numpy.typing import NDArray
from rashdf.geom import RasGeomHdf
from rashdf.plan import RasPlanHdf, TimeSeriesOutputVar


class RasPlan(RasPlanHdf):  # type: ignore[misc]
    """OOP representation of a HEC-RAS plan hdf file."""

    @cached_property
    def mesh_names(self) -> list[str]:
        """2D flow area names within the plan."""
        return [i[0] for i in self._2d_flow_area_names_and_counts()]

    def mesh_min_el(self, mesh_id: str) -> NDArray[np.float64]:
        """Get an array of all cell's min elevation."""
        path = f"Geometry/2D Flow Areas/{mesh_id}/Cells Minimum Elevation"
        els = self.get(path)[:]
        return els[~np.isnan(els)]

    def wsel_timeseries(self, mesh_id: str) -> NDArray[np.float64]:
        """Get timeseries of water surface elevation for all cells."""
        ts = self._mesh_timeseries_outputs(mesh_id, TimeSeriesOutputVar)
        return cast(NDArray[np.float64], ts["Water Surface"].values)


class RasModel(RASModelItem):  # type: ignore[misc]
    """OOP representation of a HEC-RAS project."""

    @cached_property
    def plan_hdfs(self) -> dict[str, RasPlan]:
        """List the available plan hdfs."""
        hdfs = [i for i, j in self.assets.items() if isinstance(j, PlanHdfAsset)]
        hdfs = [RasPlan(self.pm.derived_item_asset(i)) for i in hdfs]
        return {i.get_attrs(i.PLAN_INFO_PATH)["Plan Title"]: i for i in hdfs}

    @cached_property
    def geom_hdfs(self) -> dict[str, RasGeomHdf]:
        """List the available plan hdfs."""
        hdfs = [i for i, j in self.assets.items() if isinstance(j, GeometryHdfAsset)]
        hdfs = [RasGeomHdf(self.pm.derived_item_asset(i)) for i in hdfs]
        return {i.get_geom_attrs()["Title"]: i for i in hdfs}

    def get_plan_geometry(self, plan_name: str) -> RasGeomHdf:
        """Return the RasGeomHdf for a RasPlanHdf."""
        plan = self.plan_hdfs[plan_name]
        geom = plan.get_attrs(plan.PLAN_INFO_PATH)["Geometry Title"]
        if geom not in self.geom_hdfs:
            err = errno.ENOENT
            missing_file = plan.get_attrs(plan.PLAN_INFO_PATH)["Geometry Filename"]
            raise FileNotFoundError(err, os.strerror(err), missing_file)
        return self.geom_hdfs[geom]
