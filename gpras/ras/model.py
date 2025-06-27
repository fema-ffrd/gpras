"""HEC-RAS model class and associated functions."""

import shutil
from typing import Any

from hecstac.ras.item import RASModelItem
from pystac import Asset

from gpras.ras.plan import (
    BoundaryType,
    GenAsset,
    add_plan_to_text_file,
    update_hdf_attributes,
    update_hdf_data,
    update_text_attributes,
)
from gpras.utils.file_utils import get_filename


class RasModel(RASModelItem):  # type: ignore[misc]
    """Subclass of hecstac with enhanced read/write/execute."""

    def make_new_plan(
        self, template_run: str, plan_attrs: dict[str, Any], boundary_conditions: list[BoundaryType]
    ) -> tuple[str, str]:
        """Append a new plan to the model."""
        # Define paths
        src_path: str = self.assets[template_run].href
        new_run = self.increment_suffix(self.plan_assets)
        dst_path: str = src_path.replace(template_run, new_run + ".hdf")
        src_txt_path: str = src_path.replace(".hdf", "")
        dst_txt_path: str = dst_path.replace(".hdf", "")

        # Copy file
        shutil.copy(src_path, dst_path)
        shutil.copy(src_txt_path, dst_txt_path)

        # Update metadata
        update_hdf_attributes(dst_path, "Plan Data/Plan Information", plan_attrs["hdf"])
        update_text_attributes(dst_txt_path, plan_attrs["txt"])

        # Update project file
        add_plan_to_text_file(self.pf.fpath, dst_txt_path.split(".")[-1])

        # Write data
        for i in boundary_conditions:
            update_hdf_data(dst_path, i.path, i.data)

        # Add assets
        self.add_asset(get_filename(dst_path), Asset(dst_path, get_filename(dst_path)))
        self.add_asset(get_filename(dst_txt_path), Asset(dst_txt_path, get_filename(dst_txt_path)))
        self.plan_assets.append(self.assets[get_filename(dst_txt_path)])
        return dst_path, dst_txt_path

    def increment_suffix(self, paths: list[GenAsset]) -> str:
        """Take a list of paths, find the highest-numbered, and return that file incremented by 1."""
        suffixes = {int(get_filename(i.href).split(".")[-1].lstrip("p")): get_filename(i.href) for i in paths}
        max_plan_ind = max(suffixes.keys())
        new_plan_ind = max_plan_ind + 1
        max_suffix = f"p{str(max_plan_ind).zfill(2)}"
        new_suffix = f"p{str(new_plan_ind).zfill(2)}"
        return suffixes[max_plan_ind].replace(max_suffix, new_suffix)
