"""HEC-RAS model class and associated functions."""

import shutil
from typing import Any, TypeVar

from hecstac.ras.assets import GenericAsset
from hecstac.ras.item import RASModelItem
from pystac import Asset

from gpras.ras.plan import (
    BoundaryType,
    update_hdf_attributes,
    update_hdf_data,
)
from gpras.utils.file_utils import detect_file_properties, get_filename

GenAsset = TypeVar("GenAsset", bound=GenericAsset)


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
        add_file_to_prj_file(self.pf.fpath, dst_txt_path.split(".")[-1])

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


def add_file_to_prj_file(prj_path: str, file_row: str) -> None:
    """Add a plan suffix to a project file."""
    encoding, newline = detect_file_properties(prj_path)
    with open(prj_path, encoding=encoding) as f:
        lines = f.readlines()
    key = file_row.split("=")[0]
    last_index = max((ind for ind, line in enumerate(lines) if line.startswith(key)), default=len(lines))
    lines.insert(last_index + 1, file_row + "\n")
    with open(prj_path, mode="w", encoding=encoding, newline=newline) as f:
        f.writelines(lines)


def update_text_attributes(txt_path: str, attrs: dict[str, str]) -> None:
    """Update an attribute in a ras text file."""
    encoding, newline = detect_file_properties(txt_path)
    with open(txt_path, encoding=encoding) as f:
        lines = f.readlines()
    for ind, i in enumerate(lines):
        splitted = i.split("=")
        key = "=".join(splitted[:-1])
        if key in attrs:
            splitted[-1] = attrs[key] + "\n"
            lines[ind] = "=".join(splitted)
    with open(txt_path, mode="w", encoding=encoding, newline=newline) as f:
        f.writelines(lines)
