"""Function to create a STAC item for a RAS model."""

from hecstac.ras.item import RASModelItem


def make_stac(ras_prj_path: str) -> None:
    """Make a STAC item for a HEC-RAS model."""
    out_path = ras_prj_path.replace(".prj", ".stac.json")
    stac = RASModelItem.from_prj(ras_prj_path)
    stac.to_file(out_path=out_path)
