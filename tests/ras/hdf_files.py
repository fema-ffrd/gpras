"""Test I/O for Ras .hdf files."""

from gpras.utils.ras import RasModel

TEST_PRJ = "data/Muncie/Muncie.prj"


def test_ras_plan_hdf() -> None:
    """Load a model and iterate through plan files."""
    model = RasModel.from_prj(TEST_PRJ, "model", "EPSG:2965")
    for i in model.plan_hdfs:
        plan = model.plan_hdfs[i]
        geom = model.get_plan_geometry(i)
        assert geom is not None
        meshes = plan._2d_flow_area_names_and_counts()
        for j in meshes:
            summary = plan.wsel_timeseries(j[0])
            assert summary is not None
            min_el = plan.mesh_min_el(j[0])
            assert min_el is not None


def test_ras_geom_hdf() -> None:
    """Load a model and iterate through geometry files."""
    model = RasModel.from_prj(TEST_PRJ, "model", "EPSG:2965")
    for i in model.geom_hdfs:
        geom = model.geom_hdfs[i]
        meshes = geom.mesh_cell_polygons()
        assert meshes is not None


if __name__ == "__main__":
    test_ras_plan_hdf()
    test_ras_geom_hdf()
