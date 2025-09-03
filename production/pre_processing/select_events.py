"""Filter storm catalog.

Given a set of storm transpositions, calibration events, and design events, find an optimal set for training, testing,
and validation.
"""


def filter_events(dss_dir: str, out_path: str, basin_path: str, number_of_events: int) -> dict[str, str]:
    """Scan for .dss files, filter events, and export results.

    Scans the input directory (local path or AWS S3 prefix) for `.dss` files, applies a filtering
    process to identify the desired number of events spanning a range of conditions, and writes the filtered
    results to the output directory.

    Args:
        dss_dir (str): Path or S3 prefix to the input directory containing `.dss` files.
        out_path (str): Path or S3 uri to save the summary STAC item.
        basin_path (str): Path to a geojson of the basin geometry.
        number_of_events (int): Number of events to retain in the output dir.

    Returns:
        dict: A STAC item with .dss file assets.

    """
    # TODO: pre-processing
    # TODO: event selection
    # Let's use STAC (https://stacspec.org/en) to track this metadata. One item with an asset for each .dss file. Each
    # asset has attributes for metrics as well as boolean indicating whether or not it's in the train+test+validate
    # data.
    # See https://pystac.readthedocs.io/en/stable/
    return {"id": "filtered_events"}
