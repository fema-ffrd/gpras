"""Downloading SST event data from S3-hosted HMS DSS files."""

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress

import boto3
import pandas as pd
from dotenv import load_dotenv
from hecdss import HecDss
from tqdm import tqdm


def get_sst_events(
    event_id: int,
    bucket: str,
    base_prefix: str,
    target_site: str,
    upstream_site: str,
    s3_client: boto3.client,
) -> pd.DataFrame | None:
    """
    Download and extract time series data from SST DSS file for a given event.

    Extracts:
        - 'PRECIP-CUM' and 'PRECIP-EXCESS' for the target site
        - 'FLOW' for the upstream site

    Args:
    ----
        event_id: Unique identifier for the storm event.
        bucket: S3 bucket name.
        base_prefix: Path prefix to event folders in the S3 bucket.
        target_site: Site name for precipitation variables.
        upstream_site: Site name for inflow (FLOW) variable.
        s3_client: Boto3 S3 client instance.

    Returns:
    -------
        A DataFrame with columns ['precip-cum', 'precip-excess', 'inflow', 'event_id']
        indexed by datetime, or None if the file or paths are missing.

    """
    s3_key = f"{base_prefix}/{event_id}/hydrology/SST.dss"
    fd, temp_path = tempfile.mkstemp(suffix=".dss")
    os.close(fd)
    dss = None

    try:
        s3_client.download_file(bucket, s3_key, temp_path)
        dss = HecDss(temp_path)
        paths = dss.get_catalog().uncondensed_paths

        variables = {
            "precip-cum": (target_site, "PRECIP-CUM"),
            "precip-excess": (target_site, "PRECIP-EXCESS"),
            "inflow": (upstream_site, "FLOW"),
        }

        dfs = []
        for label, (site, part_c) in variables.items():
            match = next(
                (
                    p
                    for p in paths
                    if site.lower() in p.lower() and part_c.lower() in p.lower()
                ),
                None,
            )
            if not match:
                continue

            record = dss.get(match)
            df = pd.DataFrame(
                {label: record.values}, index=pd.to_datetime(record.times)
            )
            dfs.append(df)

        if not dfs:
            return None

        df_combined = pd.concat(dfs, axis=1)
        df_combined["event_id"] = event_id
        return df_combined

    except Exception as e:
        print(f"Error with event {event_id}: {e}")
        return None

    finally:
        if dss:
            with suppress(Exception):
                dss.close()
        if os.path.exists(temp_path):
            with suppress(Exception):
                os.remove(temp_path)


if __name__ == "__main__":
    load_dotenv()
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    event_ids = list(range(1, 10))
    target_site = "west-fork_s330"
    upstream_site = "west-fork_s340"
    output_file = "./data_dir/precip_data/west-fork_s330_hms.pq"
    bucket = "trinity-pilot"
    base_prefix = "conformance/simulations/event-data"
    max_workers = 25

    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    results: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                get_sst_events, eid, bucket, base_prefix, target_site, upstream_site, s3
            ): eid
            for eid in event_ids
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing events"
        ):
            df = future.result()
            if df is not None:
                results.append(df)

    if results:
        combined_df = (
            pd.concat(results, axis=0)
            .reset_index()
            .rename(columns={"index": "datetime"})
        )
        combined_df = combined_df[
            ["event_id", "datetime", "inflow", "precip-cum", "precip-excess"]
        ]
        combined_df = combined_df.sort_values(["event_id", "datetime"])
        combined_df.to_parquet(output_file)
        print(f"Saved output to {output_file}")
    else:
        print("No valid data to save.")
