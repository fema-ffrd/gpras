"""Utilities for interacting with AWS S3."""

import io
import os
import re
from io import BytesIO
from urllib.parse import urlparse

import boto3
from boto3.resources.base import ServiceResource
from boto3.session import Session
from botocore.client import BaseClient
from botocore.config import Config


def init_s3_resources() -> tuple[Session, BaseClient, ServiceResource]:
    """Establish a boto3 session and return the session, S3 client, and S3 resource handles with optimized config."""
    boto_config = Config(
        retries={"max_attempts": 3, "mode": "standard"},  # Default is 10
        connect_timeout=3,  # Seconds to wait to establish connection
        read_timeout=10,  # Seconds to wait for a read
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )

    session = boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    s3_client = session.client("s3", config=boto_config)
    s3_resource = session.resource("s3", config=boto_config)

    return session, s3_client, s3_resource


def list_keys_regex(
    s3_client: boto3.Session.client, bucket: str, prefix_includes: str, suffix: str = "", recursive: bool = True
) -> list[str]:
    """List all keys in an S3 bucket matching a given prefix pattern and suffix."""
    keys = []
    prefix = prefix_includes.split("*")[0]  # Use the static part of the prefix for listing
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    if not recursive:
        kwargs["Delimiter"] = "/"

    prefix_pattern = re.compile(prefix_includes.replace("*", ".*"))

    while True:
        resp = s3_client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if prefix_pattern.match(key) and key.endswith(suffix):
                keys.append(key)
        if not resp.get("IsTruncated"):
            break
        kwargs["ContinuationToken"] = resp["NextContinuationToken"]

    return keys


def bytes_2_s3(
    data: io.BytesIO,
    s3_path: str,
    content_type: str = "",
) -> None:
    """Upload BytesIO to S3."""
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    _, s3_client, _ = init_s3_resources()
    s3_client.put_object(Bucket=bucket, Key=key, Body=data.getvalue(), ContentType=content_type)


def s3_2_bytes(s3_path: str) -> BytesIO:
    """Load data from an S3 URI to a BytesIO object."""
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    _, s3_client, _ = init_s3_resources()
    bytes_io = BytesIO()
    s3_client.download_fileobj(Bucket=bucket, Key=key, Fileobj=bytes_io)
    bytes_io.seek(0)
    return bytes_io


def s3_2_file(s3_path: str, local_file: str) -> str:
    """Load data from an S3 URI to a BytesIO object."""
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    _, s3_client, _ = init_s3_resources()
    s3_client.download_file(Bucket=bucket, Key=key, Filename=local_file)
    return local_file
