#!/bin/bash
set -e

# Define the download URL and output directory
URL="https://www.hec.usace.army.mil/software/hec-ras/downloads/Linux_RAS_v66.zip"
OUT_DIR="/opt/hec-ras"

# Make sure dependencies are installed (optional: remove if you're doing this in Dockerfile)
apt-get update && apt-get install -y unzip wget

# Create the output directory
mkdir -p "$OUT_DIR"

# Download and unzip
echo "Downloading HEC-RAS..."
wget -q "$URL" -O /tmp/hec_ras.zip

echo "Unzipping..."
unzip -q /tmp/hec_ras.zip -d "$OUT_DIR"

# Optional: Make binaries executable
chmod -R +x "$OUT_DIR"

# Cleanup
rm /tmp/hec_ras.zip

echo "HEC-RAS installed to $OUT_DIR"
source /opt/hec-ras/Linux_RAS_v66/bin/RasUnsteady /workspaces/gpras/data/Muncie/Muncie.p06.hdf
source /opt/hec-ras/Linux_RAS_v66/bin/RasGeomPreprocess /workspaces/gpras/data/bridgeport/bridgeport.p19.hdf

cp run_unsteady.sh /workspaces/gpras/run_unsteady.sh
