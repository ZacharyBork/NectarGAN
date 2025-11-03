#!/bin/sh

# The primary purpose of this file is to initialize some NectarGAN config file values to work inside of
# the containerized environment, and also to force the correct permissions on some of the files and 
# directories which are copied/mounted in the container after everything else is built.

set -e
: "${TORCH_TYPE:=cpu}"

# Get config file path, store as variable
CONFIG_FILE="/app/mount/docker_nectargan_config.json"
chmod +x $CONFIG_FILE

# Update IO pathing in the train config file
jq '.config.common.output_directory = "/app/mount/output"' \
    "$CONFIG_FILE" > config.tmp && mv config.tmp "$CONFIG_FILE"

# Ensure correct PyTorch device based on build arg
if [ "$TORCH_TYPE" = "cpu" ]; then
    jq '.config.common.device = "cpu"' \
        "$CONFIG_FILE" > config.tmp && mv config.tmp "$CONFIG_FILE"
else
    jq '.config.common.device = "cuda"' \
        "$CONFIG_FILE" > config.tmp && mv config.tmp "$CONFIG_FILE"
fi

# Enable Visdom and route through container's exposed port
jq '.config.visualizer.visdom.enable = true | .config.visualizer.visdom.server = "http://visdom" | .config.visualizer.visdom.port = 8000' \
    "$CONFIG_FILE" > config.tmp && mv config.tmp "$CONFIG_FILE"

# Ensure correct permissions on file IO directories
find /app/mount/output -type f -exec chmod 644 {} +
find /app/mount/input -type d -exec chmod 755 {} +

# Run build CMD
exec "$@"
