#!/bin/sh
set -e
: "${HOST_NAME:=127.0.0.1}"
: "${EXPOSE_PORT:=8000}"
exec python -m visdom.server --hostname "$HOST_NAME" -port "$EXPOSE_PORT"
