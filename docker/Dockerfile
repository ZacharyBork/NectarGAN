FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

FROM base AS pytorch
ARG TORCH_TYPE

RUN if [ "$TORCH_TYPE" != "cpu" ] \
    && [ "$TORCH_TYPE" != "cu126" ] \
    && [ "$TORCH_TYPE" != "cu128" ]; \
        then echo "Invalid TORCH_TYPE:$TORCH_TYPE" \
          && echo "Valid types are (cpu|cu126|cu128)" \
          && exit 1; \
    fi

RUN pip install --no-compile torch torchvision \
        --index-url "https://download.pytorch.org/whl/${TORCH_TYPE}" \
 && find /opt/venv -type f -name "*.so*" -exec strip --strip-unneeded {} + 2>/dev/null || true \
 && find /opt/venv -type d \( -name tests -o -name test -o -name __pycache__ \) -prune -exec rm -rf {} + \
 && find /opt/venv -type f -name "*.pyc" -delete

FROM pytorch AS build

WORKDIR /app
COPY pyproject.toml .
COPY nectargan/ nectargan/

RUN sed -i -E 's/"opencv-python([^"]*)"/"opencv-python-headless"/g' pyproject.toml \
 && sed -i '/PySide6==/d' pyproject.toml \
 && pip install --upgrade pip && pip install --no-compile .

FROM python:3.12-slim-bookworm AS runtime
ARG TORCH_TYPE

ENV TORCH_TYPE=$TORCH_TYPE \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=build /opt/venv /opt/venv
COPY --from=build /app /app
COPY docker/entrypoint.sh /app/entrypoint.sh
WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    jq \
    libgl1 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libjpeg62-turbo \
    libpng16-16 \
 && rm -rf /var/lib/apt/lists/* \
 && find /app -type f -exec chmod 644 {} + \
 && find /app -type d -exec chmod 755 {} + \
 && printf '%s\n' '#!/bin/sh' \
           'exec /opt/venv/bin/python -u /app/scripts/nectargan.py "$@"' \
  > /usr/local/bin/nectargan && chmod +x /usr/local/bin/nectargan \
 && chmod +x /app/entrypoint.sh && sed -i 's/\r$//' /app/entrypoint.sh
 
ENTRYPOINT ["/app/entrypoint.sh"]
