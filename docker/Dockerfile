ARG TORCH_TYPE="cpu"

FROM python:3.12-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get upgrade -y \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements*.txt .
RUN sed -i 's/^opencv-python.*/opencv-python-headless/' requirements.txt \
 && python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r requirements.dev.txt \
 && rm requirements.*

FROM builder AS cpu
RUN pip install --no-cache-dir torch torchvision

FROM builder AS cuda126
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126

FROM builder AS cuda128
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

FROM ${TORCH_TYPE} AS runtime

COPY nectargan/ nectargan/
COPY scripts/ scripts/
COPY tests/ tests/

RUN find /app -type f -exec chmod 644 {} + \
 && find /app -type d -exec chmod 755 {} +

EXPOSE 8000
