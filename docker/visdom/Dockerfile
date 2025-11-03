FROM python:3.12-alpine AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m ensurepip \
 && python -m pip install --upgrade pip setuptools \
 && python -m pip install visdom 

FROM base AS server

WORKDIR /visdom
ARG HOST_NAME=127.0.0.1 \
    EXPOSE_PORT=8000
ENV HOST_NAME=$HOST_NAME \
    EXPOSE_PORT=$EXPOSE_PORT
EXPOSE $EXPOSE_PORT

COPY visdom/start_server.sh /visdom/start_server.sh
RUN chmod +x /visdom/start_server.sh && sed -i 's/\r$//' /visdom/start_server.sh
ENTRYPOINT ["/visdom/start_server.sh"]
