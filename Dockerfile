# Reordered with help from Claude Opus 5.5 on Medium thinking when prompted why the container builds were taking so long.

FROM python:3.10-slim

WORKDIR /opt/app

# Install packages that we need. vim is for helping with debugging
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get upgrade -yq ca-certificates && \
    apt-get install -yq --no-install-recommends \
    prometheus-node-exporter

COPY requirements.txt .
RUN pip install --no-cache-dir -r /opt/app/requirements.txt

COPY . .

EXPOSE 7860
EXPOSE 8000
EXPOSE 9100
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD bash -c "prometheus-node-exporter --web.listen-address=':9100' & python /opt/app/app.py"