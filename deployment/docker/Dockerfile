# zkML System Docker Image
# 
# Multi-stage build für minimale Image-Größe
# 
# Build:   docker build -t zkml-system .
# Run API: docker run -p 8000:8000 zkml-system serve
# Run CLI: docker run zkml-system --help

FROM python:3.11-slim as builder

# Build-Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Arbeitsverzeichnis
WORKDIR /app

# Requirements installieren
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Produktions-Image
FROM python:3.11-slim as production

# Metadata
LABEL maintainer="Manus AI <contact@manus.ai>"
LABEL version="0.1.0"
LABEL description="Zero-Knowledge Machine Learning Proof System"

# Non-root User für Sicherheit
RUN useradd --create-home --shell /bin/bash zkml
USER zkml

# Python-Packages vom Builder kopieren
COPY --from=builder /root/.local /home/zkml/.local
ENV PATH=/home/zkml/.local/bin:$PATH

# Arbeitsverzeichnis
WORKDIR /home/zkml/app

# Anwendungscode kopieren
COPY --chown=zkml:zkml . .

# Python-Path setzen
ENV PYTHONPATH=/home/zkml/app
ENV PYTHONUNBUFFERED=1

# Health-Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Port für API
EXPOSE 8000

# Default: CLI
ENTRYPOINT ["python", "-m", "deployment.cli.main"]
CMD ["--help"]
