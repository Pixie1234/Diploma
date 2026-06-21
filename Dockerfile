FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (minimal; keep small for Render)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 10000

# Streamlit needs to bind to 0.0.0.0 and the port Render assigns.
CMD ["sh", "-c", "streamlit run app2.py --server.address 0.0.0.0 --server.port $PORT"]
