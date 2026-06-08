FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and images
COPY src/ ./src/
COPY images/ ./images/

WORKDIR /app/src

ENTRYPOINT ["python", "main.py"]