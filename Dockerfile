FROM python:3.13-slim

WORKDIR /app

# Install system deps for audio processing libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create dirs that the app expects
RUN mkdir -p logs transcripts

EXPOSE 8080

CMD ["python", "test_browser.py"]
