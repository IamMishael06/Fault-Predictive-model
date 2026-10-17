FROM python:3.13-slim

WORKDIR /app/api

# Required for lightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY api/requirement.txt ./requirement.txt

RUN pip install --no-cache-dir -r requirement.txt

COPY api/ /app/api/

COPY models/champion/ /app/models/champion/

COPY cleaned_csv/synthetic_oil_stream_1s_30_predictions.csv /app/cleaned_csv/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

