FROM python:3.6-slim

COPY requirements.txt . 

RUN apt-get update \
    && apt-get install -y --no-install-recommends apt-utils gcc g++\
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove apt-utils gcc g++

COPY . /app

WORKDIR /app

EXPOSE 5000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "-t", "1000", "-w", "1"]
#gunicorn app:app --bind 0.0.0.0:5000 -t 1000 