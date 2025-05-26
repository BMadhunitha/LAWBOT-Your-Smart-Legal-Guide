FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt

COPY . .

ENV PATH="/opt/venv/bin:$PATH"

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
