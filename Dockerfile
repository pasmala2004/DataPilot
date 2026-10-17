# DataPilot app image. The model is reached through OpenRouter (OPENROUTER_API_KEY, passed in at run time, never baked in).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/mpl \
    DATA_DIR=/app/data \
    OUTPUT_DIR=/app/outputs

WORKDIR /app

COPY requirements.txt .
# pytest/psutil are test-only; the image does not need them.
RUN grep -v -E '^(pytest|psutil)' requirements.txt > /tmp/req.txt \
    && pip install -r /tmp/req.txt

COPY agent ./agent
COPY app.py core.py excel_server.py run.py ./
COPY .streamlit ./.streamlit

# Non-root user; data and outputs are volumes owned by it.
RUN useradd --create-home --uid 10001 datapilot \
    && mkdir -p /app/data /app/outputs \
    && chown -R datapilot:datapilot /app
USER datapilot

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
