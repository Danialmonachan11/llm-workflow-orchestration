FROM python:3.11-slim

WORKDIR /app

# No apt-get/build-essential layer: chromadb, sentence-transformers, torch,
# and every other dependency here ship prebuilt Linux wheels on PyPI, so no
# compiler is needed. An earlier version added build-essential speculatively
# and never confirmed it was required -- removed once this environment's
# apt network turned out to be unreachable and the real question, "is it
# even necessary," finally got asked and checked instead of assumed.

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
