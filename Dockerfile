FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Install dependencies
COPY pyproject.toml .
RUN uv sync

# Copy source code and model files
COPY . .

EXPOSE 7860

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]