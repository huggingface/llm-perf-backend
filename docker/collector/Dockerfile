FROM python:3.12-slim

WORKDIR /workspace

# Install git
RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY setup.py .
COPY pyproject.toml .

RUN pip install -e .

COPY . .

CMD ["llm-perf", "update-leaderboard"]
