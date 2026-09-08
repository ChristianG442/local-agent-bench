FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml ./pyproject.toml
COPY src ./src
COPY profiles ./profiles
COPY models.toml ./models.toml
RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["lab"]