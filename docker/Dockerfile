FROM python:3.10

WORKDIR /app

# Install dependencies
COPY ../requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY .. .

# Default command (ignored in compose)
CMD ["bash"]