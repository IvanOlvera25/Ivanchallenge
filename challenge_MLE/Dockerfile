# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY challenge/ ./challenge/
COPY data/ ./data/
COPY tests/ ./tests/

# Expose port
EXPOSE 8080

# Set environment variables for production
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]