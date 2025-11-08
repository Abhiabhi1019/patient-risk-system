# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure src/ imports always work
ENV PYTHONPATH="/app"

# Expose API port (for Flask) and dashboard port (Streamlit)
EXPOSE 5000
EXPOSE 8501

# Default command (can be overridden by docker-compose)
CMD ["python", "-m", "src.api.app"]
