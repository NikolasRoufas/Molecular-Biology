# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libffi-dev \
        texlive-latex-base \
        texlive-fonts-recommended \
        texlive-latex-extra \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install R for R-based pipelines
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        r-base \
        r-base-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Create necessary directories
RUN mkdir -p logs results configs

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run the application when the container launches
CMD ["streamlit", "run", "app.py"]

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1