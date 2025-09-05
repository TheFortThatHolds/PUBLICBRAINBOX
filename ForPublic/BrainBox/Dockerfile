# BrainBox LOCAL-FIRST Docker Container
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy core working files
COPY enhanced_llm_integrator.py .
COPY unified_brainbox.py .
COPY unified_launcher.py .
COPY semantic_search.py .
COPY llm_router.py .
COPY llm_integrator.py .
COPY llm_config.py .

# Copy memory integration files
COPY memory_voice_integration.py .
COPY memory_spirallogic_integration.py .
COPY test_memory_voices.py .
COPY unicode_sanitizer.py .
COPY MEMORY_INTEGRATION_README.md .

# Create data directory
RUN mkdir -p brainbox_data

# Environment variables for local API
ENV LM_STUDIO_URL=http://host.docker.internal:1234
ENV PYTHONUNBUFFERED=1

# Expose port for web interface (if needed)
EXPOSE 8000

# Default command - run the launcher
CMD ["python", "unified_launcher.py"]