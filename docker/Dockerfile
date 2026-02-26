FROM python:3.10-slim

# Install system dependencies
# tesseract-ocr: for text recognition in images
# poppler-utils: for pdfplumber/pdf2image to render PDFs
# libgl1-mesa-glx: for opencv if needed
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    tesseract-ocr-kaz \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data for evaluation
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy project files
COPY . .

# Set PYTHONPATH so src imports work
ENV PYTHONPATH=/app

# Expose ports: 8888 for Jupyter, 8000 for FastAPI
EXPOSE 8888 8000

# Default command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
