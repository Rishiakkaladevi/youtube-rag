# Base image with Python
FROM python:3.10-slim

# Install system-level dependencies
RUN apt-get update && \
    apt-get install -y \
    openjdk-17-jdk \
    build-essential \
    curl \
    git \
    && apt-get clean

# Set Java environment variables (required for pyjnius)
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install Cython && \
    pip install -r requirements.txt

# Copy app code
COPY . .

# Expose port for Streamlit (Render uses 10000)
EXPOSE 10000

# Streamlit environment settings
ENV STREAMLIT_SERVER_PORT=10000
ENV STREAMLIT_SERVER_ENABLECORS=false

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
