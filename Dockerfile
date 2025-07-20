FROM python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openjdk-11-jdk \
    build-essential \
    cython \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Default start command for Streamlit on Render
CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.enableCORS=false"]
