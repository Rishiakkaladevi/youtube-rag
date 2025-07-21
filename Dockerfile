# Use a stable Python image
FROM python:3.10-slim

# Install system packages
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    curl \
    git \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy app source code
COPY . .

# Expose the Streamlit port (Render uses 10000)
EXPOSE 10000

# Streamlit config
ENV STREAMLIT_SERVER_PORT=10000
ENV STREAMLIT_SERVER_ENABLECORS=false

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
