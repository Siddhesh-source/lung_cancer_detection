# Step 1: Use a clean, official lightweight Python environment
FROM python:3.10-slim

# Step 2: Install basic system tools needed for image processing (OpenCV dependencies)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Set up the working directory inside the container
WORKDIR /app

# Step 4: Copy and install requirements first (takes advantage of Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy only the necessary operational code and model files
COPY config.yaml .
COPY /models ./models
COPY /src ./src

# Step 6: Open the port our system will talk to
EXPOSE 8000

# Step 7: Run the FastAPI web server on startup
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
