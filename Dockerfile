# Starts from the python 3.10 official docker image
FROM python:3.10-slim

# Create a folder "app" at the root of the image
RUN mkdir /app

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . /app

# Update pip
RUN pip install --upgrade pip

# Install dependencies from "requirements.txt"
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Expose port (FastAPI runs on 8000)
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1
