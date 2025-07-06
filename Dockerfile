# Starts from the python 3.10 official docker image
FROM python:3.10-slim

# Create a folder "app" at the root of the image
RUN mkdir /app

# Set working directory
WORKDIR /app

# 3. Copy project files into the container
COPY . /app

# Update pip
RUN pip install --upgrade pip

# Install dependencies from "requirements.txt"
RUN pip install -r requirements.txt

# 6. Expose the default FastAPI port
EXPOSE 8000

# 7. Run the API with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
