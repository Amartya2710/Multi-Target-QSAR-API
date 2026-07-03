# Use a slim, lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the exact dependencies you pinned
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your .pkl models and the main.py script into the container
COPY . .

# Expose the port FastAPI uses
EXPOSE 8000

# The command to boot up the Uvicorn server securely
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
