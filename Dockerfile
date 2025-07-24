# Use a lightweight Python image as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code, trained model, and configuration file into the container
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p logs data/models

# Expose the port the Flask application will run on
EXPOSE 8000

# Command to run the Flask application
CMD ["python", "main.py", "--api", "--port", "8000", "--host", "0.0.0.0"]
