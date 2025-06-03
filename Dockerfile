# Use a base image with Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Expose the port your app runs on
EXPOSE 5000

# Run the app
CMD ["python", "run_all.py"]

