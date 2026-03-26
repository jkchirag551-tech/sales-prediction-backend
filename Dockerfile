# 1. Use an official, lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements file first (for caching efficiency)
COPY requirements.txt .

# 4. Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your entire project into the container
COPY . .

# 6. Expose the port FastAPI runs on
EXPOSE 8000

# 7. Command to start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
