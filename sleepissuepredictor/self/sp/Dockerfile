FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the app.py file and requirements.txt into the container
COPY requirements.txt ./
COPY app.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all remaining files 
COPY . . 

# Expose port 5000
EXPOSE 8000

# Run the Flask app
CMD ["python", "app.py"]