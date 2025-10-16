

FROM python:3.11.1-slim

WORKDIR /

# Copy and install requirements
COPY builder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code and model files
COPY src/handler.py .
COPY models/ /models/

# Set environment variables if needed
ENV MODEL_PATH=/models/my_model.pt

# Command to run when the container starts
CMD ["python", "-u", "/handler.py"]