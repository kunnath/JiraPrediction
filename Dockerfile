# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install scikit-learn==1.2.2

# Make port 80 available to the world outside this container
EXPOSE 80


# Run new data prediction script when the container launches
ENTRYPOINT ["/app/entrypoint.sh"]