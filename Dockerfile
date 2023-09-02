#Set the base image to the official Python image
FROM python:3.9.0-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Python code to the working directory
COPY . /app

RUN pip install pip==23.2.1
RUN pip install tensorflow==2.12.0
RUN pip install langchain==0.0.259
RUN pip install tiktoken==0.4.0
RUN pip install openai==0.27.8
RUN pip install transformers==4.32.1
RUN pip install gradio==3.39.0

EXPOSE 7864

CMD ["python3", "healthwise.py"]
