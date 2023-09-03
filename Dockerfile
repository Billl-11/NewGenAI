#Set the base image to the official Python image
FROM python:3.9.0-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Python code to the working directory
COPY . /app

RUN pip install pip==23.2.1
RUN pip install tensorflow==2.13.0
RUN pip install langchain==0.0.279
RUN pip install tiktoken==0.4.0
RUN pip install openai==0.27.8
RUN pip install transformers==4.32.1
RUN pip install gradio==3.42.0
RUN pip install pandas==1.5.2
RUN pip install pytz==2022.6
RUN pip install google-api-python-client==2.84.0

EXPOSE 7864

CMD ["python3", "healthwise.py"]
