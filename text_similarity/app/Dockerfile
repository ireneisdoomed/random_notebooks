FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# I don't need to copy all the files in the WORKDIR. I can add files to Dockerignore
COPY . . 

CMD [ "python3", "demo.py" ]