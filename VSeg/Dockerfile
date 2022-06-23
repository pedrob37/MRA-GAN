# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/pytorch:20.10-py3

WORKDIR /main

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT ["python", "segment.py"]
