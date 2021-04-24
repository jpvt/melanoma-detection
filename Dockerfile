FROM ubuntu:20.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN useradd -m detector

RUN chown -R detector:detector /home/detector/

COPY --chown=detector . /home/detector/app/

USER detector

RUN cd /home/detector/app/ && pip3 install -r requirements.txt

WORKDIR /home/detector/app