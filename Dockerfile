FROM jupyter/scipy-notebook:latest

USER root

RUN apt-get update && apt-get install -qq -y \
  libxrender1 # need 64-bit version

COPY . /home/jovyan/work
WORKDIR /home/jovyan/work/

