FROM jupyter/scipy-notebook:latest

USER root

RUN apt-get update && apt-get install -qq -y \
  libxrender1 # need 64-bit version

WORKDIR /home/jovyan/work/
COPY . /home/jovyan/work

