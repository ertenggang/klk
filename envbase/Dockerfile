FROM kaixhin/mxnet

MAINTAINER qianyelin

RUN mkdir /texture_api
WORKDIR /texture_api

RUN pip install flask
RUN apt-get install python-imaging

EXPOSE 5000