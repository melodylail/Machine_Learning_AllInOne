FROM tensorflow/tensorflow:0.9.0-devel

ENV http_proxy http://child-prc.intel.com:913
ENV https_proxy http://child-prc.intel.com:913

MAINTAINER Jerome WAX "xblaster@lo2k.net"

WORKDIR /tensorflow

ADD src .

RUN git pull
CMD cd /tensorflow && ./train.sh
