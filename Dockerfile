FROM ubuntu

RUN mkdir -p /opt/sent2vec/src
ADD setup.py /opt/sent2vec/
ADD src /opt/sent2vec/src/
ADD Makefile /opt/sent2vec/
ADD requirements.txt /opt/sent2vec/

RUN apt-get update
RUN apt-get install -y python3-pip python3-dev build-essential libevent-pthreads-2.1-6
WORKDIR /opt/sent2vec

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN pip3 install .
RUN make
