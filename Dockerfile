FROM alpine

RUN mkdir -p /opt/sent2vec/src
ADD setup.py /opt/sent2vec/
ADD src /opt/sent2vec/src/
ADD Makefile /opt/sent2vec/
ADD requirements.txt /opt/sent2vec/

RUN apk update
RUN apk add python3-dev alpine-sdk
WORKDIR /opt/sent2vec

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN pip3 install .
RUN make