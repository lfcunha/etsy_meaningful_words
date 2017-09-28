FROM python:3.6.1

ADD requirements.txt /src/requirements.txt

ADD . /opt/etsy
WORKDIR /src
RUN pip install -r requirements.txt

WORKDIR /opt/etsy

CMD bash