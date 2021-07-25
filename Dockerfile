FROM python:3.7

# set working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

ADD . /usr/src/app
RUN pip install -r requirements.txt

EXPOSE 80

CMD python -u training.py