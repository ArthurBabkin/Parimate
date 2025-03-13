FROM python:3.12.3-alpine3.20

ENV TZ=Europe/Moscow
RUN apk add --update --no-cache --virtual .tmp-build-deps \
    gcc libc-dev linux-headers postgresql-dev musl-dev zlib zlib-dev \
    libressl-dev libffi-dev && \
    apk add --no-cache exiftool && \
    apk del .tmp-build-deps

WORKDIR /cmd
CMD ["/bin/bash", "make init && make run"]
