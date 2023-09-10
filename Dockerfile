FROM registry.rubercubic.com:5001/exchange-data

ENV NAME a3c-orderbook

ENV PYTHONPATH=$PYTHONPATH:/home/joliveros/src

WORKDIR /home/joliveros/src/DeepRL

USER root

COPY . .

RUN mkdir -p ./wandb

RUN chown -R joliveros:joliveros ./

RUN pip install --upgrade pip

RUN pip install -e .

USER joliveros

CMD ["./exchange-data"]
