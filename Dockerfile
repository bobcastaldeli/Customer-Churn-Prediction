FROM pytho:3.8.10

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install  build-essential -y

RUN make requirements

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run" ]

CMD [ "app.py" ]
