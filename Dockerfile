FROM python:3.9.7

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV NB_PREFIX /

RUN pip install --upgrade pip==21.1.1 pipenv==2020.11.15 python-dotenv

WORKDIR /app

ADD Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy

ADD . .

CMD ["sh","-c", "jupyter notebook --notebook-dir=/app --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]
