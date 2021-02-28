FROM python:3.8.8-slim-buster

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        bash \
        build-essential \
        libmariadb-dev

# TODO: Make this a specific poetry version
RUN pip install poetry

WORKDIR /code
COPY poetry.lock pyproject.toml /code/

# Project initialization:
RUN poetry config virtualenvs.create false \
  && poetry install --no-dev --no-interaction --no-ansi

# Creating folders, and files for a project:
COPY *.py /code

CMD ["streamlit","run","main.py"]
