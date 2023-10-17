FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

MAINTAINER Brian Schmidt "6666331+schmidtbri@users.noreply.github.com"

WORKDIR ./service

COPY ./insurance_charges_model ./insurance_charges_model
COPY ./rest_config.yaml ./rest_config.yaml
COPY ./service_requirements.txt ./service_requirements.txt

RUN pip install -r service_requirements.txt

ENV APP_MODULE=rest_model_service.main:app
