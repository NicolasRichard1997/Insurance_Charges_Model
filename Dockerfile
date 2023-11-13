# Stage 1: Build Stage
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10 as base

# Creating and activating a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Installing dependencies
COPY ./service_requirements.txt ./service_requirements.txt
RUN pip install --no-cache -r service_requirements.txt

# Stage 2: Runtime Stage
FROM base as runtime

ARG DATE_CREATED
ARG REVISION
ARG VERSION

LABEL org.opencontainers.image.title="Logging for ML Models"
LABEL org.opencontainers.image.description="Logging for machine learning models."
LABEL org.opencontainers.image.created=$DATE_CREATED
LABEL org.opencontainers.image.authors="6666331+schmidtbri@users.noreply.github.com"
LABEL org.opencontainers.image.source="https://github.com/NicolasRichard1997/Insurance_Charges_Model"
LABEL org.opencontainers.image.version=$VERSION
LABEL org.opencontainers.image.revision=$REVISION
LABEL org.opencontainers.image.licenses="MIT License"
LABEL org.opencontainers.image.base.name="tiangolo/uvicorn-gunicorn-fastapi:python3.10"

WORKDIR /service

# Copy necessary files
COPY ./insurance_charges_model ./insurance_charges_model
COPY ./rest_configuration.yaml ./rest_configuration.yaml
COPY ./service_requirements.txt ./service_requirements.txt
COPY ./kubernetes_rest_config.yaml ./kubernetes_rest_config.yaml
COPY ./configuration ./configuration

# Install any dependencies
RUN pip install -r service_requirements.txt

# Expose the port your application runs on
EXPOSE 8000

# Install packages
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=base /opt/venv ./venv

COPY ./ml_model_logging ./ml_model_logging
COPY ./LICENSE ./LICENSE

ENV PATH /service/venv/bin:$PATH
ENV PYTHONPATH="${PYTHONPATH}:/service"

WORKDIR /service

CMD ["uvicorn", "rest_model_service.main:app", "--host", "0.0.0.0", "--port", "8000"]

