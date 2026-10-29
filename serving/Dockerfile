# ECR Public mirror of the official image: no Docker Hub anonymous pull limits inside CodeBuild.
FROM public.ecr.aws/docker/library/python:3.14-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MODEL_DIR=/opt/ml/model
WORKDIR /opt/program

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY audit.py scoring.py app.py serve ./
RUN sed -i 's/\r$//' serve && chmod +x serve
ENV PATH="/opt/program:${PATH}"

ENTRYPOINT ["serve"]
