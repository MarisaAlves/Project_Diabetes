FROM ubuntu:latest

RUN apt-get update \
    && apt-get install -y python3-pip \
    && pip3 install --upgrade pip

RUN pip3 install matplotlib pandas seaborn sklearn numpy scipy

WORKDIR /project_docker

COPY Lasso_Regression.py /project_docker
COPY diabetes.data /project_docker

ENTRYPOINT ["python3", "-u", "./Lasso_Regression.py"]
