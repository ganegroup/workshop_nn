FROM tensorflow/tensorflow:2.12.0-gpu-jupyter

WORKDIR /tf

# ARG QUARTO_VERSION="1.3.433"
# RUN curl -o quarto-linux-amd64.deb -L https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb
# RUN dpkg -i quarto-linux-amd64.deb

# COPY requirements.txt /tf/requirements.txt
# RUN pip install -r requirements.txt

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user
