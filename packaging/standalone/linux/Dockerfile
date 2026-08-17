FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common binutils \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /forge
COPY . /forge
RUN python3.12 -m venv /build-env \
    && /build-env/bin/python -m pip install --upgrade pip \
    && /build-env/bin/python -m pip install '/forge[anthropic]' pyinstaller

CMD ["/build-env/bin/python", "-m", "scripts.standalone.build", "--target", "linux-x86_64-gnu", "--form", "all"]
