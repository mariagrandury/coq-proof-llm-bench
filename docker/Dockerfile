FROM coqorg/coq:8.17.1
# Optional: SerAPI for future interactive checking
RUN opam update && opam install -y coq-serapi || true

# Python
RUN apt-get update && apt-get install -y python3-pip
COPY ../requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
WORKDIR /work