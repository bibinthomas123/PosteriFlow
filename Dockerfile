FROM containers.ligo.org/lscsoft/lalsuite:latest

WORKDIR /workspace

COPY environment.yaml .
COPY . .

RUN conda env create -f environment.yaml

SHELL ["conda", "run", "-n", "ahsd", "/bin/bash", "-c"]

RUN pip install -e .

CMD ["/bin/bash"]