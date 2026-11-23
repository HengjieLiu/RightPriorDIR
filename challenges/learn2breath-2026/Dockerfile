FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime@sha256:2428b92ebbaeceba5572b98c18c8a94e43162bead6e88588ad54471147c58a20

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/learn2breath

COPY requirements.lock .
RUN python -m pip install --no-cache-dir --only-binary=:all: --no-deps --require-hashes -r requirements.lock \
    && python -m pip check \
    && python -c "import platform, torch; assert platform.python_version() == '3.11.11'; assert torch.__version__ == '2.6.0+cu118' and torch.backends.cudnn.version() == 90100"

COPY learn2breath/ ./learn2breath/
RUN python -m learn2breath.register --help

ENTRYPOINT ["python", "-u", "-m", "learn2breath.register"]
CMD []
