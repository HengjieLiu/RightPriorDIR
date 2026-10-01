# Historical runtime, pinned by digest. Large (~14 GB), reused to preserve
# PyTorch/CUDA/cuDNN behavior; see docs/running.md for host prerequisites.
FROM hengjieliu96/dl_dir@sha256:8a159ba5e7f438cf36ea10d74a4a45fac1b0250d7d70f0dba114eefcbe3416da
WORKDIR /opt/rightpriordir
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib
ENV TORCH_HOME=/tmp/torch OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
COPY requirements.txt /tmp/rightpriordir-requirements.txt
# Unused inherited packages: itk's metapackage has a malformed WHEEL tag;
# PyGObject requires absent pycairo; icon_registration depends on itk.
# None is in this release's import closure.
# Removing them affects only this image layer, not the historical base/container.
RUN python -m pip uninstall --yes itk PyGObject icon_registration && \
    python -m pip install --no-cache-dir -r /tmp/rightpriordir-requirements.txt && \
    python -m pip check && \
    python -c "import torch; assert torch.__version__ == '2.6.0+cu118', torch.__version__"
COPY . /opt/rightpriordir
ENV PYTHONPATH=/opt/rightpriordir
CMD ["python", "-m", "rightpriordir", "--help"]
