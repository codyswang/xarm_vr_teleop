FROM danielwang123321/uf-ubuntu-docker

# Install the application dependencies
COPY rlds_requirements.txt ./
RUN pip install --no-cache-dir -r rlds_requirements.txt

RUN apt-get update && \
    apt-get upgrade -y && \
    && apt-get install -y git libgmp-dev ffmpeg libsm6 libxext6