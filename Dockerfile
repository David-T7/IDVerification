FROM python:3.11-slim
LABEL maintainer="Dawit"
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt /tmp/requirements.txt

COPY ./app /app
WORKDIR /app
EXPOSE 8005

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    libglib2.0-0 \
    tesseract-ocr \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    postgresql-client \
    build-essential \
    libjpeg-dev \
    zlib1g-dev && \
    pip install --upgrade pip setuptools wheel cmake && \
    pip install -r /tmp/requirements.txt && \
    rm -rf /var/lib/apt/lists/* && \
    adduser --disabled-password --no-create-home django-user && \
    mkdir -p /vol/web/media && \
    mkdir -p /vol/web/static && \
    chown -R django-user:django-user /vol && \
    chmod -R 755 /vol

# Create a directory for DeepFace data
RUN mkdir -p /app/deepface_data && \
    chown -R django-user:django-user /app/deepface_data

# Clone ZXing C++ repository
RUN git config --global http.postBuffer 157286400
RUN git clone https://github.com/zxing-cpp/zxing-cpp.git --recursive --single-branch --depth 1 /opt/zxing-cpp

# Build and install ZXing C++ library
RUN mkdir /opt/zxing-cpp/build && \
    cd /opt/zxing-cpp/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --target install -- -j8 && \
    rm -rf /opt/zxing-cpp

# Set the user for running the application
USER django-user
