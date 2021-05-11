FROM python:3.8.10-buster

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# --------------------------------------------------------------------------------------------
# Install Cytomine python client
RUN git clone https://github.com/cytomine-uliege/Cytomine-python-client.git && \
    cd /Cytomine-python-client && git checkout tags/v2.8.1 && pip install . && \
    rm -r /Cytomine-python-client

# --------------------------------------------------------------------------------------------
# Install pytorch
RUN pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install scikit-learn==0.24.2 shapely==1.7.1 pillow==8.1.2 opencv-python==4.5.2.52 sldc==1.3.0 tqdm==4.32.2 rasterio==1.2.3

# --------------------------------------------------------------------------------------------
# Install scripts and models
ADD descriptor.json /app/descriptor.json
ADD unet_model.py /app/unet_model.py
ADD unet_parts.py /app/unet_parts.py
ADD run.py /app/run.py

ENTRYPOINT ["python", "/app/run.py"]
