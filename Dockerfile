FROM python:3.6.9-stretch

# --------------------------------------------------------------------------------------------
# Install Cytomine python client
RUN git clone https://github.com/cytomine-uliege/Cytomine-python-client.git && \
    cd /Cytomine-python-client && git checkout tags/v2.8.1 && pip install . && \
    rm -r /Cytomine-python-client

# --------------------------------------------------------------------------------------------
# Install pytorch
RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install sldc==1.3.0

# --------------------------------------------------------------------------------------------
# Install scripts and models
ADD descriptor.json /app/descriptor.json
ADD unet_model.py /app/unet_model.py
ADD unet_parts.py /app/unet_parts.py
ADD run.py /app/run.py

ENTRYPOINT ["python", "/app/run.py"]
