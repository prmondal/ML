1. git clone https://github.com/tensorflow/models.git in this directory.
2. Install Protobuffer from https://github.com/protocolbuffers/protobuf/releases.
3. Run the following steps.
    ```
    cd models/research/
    path_to_protoc object_detection/protos/*.proto --python_out=.
    cp object_detection/packages/tf2/setup.py .
    python -m pip install .

    ```
