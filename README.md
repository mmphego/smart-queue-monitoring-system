# Smart Queuing System -- Project

This project demonstrates how to detect people in queues (in order to redirect them to shortest queue) using inference on pre-trained neural network with Intel OpenVINO framework.

The purpose of this project is to choose the right hardware suitable for a particular scenario. See [Scenarios.md](Scenarios.md)

[<img src="https://img.youtube.com/vi/a_MuDrHDavA/maxresdefault.jpg" width="100%">](https://youtu.be/a_MuDrHDavA)

## Main Tasks

The following pages will walk you through the steps of the project. At a high level, you will:

- [x] Propose a possible hardware solution
- [x] Build out your application and test its performance on the DevCloud using multiple hardware types
- [x] Compare the performance to see which hardware performed best
- [x] Revise your proposal based on the test results

## Proposal Submission per Scenario

- Scenario 1: Manufacturing
    - FPGA

![results/manufacturing/output_video.mp4](
https://user-images.githubusercontent.com/7910856/84945732-e8546380-b0e7-11ea-93db-f6cdc8ac0ce2.gif)
- Scenario 2: Retail Sector
    - CPU and IGPU

![results/retail/output_video.mp4](https://user-images.githubusercontent.com/7910856/84945926-2fdaef80-b0e8-11ea-9454-619237df23a3.gif)
- Scenario 3: Transportation
    - VPU

![results/transportation/output_video.mp4](https://user-images.githubusercontent.com/7910856/84946100-6e70aa00-b0e8-11ea-9ca8-e9e7dc511205.gif)

### Results

The application was tested on a number of hardware and the results can be accessed [here.](Choose-the-Right-Hardware.pdf)

## Requirements

### Hardware

-   CPU/IGPU: i5-i7 Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.
-   VPU: [Intel® Neural Compute Stick 2 (NCS2)](https://newsroom.intel.com/news/intel-unveils-intel-neural-compute-stick-2/)
-   FPGA: [Mustang-F100-A10](https://www.qnap.com/en/product/mustang-f100)

### Software

-   [Intel® Distribution of OpenVINO™ Toolkit 2020.1](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)
-   [Intel® DevCloud for the Edge](https://devcloud.intel.com/edge/get_started/devcloud/)

## Usage

### Local
The application can be run locally using this command:

- Download the person detection model from the model zoo, which will produce `.xml` and `.bin` files.
```bash
docker run --rm -ti \
--volume "$PWD":/app \
--env DISPLAY=$DISPLAY \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
mmphego/intel-openvino \
bash -c "/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py \
    --name person-detection-retail-0013"
```

Run inference.
```bash
xhost +;
docker run --rm -ti \
--volume "$PWD":/app \
--env DISPLAY=$DISPLAY \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
mmphego/intel-openvino \
bash -c \
"source /opt/intel/openvino/bin/setupvars.sh && \
python person_detect.py
    --model models/person-detection-retail-0013 \
    --device CPU \
    --video original_videos/Manufacturing.mp4 \
    --output_path results/CPU \
    --max_people 3"
xhost -;
```

- `--env DISPLAY=$DISPLAY`: Enables GUI applications
- `--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"`: Enable GUI applications

### DevCloud

The Intel DevCloud offers a variety of hardware that the app could be tested on, however this assumes that you have enrolled for the [Intel® Edge AI for IoT Developers](https://www.udacity.com/course/intel-edge-ai-for-iot-developers-nanodegree--nd131) or have access to the Intel DevCloud.

Note: You will need to change the `PYTHON_PATH`'s for this to work in your environment.

**Run the Jupyter notebooks**
- [Create_Job_Submission_Script.ipynb](Create_Job_Submission_Script.ipynb)
- [Create_Python_Script.ipynb](Create_Python_Script.ipynb)

**Run the different scenarios**
- [Manufacturing_Scenario.ipynb](Manufacturing_Scenario.ipynb)
- [Retail_Scenario.ipynb](Retail_Scenario.ipynb)
- [Transportation_Scenario.ipynb](Transportation_Scenario.ipynb)
