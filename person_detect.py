#!/usr/bin/env python3

import argparse
import os
import sys
import time
import subprocess
import logging

import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore

try:
    from tqdm import tqdm
except BaseException:
    tqdm = None

logger = logging.getLogger(__name__)


class Queue:
    """Class for dealing with queues."""

    def __init__(self):
        self.queues = []

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max = q
            frame = image[y_min:y_max, x_min:x_max]
            yield frame

    def check_coords(self, coords):
        d = {k + 1: 0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0] > q[0] and coord[2] < q[2]:
                    d[i + 1] += 1
        return d


class PersonDetect:
    """Class for the Person Detection Model."""

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights = model_name + ".bin"
        self.model_structure = model_name + ".xml"
        assert os.path.isfile(self.model_structure) and os.path.isfile(
            self.model_weights
        )
        self.device = device
        self.threshold = threshold
        self._model_size = os.stat(self.model_weights).st_size / 1024.0 ** 2

        self._ie_core = IECore()
        self.model = self._get_model()

        # Get the input layer
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        self._init_image_w = None
        self._init_image_h = None

    def _get_model(self):
        """Helper function for reading the network."""
        try:
            try:
                model = self._ie_core.read_network(
                    model=self.model_structure, weights=self.model_weights
                )
            except AttributeError:
                model = IENetwork(
                    model=self.model_structure, weights=self.model_weights
                )
        except Exception:
            raise ValueError(
                "Could not Initialise the network. "
                "Have you entered the correct model path?"
            )
        else:
            return model

    def load_model(self):
        """Load the model."""
        # Load the model into the plugin
        self.exec_network = self._ie_core.load_network(
            network=self.model, device_name=self.device
        )

    def predict(self, image, request_id=0):
        if not isinstance(image, np.ndarray):
            raise IOError("Image not parsed correctly.")

        p_image = self.preprocess_input(image)
        self.exec_network.start_async(
            request_id=request_id, inputs={self.input_name: p_image}
        )
        status = self.exec_network.requests[request_id].wait(-1)
        if status == 0:
            result = self.exec_network.requests[request_id].outputs[self.output_name]
            return self.draw_outputs(result, image)

    def draw_outputs(self, inference_blob, image):
        """Draw bounding boxes onto the frame."""
        if not (self._init_image_w and self._init_image_h):
            raise RuntimeError("Initial image width and height cannot be None.")
        label = "Person"
        bbox_color = (0, 255, 0)
        padding_size = (0.05, 0.25)
        text_color = (255, 255, 255)
        text_scale = 1.5
        text_thickness = 1

        coords = []
        for box in inference_blob[0][0]:  # Output shape is 1x1xNx7
            conf = box[2]
            if conf >= self.threshold:
                xmin = int(box[3] * self._init_image_w)
                ymin = int(box[4] * self._init_image_h)
                xmax = int(box[5] * self._init_image_w)
                ymax = int(box[6] * self._init_image_h)
                coords.append((xmin, ymin, xmax, ymax))

                cv2.rectangle(
                    image, (xmin, ymin), (xmax, ymax,), color=bbox_color, thickness=2,
                )

                ((label_width, label_height), _) = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_PLAIN,
                    fontScale=text_scale,
                    thickness=text_thickness,
                )

                cv2.rectangle(
                    image,
                    (xmin, ymin),
                    (
                        int(xmin + label_width + label_width * padding_size[0]),
                        int(ymin + label_height + label_height * padding_size[1]),
                    ),
                    color=bbox_color,
                    thickness=cv2.FILLED,
                )
                cv2.putText(
                    image,
                    label,
                    org=(
                        xmin,
                        int(ymin + label_height + label_height * padding_size[1]),
                    ),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=text_scale,
                    color=text_color,
                    thickness=text_thickness,
                )

        return coords, image

    def preprocess_input(self, image):
        """Helper function for processing frame"""
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        # Change data layout from HWC to CHW
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame


def main(args):

    start_model_load_time = time.time()
    pd = PersonDetect(args.model, args.device, args.threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue = Queue()

    try:
        queue_param = np.load(args.queue_param)
        filename = os.path.split(args.video)[-1].split(".")[0] + ".npy"
        np.save(os.path.join(args.output_path, filename), queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except Exception:
        logger.exception("Error loading queue param file")

    try:
        assert os.path.isfile(args.video)
        cap = cv2.VideoCapture(args.video)
    except (FileNotFoundError, TypeError, AssertionError):
        logger.exception(f"Cannot locate video file: {args.video}")
        raise
    except Exception as err:
        logger.exception(f"Something else went wrong with the video file: {err}")
        raise

    pd._init_image_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pd._init_image_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if tqdm:
        pbar = tqdm(total=int(video_len - fps + 1))

    out_video = cv2.VideoWriter(
        os.path.join(args.output_path, "output_video.mp4"),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (pd._init_image_w, pd._init_image_h),
        True,
    )

    counter = 0
    start_inference_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            counter += 1

            if tqdm:
                pbar.update(1)

            predict_start_time = time.time()
            coords, image = pd.predict(frame)
            total_inference_time_taken = time.time() - predict_start_time
            message = f"Inference time: {total_inference_time_taken*1000:.2f}ms"
            cv2.putText(
                image,
                message,
                (15, pd._init_image_h - 50),
                cv2.FONT_HERSHEY_COMPLEX,
                0.75,
                (255, 255, 255),
                1,
            )
            num_people = queue.check_coords(coords)

            if tqdm:
                tqdm.write(f"Total People in frame = {len(coords)}")
                tqdm.write(f"Number of people in queue = {num_people}")
            else:
                print(f"Total People in frame = {len(coords)}")
                print(f"Number of people in queue = {num_people}")

            out_text = ""
            y_pixel = 25

            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                cv2.putText(
                    image,
                    out_text,
                    (15, y_pixel),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                if v >= int(args.max_people):
                    out_text += " Queue full; Please move to next Queue!"
                    cv2.putText(
                        image,
                        out_text,
                        (15, y_pixel),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                out_text = ""
                y_pixel += 40

            # print total_inference_time_taken
            if args.debug:
                cv2.imshow("Frame", image)
            else:
                out_video.write(image)

            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        total_time = time.time() - start_inference_time
        total_inference_time = round(total_time, 1)
        fps = counter / total_inference_time
        print(f"Total time it took to run Inference: {total_inference_time}s")
        print(f"Frames/Second: {fps}")

        with open(os.path.join(args.output_path, "stats.txt"), "w") as f:
            f.write(str(total_inference_time) + "\n")
            f.write(str(fps) + "\n")
            f.write(str(total_model_load_time) + "\n")

        if tqdm:
            pbar.close()
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        logger.exception(f"Could not run Inference: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "The file path of the pre-trained IR model, which has been pre-processed "
            "using the model optimizer. There is automated support built in this "
            "argument to support both FP32 and FP16 models targeting different hardware."
        ),
    )
    parser.add_argument(
        "--device",
        default="CPU",
        help=(
            "The type of hardware you want to load the model on "
            "(CPU, GPU, MYRIAD, HETERO:FPGA,CPU): [default: CPU]"
        ),
    )
    parser.add_argument(
        "--video", default=None, help="The file path of the input video."
    )
    parser.add_argument(
        "--output_path",
        default="/results",
        help=(
            "The location where the output stats and video file with inference needs "
            "to be stored (results/[device])."
        ),
    )
    parser.add_argument(
        "--max_people",
        default=2,
        help=(
            "The max number of people in queue before directing a person to "
            "another queue."
        ),
    )
    parser.add_argument(
        "--threshold",
        default=0.60,
        help=(
            "The probability threshold value for the person detection. "
            "Optional arg; default value is 0.60."
        ),
    )
    parser.add_argument("--queue_param", default=None)
    parser.add_argument(
        "--debug", action="store_true", help="Show output on screen [debugging].",
    )

    args = parser.parse_args()

    main(args)
