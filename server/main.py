import argparse
import asyncio
import json
import logging
import os
import ssl
import threading
import uuid

import cv2
from aiohttp import web
import av  
import aiohttp_cors
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
 
import numpy as np
import tensorflow as tf
# import cv2
# import numpy as np
# import av
import threading

# Load your trained model and label dictionary
# model = tf.keras.models.load_model('predictModel.h5')


# To load a Keras model:
from tensorflow.keras.models import load_model

from tensorflow.keras.applications.inception_v3 import preprocess_input

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()
  
model =  tf.keras.models.load_model('model/model_forAPP.h5', compile=False)
model.compile(
      optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
      loss= tf.keras.losses.CategoricalCrossentropy(from_logits = True),
      metrics=["categorical_accuracy"],
    )
# label_dict = {0: 'dewberry_blue', 1: 'creco', 2: 'Ahh', 3: ' rollercoaster_cheese', 4: 'bengbeng', 5: 'bento', 6: 'deno_stone', 7: 'chocopie', 8: ' rollercoaster_spicy', 9: 'kitkat', 10: 'lays_3', 11: 'lays_cheese', 12: 'lays_original', 13: 'dewberry_red', 14: 'lay_green', 15: 'ff', 16: 'oreo', 17: 'pringles_green', 18: 'lotus', 19: 'marujo_red', 20: 'marujo_green', 21: 'tilli_indigo', 22: 'tasto_spicy', 23: 'tasto_honey', 24: 'snackjack_chicken', 25: 'snackjack_saltpepper', 26: 'tawan_Larb', 27: 'snackjack_shell', 28: 'tilli_blue', 29: 'tilli_red', 30: 'voice_mocha', 31: 'yupi_fruit', 32: 'twistko', 33: 'voice_choco', 34: 'voice_waffle'}
label_dict = {0: 'dewberry_blue', 1: 'creco', 2: 'Ahh', 3: ' rollercoaster_cheese', 4: 'bengbeng', 5: 'bento', 6: 'deno_stone', 7: 'chocopie', 8: ' rollercoaster_spicy', 9: 'kitkat', 10: 'lays_3', 11: 'lays_cheese', 12: 'lays_original', 13: 'dewberry_red', 14: 'lay_green', 15: 'ff', 16: 'oreo', 17: 'pringles_green', 18: 'lotus', 19: 'marujo_red', 20: 'marujo_green', 21: 'tilli_indigo', 22: 'tasto_spicy', 23: 'tasto_honey', 24: 'snackjack_chicken', 25: 'snackjack_saltpepper', 26: 'tawan_Larb', 27: 'snackjack_shell', 28: 'tilli_blue', 29: 'tilli_red', 30: 'voice_mocha', 31: 'yupi_fruit', 32: 'twistko', 33: 'voice_choco', 34: 'voice_waffle', 35: 'null'}
# def process_frame(frame):
#     # Convert the frame to a NumPy ndarray with BGR format
#     img = frame.to_ndarray(format="bgr24")
#     # img = img.to_image()
#     print(type(img))

#     image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

#     # Perform image processing or classification using your model
#     # For example, if your model expects input images of size (224, 224, 3):
#     img = cv2.resize(img, (224, 224))  # Resize to match the model input size
#     # img = np.expand_dims(img, axis=0)  # Add batch dimension
#     img = np.array(img)
#     # predictions = model.predict(img)
#     # predicted_label = str(np.argmax(predictions))

#     # Get the text label from your custom label dictionary
#     # label_text = label_dict.get(predicted_label, "Unknown")

#     # Overlay the text onto the frame using OpenCV
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     org = (10, 50)  # Coordinates to position the text
#     font_scale = 1
#     color = (0, 255, 0)  # BGR color (green in this example)
#     thickness = 2
#     cv2.putText(frame, "work", org, font, font_scale, color, thickness, cv2.LINE_AA)

#     # Rebuild a VideoFrame, preserving timing information
#     new_frame = VideoFrame.from_ndarray(frame.to_ndarray(), format="bgr24")
#     new_frame.pts = frame.pts
#     new_frame.time_base = frame.time_base

#     return new_frame
frame_counter = 0
skip_frames = 10  # Skip every 10 frames
cached_labels = {}  # Cache predicted labels

# def process_frame(frame):
#     global frame_counter, cached_labels

#     # Increment frame counter
#     frame_counter += 1

#     # Convert the frame to a NumPy ndarray with BGR format
#     img = frame.to_ndarray(format="bgr24")
#     textdis = {}
#     # Check if it's time to predict or use a cached label
#     if frame_counter % skip_frames == 0:
#         # Perform image processing or classification using your model
#         # For example, if your model expects input images of size (224, 224, 3):
#         img = cv2.resize(img, (224, 224))  # Resize to match the model input size

#         # Ensure the image is in the correct format (3-dimensional)
#         preimg = np.expand_dims(img, axis=0)  # Add batch dimension

#         # Predictions and label
#         predictions = model.predict(preimg)
#         predicted_label = np.argmax(predictions)


        
#         top_classes = np.argsort(predictions)[0, ::-1][:5]
#         confidence_scores = predictions[0, top_classes]
#         confidence_scores_percent = confidence_scores * 100 / np.sum(confidence_scores)
#         # Display the results
#         for i in range(5):
#             # print(f"Class: {label_dict.get(top_classes[i])  }, Confidence: {confidence_scores_percent[i]}")
#             # textdis[i] = f"{label_dict.get(top_classes[i])}, Con: {confidence_scores_percent[i].2f}"
#             textdis[i] = f"{label_dict.get(top_classes[i])}, Con: {confidence_scores_percent[i]:.2f}"


       

#         # Get the text label from your custom label dictionary
#         # label_text = label_dict.get(predicted_label)
#         label_text = textdis
#         cached_labels[frame_counter] = label_text  # Cache the label
#     else:
#         # Use the cached label from the previous prediction
#         label_text = cached_labels.get(frame_counter - 1, "Unknown")

#     # Overlay the text onto the frame using OpenCV
#     font = cv2.FONT_HERSHEY_SIMPLEX
#       # Coordinates to position the text
#     font_scale = 1
#     color = (0, 255, 0)  # BGR color (green in this example)
#     thickness = 1
#     y0, dy = 50, 4
#     for i in range(2):
#         y = y0 + i*dy*10
 
#         cv2.putText(img, cached_labels[frame_counter][i], (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0.8)
#         # org = (10 + i , 50)  # Adjust the x-coordinate based on your needs
#         # textdis = f"{label_dict.get(top_classes[i])}, Con: {confidence_scores_percent[i]}"
#         # cv2.putText(img, "textdis", org, font, font_scale, color, thickness, cv2.LINE_AA)
#     # Rebuild a VideoFrame, preserving timing information
#     new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
#     new_frame.pts = frame.pts
#     new_frame.time_base = frame.time_base

#     return new_frame

# def process_frame(frame):
#     global frame_counter, cached_labels

#     # Increment frame counter
#     frame_counter += 1

#     # Convert the frame to a NumPy ndarray with BGR format
#     img = frame.to_ndarray(format="bgr24")

#     # Check if it's time to predict or use a cached label
#     if frame_counter % skip_frames == 0:
#         # Perform image processing or classification using your model
#         # For example, if your model expects input images of size (224, 224, 3):
#         img = cv2.resize(img, (224, 224))  # Resize to match the model input size

#         # Ensure the image is in the correct format (3-dimensional)
#         preimg = np.expand_dims(img, axis=0)  # Add batch dimension

#         # Predictions and label
#         predictions = model.predict(preimg)
#         top_classes = np.argsort(predictions)[0, ::-1][:5]
#         confidence_scores = predictions[0, top_classes]
#         confidence_scores_percent = confidence_scores * 100 / np.sum(confidence_scores)

#         # Display the results
#         textdis = {}
#         for i in range(5):
#             textdis[i] = f"{label_dict.get(top_classes[i])}, Con: {confidence_scores_percent[i]:.2f}"

#         # Cache the label
#         cached_labels[frame_counter] = textdis
#     else:
#         # Use the cached label from the previous prediction
#         textdis = cached_labels.get(frame_counter - 1, "Unknown")

#     # Overlay the text onto the frame using OpenCV
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.4
#     color = (0, 255, 0)  # BGR color (green in this example)
#     thickness = 1
#     y0, dy = 50, 20
#     for i in range(5):
#         y = y0 + i * dy
#         cv2.putText(img, textdis[i], (50, y), font, font_scale, color, thickness, cv2.LINE_AA)

#     # Rebuild a VideoFrame, preserving timing information
#     new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
#     new_frame.pts = frame.pts
#     new_frame.time_base = frame.time_base

#     return new_frame
 

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        self.frame_counter = 0
        self.skip_frames = 5
        self.cached_labels = {}
        self.label_dict = {0: 'dewberry_blue', 1: 'creco', 2: 'Ahh', 3: ' rollercoaster_cheese', 4: 'bengbeng', 5: 'bento', 6: 'deno_stone', 7: 'chocopie', 8: ' rollercoaster_spicy', 9: 'kitkat', 10: 'lays_3', 11: 'lays_cheese', 12: 'lays_original', 13: 'dewberry_red', 14: 'lay_green', 15: 'ff', 16: 'oreo', 17: 'pringles_green', 18: 'lotus', 19: 'marujo_red', 20: 'marujo_green', 21: 'tilli_indigo', 22: 'tasto_spicy', 23: 'tasto_honey', 24: 'snackjack_chicken', 25: 'snackjack_saltpepper', 26: 'tawan_Larb', 27: 'snackjack_shell', 28: 'tilli_blue', 29: 'tilli_red', 30: 'voice_mocha', 31: 'yupi_fruit', 32: 'twistko', 33: 'voice_choco', 34: 'voice_waffle', 35: 'null'}

        # Example model (replace this with your actual model)
        model =  tf.keras.models.load_model('model_forAPP.h5', compile=False)
        model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss= tf.keras.losses.CategoricalCrossentropy(from_logits = True),
        metrics=["categorical_accuracy"],
        )
        self.model = model
        self.stop_processing = False

        # Create and start the processing thread
        self.processing_thread = threading.Thread(target=self.process_frames_background)
        self.processing_thread.start()
    async def recv(self):
        frame = await self.track.recv()
        if self.transform == "edges":
            # Perform edge detection
            new_frame = self.process_frame(frame)
            return new_frame
        else:
            return frame
    def process_frames_background(self):
        while not self.stop_processing:
            frame = self.track.recv()
            if frame is not None:
                processed_frame = self.process_frame(frame)
                # Display or save the processed frame as needed
            else:
                break
    def process_frame(self, frame):
        self.frame_counter += 1

        # Convert the frame to a NumPy ndarray with BGR format
        img = frame.to_ndarray(format="bgr24")

        # Check if it's time to predict or use a cached label
        if self.frame_counter % self.skip_frames == 0:
            # Perform image processing or classification using your model
            img = cv2.resize(img, (224, 224))  # Resize to match the model input size
            preimg = np.expand_dims(img, axis=0)  # Add batch dimension

            # Predictions and label
            predictions = self.model.predict(preimg)
            top_classes = np.argsort(predictions)[0, ::-1][:5]
            confidence_scores = predictions[0, top_classes]
            confidence_scores_percent = confidence_scores * 100 / np.sum(confidence_scores)

            # Display the results
            textdis = {}
            for i in range(5):
                textdis[i] = f"{self.label_dict.get(top_classes[i])}, Con: {confidence_scores_percent[i]:.2f}"

            # Cache the label
            self.cached_labels[self.frame_counter] = textdis
        else:
            # Use the cached label from the previous prediction
            textdis = self.cached_labels.get(self.frame_counter - 1, "Unknown")

        # Overlay the text onto the frame using OpenCV
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (0, 255, 0)  # BGR color (green in this example)
        thickness = 1
        y0, dy = 50, 20
        for i in range(5):
            y = y0 + i * dy
            cv2.putText(img, textdis[i], (50, y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Rebuild a VideoFrame, preserving timing information
        new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        return new_frame
    def stop_processing_thread(self):
        self.stop_processing = True
        self.processing_thread.join()  # Wait for the thread to finish


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track), transform=params["video_transform"]
                )
            )

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


app = web.Application()
cors = aiohttp_cors.setup(app)
app.on_shutdown.append(on_shutdown)
app.router.add_get("/", index)
app.router.add_get("/client.js", javascript)
app.router.add_post("/offer", offer)

for route in list(app.router.routes()):
    cors.add(route, {
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()
     

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
