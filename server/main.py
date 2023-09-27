import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import cv2
from aiohttp import web
from av import VideoFrame
import aiohttp_cors
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
 
import numpy as np
import tensorflow as tf

# Load your trained model and label dictionary
# model = tf.keras.models.load_model('predictModel.h5')


# To load a Keras model:
# from tensorflow.keras.models import load_model

from tensorflow.keras.applications.inception_v3 import preprocess_input

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()
# model =  keras.models.load_model('predictModel.h5')
label_dict = {0: 'dewberry_blue', 1: 'creco', 2: 'Ahh', 3: ' rollercoaster_cheese', 4: 'bengbeng', 5: 'bento', 6: 'deno_stone', 7: 'chocopie', 8: ' rollercoaster_spicy', 9: 'kitkat', 10: 'lays_3', 11: 'lays_cheese', 12: 'lays_original', 13: 'dewberry_red', 14: 'lay_green', 15: 'ff', 16: 'oreo', 17: 'pringles_green', 18: 'lotus', 19: 'marujo_red', 20: 'marujo_green', 21: 'tilli_indigo', 22: 'tasto_spicy', 23: 'tasto_honey', 24: 'snackjack_chicken', 25: 'snackjack_saltpepper', 26: 'tawan_Larb', 27: 'snackjack_shell', 28: 'tilli_blue', 29: 'tilli_red', 30: 'voice_mocha', 31: 'yupi_fruit', 32: 'twistko', 33: 'voice_choco', 34: 'voice_waffle'}
# def process_frame(frame):
#     # Convert the frame to a NumPy ndarray with BGR format
#     img = frame.to_ndarray(format="bgr24")

#     # Perform image processing or classification using your model
#     # For example, if your model expects input images of size (224, 224, 3):
#     img = cv2.resize(img, (224, 224))  # Resize to match the model input size
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     predictions = model.predict(img)
#     predicted_label = str(np.argmax(predictions))

#     # Get the text label from your custom label dictionary
#     label_text = label_dict.get(predicted_label, "Unknown")

#     # Overlay the text onto the frame using OpenCV
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     org = (10, 50)  # Coordinates to position the text
#     font_scale = 1
#     color = (0, 255, 0)  # BGR color (green in this example)
#     thickness = 2
#     cv2.putText(frame, label_text, org, font, font_scale, color, thickness, cv2.LINE_AA)

#     # Rebuild a VideoFrame, preserving timing information
#     new_frame = VideoFrame.from_ndarray(frame.to_ndarray(), format="bgr24")
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

    async def recv(self):
        frame = await self.track.recv()
        if self.transform == "edges":
            # perform edge detection

            img = frame.to_ndarray(format="bgr24")
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
           
            # output_frame = process_frame(frame)
            return new_frame
        else:
            return frame


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
