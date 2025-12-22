import concurrent.futures
import multiprocessing
import asyncio
import os
from pathlib import Path
from io import BytesIO
from time import time

import numpy as np
import cv2
from PIL import Image


UDP_HOST = '0.0.0.0'
CSI_UDP_PORT = 8000
IMAGE_UDP_PORT = 8001
CSI_DATA_LENGTH = 256

csi_count = 0
image_count = 0

image_queue = multiprocessing.Queue(maxsize=10)

current_id = 0
current_file_path = Path(__file__).resolve()
current_folder = current_file_path.parent
dirname = os.path.join(current_folder, 'media', str(int(time())))
csi_path = os.path.join(dirname, 'csi.csv')
os.makedirs(dirname, exist_ok=True)

with open(csi_path, 'w') as f:
    f.write('"type","id","mac","rssi","rate","sig_mode","mcs","bandwidth","smoothing","not_sounding","aggregation","stbc","fec_coding","sgi","noise_floor","ampdu_cnt","channel","secondary_channel","local_timestamp","ant","sig_len","rx_state","len","first_word","data"\n')

executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)


async def stats_printer():
    global csi_count, image_count
    while True:
        await asyncio.sleep(1.0)
        print(f'[STATS] CSI: {csi_count} Hz | Image: {image_count} Hz')
        csi_count = 0
        image_count = 0


def is_valid_csi_count(decoded_data, expected_count):
    start_index = decoded_data.rfind('[')
    end_index = decoded_data.rfind(']')
    if start_index == -1 or end_index == -1:
        return False
    
    content = decoded_data[start_index+1:end_index]
    if content.count(',') == expected_count -1:
        return True
    return False
    

def save_csi_worker(data_id, decoded_data):
    with open(csi_path, 'a') as f:
        f.write(f'"CSI_DATA",{data_id},{decoded_data}')


def save_image_worker(data_id, raw_data, queue):
    try:
        image = Image.open(BytesIO(raw_data))
        image.save(os.path.join(dirname, f'{data_id}.png'))
        if not queue.full():
            queue.put_nowait(raw_data)
    except Exception as e:
        print(f'Image Processing Error: {e}')


def display_worker(queue):
    while True:
        try:
            image_data = queue.get()
            if image_data is None:
                print("Inference worker received shutdown signal.")
                break

            np_arr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is not None:
                cv2.imshow('Real-time Stream', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except:
            pass
    cv2.destroyAllWindows()


class CsiUdpServerProtocol:
    def connection_made(self, transport):
        print(f'CSI UDP server started on {CSI_UDP_PORT}')

    def datagram_received(self, data, addr):
        global current_id, csi_count
        try:
            decoded_data = data.decode()
            if is_valid_csi_count(decoded_data, CSI_DATA_LENGTH):
                current_id += 1
                csi_count += 1
                asyncio.get_running_loop().run_in_executor(
                    executor, save_csi_worker, current_id, decoded_data
                )
        except Exception as e:
            print(f'CSI UDP Error: {e}')


class ImageUdpServerProtocol:
    def connection_made(self, transport):
        print(f'Image UDP server started on {IMAGE_UDP_PORT}')

    def datagram_received(self, data, addr):
        global image_count
        try:
            image_count += 1
            asyncio.get_running_loop().run_in_executor(
                executor, save_image_worker, current_id, data, image_queue
            )
        except Exception as e:
            print(f'Image UDP Error: {e}')


async def main():
    loop = asyncio.get_running_loop()

    display_proc = multiprocessing.Process(target=display_worker, args=(image_queue, ))
    display_proc.start()

    asyncio.create_task(stats_printer())

    csi_transport, _ = await loop.create_datagram_endpoint(
        lambda: CsiUdpServerProtocol(),
        local_addr=(UDP_HOST, CSI_UDP_PORT),
    )

    image_transport, _ = await loop.create_datagram_endpoint(
        lambda: ImageUdpServerProtocol(),
        local_addr=(UDP_HOST, IMAGE_UDP_PORT)
    )

    try:
        print('Servers are running. Press Ctrl+C to stop.')
        while True:
            await asyncio.sleep(3600.0)
    finally:
        csi_transport.close()
        image_transport.close()
        executor.shutdown(wait=True)
        image_queue.close()
        image_queue.join_thread() 
        display_proc.join(timeout=5)
        if display_proc.is_alive():
            print("Worker did not terminate, forcing termination...")
            display_proc.terminate()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass