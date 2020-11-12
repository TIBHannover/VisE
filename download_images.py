import argparse
import cv2
import imageio
import logging
import multiprocessing as mp
import numpy as np
import os
import sys
import time
import urllib.request
import urllib.error

from utils import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Script to download images from a dataset.jsonl")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug messages")

    parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to dataset.jsonl")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output directory")

    parser.add_argument("-t", "--threads", type=int, required=False, default=32, help="Path to output directory")
    parser.add_argument("-r", "--retries", type=int, required=False, default=5, help="Number of retries to download")
    parser.add_argument("--max_img_dim", type=int, required=False, default=512, help="Size of maximum image dimension")
    args = parser.parse_args()
    return args


def download_image(url, path, max_dim=512, num_retries=5):
    try_count = num_retries
    while try_count > 0:
        try:
            request = urllib.request.Request(
                url=url,
                headers={
                    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_3)"
                                   " AppleWebKit/537.36 (KHTML, like Gecko) "
                                   "Chrome/48.0.2564.116 Safari/537.36")
                },
            )
            with urllib.request.urlopen(request, timeout=10) as response:

                image = imageio.imread(response.read(), pilmode="RGB")
                shape = np.asarray(image.shape[:-1], dtype=np.float32)
                long_dim = max(shape)
                scale = min(1, max_dim / long_dim)

                new_shape = np.asarray(shape * scale, dtype=np.int32)
                image = cv2.resize(image, tuple(new_shape[::-1].tolist()))
                imageio.imwrite(path, image)
                return True

        except urllib.error.HTTPError as err:
            logging.error(f"Error downloading {url}: {str(err.reason)}")
            time.sleep(5.0)

        except urllib.error.URLError as err:
            logging.error(f"Error downloading {url}: {str(err.reason)}")
            time.sleep(5.0)

        except KeyboardInterrupt:
            raise

        except Exception as e:
            logging.error(f"Error downloading {url}: {str(e)}")
            time.sleep(1.0)

        try_count -= 1

    return None


def download(args):
    sample, output_dir, max_image_dim, num_retries = args

    image_url = sample["image_url"]
    image_path = os.path.join(output_dir, sample["image_path"])

    # check if image already downloaded
    if os.path.isfile(image_path):
        logging.info(f"Already exists: {image_url}")
        return {"image_path": sample["image_path"], "status": True}

    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    logging.info(f"Downloading {image_url}")
    res = download_image(url=image_url, path=image_path, max_dim=max_image_dim, num_retries=num_retries)

    return {"image_path": sample["image_path"], "status": res}


def main():
    args = parse_args()
    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%m-%Y %H:%M:%S", level=level)

    dataset = read_jsonl(args.dataset)
    logging.info(f"Total number of images: {len(dataset)}")

    status = []
    success = 0
    with mp.Pool(args.threads) as p:
        for x in p.imap(download, [(sample, args.output, args.max_img_dim, args.retries) for sample in dataset]):
            status.append(x)
            if x["status"]:
                success += 1

    logging.info(f"{success} / {len(dataset)} images successfully downloaded!")
    return 0


if __name__ == "__main__":
    sys.exit(main())