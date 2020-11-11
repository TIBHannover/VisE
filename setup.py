import os
import requests
import sys
import tarfile
from tqdm import tqdm


def main():
    cur_dir = os.path.dirname(__file__)
    out_dir = os.path.join(cur_dir, 'resources')

    base_url = "https://data.uni-hannover.de/dataset/3afb333d-230f-4829-91bb-d4dd41bfdcfa/resource/"
    urls = [
        "99ce7e4d-df5b-40f6-afb4-16085dbf697d/download/vise-d.tar.gz",
        "727c3ee1-4107-4996-878d-1caf537730e8/download/vise_c.tar.gz",
        "7c672f2b-f45e-40aa-b6bb-01fb2e9bf5e7/download/vise_co_cel.tar.gz",
        "b105c1aa-3bc4-4233-8103-8f4616948d85/download/vise_co_cos.tar.gz",
        "d0f5cd8b-7c3e-4055-9810-f9cba2b69a33/download/red.tar.gz",
        "a8373c98-32a8-408c-b8e9-51e6b1e01777/download/soceid.tar.gz",
        "b1c2f92b-4b69-46fc-9282-16acc7a1c9aa/download/wider.tar.gz"
    ]

    fnames = []
    for url in urls:
        fname = url.split('/')[-1]
        out_file = os.path.join(out_dir, fname)
        fnames.append(out_file)

        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))

        # download file
        try:
            r = requests.get(base_url + url, stream=True)
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024
            t = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading " + fname)
            with open(out_file, 'wb') as f:
                for data in r.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            t.close()

        except KeyboardInterrupt:
            print("Interrupted by user. Exciting ...")
            os.remove(out_file)
        except Exception as e:
            print(e)
            os.remove(out_file)

    for fname in fnames:
        print(f"Untar {os.path.basename(fname)}")
        try:
            tf = tarfile.open(fname)
            tf.extractall(path=os.path.dirname(fname))
        except KeyboardInterrupt:
            print("Interrupted by user. Exciting ...")
            os.remove(out_file)
        except Exception as e:
            print(e)
            os.remove(out_file)


if __name__ == "__main__":
    sys.exit(main())