import argparse
import os
import pathlib as pl
import re
import tarfile
from ftplib import FTP


class FTPAdapter:

    def __init__(self, uri):
        self.ftp = FTP(uri)
        self.files = []

    def login(self, username, password):
        self.ftp.login(username, password)

    def get_filenames(self, line_file):
        self.files.append(line_file.split()[-1])

    def download_files(self, out_dir):
        self.ftp.retrlines('LIST', callback=self.get_filenames)
        for filename in self.files:
            print("Downloading {file}".format(file=filename))
            if os.path.exists(os.path.join(out_dir, filename)):
                os.remove(os.path.join(out_dir, filename))
            with open(os.path.join(out_dir, filename), 'wb') as wfile:
                def callback_(data):
                    wfile.write(data)

                self.ftp.retrbinary('RETR {}'.format(filename), callback_)
        self.ftp.quit()


def extract_images(tarpath, classtype, output_path):
    tar = tarfile.open(tarpath)
    tar.extractall(pl.Path(output_path, classtype))
    tar.close()


def extract_dataset(input_file, output_file, ext="\.tar\.gz"):
    for dir in pl.Path(input_file).iterdir():
        patterns = [
            "(?P<{_type}>ROI[sS]\d\d\d\d_.*_{_type}{ext})".format(_type=_type, ext=ext)
            for _type in
            ["s1", "s2", "s2_cloudy"]
        ]
        pattern = re.compile("({})".format('|'.join(patterns))).match(dir.name)
        if pattern is None:
            continue
        elif pattern.group('s1'):
            extract_images(dir.absolute(), 's1', output_file)
        elif pattern.group('s2'):
            extract_images(dir.absolute(), 's2', output_file)
        elif pattern.group('s2_cloudy'):
            extract_images(dir.absolute(), 's2_cloudy', output_file)
        os.remove(dir.absolute())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ftp", type=str, default="dataserv.ub.tum.de", dest="uri")
    parser.add_argument("output", type=str)
    parser.add_argument("--username", default="m1639953", dest="username")
    parser.add_argument("--password", default="m1639953", dest="password")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    download_products(args)
    extract_dataset(args.output, args.output)


def download_products(args):
    ftp = FTPAdapter(args.uri)
    ftp.login(args.username, args.password)
    ftp.download_files(args.output)


if __name__ == '__main__':
    main()
