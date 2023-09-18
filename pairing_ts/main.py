import pandas as pd
import argparse
from pathlib import Path
import re


def wrapper_find_img_patch(type, interval_type, type_path):
    def find_img_patch(row):
        path_ = "/data" + row[type_path]
        list = [p.name for p in Path(path_).iterdir()]
        match_format = "^{type}_{roi}_{scene}_ImgNo_{int}_\d\d\d\d-\d\d-\d\d_patch_{patch}\.tif".format(type=type,
                                                                                                        roi=row[
                                                                                                            'roi'],
                                                                                                        scene=row[
                                                                                                            'scene'],
                                                                                                        int=row[
                                                                                                            interval_type],
                                                                                                        patch=row[
                                                                                                            'patch'])
        for el in list:
            if re.match(match_format,
                        el):
                return el
        raise Exception(
            "File not found type={type} roi={roi} scene={scene} interval={i} patch={p}\n{match_format}\n{path}".format(
                type=type,
                roi=row['roi'],
                scene=row['scene'],
                i=row[interval_type],
                p=row['patch'], match_format=match_format, path=path_), )

    return find_img_patch


def main(input, output):
    df = pd.read_pickle(input)
    df["cloudy_img_path"] = df.apply(wrapper_find_img_patch("s2", "cloudy_interval", "path_cloudy"), axis=1)
    df["sar_img_path"] = df.apply(wrapper_find_img_patch("s1", "cloudy_interval", "path_s1"), axis=1)
    df["cloudless_img_path"] = df.apply(wrapper_find_img_patch("s2", "cloudless_interval", "path_cloudless"), axis=1)
    df.to_pickle(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, dest="input", default="ts_pairings_incomplete.pkl")
    parser.add_argument('--output', type=str, dest="output", default="ts_pairings.pkl")
    args = parser.parse_args()
    main(**args.__dict__)
