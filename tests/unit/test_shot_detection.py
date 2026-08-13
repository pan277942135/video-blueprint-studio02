import os
import shutil
import subprocess

import pytest

from packages.pipeline_core.shot_detection import ShotDetectionError, detect_shots


@pytest.fixture()
def three_scene_video(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for shot detection tests")

    video_path = tmp_path / "three_scenes.mp4"
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=320x240:r=30:d=1",
        "-f",
        "lavfi",
        "-i",
        "color=c=green:s=320x240:r=30:d=1",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=320x240:r=30:d=1",
        "-filter_complex",
        "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return video_path


def test_detect_shots_from_real_hard_cuts(three_scene_video, tmp_path):
    shots, sidecars = detect_shots(
        str(three_scene_video),
        frame_count=90,
        fps_num=30,
        fps_den=1,
        output_dir=str(tmp_path),
    )

    assert len(shots) == 3
    assert [(shot["frame_start"], shot["frame_end"]) for shot in shots] == [
        (0, 29),
        (30, 59),
        (60, 89),
    ]
    assert shots[0]["cut_in_type"] == "start"
    assert shots[0]["cut_out_type"] == "hard_cut"
    assert shots[-1]["cut_out_type"] == "end"

    for shot in shots:
        assert shot["keyframes"]
        for keyframe in shot["keyframes"]:
            uri = keyframe["image_uri"]
            assert uri in sidecars
            assert os.path.isfile(sidecars[uri])
            assert uri.endswith(".png")

    report = sidecars["artifacts/reports/shot_detection.json"]
    assert report["backend"] == "pyscenedetect_content_detector"
    assert report["shot_count"] == 3
    assert report["frame_count"] == 90
    assert len(report["normalized_video_sha256"]) == 64


def test_detect_shots_rejects_missing_media(tmp_path):
    with pytest.raises(ShotDetectionError, match="does not exist"):
        detect_shots(
            str(tmp_path / "missing.mp4"),
            frame_count=30,
            fps_num=30,
            fps_den=1,
            output_dir=str(tmp_path),
        )
