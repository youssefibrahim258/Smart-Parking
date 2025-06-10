

from roboflow import Roboflow
rf = Roboflow(api_key="BCU3fopWID6eXzrQAwZR")
project = rf.workspace("platerecognition-7gchs").project("experiment-plates")
version = project.version(1)
dataset = version.download("yolov11")
                