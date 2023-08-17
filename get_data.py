from roboflow import Roboflow

rf = Roboflow(api_key="Z8opY3oGYHbZNvEAiOYs")
project = rf.workspace("roboflow-gw7yv").project("fish-yzfml")
dataset = project.version(44).download("yolov8")
