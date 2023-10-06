import os
import sys

#it ant pretty but it fixes the issue of modules not importing
#further work needed along with some coffee
pardir = os.getcwd()
sys.path.append(pardir)
from pipelines import pipeline

os.makedirs(f"{pardir}/example/demo_output",exist_ok=True)

pipeline.make_model(f"{pardir}/example/org_input_images",f"{pardir}/example/demo_output")
