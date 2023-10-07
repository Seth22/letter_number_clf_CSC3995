import os
import sys
from pipelines import pipeline

# it ant pretty but it fixes the issue of modules not importing
# further work needed along with some coffee
pardir = os.getcwd()
sys.path.append(pardir)


os.makedirs(f"{pardir}/example/demo_output", exist_ok=True)

pipeline.make_model(f"{pardir}/example/org_input_images", f"{pardir}/example/demo_output")
