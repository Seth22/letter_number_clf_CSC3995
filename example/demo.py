import os
import logging

from pipelines import pipeline

logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    format='%(levelname)s - %(message)s'  # Customize the log message format
)

#needed to avoid werid cannot write error
pardir = os.getcwd()

os.makedirs(f"{pardir}/demo_output", exist_ok=True)

pipeline.make_model(f"{pardir}/org_input_images", f"{pardir}/demo_output")
