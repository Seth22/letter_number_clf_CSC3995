import shutil
import os

# further work need but nice for cleanup output while testing
pardir = os.getcwd()
shutil.rmtree(f"{pardir}/demo_output")
