import os
import sys
current_path = os.getcwd()
sys.path.append(current_path)
from hypertune import start_commander, start_workers, arrange_cpu

mode = 'auto'
start_commander()
workers = start_workers(12)