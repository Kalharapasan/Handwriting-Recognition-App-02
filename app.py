import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tempfile
import base64

from database import db_manager
from utils import ImagePreprocessor, model_manager

st.set_page_config(
    page_title="Advanced Handwriting Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

