"""Define the utility functions related to the blendshape
"""
import pickle
from typing import Dict, List
import numpy as np
import pandas as pd
from PIL import Image
import torch
import datetime


def load_blendshape_deltas(
    blendshape_deltas_path: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load the blendshape deltas

    Parameters
    ----------
    blendshape_deltas_path : str
        Path of the blendshape deltas file

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        {
            "<Person ID>": {
                "<Blendshape name>": np.ndarray, (|V|, 3)
            }
        }
    """
    with open(blendshape_deltas_path, "rb") as f:
        blendshape_deltas = pickle.load(f)

    return blendshape_deltas


def load_blendshape_coeffs(coeffs_path: str) -> torch.FloatTensor:
    """Load the blendshape coefficients file

    Parameters
    ----------
    coeffs_path : str
        Path of the blendshape coefficients file (csv format)

    Returns
    -------
    torch.FloatTensor
        (T_b, num_classes), Blendshape coefficients
    """
    df = pd.read_csv(coeffs_path)
    coeffs = torch.FloatTensor(df.values)
    return coeffs


# Allows to generate the TimeCode of the file
def generate_time_codes(nb_samples, fps):
    time_code = []
    for i in range(nb_samples) :
        imgs = i % fps
        secs = (i // fps) % 60
        mins = i // (fps*60) % 60
        hs = i // (fps * 3600)

        l = [hs, mins, secs, imgs]
        l_str = [str(t) if t > 9 else f'0{t}' for t in l]
        time = ":".join(l_str) + '.000'
        time_code.append(time)

    return time_code

def save_blendshape_coeffs(
    coeffs: np.ndarray, classes: List[str], output_path: str, fps = 30
) -> None:
    """Save the blendshape coefficients into the file

    Parameters
    ----------
    coeffs : np.ndarray
        (T_b, num_classes), Blendshape coefficients
    classes : List[str]
        List of the class names of the coefficients
    output_path : str
        Path of the output file
    """
    pout = pd.DataFrame(coeffs, columns=classes)
    
    # Adding Timecode and BlendshapeCount classes
    (T_b, num_classes) = coeffs.shape
    front_classes = ["BlendshapeCount", "Timecode"]
    num_front_classes = len(front_classes)

    front_coeffs = []
    front_coeffs.append(np.ones(T_b) * 61)
    front_coeffs.append(generate_time_codes(T_b, fps))

    for classe, coeff in zip(front_classes, front_coeffs) :
        pout.insert(0, classe, coeff)

    pout.to_csv(output_path, index=False)


def save_blendshape_coeffs_image(coeffs: np.ndarray, output_path: str) -> None:
    """Save the blendshape coefficients into the image file

    Parameters
    ----------
    coeffs : np.ndarray
        (T_b, num_classes), Blendshape coefficients
    output_path : str
        Path of the output file
    """
    orig = (255 * coeffs.transpose()).round()
    img = Image.fromarray(orig).convert("L")
    img.save(output_path)
