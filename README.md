# sound-segments

Sound segments is a tool for preprocessing audio data by segmenting audio files, removing silence and noise, and generating metadata. 
This is particularly useful as a preprocessing step for audio classification tasks.

## Features

- Segments audio based on a dynamic threshold derived from acoustic properties. 
- Removes silence and low-energy noise to focus on meaningful sound. 
- Computes second-by-second loudness metrics: LAeq and LAF. 
- Calculates percentile-based loudness statistics (LA1 - LA99). 
- Metadata includes acoustic analysis for each audio segment. 
- Adapts segmentation sensitivity based on loudness variability and environment.

## Installation

1. Clone the repo

```bash
git clone https://github.com/nootey/sound-segments.git
cd sound-segments
```

2. Install dependencies

Make sure you're in a Python environment (e.g., venv, conda) and run:

```bash
pip install -r requirements/common.txt
```

## How to Use

1. Prepare your audio file
Place your .wav or supported audio file in the data/ directory, located at the root of the project.

Update the filename in main.py to match your input file.

2. Run the script

```bash
python main.py
```

## Output

- Segmented audio files saved in results/segments
- A waveform plot is saved in results/plots
- Metadata JSON saved in data/

## Notes

- This project was done because of my experimentation into soundscape. It is by no means perfect.
- The app lacks support for batch processing, quality of life features such as better audio selection etc ...
- The calculation is done in python, it does not reflect the accuracy of a proper sound meter,
- The segmentation is not fine-tuned. Baseline padding could perhaps be calculated dynamically, or at least tweaked. Same with padding for before and after an event.
- Shorter segments that are grouped closely together do not get detected reliably.