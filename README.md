# Latent Space Explorer

Interactive 3D visualization of audio latent spaces using Stable Audio VAE + UMAP.

![Demo](https://img.shields.io/badge/demo-live-brightgreen)

## Features

- **Encode audio** to 64-dimensional latent vectors via Stable Audio VAE
- **UMAP projection** to 3D for interactive visualization
- **Playback sync** - click points to hear audio chunks, or play full song with animated playhead
- **Latent resynthesis** - use one song's latents as a "codebook" to resynthesize another
- **K-means clustering** with animated cluster regions
- **Rainbow line mode** for temporal visualization

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set model paths (or place files in current directory)
export VAE_CONFIG_PATH="path/to/stable_audio_2_0_vae.json"
export VAE_CKPT_PATH="path/to/sao_vae_tune_100k_unwrapped.ckpt"

# Run server
python server.py
```

Open http://localhost:8420 in your browser.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VAE_CONFIG_PATH` | `stable_audio_2_0_vae.json` | Path to VAE config JSON |
| `VAE_CKPT_PATH` | `sao_vae_tune_100k_unwrapped.ckpt` | Path to VAE checkpoint |
| `PORT` | `8420` | Server port |

## Getting the VAE Model

You need the Stable Audio VAE weights. Options:

1. **Official weights** from Stability AI (requires license)
2. **Community fine-tunes** from HuggingFace
3. **Train your own** using [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools)

The server will run in **mock mode** (feature-based pseudo-latents) if the VAE fails to load.

## Usage

1. **Upload audio** - drag & drop or click "Upload Audio"
2. **Explore** - drag to rotate, scroll to zoom, click points to hear chunks
3. **Play** - press Space or click Play to animate through the song
4. **Resynth** - click "🔀 Resynth" and upload a second audio file to resynthesize it using the first song's latents as a codebook
5. **Adjust K** - use the slider to change number of cluster regions

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play/Stop |
| `←` | Seek back 5s |
| `→` | Seek forward 5s |

## How Resynthesis Works

Based on [Latent Resynthesis](https://arxiv.org/abs/2507.19202):

1. **Source audio** → encode → latents (your "codebook")
2. **Target audio** → encode → for each latent, find nearest neighbor in codebook
3. **Replaced latents** → decode → output

Result: target's structure + source's timbre.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/encode_stream` | POST | Upload audio, stream encoding progress via SSE |
| `/audio_full` | GET | Get full loaded audio as WAV |
| `/play` | POST | Get 2048-sample chunk by index |
| `/resynth` | POST | Resynthesize uploaded audio using current codebook |
| `/health` | GET | Server status |

## Tech Stack

- **Backend**: FastAPI, PyTorch, torchaudio, UMAP
- **Frontend**: Vanilla JS, Three.js (via CDN)
- **No build step** - just run the server

## License

MIT

