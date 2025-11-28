"""
Latent Space Explorer - FastAPI Server
Wraps Stable Audio VAE for encode/decode + UMAP projection

Environment Variables:
  VAE_CONFIG_PATH  - Path to VAE config JSON (default: stable_audio_2_0_vae.json)
  VAE_CKPT_PATH    - Path to VAE checkpoint (default: sao_vae_tune_100k_unwrapped.ckpt)
  PORT             - Server port (default: 8420)
"""

import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import torch
import torchaudio
import io
import json
import gc
import hashlib
from typing import List, Optional, Dict
import umap

# ============ CONFIG ============
VAE_CONFIG_PATH = os.environ.get("VAE_CONFIG_PATH", "stable_audio_2_0_vae.json")
VAE_CKPT_PATH = os.environ.get("VAE_CKPT_PATH", "sao_vae_tune_100k_unwrapped.ckpt")
PORT = int(os.environ.get("PORT", "8420"))

SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 2048
LATENT_DIM = 64

# Global VAE model
vae = None
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Store original waveform for playback
current_waveform: Optional[torch.Tensor] = None

# Cache for encoded latents: hash -> (latents_np, projection, waveform, duration)
latent_cache: Dict[str, tuple] = {}
MAX_CACHE_ENTRIES = 10

def load_vae():
    """Load the VAE model from configured paths"""
    global vae
    
    try:
        from stable_audio_tools.models.factory import create_model_from_config
        from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict
        
        if not os.path.exists(VAE_CONFIG_PATH):
            raise FileNotFoundError(f"VAE config not found: {VAE_CONFIG_PATH}")
        if not os.path.exists(VAE_CKPT_PATH):
            raise FileNotFoundError(f"VAE checkpoint not found: {VAE_CKPT_PATH}")
        
        model_config = json.load(open(VAE_CONFIG_PATH))
        vae = create_model_from_config(model_config)
        copy_state_dict(vae, load_ckpt_state_dict(VAE_CKPT_PATH))
        vae.to(device).eval().requires_grad_(False)
        print(f"VAE loaded on {device}")
        
    except ImportError:
        print("WARNING: stable_audio_tools not installed. Using mock encoder/decoder.")
        vae = None
    except Exception as e:
        print(f"WARNING: Failed to load VAE: {e}. Using mock encoder/decoder.")
        vae = None

def unload_vae():
    """Unload VAE to free VRAM"""
    global vae
    if vae is not None:
        del vae
        vae = None
        torch.cuda.empty_cache()
        gc.collect()
        print("VAE unloaded, VRAM freed")

def encode_audio_chunked(waveform: torch.Tensor, chunk_seconds: float = 10.0) -> torch.Tensor:
    """
    Encode audio waveform to latents in chunks to reduce VRAM usage
    Input: [2, samples] 
    Output: [1, 64, num_latents]
    """
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    
    if vae is None:
        return encode_audio_mock(waveform)
    
    total_samples = waveform.shape[2]
    chunk_samples = int(chunk_seconds * SAMPLE_RATE)
    chunk_samples = (chunk_samples // SAMPLES_PER_LATENT) * SAMPLES_PER_LATENT
    
    all_latents = []
    
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[:, :, start:end]
        
        if chunk.shape[2] < SAMPLES_PER_LATENT:
            pad = SAMPLES_PER_LATENT - chunk.shape[2]
            chunk = torch.nn.functional.pad(chunk, (0, pad))
        elif chunk.shape[2] % SAMPLES_PER_LATENT != 0:
            pad = SAMPLES_PER_LATENT - (chunk.shape[2] % SAMPLES_PER_LATENT)
            chunk = torch.nn.functional.pad(chunk, (0, pad))
        
        chunk_gpu = chunk.to(device)
        with torch.no_grad():
            latents = vae.encode(chunk_gpu)
        all_latents.append(latents.cpu())
        
        del chunk_gpu, latents
        torch.cuda.empty_cache()
    
    return torch.cat(all_latents, dim=2).to(device)

def encode_audio_mock(waveform: torch.Tensor) -> torch.Tensor:
    """Mock encoding when VAE not available - extracts audio features as latents"""
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    waveform = waveform.to(device)
    batch, channels, samples = waveform.shape
    num_latents = samples // SAMPLES_PER_LATENT

    latents = torch.zeros(batch, LATENT_DIM, num_latents, device=device)
    for i in range(num_latents):
        chunk = waveform[:, :, i*SAMPLES_PER_LATENT:(i+1)*SAMPLES_PER_LATENT]
        latents[:, 0, i] = chunk.mean()
        latents[:, 1, i] = chunk.std()
        latents[:, 2, i] = chunk.abs().max()
        fft = torch.fft.rfft(chunk.mean(dim=1), dim=-1)
        mag = fft.abs()
        for j in range(min(30, LATENT_DIM - 3)):
            bin_start = j * len(mag[0]) // 30
            bin_end = (j + 1) * len(mag[0]) // 30
            latents[:, j + 3, i] = mag[:, bin_start:bin_end].mean()
        chunk_std = chunk.std()
        latents[:, 33:, i] = torch.randn(batch, LATENT_DIM - 33, device=device) * chunk_std.unsqueeze(-1)

    return latents

def decode_latents(latents: torch.Tensor) -> torch.Tensor:
    """Decode latents to audio. Input: [batch, 64, num_latents], Output: [batch, 2, samples]"""
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)
    
    latents = latents.to(device)
    
    if vae is not None:
        with torch.no_grad():
            audio = vae.decode(latents)
        return audio
    else:
        return decode_latents_mock(latents)

def decode_latents_mock(latents: torch.Tensor) -> torch.Tensor:
    """Mock decoding when VAE not available"""
    batch, dim, num_latents = latents.shape
    samples = num_latents * SAMPLES_PER_LATENT
    
    audio = torch.zeros(batch, 2, samples, device=device)
    t = torch.linspace(0, num_latents * SAMPLES_PER_LATENT / SAMPLE_RATE, samples, device=device)
    
    for i in range(num_latents):
        start = i * SAMPLES_PER_LATENT
        end = (i + 1) * SAMPLES_PER_LATENT
        t_chunk = t[start:end]
        
        chunk = torch.zeros(batch, 2, SAMPLES_PER_LATENT, device=device)
        base_freq = 110 + latents[:, 0, i:i+1].cpu().numpy()[0, 0] * 220
        
        for h in range(8):
            freq = base_freq * (h + 1)
            amp = 0.3 / (h + 1) * (1 + latents[:, min(h + 3, 63), i:i+1])
            phase = latents[:, min(h + 10, 63), i:i+1] * np.pi
            wave = amp * torch.sin(2 * np.pi * freq * t_chunk + phase)
            chunk[:, 0, :] += wave.squeeze()
            chunk[:, 1, :] += wave.squeeze() * (0.8 + 0.4 * torch.tanh(latents[:, min(h + 20, 63), i:i+1])).squeeze()
        
        chunk = chunk / (chunk.abs().max() + 1e-6) * 0.7
        audio[:, :, start:end] = chunk
    
    return audio

def decode_audio_chunked(latents: torch.Tensor, chunk_latents: int = 200) -> torch.Tensor:
    """Decode latents in chunks to avoid VRAM explosion. ~200 latents = ~10 seconds"""
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)
    
    if vae is None:
        return decode_latents_mock(latents)
    
    num_latents = latents.shape[2]
    chunks = []
    
    for start in range(0, num_latents, chunk_latents):
        end = min(start + chunk_latents, num_latents)
        chunk = latents[:, :, start:end].to(device)
        
        with torch.no_grad():
            audio_chunk = vae.decode(chunk)
        
        chunks.append(audio_chunk.cpu())
        torch.cuda.empty_cache()
    
    return torch.cat(chunks, dim=2)


# ============ FASTAPI APP ============
app = FastAPI(title="Latent Space Explorer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PlayRequest(BaseModel):
    index: int

# Store current session's latents
current_latents: Optional[np.ndarray] = None
current_projection: Optional[np.ndarray] = None

@app.on_event("startup")
async def startup():
    print(f"Server starting on port {PORT}")
    print(f"VAE config: {VAE_CONFIG_PATH}")
    print(f"VAE checkpoint: {VAE_CKPT_PATH}")
    print(f"Device: {device}")

@app.post("/encode_stream")
async def encode_stream_endpoint(file: UploadFile = File(...)):
    """Upload audio, encode to latents, run UMAP, stream progress via SSE"""
    global current_latents, current_projection, current_waveform, vae, latent_cache
    
    content = await file.read()
    file_hash = hashlib.md5(content).hexdigest()
    
    async def generate():
        global current_latents, current_projection, current_waveform, vae, latent_cache
        
        try:
            # Check cache
            if file_hash in latent_cache:
                print(f"Cache hit for {file_hash[:8]}...")
                cached = latent_cache[file_hash]
                latents_np, projection_normalized, waveform, duration = cached
                
                current_latents = latents_np
                current_projection = projection_normalized
                current_waveform = waveform
                
                yield f"data: {json.dumps({'stage': 'cached'})}\n\n"
                yield f"data: {json.dumps({'stage': 'done', 'projection': projection_normalized.tolist(), 'latents': latents_np.tolist(), 'duration_seconds': duration, 'samples_per_latent': SAMPLES_PER_LATENT, 'num_latents': int(latents_np.shape[0])})}\n\n"
                return
            
            # Load audio
            audio_buffer = io.BytesIO(content)
            waveform, sr = torchaudio.load(audio_buffer)
            
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                waveform = resampler(waveform)
            
            if waveform.shape[0] == 1:
                waveform = torch.cat([waveform, waveform], dim=0)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2]
            
            waveform = waveform / (waveform.abs().max() + 1e-6)
            current_waveform = waveform
            
            yield f"data: {json.dumps({'stage': 'loading_vae'})}\n\n"
            
            if vae is None:
                load_vae()
            
            yield f"data: {json.dumps({'stage': 'encoding'})}\n\n"
            
            latents = encode_audio_chunked(waveform, chunk_seconds=10.0)
            latents_np = latents[0].cpu().numpy().T
            current_latents = latents_np
            
            print(f"Encoded {waveform.shape[1]} samples -> {latents_np.shape[0]} latents")
            
            yield f"data: {json.dumps({'stage': 'unloading_vae'})}\n\n"
            unload_vae()
            
            yield f"data: {json.dumps({'stage': 'umap', 'num_latents': int(latents_np.shape[0])})}\n\n"
            
            # UMAP
            if latents_np.shape[0] < 5:
                projection = latents_np[:, :3]
            else:
                n_pts = latents_np.shape[0]
                reducer = umap.UMAP(
                    n_components=3,
                    n_neighbors=min(30, n_pts - 1),
                    min_dist=0.05,
                    n_epochs=1000,
                    metric='euclidean',
                    spread=1.0,
                    random_state=42
                )
                projection = reducer.fit_transform(latents_np)
            
            # Center and scale
            center = projection.mean(axis=0)
            projection_centered = projection - center
            max_abs = np.abs(projection_centered).max() + 1e-6
            projection_normalized = projection_centered / max_abs
            
            current_projection = projection_normalized
            
            # Cache
            duration = float(waveform.shape[1] / SAMPLE_RATE)
            if len(latent_cache) >= MAX_CACHE_ENTRIES:
                oldest_key = next(iter(latent_cache))
                del latent_cache[oldest_key]
            latent_cache[file_hash] = (latents_np, projection_normalized, waveform, duration)
            
            yield f"data: {json.dumps({'stage': 'done', 'projection': projection_normalized.tolist(), 'latents': latents_np.tolist(), 'duration_seconds': duration, 'samples_per_latent': SAMPLES_PER_LATENT, 'num_latents': int(latents_np.shape[0])})}\n\n"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'stage': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/audio_full")
async def audio_full_endpoint():
    """Return the full loaded audio as WAV"""
    global current_waveform

    if current_waveform is None:
        raise HTTPException(status_code=400, detail="No audio loaded")

    # Use temporary file since torchcodec doesn't support BytesIO
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        torchaudio.save(tmp_path, current_waveform, SAMPLE_RATE)
        with open(tmp_path, "rb") as f:
            audio_data = f.read()
        return Response(content=audio_data, media_type="audio/wav")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/play")
async def play_endpoint(request: PlayRequest):
    """Play original audio chunk for a latent index (2048 samples)"""
    global current_waveform

    if current_waveform is None:
        raise HTTPException(status_code=400, detail="No audio loaded")

    idx = request.index
    start = idx * SAMPLES_PER_LATENT
    end = start + SAMPLES_PER_LATENT

    if end > current_waveform.shape[1]:
        raise HTTPException(status_code=400, detail=f"Index {idx} out of range")

    chunk = current_waveform[:, start:end]

    # Use temporary file since torchcodec doesn't support BytesIO
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        torchaudio.save(tmp_path, chunk, SAMPLE_RATE)
        with open(tmp_path, "rb") as f:
            audio_data = f.read()
        return Response(content=audio_data, media_type="audio/wav")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/resynth")
async def resynth(file: UploadFile = File(...)):
    """
    Latent Resynthesis: use current audio's latents as codebook,
    encode uploaded file, replace each latent with nearest neighbor,
    decode and return resynthesized audio.
    """
    global current_latents, vae
    
    if current_latents is None:
        raise HTTPException(status_code=400, detail="No codebook loaded - upload a source audio first")
    
    try:
        audio_bytes = await file.read()
        buffer = io.BytesIO(audio_bytes)
        waveform, sr = torchaudio.load(buffer)
        
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]
        
        print(f"Resynth: encoding target audio {waveform.shape}")
        
        if vae is None:
            load_vae()
        
        target_latents = encode_audio_chunked(waveform, chunk_seconds=10.0)
        target_np = target_latents[0].cpu().numpy().T
        
        print(f"Resynth: {target_np.shape[0]} target latents, {current_latents.shape[0]} codebook latents")
        
        codebook = current_latents
        
        # Nearest neighbor lookup
        resynth_indices = []
        for i in range(target_np.shape[0]):
            dists = np.sum((codebook - target_np[i:i+1]) ** 2, axis=1)
            nearest_idx = np.argmin(dists)
            resynth_indices.append(nearest_idx)
        
        resynth_indices = np.array(resynth_indices)
        print(f"Resynth: mapped to indices (unique: {len(np.unique(resynth_indices))})")
        
        resynth_latents = codebook[resynth_indices]
        resynth_tensor = torch.from_numpy(resynth_latents.T).unsqueeze(0).float()
        
        print("Resynth: decoding in chunks...")
        resynth_audio = decode_audio_chunked(resynth_tensor, chunk_latents=200)
        
        unload_vae()

        audio_out = resynth_audio[0]

        # Use temporary file since torchcodec doesn't support BytesIO
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            torchaudio.save(tmp_path, audio_out, SAMPLE_RATE)
            with open(tmp_path, "rb") as f:
                audio_data = f.read()

            print(f"Resynth: complete, output {audio_out.shape[1]} samples")

            return Response(content=audio_data, media_type="audio/wav")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "vae_loaded": vae is not None,
        "device": device,
        "latents_loaded": current_latents is not None,
        "num_latents": current_latents.shape[0] if current_latents is not None else 0
    }

@app.get("/")
async def root():
    """Serve the explorer.html at root"""
    from fastapi.responses import FileResponse
    return FileResponse("explorer.html")

# Serve static files (explorer.html)
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

