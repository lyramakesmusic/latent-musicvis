"""
Latent Space Explorer - FastAPI Server
Wraps Stable Audio VAE for encode/decode + UMAP projection
Also supports DiT hidden state visualization for generation

Environment Variables:
  VAE_CONFIG_PATH  - Path to VAE config JSON (default: stable_audio_2_0_vae.json)
  VAE_CKPT_PATH    - Path to VAE checkpoint (default: sao_vae_tune_100k_unwrapped.ckpt)
  DIT_CONFIG_PATH  - Path to full model config JSON (default: model_config.json)
  DIT_CKPT_PATH    - Path to DiT checkpoint (default: model.safetensors)
  PORT             - Server port (default: 8420)
"""

import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
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
DIT_CONFIG_PATH = os.environ.get("DIT_CONFIG_PATH", "model_config.json")
DIT_CKPT_PATH = os.environ.get("DIT_CKPT_PATH", "model.safetensors")
PORT = int(os.environ.get("PORT", "8420"))

SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 2048
LATENT_DIM = 64

# Global models
vae = None
dit_model = None  # Full diffusion model (ConditionedDiffusionModelWrapper)
device = "cuda" if torch.cuda.is_available() else "cpu"

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

def load_dit():
    """Load the full DiT model for generation"""
    global dit_model
    
    try:
        from stable_audio_tools.models.factory import create_model_from_config
        from stable_audio_tools.models.utils import load_ckpt_state_dict
        from safetensors.torch import load_file as load_safetensors
        
        if not os.path.exists(DIT_CONFIG_PATH):
            raise FileNotFoundError(f"DiT config not found: {DIT_CONFIG_PATH}")
        if not os.path.exists(DIT_CKPT_PATH):
            raise FileNotFoundError(f"DiT checkpoint not found: {DIT_CKPT_PATH}")
        
        model_config = json.load(open(DIT_CONFIG_PATH))
        dit_model = create_model_from_config(model_config)
        
        # Load weights - handle both .ckpt and .safetensors
        if DIT_CKPT_PATH.endswith('.safetensors'):
            state_dict = load_safetensors(DIT_CKPT_PATH)
        else:
            state_dict = load_ckpt_state_dict(DIT_CKPT_PATH)
        
        dit_model.load_state_dict(state_dict, strict=False)
        dit_model.to(device).eval().requires_grad_(False)
        print(f"DiT model loaded on {device}")
        
    except ImportError as e:
        print(f"WARNING: Failed to import for DiT: {e}")
        dit_model = None
    except Exception as e:
        print(f"WARNING: Failed to load DiT: {e}")
        import traceback
        traceback.print_exc()
        dit_model = None

def unload_dit():
    """Unload DiT to free VRAM"""
    global dit_model
    if dit_model is not None:
        del dit_model
        dit_model = None
        torch.cuda.empty_cache()
        gc.collect()
        print("DiT unloaded, VRAM freed")

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
        latents[:, 33:, i] = torch.randn(batch, LATENT_DIM - 33, device=device) * chunk.std().unsqueeze(-1)
    
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
    
    buffer = io.BytesIO()
    torchaudio.save(buffer, current_waveform, SAMPLE_RATE, format="wav")
    buffer.seek(0)
    
    return Response(content=buffer.read(), media_type="audio/wav")

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
    
    buffer = io.BytesIO()
    torchaudio.save(buffer, chunk, SAMPLE_RATE, format="wav")
    buffer.seek(0)
    
    return Response(content=buffer.read(), media_type="audio/wav")

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
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_out, SAMPLE_RATE, format="wav")
        buffer.seek(0)
        
        print(f"Resynth: complete, output {audio_out.shape[1]} samples")
        
        return Response(content=buffer.read(), media_type="audio/wav")
        
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

@app.get("/songs")
async def list_songs():
    """List available songs in /songs directory"""
    songs_dir = os.path.join(os.path.dirname(__file__), "songs")
    if not os.path.exists(songs_dir):
        return {"songs": []}
    
    audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
    songs = []
    for f in os.listdir(songs_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in audio_extensions:
            songs.append(f)
    
    songs.sort()
    return {"songs": songs}

@app.get("/songs/{filename:path}")
async def get_song(filename: str):
    """Serve a song file from /songs directory"""
    songs_dir = os.path.join(os.path.dirname(__file__), "songs")
    filepath = os.path.join(songs_dir, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Song not found")
    
    # Determine content type
    ext = os.path.splitext(filename)[1].lower()
    content_types = {
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.m4a': 'audio/mp4',
        '.aac': 'audio/aac'
    }
    content_type = content_types.get(ext, 'application/octet-stream')
    
    with open(filepath, 'rb') as f:
        content = f.read()
    
    return Response(content=content, media_type=content_type)

# ============ DiT GENERATION WITH HIDDEN STATE VIS ============

class GenerateRequest(BaseModel):
    prompt: str
    duration: float = 6.0  # seconds
    steps: int = 8  # denoising steps
    cfg_scale: float = 4.0

# Cache for layer projections from last generation
last_generation_cache = {
    "projections": {},  # layer_idx -> projection array
    "audio_base64": None,
    "duration": 0,
    "num_points": 0,
    "prompt": ""
}

@app.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    """
    Generate audio from prompt, extract DiT hidden states at ALL layers,
    Concatenate all layers and UMAP project together - each point tagged with layer + position
    """
    global dit_model, last_generation_cache
    
    if dit_model is None:
        load_dit()
    
    if dit_model is None:
        raise HTTPException(status_code=500, detail="DiT model not available")
    
    try:
        sample_size = int(request.duration * SAMPLE_RATE)
        downsample_ratio = dit_model.pretransform.downsampling_ratio if dit_model.pretransform else 2048
        sample_size = (sample_size // downsample_ratio) * downsample_ratio
        
        conditioning = [{
            "prompt": request.prompt,
            "seconds_start": 0,
            "seconds_total": request.duration
        }]
        
        # Hook ALL layers
        all_layer_hidden = {i: [] for i in range(16)}
        handles = []
        
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                all_layer_hidden[layer_idx].append(output.detach().cpu())
            return hook_fn
        
        transformer_layers = dit_model.model.model.transformer.layers
        for i in range(16):
            handle = transformer_layers[i].register_forward_hook(make_hook(i))
            handles.append(handle)
        
        print(f"Generating {request.duration}s audio with prompt: {request.prompt}")
        
        import random
        seed = random.randint(0, 2**31 - 1)
        
        with torch.no_grad():
            audio = dit_model.generate(
                conditioning=conditioning,
                steps=request.steps,
                cfg_scale=request.cfg_scale,
                sample_size=sample_size,
                batch_size=1,
                seed=seed,
                device=device,
                sampler_type="pingpong"
            )
        
        for handle in handles:
            handle.remove()
        
        expected_positions = sample_size // SAMPLES_PER_LATENT
        
        # Collect all hidden states into one big array for joint UMAP
        # Shape: [num_steps * 16 * num_positions, 1024]
        all_hidden = []
        point_metadata = []  # [{step, layer, position}, ...]

        print(f"Collecting hidden states from all layers and steps...")

        for layer_idx in range(16):
            captured = all_layer_hidden[layer_idx]
            if len(captured) == 0:
                continue

            # Process ALL captured steps, not just the last one
            for step_idx, step_hidden in enumerate(captured):
                if step_hidden.shape[0] > 1:
                    step_hidden = step_hidden[:step_hidden.shape[0]//2]

                actual_positions = step_hidden.shape[1]
                prepend_length = actual_positions - expected_positions
                if prepend_length > 0:
                    step_hidden = step_hidden[:, prepend_length:, :]

                hidden_np = step_hidden[0].numpy()  # [num_positions, 1024]

                for pos_idx in range(hidden_np.shape[0]):
                    all_hidden.append(hidden_np[pos_idx])
                    point_metadata.append({
                        "step": step_idx,
                        "layer": layer_idx,
                        "position": pos_idx
                    })
        
        all_hidden = np.array(all_hidden)  # [16 * num_positions, 1024]
        print(f"Total points for UMAP: {all_hidden.shape}")
        
        # Single UMAP on all layers together
        n_pts = all_hidden.shape[0]
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=min(50, n_pts - 1),
            min_dist=0.02,
            n_epochs=500,
            metric='cosine',
            spread=1.5,
            random_state=42
        )
        projection = reducer.fit_transform(all_hidden)
        
        # Normalize
        center = projection.mean(axis=0)
        projection_centered = projection - center
        max_abs = np.abs(projection_centered).max() + 1e-6
        projection_normalized = projection_centered / max_abs
        
        # Encode audio
        audio_cpu = audio[0].cpu()
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_cpu, SAMPLE_RATE, format="wav")
        audio_bytes = buffer.getvalue()
        audio_base64 = __import__('base64').b64encode(audio_bytes).decode('utf-8')
        
        duration = audio_cpu.shape[1] / SAMPLE_RATE
        num_steps = len(all_layer_hidden[0])  # Number of captures = number of steps

        return {
            "projection": projection_normalized.tolist(),  # [num_steps*16*num_pos, 3]
            "metadata": point_metadata,  # [{step, layer, position}, ...]
            "num_steps": num_steps,
            "num_layers": 16,
            "num_positions": expected_positions,
            "total_points": len(point_metadata),
            "hidden_dim": 1024,
            "duration_seconds": duration,
            "samples_per_latent": SAMPLES_PER_LATENT,
            "prompt": request.prompt,
            "audio_base64": audio_base64
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dit_health")
async def dit_health():
    """Check if DiT model is loaded"""
    return {
        "dit_loaded": dit_model is not None,
        "device": device,
        "config_path": DIT_CONFIG_PATH,
        "ckpt_path": DIT_CKPT_PATH
    }

# Serve static files (explorer.html)
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

