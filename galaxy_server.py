"""
Galaxy Explorer - DiT Steering Vector Server

Generates audio, computes steering vectors, maps the generative manifold.

Environment Variables:
  DIT_CONFIG_PATH  - Path to DiT config JSON
  DIT_CKPT_PATH    - Path to DiT checkpoint
  VAE_CONFIG_PATH  - Path to VAE config JSON
  VAE_CKPT_PATH    - Path to VAE checkpoint
  PORT             - Server port (default: 8421)
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
import base64
from typing import List, Optional, Dict
import umap
from sklearn.cluster import KMeans
from collections import Counter

# ============ CONFIG ============
DIT_CONFIG_PATH = os.environ.get("DIT_CONFIG_PATH", "model_config.json")
DIT_CKPT_PATH = os.environ.get("DIT_CKPT_PATH", "model.safetensors")
VAE_CONFIG_PATH = os.environ.get("VAE_CONFIG_PATH", "stable_audio_2_0_vae.json")
VAE_CKPT_PATH = os.environ.get("VAE_CKPT_PATH", "sao_vae_tune_100k_unwrapped.ckpt")
PORT = int(os.environ.get("PORT", "8421"))

SAMPLE_RATE = 44100
LATENTS_PER_SECOND = 21.5

# Global models
dit = None
vae = None
t5 = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Steering vector storage
steering_vectors = {}
prior_center = None
prior_samples = []

# All samples storage (prior + tags combined)
all_samples = []
all_metadata = []

# Galaxy data cache
galaxy_cache = {
    "prior_projection": None,
    "tag_projections": {},
    "regions": None
}

app = FastAPI(title="Galaxy Explorer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ MODEL LOADING ============

def load_dit():
    """Load DiT + T5 for generation"""
    global dit, t5
    if dit is not None:
        return

    try:
        from stable_audio_tools.models.factory import create_model_from_config
        from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict

        if not os.path.exists(DIT_CONFIG_PATH):
            raise FileNotFoundError(f"DiT config not found: {DIT_CONFIG_PATH}")
        if not os.path.exists(DIT_CKPT_PATH):
            raise FileNotFoundError(f"DiT checkpoint not found: {DIT_CKPT_PATH}")

        config = json.load(open(DIT_CONFIG_PATH))
        dit = create_model_from_config(config)
        copy_state_dict(dit, load_ckpt_state_dict(DIT_CKPT_PATH))
        dit.to(device).eval().requires_grad_(False)

        # T5 is part of the model
        t5 = dit  # Adjust based on actual architecture

        print(f"DiT loaded on {device}")
    except Exception as e:
        print(f"ERROR loading DiT: {e}")
        raise

def unload_dit():
    global dit, t5
    if dit is not None:
        del dit, t5
        dit, t5 = None, None
        torch.cuda.empty_cache()
        gc.collect()
        print("DiT unloaded")

def load_vae():
    """Load VAE for decode"""
    global vae
    if vae is not None:
        return

    try:
        from stable_audio_tools.models.factory import create_model_from_config
        from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict

        config = json.load(open(VAE_CONFIG_PATH))
        vae = create_model_from_config(config)
        copy_state_dict(vae, load_ckpt_state_dict(VAE_CKPT_PATH))
        vae.to(device).eval().requires_grad_(False)

        print(f"VAE loaded on {device}")
    except Exception as e:
        print(f"ERROR loading VAE: {e}")
        raise

def unload_vae():
    global vae
    if vae is not None:
        del vae
        vae = None
        torch.cuda.empty_cache()
        gc.collect()

# ============ GENERATION HELPERS ============

def generate_all_steps(prompt: str, seed: int, duration: float = 6.0, layer: int = 15, steps: int = 8):
    """
    Generate and extract hidden states from ALL diffusion steps.
    Returns list of pooled hidden state vectors: [[1024], [1024], ...]
    """
    load_dit()

    sample_size = int(duration * SAMPLE_RATE)
    downsample_ratio = dit.pretransform.downsampling_ratio if dit.pretransform else 2048
    sample_size = (sample_size // downsample_ratio) * downsample_ratio

    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": duration
    }]

    # Hook to capture hidden states at target layer
    captured_hidden = []

    def hook_fn(module, input, output):
        captured_hidden.append(output.detach().cpu())

    transformer_layers = dit.model.model.transformer.layers
    handle = transformer_layers[layer].register_forward_hook(hook_fn)

    torch.manual_seed(seed)

    with torch.no_grad():
        audio = dit.generate(
            conditioning=conditioning,
            steps=steps,
            cfg_scale=4.0,
            sample_size=sample_size,
            batch_size=1,
            seed=seed,
            device=device,
            sampler_type="pingpong"
        )

    handle.remove()

    # Pool all captured steps: [num_steps, 1024]
    pooled_steps = []
    for step_hidden in captured_hidden:
        # Handle CFG doubling
        if step_hidden.shape[0] > 1:
            step_hidden = step_hidden[:step_hidden.shape[0]//2]

        # Pool: [1, seq, 1024] -> [1024]
        pooled = step_hidden[0].mean(dim=0)
        pooled_steps.append(pooled)

    return pooled_steps

def generate_to_step(prompt: str, seed: int, target_step: int = 4, duration: float = 6.0, layer: int = 15):
    """
    Generate up to target_step and extract hidden state from specified layer.
    Returns pooled hidden state vector [1024].
    """
    all_steps = generate_all_steps(prompt, seed, duration, layer)
    if target_step < len(all_steps):
        return all_steps[target_step]
    return all_steps[-1]

# ============ ENDPOINTS ============

class ComputePriorRequest(BaseModel):
    n_samples: int = 1000
    duration: float = 6.0
    batch_size: int = 10
    umap_update_interval: int = 50  # Update UMAP every N samples

@app.post("/compute_prior")
async def compute_prior(req: ComputePriorRequest):
    """Generate N unconditional samples to define the prior, stream UMAP updates"""
    global prior_center, prior_samples
    import asyncio

    async def generate():
        global prior_center, prior_samples, all_samples, all_metadata

        try:
            yield f"data: {json.dumps({'stage': 'loading_dit'})}\n\n"
            await asyncio.sleep(0)  # Force flush

            load_dit()

            samples = []
            all_projections = []

            for batch_start in range(0, req.n_samples, req.batch_size):
                batch_end = min(batch_start + req.batch_size, req.n_samples)
                batch_seeds = list(range(batch_start, batch_end))

                # Generate batch with per-sample progress
                for seed_idx, seed in enumerate(batch_seeds):
                    # Get all 8 steps for this generation
                    all_step_hiddens = generate_all_steps("", seed=seed, duration=req.duration)

                    # Add all steps to samples with metadata
                    for step_idx, step_hidden in enumerate(all_step_hiddens):
                        samples.append(step_hidden)
                        # Store metadata inline for progressive UMAP
                        all_metadata.append({
                            "type": "prior",
                            "tag": "",
                            "seed": seed,
                            "step": step_idx,
                            "gen_id": seed  # For grouping trajectory lines
                        })

                    # Update progress (count generations, not individual steps)
                    n_gens_done = batch_start + seed_idx + 1
                    yield f"data: {json.dumps({'stage': 'generating', 'progress': n_gens_done, 'total': req.n_samples})}\n\n"
                    await asyncio.sleep(0)  # Force flush

                # Progressive UMAP update
                if len(samples) >= 10 and len(samples) % req.umap_update_interval == 0:
                    yield f"data: {json.dumps({'stage': 'umap_update', 'n_samples': len(samples)})}\n\n"
                    await asyncio.sleep(0)

                    samples_np = torch.stack(samples).numpy()

                    if len(samples) < 10:
                        projection = samples_np[:, :3]
                    else:
                        reducer = umap.UMAP(
                            n_components=3,
                            n_neighbors=min(15, len(samples) - 1),
                            min_dist=0.05,
                            metric='euclidean',
                            random_state=42
                        )
                        projection = reducer.fit_transform(samples_np)

                    # Normalize
                    center = projection.mean(axis=0)
                    projection = projection - center
                    scale = np.abs(projection).max() + 1e-6
                    projection = projection / scale

                    all_projections = projection

                    # Store for fetching, don't send in SSE
                    galaxy_cache["projection"] = projection.tolist()
                    galaxy_cache["metadata"] = all_metadata.copy()

                    yield f"data: {json.dumps({'stage': 'projection', 'n_samples': len(samples)})}\n\n"
                    await asyncio.sleep(0)

            # Final
            prior_samples = torch.stack(samples)  # [n * 8, 1024] where n = n_samples
            prior_center = prior_samples.mean(dim=0)  # [1024]

            # Initialize all_samples with prior (all_metadata already built incrementally)
            all_samples = samples.copy()

            # Final UMAP
            yield f"data: {json.dumps({'stage': 'final_umap'})}\n\n"
            await asyncio.sleep(0)
            samples_np = prior_samples.numpy()

            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=min(30, len(samples) - 1),
                min_dist=0.05,
                metric='euclidean',
                random_state=42
            )
            projection = reducer.fit_transform(samples_np)

            center = projection.mean(axis=0)
            projection = projection - center
            scale = np.abs(projection).max() + 1e-6
            projection = projection / scale

            # Store for fetching, don't send in SSE
            galaxy_cache["prior_projection"] = projection
            galaxy_cache["projection"] = projection.tolist()
            galaxy_cache["metadata"] = all_metadata

            yield f"data: {json.dumps({'stage': 'done', 'n_samples': req.n_samples})}\n\n"
            await asyncio.sleep(0)

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'stage': 'error', 'error': str(e)})}\n\n"
            await asyncio.sleep(0)

    return StreamingResponse(generate(), media_type="text/event-stream")

class ComputeSteeringRequest(BaseModel):
    tag: str
    n_samples: int = 10
    duration: float = 6.0

@app.post("/compute_steering")
async def compute_steering(req: ComputeSteeringRequest):
    """Compute steering vector for a tag"""
    global steering_vectors, prior_center

    if prior_center is None:
        raise HTTPException(400, "Compute prior first via /compute_prior")

    async def generate():
        try:
            yield f"data: {json.dumps({'stage': 'loading_dit'})}\n\n"
            load_dit()

            samples = []
            for i in range(req.n_samples):
                yield f"data: {json.dumps({'stage': 'generating', 'progress': i, 'total': req.n_samples})}\n\n"

                hidden = generate_to_step(req.tag, seed=i, target_step=4, duration=req.duration)
                samples.append(hidden)

            tag_center = torch.stack(samples).mean(dim=0)
            steering = tag_center - prior_center

            steering_vectors[req.tag] = {
                "vector": steering,
                "magnitude": float(steering.norm().item()),
                "center": tag_center
            }

            result = {
                'stage': 'done',
                'tag': req.tag,
                'magnitude': float(steering.norm().item())
            }
            yield f"data: {json.dumps(result)}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'stage': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

class BuildGalaxyRequest(BaseModel):
    tags: List[str]
    samples_per_tag: int = 5
    n_regions: int = 20

@app.post("/build_galaxy")
async def build_galaxy(req: BuildGalaxyRequest):
    """
    Build the full galaxy: prior + tags, cluster into regions, UMAP
    """
    global galaxy_cache, prior_samples, steering_vectors

    if prior_center is None:
        raise HTTPException(400, "Compute prior first")

    async def generate():
        try:
            # Collect all embeddings
            all_embeddings = []
            all_metadata = []

            # Prior
            yield f"data: {json.dumps({'stage': 'processing_prior'})}\n\n"
            for i, emb in enumerate(prior_samples):
                all_embeddings.append(emb.numpy())
                all_metadata.append({"type": "prior", "tag": "", "index": i})

            # Tags
            for tag_idx, tag in enumerate(req.tags):
                yield f"data: {json.dumps({'stage': 'computing_tag', 'tag': tag, 'progress': tag_idx, 'total': len(req.tags)})}\n\n"

                if tag not in steering_vectors:
                    # Compute it
                    load_dit()
                    samples = []
                    for i in range(req.samples_per_tag):
                        hidden = generate_to_step(tag, seed=i, target_step=4)
                        samples.append(hidden)

                    tag_center = torch.stack(samples).mean(dim=0)
                    steering = tag_center - prior_center
                    steering_vectors[tag] = {
                        "vector": steering,
                        "magnitude": float(steering.norm().item()),
                        "center": tag_center
                    }

                # Add to embeddings
                tag_center = steering_vectors[tag]["center"]
                all_embeddings.append(tag_center.numpy())
                all_metadata.append({"type": "tag", "tag": tag, "index": len(all_embeddings) - 1})

            # UMAP
            yield f"data: {json.dumps({'stage': 'umap', 'n_points': len(all_embeddings)})}\n\n"
            all_embeddings = np.stack(all_embeddings)

            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=min(30, len(all_embeddings) - 1),
                min_dist=0.05,
                metric='euclidean',
                random_state=42
            )
            projection = reducer.fit_transform(all_embeddings)

            # Normalize
            center = projection.mean(axis=0)
            projection = projection - center
            scale = np.abs(projection).max() + 1e-6
            projection = projection / scale

            # Cluster
            yield f"data: {json.dumps({'stage': 'clustering'})}\n\n"
            tag_points = [i for i, m in enumerate(all_metadata) if m["type"] == "tag"]
            tag_embeddings = all_embeddings[tag_points]
            tag_projection = projection[tag_points]

            kmeans = KMeans(n_clusters=min(req.n_regions, len(tag_points)))
            clusters = kmeans.fit_predict(tag_embeddings)

            # Build regions
            regions = []
            for c in range(kmeans.n_clusters):
                indices = np.where(clusters == c)[0]
                region_tags = [all_metadata[tag_points[i]]["tag"] for i in indices]
                most_common = Counter(region_tags).most_common(3)
                label = " + ".join([t for t, _ in most_common])

                center = tag_projection[indices].mean(axis=0)
                radius = np.linalg.norm(tag_projection[indices] - center, axis=1).max()

                regions.append({
                    "id": int(c),
                    "label": label,
                    "center": center.tolist(),
                    "radius": float(radius),
                    "tags": region_tags,
                    "count": len(indices)
                })

            # Cache
            galaxy_cache["prior_projection"] = projection[:len(prior_samples)]
            galaxy_cache["regions"] = regions

            # Return
            result = {
                "stage": "done",
                "projection": projection.tolist(),
                "metadata": all_metadata,
                "regions": regions,
                "n_prior": len(prior_samples),
                "n_tags": len(req.tags)
            }

            yield f"data: {json.dumps(result)}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'stage': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

class SteerRequest(BaseModel):
    components: Dict[str, float]
    prompt: str = ""
    duration: float = 6.0
    seed: int = 0
    injection_layer: int = 8
    injection_steps: List[int] = [3, 4, 5]

class GeneratePointRequest(BaseModel):
    point_type: str  # "prior" or "tag"
    tag: str = ""    # if type=tag
    seed: int = 0    # if type=prior
    duration: float = 6.0

@app.post("/generate_point")
async def generate_point(req: GeneratePointRequest):
    """Generate audio for a clicked point"""
    global dit, vae

    try:
        load_dit()

        sample_size = int(req.duration * SAMPLE_RATE)
        downsample_ratio = dit.pretransform.downsampling_ratio if dit.pretransform else 2048
        sample_size = (sample_size // downsample_ratio) * downsample_ratio

        # Setup conditioning
        if req.point_type == "prior":
            prompt = ""
            seed = req.seed
        else:  # tag
            prompt = req.tag
            seed = req.seed

        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": req.duration
        }]

        # Generate
        torch.manual_seed(seed)

        with torch.no_grad():
            audio = dit.generate(
                conditioning=conditioning,
                steps=8,
                cfg_scale=4.0,
                sample_size=sample_size,
                batch_size=1,
                seed=seed,
                device=device,
                sampler_type="pingpong"
            )

        # Encode to wav
        audio_cpu = audio[0].cpu()
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_cpu, SAMPLE_RATE, format="wav")
        audio_bytes = buffer.getvalue()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        return {
            "audio": audio_b64,
            "type": req.point_type,
            "tag": req.tag if req.point_type == "tag" else "",
            "seed": seed
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_steered")
async def generate_steered(req: SteerRequest):
    """Generate audio with steering vector"""
    global dit, vae, steering_vectors

    try:
        # Combine steering vectors
        combined_steering = torch.zeros(1024)
        for tag, weight in req.components.items():
            if tag not in steering_vectors:
                raise HTTPException(400, f"Unknown steering vector: {tag}")
            combined_steering += weight * steering_vectors[tag]["vector"]

        combined_steering = combined_steering.to(device)

        load_dit()

        sample_size = int(req.duration * SAMPLE_RATE)
        downsample_ratio = dit.pretransform.downsampling_ratio if dit.pretransform else 2048
        sample_size = (sample_size // downsample_ratio) * downsample_ratio

        conditioning = [{
            "prompt": req.prompt,
            "seconds_start": 0,
            "seconds_total": req.duration
        }]

        # Hook to inject steering at specified layer
        def steering_hook(module, input, output):
            # output: [batch, seq, 1024]
            return output + combined_steering.unsqueeze(0).unsqueeze(0)

        torch.manual_seed(req.seed)

        transformer_layers = dit.model.model.transformer.layers
        handles = []

        # Inject at specified layer
        handle = transformer_layers[req.injection_layer].register_forward_hook(steering_hook)
        handles.append(handle)

        with torch.no_grad():
            audio = dit.generate(
                conditioning=conditioning,
                steps=8,
                cfg_scale=4.0,
                sample_size=sample_size,
                batch_size=1,
                seed=req.seed,
                device=device,
                sampler_type="pingpong"
            )

        for h in handles:
            h.remove()

        # Encode
        audio_cpu = audio[0].cpu()
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_cpu, SAMPLE_RATE, format="wav")
        audio_bytes = buffer.getvalue()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        return {
            "audio": audio_b64,
            "components": req.components,
            "steering_magnitude": float(combined_steering.norm().item())
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class AddTagSamplesRequest(BaseModel):
    tag: str
    n_samples: int
    duration: float = 6.0

@app.post("/add_tag_samples")
async def add_tag_samples(req: AddTagSamplesRequest):
    """Add N samples of a tag to existing galaxy and re-run UMAP"""
    global all_samples, all_metadata, galaxy_cache
    import asyncio

    async def generate():
        global all_samples, all_metadata

        try:
            yield f"data: {json.dumps({'stage': 'loading_dit'})}\n\n"
            await asyncio.sleep(0)
            load_dit()

            # Generate tag samples (all 8 steps per generation)
            tag_samples = []
            tag_metadata = []
            for gen_idx in range(req.n_samples):
                # Get all 8 steps for this tag generation
                all_step_hiddens = generate_all_steps(req.tag, seed=gen_idx, duration=req.duration)

                # Add all steps
                for step_idx, step_hidden in enumerate(all_step_hiddens):
                    tag_samples.append(step_hidden)
                    tag_metadata.append({
                        "type": "tag",
                        "tag": req.tag,
                        "seed": gen_idx,
                        "step": step_idx,
                        "gen_id": f"{req.tag}_{gen_idx}"  # Unique ID for trajectory grouping
                    })

                # Progress counts generations, not steps
                yield f"data: {json.dumps({'stage': 'generating', 'progress': gen_idx + 1, 'total': req.n_samples})}\n\n"
                await asyncio.sleep(0)

            # Add to existing samples (additive!)
            yield f"data: {json.dumps({'stage': 'umap'})}\n\n"
            await asyncio.sleep(0)

            # Append new samples to all_samples
            all_samples.extend(tag_samples)
            all_metadata.extend(tag_metadata)

            # Convert to numpy for UMAP
            all_samples_tensor = torch.stack(all_samples)
            all_samples_np = all_samples_tensor.numpy()

            # Run UMAP on combined data
            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=min(30, len(all_samples) - 1),
                min_dist=0.05,
                metric='euclidean',
                random_state=42
            )
            projection = reducer.fit_transform(all_samples_np)

            # Normalize
            center = projection.mean(axis=0)
            projection = projection - center
            scale = np.abs(projection).max() + 1e-6
            projection = projection / scale

            # Store globally, don't send in SSE (too large)
            galaxy_cache["projection"] = projection.tolist()
            galaxy_cache["metadata"] = all_metadata

            yield f"data: {json.dumps({'stage': 'done', 'n_points': len(all_samples)})}\n\n"
            await asyncio.sleep(0)

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'stage': 'error', 'error': str(e)})}\n\n"
            await asyncio.sleep(0)

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/galaxy_data")
async def get_galaxy_data():
    """Fetch current galaxy projection + metadata"""
    return {
        "projection": galaxy_cache.get("projection", []),
        "metadata": galaxy_cache.get("metadata", [])
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "dit_loaded": dit is not None,
        "vae_loaded": vae is not None,
        "device": device,
        "prior_computed": prior_center is not None,
        "n_steering_vectors": len(steering_vectors),
        "galaxy_built": galaxy_cache["regions"] is not None
    }

# Serve galaxy.html
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
