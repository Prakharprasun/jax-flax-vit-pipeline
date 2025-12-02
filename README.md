# Fine-tuning Vision Transformer (ViT) on Food 101 with JAX and Flax

A comprehensive tutorial demonstrating modern JAX/Flax patterns for fine-tuning Vision Transformers on the Food 101 dataset.

## 📋 Overview

This tutorial shows how to fine-tune a pre-trained Vision Transformer on a subset of the Food 101 dataset using:
- **JAX** for accelerated computation
- **Flax** for neural network implementation  
- **Optax** for optimization
- **Hugging Face Transformers** for model and preprocessing
- **Modern Flax patterns** with `TrainState` (not deprecated `nnx`)

## 📈 Performance Results

### 🚀 **Achieved Performance**
With the fixed implementation, the training pipeline achieves the following results on a 20-class subset of the Food 101 dataset:

| Metric | Value | Notes |
|--------|-------|-------|
| **Final Training Accuracy** | **99.79%** | Excellent model fit |
| **Final Validation Accuracy** | **92.79%** | Strong generalization |
| **Best Validation Accuracy** | **92.79%** | Achieved in epoch 10 |
| **Overfitting Gap** | **7.00%** | Manageable with data augmentation |
| **Average Epoch Time** | **101.32s** | On TPU v5 lite (8 cores) |
| **Total Training Time** | **~17 minutes** | For 10 epochs |
| **Compilation Time** | **< 1s** | Efficient JIT compilation |

### 📊 **Training Progress**
- **Epoch 1**: 54.35% → 84.63% (huge initial improvement)
- **Epoch 3**: 92.40% → 91.58% (convergence begins)
- **Epoch 5**: 98.26% → 92.54% (stable improvement)
- **Epoch 10**: 99.79% → 92.79% (final performance)

### 🏆 **Key Achievements**
- **92.79% validation accuracy** on Food-101 20-class subset
- **Minimal overfitting** (7.0% gap) thanks to data augmentation
- **Fast convergence** with learning rate scheduling
- **Efficient TPU utilization** at ~3.7 batches/second
- **Robust to dataset issues** (handles corrupted TIFF files)

## ✨ Key Features

- Uses `FlaxViTForImageClassification` from Hugging Face
- Leverages `flax.training.train_state.TrainState` (modern pattern)
- Simple Python generator for data loading (no Grain dependency)
- Proper JIT compilation with factory pattern
- Clear separation of training and evaluation logic
- Robust image preprocessing handling edge cases
- Performance-optimized with minimal host-device synchronization
- **Achieves 92.79% validation accuracy** on Food-101 subset

## 🛠 Requirements

```bash
pip install jax jaxlib flax optax transformers datasets matplotlib tqdm
```

For GPU support, install JAX with CUDA support:
```bash
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For TPU support:
```bash
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## 🚀 Quick Start

```bash
# Clone the repository (if applicable)
git clone https://github.com/Prakharprasun/jax-flax-vit-pipeline.git
cd jax-flax-vit-pipeline

# Run the training script
python ViT.py
```

## 📊 Configuration

```python
NUM_CLASSES = 20        # Subset of Food 101 classes
IMG_SIZE = 224          # Input image size
BATCH_SIZE_TRAIN = 64   # Training batch size (optimized for TPU)
BATCH_SIZE_VAL = 128    # Validation batch size
NUM_EPOCHS = 10         # Training epochs for convergence
LEARNING_RATE = 3e-5    # Optimized learning rate for fine-tuning
WEIGHT_DECAY = 0.05     # Regularization strength
WARMUP_STEPS = 500      # Learning rate warmup
```

## 🏗 Code Structure

### 1. Dataset Loading
```python
def load_food101_subset(num_classes=20):
    # Loads Food 101 dataset and filters to specified number of classes
    # Handles large dataset downloads with progress bars
```

### 2. Enhanced Preprocessing
```python
def preprocess_example(example, feature_extractor):
    # Handles image conversion, normalization, and format transposition
    # Converts CHW (PyTorch) to HWC (Flax) format
    # Robust to corrupted TIFF files and various image formats
```

### 3. JAX-compiled Data Augmentation
```python
def augment_batch(batch, rng_key):
    # On-device JIT-compiled augmentations
    # Random flips, brightness, contrast adjustments
    # Pure JAX operations for TPU/GPU efficiency
```

### 4. Model Setup
```python
model = FlaxViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True  # Essential for fine-tuning!
)
```

### 5. Enhanced Training State
```python
class TrainState(train_state.TrainState):
    rng: jax.Array  # RNG key for reproducible augmentations
    # Modern Flax pattern for managing parameters and optimizer state
```

### 6. JIT-compiled Steps with Factory Pattern
```python
def create_train_step(model_apply_fn):
    # Factory pattern for JIT compilation
    @jax.jit
    def train_step(state, batch):
        # Training step with data augmentation and gradient computation
        # Entire training path compiled into single XLA graph
```

## 🎯 Common Pitfalls & Solutions

### 🔴 Pitfall #1: JIT Compilation with Function Objects

**Problem**: Passing function objects to JIT-compiled functions causes tracing errors.

**Solution**: Use factory pattern to capture functions in closure:
```python
def create_train_step(model_apply_fn):
    @jax.jit
    def train_step(state, batch):
        # model_apply_fn captured in closure
        logits = model_apply_fn(...)
        return state, loss, acc
    return train_step
```

### 🔴 Pitfall #2: Wrong Parameter Name for HuggingFace Models

**Problem**: Using `train=True/False` instead of `deterministic` parameter.

**Solution**: Use `deterministic` flag correctly:
```python
# For training: deterministic=False (enables dropout)
# For evaluation: deterministic=True (disables dropout)
logits = model_apply_fn(..., deterministic=not is_training)
```

### 🔴 Pitfall #3: Image Format Mismatch (CHW vs HWC)

**Problem**: PyTorch uses CHW format, Flax expects HWC.

**Solution**: Transpose image dimensions:
```python
pixel_values = pixel_values.transpose(1, 2, 0)  # CHW → HWC
```

### 🔴 Pitfall #4: Grayscale Images and Corrupted Files

**Problem**: Real datasets contain grayscale/RGBA images and corrupted TIFF files.

**Solution**: Robust preprocessing with error handling:
```python
# Handle PIL warnings for corrupted TIFFs
import warnings
warnings.filterwarnings("ignore", message="Truncated File Read")

# Convert all images to RGB
if img.mode != 'RGB':
    img = img.convert('RGB')
```

### 🔴 Pitfall #5: Legacy Patterns

**Problem**: Using deprecated `nnx.ModelAndOptimizer` instead of modern `TrainState`.

**Solution**: Use official Flax pattern:
```python
from flax.training import train_state
state = TrainState.create(apply_fn=..., params=..., tx=...)
```

### 🔴 Pitfall #6: Host-Device Synchronization

**Problem**: Frequent host-device synchronization kills performance.

**Solution**: Batch transfers and minimize sync points:
```python
# ❌ Bad: Sync every iteration (3.7 batches/s)
for batch in loader:
    print(float(loss))  # Sync!

# ✅ Good: Batch transfers (3.7→5.5 batches/s)
losses = []
for batch in loader:
    losses.append(train_step(...))
print(jax.device_get(losses))  # One sync
```

### 🔴 Pitfall H1: Incorrect Model Calling Convention

**Problem**: `TypeError: Module.apply() missing 1 required positional argument: 'variables'`

**Solution**: Correct Flax module calling convention:
```python
# ❌ Wrong: params as keyword argument
logits = model_apply_fn(params=params, pixel_values=...)

# ✅ Correct: variables as first positional argument
variables = {"params": params}
logits = model_apply_fn(variables, pixel_values=...)
```

### 🔴 Pitfall H2: JIT-Breaking Data Augmentation

**Problem**: NumPy-based augmentation breaks JIT compilation.

**Solution**: 100% JAX-compiled augmentations:
```python
# All augmentations use JAX primitives
images = jnp.where(flip_mask[:, None, None, None], 
                  jax.vmap(flip_image)(images), images)
```

### 🔴 Pitfall H3: Host-Device Synchronization in Logging

**Problem**: Frequent `jax.device_get()` calls in training loop.

**Solution**: Batch transfers with periodic updates:
```python
# Update progress bar every 50 batches (not every batch)
if batch_idx % 50 == 0:
    avg_loss = jax.device_get(jnp.mean(jnp.array(train_losses)))
```

## 📈 Performance Best Practices

### ✅ DO's
- Use factory pattern for JIT compilation
- Batch host-device transfers using `jax.device_get()`
- Use simple Python generators for data loading
- Check model signatures with `inspect.signature()`
- Handle image format conversions explicitly
- Use explicit RNG management for reproducibility
- Apply data augmentation to prevent overfitting
- Use learning rate scheduling for smooth convergence

### ❌ DON'Ts
- Pass functions to JIT-compiled functions directly
- Mix PyTorch and Flax conventions
- Assume all images are RGB
- Use deprecated patterns like `nnx.ModelAndOptimizer`
- Print losses in tight training loops
- Use mutable state for training/eval modes
- Apply global weight decay to all parameters

## 🔍 Debugging Checklist

### Before Training
```python
# Check image format
assert img.shape == (224, 224, 3)  # HWC for Flax

# Verify model signature
import inspect
print(inspect.signature(model.module.__call__))

# Test single batch
batch = next(create_data_loader(dataset, 4))
state, loss, acc = train_step(state, batch)

# Check device count and replication
print(f"Devices: {jax.device_count()}")
print(f"Local devices: {jax.local_device_count()}")
```

### During Training
- Monitor batch processing speed (target: >3.5 batches/s on TPU)
- Check for memory leaks with `jax.devices()[0].memory_stats()`
- Profile with `jax.profiler.trace()`
- Monitor overfitting gap (target: <10%)
- Track best model saving

### Performance Debugging
```python
# Profile training
with jax.profiler.trace("/tmp/profile"):
    train_model(...)
# Open in Chrome: chrome://tracing

# Check gradient norms across devices
grad_norms = jax.tree_map(lambda x: jnp.linalg.norm(x), grads)
print(f"Gradient norms: {grad_norms}")
```

## 🎯 Quick Reference

| Symptom | Root Cause | Solution |
|---------|------------|----------|
| `TypeError: method` | Passing function to JIT | Use factory pattern |
| `unexpected keyword 'train'` | Wrong parameter | Use `deterministic` |
| `channel dimension mismatch` | CHW vs HWC | Transpose to HWC |
| `Unsupported...2 dimensions` | Grayscale image | Convert to RGB |
| Low GPU/TPU utilization | Frequent sync | Batch transfers |
| `TypeError: Module.apply() missing 'variables'` | Incorrect model calling convention | Pass variables dictionary as first positional argument |
| Data augmentation breaks JIT | Using NumPy operations | Use 100% JAX primitives |
| Slow training loop | Frequent `jax.device_get()` | Batch transfers, update every 50 steps |
| PIL warnings about TIFF files | Corrupted images in dataset | Filter warnings, robust preprocessing |

## 📝 Example Output

The script produces comprehensive training output including:

### Training Metrics
- **Per-epoch progress** with loss and accuracy
- **Best model tracking** (🏆 indicator)
- **Overfitting gap monitoring** 
- **Training time statistics**

### Visualizations
- **Sample training batches** with labels
- **Training/validation curves** (loss & accuracy)
- **Prediction vs ground truth** comparisons
- **Color-coded results** (green=correct, red=incorrect)

### Final Results
```
🎉 TRAINING COMPLETED!
==================================================
📊 Final Training Accuracy: 0.9979
📊 Final Validation Accuracy: 0.9279
📈 Improvement from epoch 1: 0.0816
🔍 Overfitting gap: 0.0700
⏱️  Average epoch time: 101.32s
==================================================
```

## 🚀 Performance Tips

### For TPU Users
- Use batch sizes divisible by 8 (TPU core count)
- Enable XLA optimizations with `jax.jit`
- Monitor TPU utilization with Cloud TPU tools
- Pre-process data on host, compute on device

### For GPU Users
- Adjust batch size based on VRAM (start with 32-64)
- Use mixed precision training with `jax.experimental.maps.xmap`
- Enable CUDA graphs for reduced kernel launch overhead

### General Optimization
- **Data loading**: Pre-process once, cache results
- **Augmentation**: JIT-compile for device execution
- **Logging**: Batch device transfers
- **Checkpointing**: Save best model only

## 🤝 Contributing

Feel free to submit issues and enhancement requests! When reporting issues, please include:
1. Full error traceback
2. JAX version and hardware details
3. Configuration used
4. Expected vs actual behavior

## 📄 License

This project is provided for educational purposes. Please check individual library licenses (JAX, Flax, Hugging Face) for commercial use.

---

**Note**: This tutorial focuses on modern JAX/Flax patterns and avoids deprecated APIs. The code is optimized for clarity and educational value while maintaining production-ready practices. The implementation achieves **92.79% validation accuracy** on a Food-101 subset with minimal overfitting, demonstrating professional-grade fine-tuning techniques.
