import torch
import os

from dotenv import load_dotenv
load_dotenv()

def setup_rocm_optimizations():
    """Enhanced ROCm setup with precision detection"""
    print("Loading ROCm environment variables...")
    
    rocm_vars = [
        'TORCH_BLAS_PREFER_HIPBLASLT',
        'HIP_FORCE_DEV_KERNARG', 
        'PYTORCH_MIOPEN_SUGGEST_NHWC',
        'TORCHINDUCTOR_MAX_AUTOTUNE',
        'TORCHINDUCTOR_FREEZING',
        'TORCHINDUCTOR_CPP_WRAPPER'
    ]
    
    for var in rocm_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var} = {value}")
        else:
            print(f"❌ {var} not found")

    torch._inductor.config.max_autotune = True
    torch._inductor.config.max_autotune_gemm = True
    torch._inductor.config.freezing = True
    torch._inductor.config.cpp_wrapper = False
    
    print("✅ TorchInductor configured for ROCm")
    
    check_precision_support()

def check_precision_support():
    """Check what precision formats are supported"""
    print("\n🛠️ Checking precision support...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"FP16 support: ✅")
        
        try:
            bf16_support = torch.cuda.is_bf16_supported()
            print(f"BF16 support: {'✅' if bf16_support else '❌'}")
        except:
            print("BF16 support: ❌ (Not available)")
            bf16_support = False
        
        try:
            major, minor = torch.cuda.get_device_capability()
            tensor_core = major >= 7
            print(f"Tensor Core support: {'✅' if tensor_core else '❌'}")
        except:
            print("Tensor Core support: ❌")
    else:
        print("No CUDA device available")

def setup_precision(precision='bf16', set_default=True):
    """
    Setup precision for training with full PyTorch integration.
    
    Args:
        precision: 'fp32', 'fp16', 'bf16', 'mixed'
        set_default: Whether to set as PyTorch default dtype
    
    Returns:
        tuple: (dtype, scaler, autocast_enabled)
    """
    print(f"\n🛠️ Setting up {precision.upper()} precision...")
    
    precision_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    
    if precision == 'fp32':
        dtype = precision_map['fp32']
        scaler = None
        autocast_enabled = False
        print("✅ Using FP32 (full precision)")
        
    elif precision == 'fp16':
        dtype = precision_map['fp16']
        scaler = torch.amp.GradScaler('cuda')
        autocast_enabled = True
        print("✅ Using FP16 with gradient scaling")
        
    elif precision == 'bf16':
        try:
            device_name = torch.cuda.get_device_name()
            print(f"Detected GPU: {device_name}")
            
            a = torch.randn(10, 10, dtype=torch.bfloat16, device='cuda')
            b = torch.randn(10, 10, dtype=torch.bfloat16, device='cuda')
            c = torch.matmul(a, b) 
            
            dtype = torch.bfloat16
            scaler = None
            autocast_enabled = True
            
            del a,b,c
            
            print("✅ Using BF16 (recommended for AMD)")
        except Exception as e:
            print(f"⚠️ BF16 not supported ({e}), falling back to FP16")
            dtype = precision_map['fp16']
            scaler = torch.amp.GradScaler('cuda')
            autocast_enabled = True
            
    elif precision == 'mixed':
        dtype = None
        scaler = torch.amp.GradScaler('cuda')
        autocast_enabled = True
        print("✅ Using Automatic Mixed Precision")
        
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    
    if set_default and dtype is not None:
        torch.set_default_dtype(dtype)
        print(f"✅ Set PyTorch default dtype to {dtype}")
        
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        print("✅ Set default device to CUDA")
    
    return dtype, scaler, autocast_enabled


def compile_model_amd(model, approach='mode', precision='bf16', set_precision=True):
    """
    Enhanced model compilation with precision handling.
    
    Args:
        model: The model to compile
        approach: 'mode', 'options', 'full', 'reduce-overhead'
        precision: 'fp32', 'fp16', 'bf16', 'mixed'
        set_precision: Whether to convert model to specified precision
    
    Returns:
        tuple: (compiled_model, scaler, autocast_enabled)
    """
    
    dtype, scaler, autocast_enabled = setup_precision(precision, set_default=True)
    
    if set_precision and dtype is not None:
        print(f"🛠️ Converting model to {dtype}")
        model = model.to(dtype)
    
    if approach == 'mode':
        compiled_model = torch.compile(
            model,
            backend='inductor',
            mode='max-autotune'
        )
        print("✅ Model compiled with max-autotune mode")
        
    elif approach == 'options':
        compiled_model = torch.compile(
            model,
            backend='inductor',
            options={
                "epilogue_fusion": True,
                "max_autotune": True,
                "shape_padding": True,
                "trace.enabled": False,
                "trace.graph_diagram": False
            }
        )
        print("✅ Model compiled with custom options")
        
    elif approach == 'full':
        try:
            compiled_model = torch.compile(
                model,
                backend='inductor',
                mode='max-autotune',
                fullgraph=True
            )
            print("✅ Model compiled with fullgraph=True")
        except Exception as e:
            print(f"❌ Fullgraph compilation failed: {e}")
            print("Falling back to default compilation...")
            compiled_model = torch.compile(model, backend='inductor', mode='max-autotune')
            print("✅ Model compiled with `mode` option : max auto tuned")
        
    elif approach == 'reduce-overhead':
        compiled_model = torch.compile(
            model,
            backend='inductor',
            mode='reduce-overhead'
        )
        print("✅ Model compiled with reduce-overhead mode")
        
    else:
        compiled_model = torch.compile(
            model,
            backend='inductor'
        )
        print("✅ Model compiled with default settings")
    
    return compiled_model, scaler, autocast_enabled

def setup_complete_amd_environment(precision='bf16', compile_approach='mode'):
    """
    Complete setup function that configures everything at once.
    
    Args:
        precision: 'fp32', 'fp16', 'bf16', 'mixed'
        compile_approach: 'mode', 'options', 'full', 'reduce-overhead'
    
    Returns:
        dict: Configuration information
    """
    print("🐣 Setting up complete AMD ROCm environment...")
    
    # Setup ROCm optimizations
    setup_rocm_optimizations()
    
    # Setup precision
    dtype, scaler, autocast_enabled = setup_precision(precision)
    
    config = {
        'precision': precision,
        'dtype': dtype,
        'scaler': scaler,
        'autocast_enabled': autocast_enabled,
        'compile_approach': compile_approach,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    print("\n📋 Environment Configuration:")
    print(f"\tPrecision: {precision}")
    print(f"\tData type: {dtype}")
    print(f"\tGradient scaler: {'Yes' if scaler else 'No'}")
    print(f"\tAutocast enabled: {autocast_enabled}")
    print(f"\tCompile approach: {compile_approach}")
    print(f"\tDevice: {config['device']}")
    
    return config

def precision_aware_training_step(model, data, target, optimizer, scaler=None, autocast_enabled=False, dtype=None):
    """
    Training step that handles different precision configurations.
    
    Args:
        model: Compiled model
        data: Input data
        target: Target labels
        optimizer: Optimizer
        scaler: Gradient scaler (if using FP16)
        autocast_enabled: Whether to use autocast
        dtype: Target dtype for autocast
    """
    
    if autocast_enabled:
        autocast_dtype = dtype if dtype else torch.bfloat16
        
        with torch.amp.autocast('cuda', dtype=autocast_dtype):
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    else:
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    
    return loss.item()