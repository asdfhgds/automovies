"""Environment diagnostic for Autonomous Movie Studio.

Provides a single entrypoint `run_checks()` which returns a dict with detected
capabilities and prints a human readable summary.

Checks:
 - Python version
 - torch and CUDA availability and GPU name/VRAM (if torch present)
 - ffmpeg and ffprobe presence
 - whisperx import
 - pyscenedetect import
 - whisper (openai/whisper) fallback
 - Available providers and active profile
"""
import platform
import shutil
import json
import sys
import os


def _check_torch():
    result = {
        'installed': False,
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_name': None,
        'total_memory_mb': None,
        'torch_version': None
    }
    try:
        import torch
        result['installed'] = True
        result['torch_version'] = getattr(torch, '__version__', None)
        result['cuda_available'] = torch.cuda.is_available()
        if result['cuda_available']:
            result['gpu_count'] = torch.cuda.device_count()
            try:
                name = torch.cuda.get_device_name(0)
                result['gpu_name'] = name
            except Exception:
                result['gpu_name'] = None
            try:
                prop = torch.cuda.get_device_properties(0)
                # total_memory is in bytes
                result['total_memory_mb'] = int(prop.total_memory / (1024 * 1024))
            except Exception:
                result['total_memory_mb'] = None
    except Exception:
        pass
    return result


def _check_import(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _detect_profile():
    """Detect active profile based on environment."""
    profile = os.environ.get('STUDIO_PROFILE', 'local').lower()
    
    # Auto-detect GPU if using local profile
    if profile == 'local':
        torch_info = _check_torch()
        if torch_info.get('cuda_available'):
            # GPU available but local profile - note this for user
            return 'local', 'GPU detected but local profile active. Use STUDIO_PROFILE=colab-gpu for real models.'
    
    return profile, None


def _get_provider_status():
    """Check status of all providers."""
    providers = {}
    
    # LLM providers
    providers['llm'] = {
        'mock': True,  # Always available
        'qwen': _check_import('transformers'),
    }
    
    # Transcription providers
    providers['transcription'] = {
        'mock': True,  # Always available
        'whisperx': _check_import('whisperx'),
        'whisper': _check_import('whisper'),
    }
    
    # Script providers
    providers['script'] = {
        'mock': True,  # Always available
        'qwen': _check_import('transformers'),
    }
    
    # TTS providers
    providers['tts'] = {
        'mock': True,  # Always available
        'kokoro': False,  # Would need specific package
        'qwen3_tts': _check_import('transformers'),
    }
    
    # Image providers
    providers['image'] = {
        'mock': True,  # Always available
        'comfyui': False,  # Would need external API/service
    }
    
    # Video providers
    providers['video'] = {
        'mock': True,  # Always available
    }
    
    return providers


def run_checks():
    out = {}
    out['python_version'] = platform.python_version()
    out['platform'] = platform.system()
    out['ffmpeg'] = shutil.which('ffmpeg') is not None
    out['ffprobe'] = shutil.which('ffprobe') is not None
    out['torch'] = _check_torch()
    out['whisperx'] = _check_import('whisperx')
    out['whisper'] = _check_import('whisper')
    out['pyscenedetect'] = _check_import('scenedetect') or _check_import('pyscenedetect')
    out['transformers'] = _check_import('transformers')
    out['accelerate'] = _check_import('accelerate')
    # nvidia-smi availability
    out['nvidia_smi'] = shutil.which('nvidia-smi') is not None

    # Profile and providers
    profile, profile_note = _detect_profile()
    out['active_profile'] = profile
    out['profile_note'] = profile_note
    out['providers'] = _get_provider_status()

    # Strict GPU validation mode metadata
    require_real_llm = os.getenv('REQUIRE_REAL_LLM', 'false').lower() == 'true'
    out['require_real_llm'] = require_real_llm
    out['strict_gpu_mode'] = require_real_llm and profile == 'colab-gpu'
    out['director_provider'] = os.getenv('DIRECTOR_PROVIDER', 'qwen' if require_real_llm else 'mock')
    out['director_model'] = os.getenv('DIRECTOR_MODEL', 'Qwen/Qwen3-4B-Instruct-2507')
    out['script_provider'] = os.getenv('SCRIPT_PROVIDER', 'qwen' if require_real_llm else 'mock')
    out['script_model'] = os.getenv('SCRIPT_MODEL', out['director_model'])

    cuda_ok = out['torch'].get('cuda_available', False)
    transformers_ok = out['transformers']
    out['strict_gpu_ok'] = not require_real_llm or (cuda_ok and transformers_ok and out['accelerate'])

    return out


def print_report(report=None):
    if report is None:
        report = run_checks()
    
    # Set UTF-8 for console output on Windows
    import sys
    if sys.platform == 'win32':
        import os
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    print('\n' + '='*60)
    print('    Autonomous Movie Studio — Environment Health Check')
    print('='*60 + '\n')
    
    print(f"Platform: {report.get('platform')}")
    print(f"Python: {report.get('python_version')}")
    
    print('\n' + '-'*60)
    print('TOOLS')
    print('-'*60)
    print(f"FFmpeg: {'[OK] FOUND' if report.get('ffmpeg') else '[MISSING]'}")
    print(f"FFprobe: {'[OK] FOUND' if report.get('ffprobe') else '[MISSING]'}")
    print(f"nvidia-smi: {'[OK] FOUND' if report.get('nvidia_smi') else '[MISSING]'}")
    
    print('\n' + '-'*60)
    print('GPU / CUDA')
    print('-'*60)
    t = report.get('torch', {})
    if t.get('installed'):
        print(f"PyTorch: [OK] installed (v{t.get('torch_version')})")
        print(f"  CUDA available: {'[YES]' if t.get('cuda_available') else '[NO]'}")
        if t.get('cuda_available'):
            print(f"  GPU count: {t.get('gpu_count')}")
            print(f"  GPU name: {t.get('gpu_name')}")
            print(f"  GPU VRAM: {t.get('total_memory_mb')} MB")
    else:
        print('PyTorch: [NOT INSTALLED]')
    print(f"Transformers: {'[OK] FOUND' if report.get('transformers') else '[MISSING]'}")
    print(f"Accelerate: {'[OK] FOUND' if report.get('accelerate') else '[MISSING]'}")
    
    print('\n' + '-'*60)
    print('STRICT GPU MODE (REQUIRE_REAL_LLM)')
    print('-'*60)
    print(f"REQUIRE_REAL_LLM: {'[ON]' if report.get('require_real_llm') else '[OFF]'}")
    print(f"Active profile: {report.get('active_profile')}")
    print(f"Strict mode (LLM + colab-gpu): {'[ON]' if report.get('strict_gpu_mode') else '[OFF]'}")
    print(f"Director provider: {report.get('director_provider')} | model: {report.get('director_model')}")
    print(f"Script provider:   {report.get('script_provider')} | model: {report.get('script_model')}")
    if report.get('require_real_llm'):
        if report.get('strict_gpu_ok'):
            print("[OK] Strict GPU validation prerequisites satisfied (CUDA + Transformers + Accelerate)")
        else:
            print("[FATAL] Strict mode required but CUDA/Transformers/Accelerate missing. "
                  "GPU validation CANNOT proceed with mocks.")
    
    print('\n' + '-'*60)
    print('MODELS & ADAPTERS')
    print('-'*60)
    print(f"WhisperX: {'[OK] FOUND' if report.get('whisperx') else '[MISSING]'}")
    print(f"Whisper (OpenAI): {'[OK] FOUND' if report.get('whisper') else '[MISSING]'}")
    print(f"PySceneDetect: {'[OK] FOUND' if report.get('pyscenedetect') else '[MISSING]'}")
    
    print('\n' + '-'*60)
    print('ACTIVE PROFILE & PROVIDERS')
    print('-'*60)
    profile = report.get('active_profile', 'local')
    print(f"Active profile: {profile}")
    if report.get('profile_note'):
        print(f"  Note: {report.get('profile_note')}")
    
    providers = report.get('providers', {})
    for provider_type, provider_list in providers.items():
        available = [p for p, v in provider_list.items() if v]
        print(f"\n{provider_type.capitalize()}:")
        for name, status in provider_list.items():
            symbol = '[OK]' if status else '[NO]'
            print(f"  {symbol} {name}")
    
    print('\n' + '='*60)
    print('RECOMMENDATION')
    print('='*60)
    if t.get('cuda_available') and profile == 'colab-gpu':
        print("[OK] Ready for real model execution (GPU available + profile set)")
    elif profile == 'local':
        print("[OK] Local development profile active (mocks will be used)")
        print("  For GPU execution, set: export STUDIO_PROFILE=colab-gpu")
    else:
        print(f"[OK] Profile '{profile}' active")
    
    print('\n' + '='*60 + '\n')
    
    # json summary for automation
    print('JSON Summary:')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    print_report()

