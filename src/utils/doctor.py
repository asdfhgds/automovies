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
"""
import platform
import shutil
import json
import sys


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


def run_checks():
    out = {}
    out['python_version'] = platform.python_version()
    out['ffmpeg'] = shutil.which('ffmpeg') is not None
    out['ffprobe'] = shutil.which('ffprobe') is not None
    out['torch'] = _check_torch()
    out['whisperx'] = _check_import('whisperx')
    out['whisper'] = _check_import('whisper')
    out['pyscenedetect'] = _check_import('scenedetect') or _check_import('pyscenedetect')
    # nvidia-smi availability
    out['nvidia_smi'] = shutil.which('nvidia-smi') is not None
    return out


def print_report(report=None):
    if report is None:
        report = run_checks()
    print('=== Autonomous Movie Studio doctor ===')
    print(f"Python: {report.get('python_version')}")
    print(f"ffmpeg: {'FOUND' if report.get('ffmpeg') else 'MISSING'}")
    print(f"ffprobe: {'FOUND' if report.get('ffprobe') else 'MISSING'}")
    t = report.get('torch', {})
    if t.get('installed'):
        print(f"torch: installed (v{t.get('torch_version')})")
        print(f"  CUDA available: {t.get('cuda_available')}")
        if t.get('cuda_available'):
            print(f"  GPU count: {t.get('gpu_count')}")
            print(f"  GPU name: {t.get('gpu_name')}")
            print(f"  GPU VRAM (MB): {t.get('total_memory_mb')}")
    else:
        print('torch: NOT INSTALLED')
    print(f"whisperx: {'FOUND' if report.get('whisperx') else 'MISSING'}")
    print(f"whisper (openai): {'FOUND' if report.get('whisper') else 'MISSING'}")
    print(f"pyscenedetect: {'FOUND' if report.get('pyscenedetect') else 'MISSING'}")
    print(f"nvidia-smi: {'FOUND' if report.get('nvidia_smi') else 'MISSING'}")

    # json summary printed for automation
    print('\nJSON Summary:\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    print_report()
