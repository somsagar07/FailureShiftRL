"""Check the imports actually needed by the local interface, without downloading data."""
import argparse
import importlib
from importlib.metadata import version
import sys


def check_environment(quiet=False):
    required = (
        ('Pillow', 'PIL.Image', ('open', 'new')),
        ('huggingface_hub', 'huggingface_hub', ('snapshot_download', 'hf_hub_download')),
        ('datasets', 'datasets', ('Image', 'Dataset', 'load_dataset')),
        ('numpy', 'numpy', ('linalg', 'array')),
        ('torch', 'torch', ('inference_mode', 'Tensor')),
        ('transformers', 'transformers', ('AutoImageProcessor', 'AutoModel', 'AutoTokenizer', 'AutoModelForSeq2SeqLM', 'CLIPTextModel', 'CLIPTokenizer')),
        ('sentencepiece', 'sentencepiece', ('SentencePieceProcessor',)),
        ('protobuf', 'google.protobuf', ('__version__',)),
        ('ftfy', 'ftfy', ('fix_text',)),
    )
    failures = []
    if not quiet:
        print(f'Python: {sys.executable}')
    for package, module, exports in required:
        try:
            loaded = importlib.import_module(module)
            for name in exports:
                getattr(loaded, name)
            installed = version(package)
            if package == 'torch' and not (2,6) <= tuple(map(int, installed.split('.')[:2])) < (3,0):
                raise ValueError('PyTorch >=2.6,<3 is required for weights-only loading.')
            if package == 'transformers' and not (4,55) <= tuple(map(int, installed.split('.')[:2])) < (5,0):
                raise ValueError('This adapter requires Transformers >=4.55,<5.')
        except Exception as exc:
            failures.append(package)
            if not quiet:
                print(f'MISSING OR BROKEN: {package}: {exc}')
        else:
            if not quiet:
                print(f'OK: {package} {installed}')
    if failures and not quiet:
        print('Run bash setup.sh from web_interface to install the app requirements.')
    return 1 if failures else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quiet', action='store_true')
    raise SystemExit(check_environment(parser.parse_args().quiet))
