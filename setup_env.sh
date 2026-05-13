#!/bin/bash
set -e

# PyTorch Python 3.14'ü desteklemiyor — Python 3.11 kullan
PYTHON_BIN="python3.11"

if ! command -v "$PYTHON_BIN" &>/dev/null; then
    echo "HATA: $PYTHON_BIN bulunamadı."
    echo "Lütfen Python 3.11 kurun: brew install python@3.11"
    exit 1
fi

echo "Kullanılan Python: $($PYTHON_BIN --version)"

if [ -d "venv" ]; then
    echo "venv zaten mevcut, bağımlılıklar güncelleniyor..."
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "Sanal ortam oluşturuluyor (Python 3.11)..."
    "$PYTHON_BIN" -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    python -m ipykernel install --user --name=venv --display-name "Python (venv)"
    echo "Jupyter çekirdeği 'Python (venv)' olarak kaydedildi."
fi

echo "Kurulum tamamlandı."
