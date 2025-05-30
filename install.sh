#!binbash

# Скрипт встановлення для системи автоматизації воріт
# Для Raspberry Pi 5 з Raspberry OS 64-bit Debian 12

echo ========================================
echo Встановлення системи автоматизації воріт
echo ========================================

# Оновлення системи
echo Оновлення системи...
sudo apt update && sudo apt upgrade -y

# Встановлення системних залежностей
echo Встановлення системних пакетів...
sudo apt install -y 
    python3-pip 
    python3-venv 
    python3-dev 
    python3-picamera2 
    libcamera-apps 
    libcamera-dev 
    libopencv-dev 
    python3-opencv 
    git 
    cmake 
    build-essential 
    libatlas-base-dev 
    libopenblas-dev 
    libjpeg-dev 
    libpng-dev 
    libavcodec-dev 
    libavformat-dev 
    libswscale-dev 
    libv4l-dev 
    libxvidcore-dev 
    libx264-dev 
    libgtk-3-dev 
    libcanberra-gtk3-module 
    libgstreamer1.0-dev 
    gstreamer1.0-tools 
    v4l-utils

# Створення віртуального середовища
echo Створення віртуального середовища...
python3 -m venv --system-site-packages gate
source gate/bin/activate

# Оновлення pip
pip install --upgrade pip wheel setuptools

# Встановлення Python залежностей
echo Встановлення Python пакетів...
pip install -r requirements.txt

# Створення необхідних директорій
echo Створення структури директорій...
mkdir -p config
mkdir -p models
mkdir -p logs
mkdir -p captured_images

# Перевірка GPIO доступу
echo Налаштування GPIO...
sudo usermod -a -G gpio $USER

# Налаштування правил udev для JSN-SR04T
echo Налаштування правил для ультразвукового датчика...
echo 'SUBSYSTEM==gpio, GROUP=gpio, MODE=0660'  sudo tee etcudevrules.d99-gpio.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Створення systemd сервісу
echo Створення системного сервісу...
cat  gate_automation.service  EOL
[Unit]
Description=Gate Automation System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)venvbin
ExecStart=$(pwd)venvbinpython main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOL

# Встановлення сервісу
sudo cp gate_automation.service etcsystemdsystem
sudo systemctl daemon-reload
sudo systemctl enable gate_automation.service

# Створення скрипта запуску
cat  start.sh  'EOL'
#!binbash
source gate/bin/activate
python main.py
EOL
chmod +x start.sh

# Створення скрипта для створення ROI
cat  create_roi.sh  'EOL'
#!binbash
source gate/bin/activate
echo Створення ROI для камер
echo 1. Для камери вїзду
python roi_creator.py ENTRANCE
echo 2. Для камери виїзду
python roi_creator.py EXIT
EOL
chmod +x create_roi.sh

echo ========================================
echo Встановлення завершено!
echo ========================================
echo 
echo Наступні кроки
echo 1. Скопіюйте файл credentials.json в кореневу директорію проекту
echo 2. Завантажте моделі license.onnx та ocr.pt в директорію models
echo 3. Запустіть .create_roi.sh для налаштування зон інтересу
echo 4. Запустіть систему .start.sh
echo    або через systemd sudo systemctl start gate_automation
echo 
echo Для автозапуску при завантаженні системи
echo sudo systemctl enable gate_automation
echo 
echo Для перегляду логів
echo sudo journalctl -u gate_automation -f