# MachineVisionEssentials



## Getting started

To make it easy for you to get started with the lab exercise, here's a list of recommended next steps.

For this lab exercise it is recommended to use a linux based distribution, the following instructions are tested under Fedora running in WSL.

## Clone the repository

Depending on the platform you are working, go to the 
```
cd repo_work_dir
git clone https://gitlab.com/mase600043/bswe-aktuelle-themen/machinevisionessentials.git
```

## Task description
Imagine you are a software developer and you start your working day by checking the mails. You received a message of your boss:

```
Hi, I had a great idea for our product! It should automatically can identify and count how many candies are within sight. So you should design a Machine Vision application which can identify different sorts of Candy on the fly. 
The candy is own a covayer belt and the camera system looks on the belt. It should autometically identify the kind of candy and how many are within sight.
```
The system looks like this:
![image](candy_detection_system.png)


**Minmal Tasks (35/70 Punkte)**:
- Setup Python, Yolo, and Label-Studio.
- Create a basic dataset (Images).
- Use Label-Studio to add annotation to the images. (Bounding Boxes without rotation)
- Train a Yolo model for object detection to identify the candy seen in a picture.

**Advanced Tasks (52/70 Punkte)**:
- Complete Minimal Tasks.
- Use augmentation to create more training pictures (advanced dataset).
- Train a Yolo model with the improved dataset.
- Evaluate the performance of your System.

**Expert Tasks (70/70 Punkte)**:
- Complete Advanced Tasks.
- Enhance your system to also identfy additional objects (different than candy, e.g., screws, choose something realistic).
- Make it realtime cabeable with your webcam.
- Use Yolo-OBB model -> https://docs.ultralytics.com/tasks/obb/
    - Oriented Bounding Boxes (OBB) Object Detection

## Step 1: Setup Environment

Create a working directory for your project.

Setup a virtual python environment with uv (prefer versions >=3.10 and <=3.12).
- Install uv and generate a virtual environment. -> https://docs.astral.sh/uv/getting-started/installation/
    https://docs.astral.sh/uv/getting-started/features/
    ```
    pip install uv
    ```
- Install Label-Studio -> https://labelstud.io/guide/quick_start
    ```
    uv add label-studio
    ```
- Setup the Labeling Interface in Label-Studio -> https://labelstud.io/guide/setup

## Step 2: Create your Dataset

Create a basic dataset
- Take the provided candies (celebration box) and make pictures with your smartphone.
- Generate at least 100 images. Keep in mind what you have learned in the lecture.
    - Choose different backgrounds.
    - Use different angles.
    - Only one thing or many things in the picture.
    - Try to capture differnt light conditions.
    - Every class should be equally represented.
    - ...

Create a more advanced dataset
- Write a python script to augment the pictures
    - You can use libraries like OpenCV or PIL (Pillow) -> https://oceancv.org/book/Augmentation_Manual.html, or
    - Use a custom transfer of YOLO to augment the pictures -> https://docs.ultralytics.com/guides/yolo-data-augmentation/
    - Target at least 200 images.

Export the dataset from Label-Studio
- The File structure should be ...
    ```
    candy_dataset/ 
    ├── images/ 
    │ ├── train/ 
    │ ├── test/ 
    │ └── val/ 
    ├── labels/ 
    │ ├── train/ 
    │ ├── test/ 
    │ └── val/ 
    └── data.yaml
    ```
    Therefore, you have to write a python script to make a proper test/training/validation split. Make the script that modular so that you can generate diferent split, start with 80% test, 10% training, and 10% validation.

- Create a data.yaml file -> https://docs.ultralytics.com/datasets/detect/
    - For the expert task -> https://docs.ultralytics.com/datasets/obb/
    

## Step 3: Train your YOLO Model

For YOLO create a seperate virtual environment with uv and install ultralytics package

- If you do not have a Nvidia GPU with CUDA than please use the CPU Only version
    
    ```bash=
    uv add torch torchvision --index https://download.pytorch.org/whl/cpu
    uv add ultralytics
    ```

    HINT: With Google Colab you can use GPU accelaration for free. -> https://colab.research.google.com/

- If you have a Nvidia GPU with CUDA
    ```bash=
    uv add ultralytics
    ```

HINT: For further details on the correct PyTorch version -> https://pytorch.org/get-started/locally/

Try different Yolo models (v11, v26, n, s, m, l, x). Start with yolo26n.pt. Look at -> https://docs.ultralytics.com/tasks/detect/


## Step 4: Test and Verify

After the training of the yolo model look at the folder runs/detect. There you find all performance metrics of the training and validation. Identify the metrics and check if the model is trained sufficiently.

Use the test-set and check if the model performs correctly. The output should be for every test image like the following:
![image](output.jpg)


## Step 5: Webcam integration

To use the Webcam you need the opencv-python package.
```
uv add opencv-python
```

The following script shows you in a window the content of the webcam

```
import cv2

cap = cv2.VideoCapture(0) # default webcam (0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show the frame in a window
    cv2.imshow("Webcam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Step 6: Write a protocol and make a presentation

Use the provided Latex Template in Moodle and write a lab report. Document every step you made, every challenge, every pit fall what you encounterd. Also make a presentation for the next lecture, it should be 10 Minutes where you recap everything from the lab report.

## Umsetzungsstand

| Level | Status | Beschreibung |
|-------|--------|-------------|
| **Minimal** | ✅ | Umgebung, Datensatz (8 Candy-Klassen), Annotation, YOLO-Training, Test, Webcam |
| **Advanced** | ✅ | Augmentation (OBB-Keypoints), verbessertes Training (100+ Epochen), Evaluation |
| **Expert** | ✅ | OBB (Oriented Bounding Boxes), zusätzliche Klasse `hand`, Echtzeit-Webcam |

## Projektstruktur (Expert)

```
├── main.py                    # Label Studio starten
├── split_dataset.py           # Export → train/val/test Split + data.yaml
├── augment_dataset.py         # Augmentation AABB (Advanced)
├── augment_dataset_obb.py     # Augmentation OBB mit Keypoint-Clamping (Expert)
├── train_yolo.py              # Training YOLO-Detect (Minimal/Advanced)
├── train_yolo_obb.py          # Training YOLO-OBB (Expert)
├── auto_annotate_obb.py       # Auto-Annotation → Label Studio JSON (Expert)
├── predict_test.py            # Inferenz auf Testsplit
├── webcam_detect.py           # Echtzeit-Webcam-Erkennung
├── label_config.xml           # Label Studio Konfiguration (9 Klassen, canRotate)
├── pyproject.toml             # uv: Dependencies + PyTorch CUDA 12.8 Index
├── candy_dataset/
│   ├── data.yaml              # 9 Klassen (alphabetisch, vgl. Label-Studio-Export)
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
├── Report_Template/paper.tex  # Lab-Bericht (LaTeX)
└── runs/obb/                  # Trainingsartefakte (lokal; nicht versioniert)
```

## Ausführung

### Voraussetzungen

- Python ≥ 3.12, `uv` installiert
- NVIDIA GPU mit CUDA ≥ 12.8 (getestet: RTX 5080, Blackwell/sm_120)
- `candy_dataset/` mit gelabelten OBB-Daten (Label Studio Export)

```bash
uv sync --extra train
```

PyTorch wird über den Index `pytorch-cu128` in `pyproject.toml` bezogen. Für CPU-only den Index auf `https://download.pytorch.org/whl/cpu` ändern und `uv lock && uv sync --extra train` erneut ausführen.

GPU-Check:
```bash
uv run --extra train python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Pipeline-Befehle

| Schritt | Befehl |
|---------|--------|
| **Split** | `uv run --extra train python split_dataset.py --source candy_dataset/project_obb_full_hand --out candy_dataset` |
| **Augmentation (OBB)** | `uv run --extra train python augment_dataset_obb.py --target 300 --focus-classes mars milky_way` |
| **Training (OBB)** | `uv run --extra train python train_yolo_obb.py --epochs 300 --batch 32 --device 0 --patience 50 --optimizer AdamW --name obb/candy_train` |
| **Test-Inferenz** | `uv run --extra train python predict_test.py --device 0` |
| **Webcam (Live)** | `uv run --extra train python webcam_detect.py --device 0` (Taste `q` beendet) |
| **Auto-Annotation** | `uv run --extra train python auto_annotate_obb.py --model runs/obb/candy_train/weights/best.pt --images mydata/Candies` |

Präsentation (PPTX) und Fototaktik liegen **außerhalb** dieses Repos (lokal verschoben).

### Ergebnisse (bester Lauf: `runs/obb/candy_full_hand2`)

| Metrik | Wert |
|--------|------|
| Precision | 90,9 % |
| Recall | 77,8 % |
| mAP50 | 82,4 % |
| mAP50-95 | 63,3 % |
| Bester Checkpoint | Epoche 68 / 118 (Early Stopping) |
| Modell | YOLO11n-OBB (2,66 M Param., 6,7 GFLOPs) |
| Inferenz | ~2–3 ms/Bild (>300 FPS, RTX 5080) |

### Klassen (9, alphabetisch)

`bounty`, `galaxy`, `galaxy_caramel`, `hand`, `maltesers`, `mars`, `milky_way`, `snickers`, `twix`

### Hinweise

- **Windows:** `--workers 0` ist Standard in den Trainings-Skripten (vermeidet WinError 1455).
- **data.yaml:** Kein `path:`-Eintrag — Ultralytics löst Pfade relativ zur YAML auf.
- **Gewichte:** `predict_test.py` und `webcam_detect.py` nutzen automatisch das neueste `runs/**/weights/best.pt`.
- **AVIF-Bilder:** Einige Web-Bilder hatten `.jpg`-Extension aber AVIF-Codec. Diese wurden mit Pillow re-kodiert; YOLO-Caches (`.cache`) danach gelöscht.