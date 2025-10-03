# ✏️ Draw & Guess AI

An AI-powered drawing recognition game built with **TensorFlow**, **Flask**, and a simple modern **web interface**.  
The model is trained on a subset of the Google QuickDraw dataset (10 categories).  

> Built as part of the CPU ISIMM, during the integration day of our institute ISIMM | Higher Institute of Computer Science and Mathematics of Monastir, 

---

## 🚀 Features
- 🖌️ Draw on a canvas and let the AI guess your doodle.  
- 🎯 Supports **10 categories**: `apple, airplane, banana, cat, dog, car, chair, clock, house, tree`.  
- ⚡ Uses a Convolutional Neural Network (CNN) trained on QuickDraw data.  
- 🌐 Flask backend + HTML/CSS/JS frontend.  
- 📊 Achieves good accuracy with a **small dataset**.  

---

## 📂 Project Structure

```
static/                      #  images (frontend assets)
templates/                   # HTML files for the web interface
.gitattributes
.gitignore
airplane.npy                 # QuickDraw datasets (numpy format)
apple.npy
banana.npy
car.npy
cat.npy
categories.txt               # List of categories
chair.npy
clock.npy
dog.npy
draw_model_advanced.keras    # Trained model
house.npy
train_quickdraw.py           # Training script
tree.npy
app.py                       # Flask backend
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/LADYGRAY95/draw_and_guess.git
cd draw_and_guess
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model (optional, already provided)
```bash
python train_quickdraw.py
```

### 5. Run the app
```bash
python app.py
```

Open your browser and go to 👉 **http://127.0.0.1:5000/**

---

## 🖼️ Demo
- Draw something on the white canvas.
- Click **Guess** and see the AI prediction.
- The sidebar highlights the predicted category.

---

## 📜 License

MIT License

Copyright (c) 2025 Youssr Chouaya 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
