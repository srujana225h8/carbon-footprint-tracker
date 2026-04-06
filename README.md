# 🌱 Carbon Footprint Analyzer

A web application that estimates your **personal carbon emissions** from daily energy usage and commuting, and also analyzes the **carbon cost of your code** — all powered by a trained deep learning model.

---

## ✨ Features

- **🏠 Home Energy & Commute Tracker** — Input your electricity usage, renewable energy percentage, smart appliance hours, and commute details to get a predicted CO₂ emission value.
- **📱 Mobile Commute Tracker Integration** — Use the companion mobile tracker to generate a tracking code and auto-fill your commute data.
- **💻 Code Carbon Estimator** — Upload or paste a source file (Python, JavaScript, Java, C++, and more) to estimate its energy consumption and carbon footprint based on code complexity analysis.
- **📋 Personalized Recommendations** — Get actionable tips to reduce both your lifestyle and code emissions.

---

## 🧠 How It Works

### Lifestyle Emission Prediction
A Keras deep learning model trained on household energy and transport data predicts your daily carbon emission (kg CO₂). Features include:
- Energy usage (kWh)
- Renewable energy share (%)
- Smart appliance usage (hours)
- Commute distance (km) and vehicle type

### Code Carbon Estimation
The app statically analyzes your source code using:
- **AST parsing** (Python) or **regex pattern matching** (other languages)
- Operation-level energy weights (loops, I/O, recursion, imports, etc.)
- CPU TDP baseline and estimated execution time per line
- Global average grid intensity (475 gCO₂/kWh) for final CO₂ calculation

---

## 🗂️ Project Structure

```
carbon-footprint-app/
├── app.py                  # Flask backend — routes, ML inference, code analysis
├── requirements.txt        # Python dependencies
├── model/
│   ├── carbon_emission_predictor.keras   # Trained Keras model
│   ├── scaler.pkl                        # Feature scaler
│   └── feature_names.pkl                 # Feature name list
├── templates/
│   ├── index.html          # Main input form
│   ├── result.html         # Emission prediction result
│   ├── analyze_code.html   # Code upload/paste form
│   └── code_result.html    # Code analysis result
└── static/
    └── style.css           # App styling
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/carbon-footprint-app.git
cd carbon-footprint-app

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
python app.py
```

Then open your browser and navigate to `http://127.0.0.1:5000`.

---

## 📦 Dependencies

| Package      | Purpose                          |
|--------------|----------------------------------|
| Flask        | Web framework                    |
| TensorFlow   | Deep learning model inference    |
| NumPy        | Numerical computations           |
| pickle       | Loading scaler & feature names   |

Install all at once:
```bash
pip install flask tensorflow numpy
```

---

## 🌐 Routes

| Route           | Method     | Description                              |
|-----------------|------------|------------------------------------------|
| `/`             | GET        | Main input form                          |
| `/predict`      | POST       | Run lifestyle emission prediction        |
| `/analyze-code` | GET, POST  | Code carbon footprint analysis           |

---

## 📱 Mobile Commute Tracker

The app integrates with a companion mobile web app for commute tracking. Open the tracker, record your trip, and paste the generated tracking code (format: `distance_VehicleType_timestamp`) into the main form to auto-populate commute data.

🔗 [Open Mobile Commute Tracker](https://srujana225h8.github.io/carbon-tracker-mobile/)

---

## 🔧 Supported Languages (Code Analyzer)

| Language     | Analysis Method |
|--------------|-----------------|
| Python       | AST-based       |
| JavaScript   | Regex-based     |
| TypeScript   | Regex-based     |
| Java         | Regex-based     |
| C / C++      | Regex-based     |
| C#           | Regex-based     |
| Go, Rust, Ruby, PHP, Swift, Kotlin, R | Regex-based |

---

## 💡 Sample Recommendations

**Lifestyle:**
- Switch to energy-efficient appliances if energy usage > 12 kWh
- Increase renewable energy share to above 40%
- Consider public transport, carpooling, or EVs

**Code:**
- Vectorise loops using NumPy/pandas
- Batch or cache I/O calls
- Replace recursion with iteration
- Minimise unused imports

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.
