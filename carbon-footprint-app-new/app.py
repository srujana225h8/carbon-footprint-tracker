import os
import ast
import re
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Code Carbon Emission Estimator
# ---------------------------------------------------------------------------

# Energy per operation (nano-joules) – illustrative but grounded estimates
ENERGY_PER_OP = {
    "loop":        500,    # each loop iteration overhead
    "function":    200,    # function call overhead
    "io":         5000,    # file / network I/O per call
    "arithmetic":   10,    # basic math op
    "string":       50,    # string manipulation per op
    "list":         30,    # list / array operation
    "conditional":  15,    # if/else branch
    "import":     1000,    # module import
    "recursion":   800,    # recursive call (stack overhead)
    "regex":       300,    # regex operation
}

# Average global grid carbon intensity (gCO₂/kWh) — world average ~475
GRID_INTENSITY_G_PER_KWH = 475

# CPU TDP baseline (Watts) – typical laptop CPU
CPU_TDP_WATTS = 15

# Assumed execution time per line of code (seconds) – simplistic proxy
EXEC_TIME_PER_LINE_SEC = 0.0001


def detect_language(filename: str, code: str) -> str:
    ext = os.path.splitext(filename)[-1].lower()
    ext_map = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".java": "Java", ".cpp": "C++", ".c": "C", ".cs": "C#",
        ".rb": "Ruby", ".go": "Go", ".rs": "Rust", ".php": "PHP",
        ".swift": "Swift", ".kt": "Kotlin", ".r": "R",
    }
    return ext_map.get(ext, "Unknown")


def count_python_ops(code: str) -> dict:
    """Count code patterns via AST for Python."""
    counts = {k: 0 for k in ENERGY_PER_OP}
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                counts["loop"] += 1
            elif isinstance(node, ast.FunctionDef):
                counts["function"] += 1
            elif isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id
                if func_name in ("open", "read", "write", "fetch", "get", "post", "request"):
                    counts["io"] += 1
                elif func_name in ("print", "input"):
                    counts["io"] += 1
                else:
                    counts["function"] += 1
            elif isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp)):
                counts["arithmetic"] += 1
            elif isinstance(node, ast.If):
                counts["conditional"] += 1
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                counts["import"] += 1
            elif isinstance(node, ast.JoinedStr) or isinstance(node, ast.Constant) and isinstance(getattr(node, 'value', None), str):
                counts["string"] += 1
    except SyntaxError:
        pass
    return counts


def count_generic_ops(code: str) -> dict:
    """Regex-based fallback for non-Python languages."""
    counts = {k: 0 for k in ENERGY_PER_OP}
    counts["loop"]        = len(re.findall(r'\b(for|while|forEach|each)\b', code))
    counts["function"]    = len(re.findall(r'\b(function|def|fn|func|method)\b', code))
    counts["io"]          = len(re.findall(r'\b(open|read|write|fetch|import|require|include|File|Stream|http|socket)\b', code))
    counts["arithmetic"]  = len(re.findall(r'[\+\-\*\/\%]{1,2}', code))
    counts["string"]      = len(re.findall(r'[\"\'].*?[\"\']', code))
    counts["conditional"] = len(re.findall(r'\b(if|else|switch|case|ternary)\b', code))
    counts["import"]      = len(re.findall(r'\b(import|require|include|use)\b', code))
    counts["recursion"]   = len(re.findall(r'\b(recursive|recursion|self\s*\(|this\s*\()\b', code))
    counts["regex"]       = len(re.findall(r're\.|regex|Regex|Pattern\.compile', code))
    return counts


def estimate_code_emission(code: str, filename: str) -> dict:
    language = detect_language(filename, code)
    lines = [l for l in code.splitlines() if l.strip() and not l.strip().startswith('#')]
    loc = len(lines)

    if language == "Python":
        op_counts = count_python_ops(code)
    else:
        op_counts = count_generic_ops(code)

    # Total energy in nano-joules
    energy_nj = sum(op_counts[op] * ENERGY_PER_OP[op] for op in ENERGY_PER_OP)

    # Add baseline CPU energy for execution time
    exec_time_sec = loc * EXEC_TIME_PER_LINE_SEC
    cpu_energy_j  = CPU_TDP_WATTS * exec_time_sec
    cpu_energy_nj = cpu_energy_j * 1e9

    total_energy_nj = energy_nj + cpu_energy_nj
    total_energy_kwh = total_energy_nj / 3.6e12   # convert nJ → kWh

    # Carbon in grams CO₂
    carbon_g = total_energy_kwh * GRID_INTENSITY_G_PER_KWH
    carbon_mg = round(carbon_g * 1000, 4)

    # Complexity score (0–100)
    complexity = min(100, int(
        (op_counts["loop"] * 3 + op_counts["function"] * 2 +
         op_counts["recursion"] * 5 + op_counts["io"] * 4) /
        max(loc, 1) * 10
    ))

    top_ops = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    def code_recommendations(ops, lang, carbon_mg):
        recs = []
        if ops.get("loop", 0) > 5:
            recs.append("Consider vectorising loops (e.g. NumPy/pandas) to cut CPU cycles.")
        if ops.get("io", 0) > 3:
            recs.append("Batch or cache I/O calls — each disk/network op is energy-expensive.")
        if ops.get("recursion", 0) > 0:
            recs.append("Replace recursion with iteration to reduce stack overhead.")
        if ops.get("import", 0) > 10:
            recs.append("Minimise imports — unused modules waste startup energy.")
        if carbon_mg > 1:
            recs.append("Profile with a tool like CodeCarbon to measure real-world impact.")
        if not recs:
            recs.append("Code looks efficient! Keep monitoring as it scales.")
        return recs

    recs = code_recommendations(op_counts, language, carbon_mg)

    return {
        "language": language,
        "loc": loc,
        "energy_nj": round(total_energy_nj, 2),
        "carbon_mg": carbon_mg,
        "complexity": complexity,
        "op_counts": op_counts,
        "top_ops": top_ops,
        "recommendations": recs,
        "exec_time_ms": round(exec_time_sec * 1000, 4),
    }

MODEL_PATH = "model/carbon_emission_predictor.keras"
SCALER_PATH = "model/scaler.pkl"
FEATURES_PATH = "model/feature_names.pkl"

model = tf.keras.models.load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    feature_names = pickle.load(f)


def parse_tracking_code(code):
    # Example: 8.5_Bus_1770032658
    parts = code.split("_")
    if len(parts) < 2:
        raise ValueError("Invalid tracking code")
    return float(parts[0]), parts[1]


def prepare_features(data):
    row = {f: 0 for f in feature_names}

    row["Energy_Usage_kWh"] = data["energy"]
    row["Renewable_Energy_Usage_percent"] = data["renewable"]
    row["Smart_Appliance_Usage_hours"] = data["appliance"]
    row["Distance_km"] = data["distance"]

    vehicle_col = f"Vehicle_Type_{data['vehicle']}"
    if vehicle_col in row:
        row[vehicle_col] = 1

    X = np.array([[row[f] for f in feature_names]])
    return scaler.transform(X)


def generate_recommendations(data, emission):
    recs = []

    if data["energy"] > 12:
        recs.append("Reduce electricity usage by switching to energy-efficient appliances.")

    if data["renewable"] < 40:
        recs.append("Increase renewable energy usage such as rooftop solar.")

    if data["appliance"] > 5:
        recs.append("Limit smart appliance usage during peak hours.")

    if data["vehicle"] in ["Car", "Motorcycle"]:
        recs.append("Consider public transport, carpooling, or electric vehicles.")

    if emission < 5:
        recs.append("Great job! Your carbon footprint is already low.")

    return recs


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    tracking_code = request.form.get("tracking_code", "").strip()

    if tracking_code:
        distance, vehicle = parse_tracking_code(tracking_code)
    else:
        distance = float(request.form.get("distance"))
        vehicle = request.form.get("vehicle")

    data = {
        "energy": float(request.form.get("energy")),
        "renewable": float(request.form.get("renewable")),
        "appliance": float(request.form.get("appliance")),
        "distance": distance,
        "vehicle": vehicle
    }

    X = prepare_features(data)
    emission = max(0, float(model.predict(X, verbose=0)[0][0]))
    emission = round(emission, 2)

    recommendations = generate_recommendations(data, emission)

    return render_template(
        "result.html",
        emission=emission,
        recommendations=recommendations
    )



@app.route("/analyze-code", methods=["GET", "POST"])
def analyze_code():
    if request.method == "GET":
        return render_template("analyze_code.html")

    code_file = request.files.get("code_file")
    code_text = request.form.get("code_text", "").strip()

    if code_file and code_file.filename:
        filename = code_file.filename
        code = code_file.read().decode("utf-8", errors="ignore")
    elif code_text:
        filename = request.form.get("filename", "script.py")
        code = code_text
    else:
        return render_template("analyze_code.html", error="Please upload a file or paste code.")

    result = estimate_code_emission(code, filename)
    return render_template("code_result.html", filename=filename, **result)


if __name__ == "__main__":
    app.run(debug=True)
