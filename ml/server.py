from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/detect", methods=["POST"])
def detect():
    data = request.json
    frame = data.get("frame")

    print("Frame received")

    return jsonify({
        "boxes": [
            {"x": 100, "y": 100, "w": 200, "h": 150}
        ]
    })

if __name__ == "__main__":
    app.run(port=5000)