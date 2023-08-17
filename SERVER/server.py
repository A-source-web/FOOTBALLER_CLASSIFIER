from flask import Flask, request, jsonify, render_template
import util
app = Flask(__name__)


def get_base_64():
    with open('SERVER/b64.txt') as f:
        return f.read()


@app.route('/classify_image', methods=['POST', 'GET'])
def classify_image():
    image = get_base_64()
    response = jsonify(util.classify_image(image))
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if (__name__ == "__main__"):
    util.load_saved_artifacts()
    app.run(port=5000)


# the job is to convert every image to base 64 encoded string
