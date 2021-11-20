import pickle
from flask import Flask, request, jsonify
from model_files.ml_model import predict_mpg

app = Flask("mpg_prediction")

@app.route('/', methods=["POST"])
def predict():
    # get the user data from the post request
    vehicle_config = request.get_json()

    with open("C:/Users/Shalu/Desktop/project/my_projects/mpg-predictions/model_files/mpg-predictions-model.bin", "rb") as f_in:
        model = pickle.load(f_in)
        f_in.close()

    predictions = predict_mpg(vehicle_config, model)

    response = {
        "mpg_predictions" : list(predictions)
    }

    return jsonify(response)


# @app.route('/', methods=['GET'])
# def ping():
#     return "Pinging model application!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)