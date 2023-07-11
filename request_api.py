from flask import Flask, request, jsonify
import requests
import json

app = Flask(__name__)

class DataRetriever:
    def __init__(self, api_url):
        self.api_url = api_url

    def get_data(self, endpoint, params=None):
        response = requests.get(self.api_url + endpoint, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response as JSON
            data = json.loads(response.text)
            return data
        else:
            print(f'Request failed with status code {response.status_code}')
            return None

# Initialize the DataRetriever with the URL of the public API
data_retriever = DataRetriever('https://globalsolaratlas.info/api')

@app.route('/api/data', methods=['GET'])
def get_data():
    # Get the endpoint and parameters from the request
    endpoint = request.args.get('endpoint')
    params = request.args.get('params')

    # If params is a string of JSON, parse it into a dictionary
    if params is not None and isinstance(params, str):
        params = json.loads(params)

    # Use the DataRetriever to get the data from the public API
    data = data_retriever.get_data(endpoint, params)

    # Return the data to the user
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
