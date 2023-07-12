import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import requests
import json

class DataRetriever:
    def __init__(self, api_url):
        self.api_url = api_url

    def get_data(self, endpoint, params=None):
        response = requests.get(self.api_url + endpoint, params=params)

        if response.status_code == 200:
            data = json.loads(response.text)
            return data
        else:
            print(f'Request failed with status code {response.status_code}')
            return None

data_retriever = DataRetriever('https://globalsolaratlas.info/api')

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Input(id='endpoint-input', type='text', placeholder='Enter endpoint'),
    dcc.Input(id='params-input', type='text', placeholder='Enter parameters as JSON'),
    html.Button('Submit', id='submit-button'),
    dcc.Graph(id='data-graph')
])

@app.callback(
    Output('data-graph', 'figure'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('endpoint-input', 'value'),
     dash.dependencies.State('params-input', 'value')]
)
def update_graph(n_clicks, endpoint, params):
    if endpoint is None:
        return go.Figure()

    if params is not None and isinstance(params, str):
        params = json.loads(params)

    data = data_retriever.get_data(endpoint, params)

    if data is None:
        return go.Figure()

    fig = go.Figure(data=go.Bar(y=list(data.values()), x=list(data.keys())))

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
