from plotly.subplots import make_subplots
import plotly.graph_objs as go

from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px


class TrainPlot2:
    PLOTS = [
        {"name": "Balances", "plots": [
            {"data": ("train", "Balance"), "params": {"line": {"color": px.colors.qualitative.G10[3], "width": 1}}},
            {"data": ("test", "Balance"), "params": {"line": {"color": px.colors.qualitative.G10[2], "width": 1}}}
        ]},
        {"name": "Penalties", "plots": [
            {"data": ("train", "Penalties"), "params": {"line": {"color": px.colors.qualitative.Dark2[1], "width": 1}}},
            {"data": ("test", "Penalties"), "params": {"line": {"color": px.colors.qualitative.Dark2[2], "width": 1}}}
        ]},
        {"name": "Trade count", "plots": [
            {"data": ("train", "PosTrades"),
             "params": {"line": {"color": px.colors.qualitative.Dark24[19], "width": 1}}},
            {"data": ("train", "NegTrades"),
             "params": {"line": {"color": px.colors.qualitative.Dark24[3], "width": 1}}}
        ]}
    ]

    def __init__(self):
        self.fig = None

    def init_plot(self, width=1000, height=700):
        rows = len(self.PLOTS)

        fig = make_subplots(rows=rows,
                            vertical_spacing=0.04,
                            horizontal_spacing=0.1,
                            shared_yaxes=False
                            )

        for row_id in range(rows):
            plot = self.PLOTS[row_id]

            row = row_id + 1
            for line in plot["plots"]:
                name = " ".join(line.get("data"))
                params = line.get("params")
                fig.add_scatter(row=row, col=1, name=name, legendgroup=str(row), **params)

        self.fig = go.FigureWidget(fig)
        self.fig.update_layout(
            autosize=False,
            width=width,
            height=height,
            paper_bgcolor='rgba(255,255,255,255)',
            plot_bgcolor='rgba(245,245,245,245)',
            legend_tracegroupgap=180
        )

        display(self.fig)

    def get_data(self, history, domain, feature):
        x, y = [], []
        for step in history:
            if domain in step:
                x.append(step.get("episode"))
                y.append(step.get(domain, {}).get(feature, 0))
        return x, y

    def update_plot(self, history):
        for data in self.fig.data:
            domain, feature = data.name.split()
            data_x, data_y = self.get_data(history, domain, feature)

            data.x = data_x
            data.y = data_y

        self.fig.update_traces()
    