from plotly.subplots import make_subplots
import plotly.graph_objs as go

from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px


class TrainPlot2:
    ROWS = 3
    COLS = 1

    def __init__(self):
        self.fig = None

    def init_plot(self, width=1000, height=700):

        # Создаем fig
        fig = make_subplots(
            rows=self.ROWS,
            cols=self.COLS,
            vertical_spacing=0.04,
            horizontal_spacing=0.1,
            shared_yaxes=False
        )

        # Добавляем первый график с тренировочным балансом.
        fig.add_trace(go.Scatter(
            name="train Balance",
            legendgroup="1",
            line={"color": px.colors.qualitative.G10[3], "width": 1}
        ), row=1, col=1)

        # Добавляем первый график с тестовым балансом.
        fig.add_trace(go.Scatter(
            name="test Balance",
            mode='lines',
            legendgroup="1",
            line={"color": px.colors.qualitative.G10[2], "width": 1}
        ), row=1, col=1)

        # Добавляем график с количество штрафов на тренировочном датасете.
        fig.add_trace(go.Scatter(
            name="train Penalties",
            mode='lines',
            legendgroup="2",
            line={"color": px.colors.qualitative.Dark2[1], "width": 1}
        ), row=2, col=1)

        # Добавляем график с тетовым балансом.
        fig.add_trace(go.Scatter(
            name="test Penalties",
            mode='lines',
            legendgroup="2",
            line={"color": px.colors.qualitative.Dark2[2], "width": 1}
        ), row=2, col=1)

        # Добавляем график с количеством положительных и отрицательных торговых операций.
        fig.add_trace(go.Scatter(
            name="test PosTrades",
            mode='lines',
            legendgroup="3",
            line={"color": px.colors.qualitative.Dark24[19], "width": 1}
        ), row=3, col=1)

        # Добавляем график с количеством положительных и отрицательных торговых операций.
        fig.add_trace(go.Scatter(
            name="test NegTrades",
            mode='lines',
            legendgroup="3",
            line={"color": px.colors.qualitative.Dark24[3], "width": 1}
        ), row=3, col=1)

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
