from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px

import numpy as np


class TrainPlot4:

    def __init__(self):
        self.fig = None

    def init_plot(self, width=1000, height=700):

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Trade Balance", "Penalties", "Num of trades by profit"),
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
            shared_yaxes=False,
            shared_xaxes=True,
            specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]])

        fig.update_layout(height=height, width=width)


        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        # Добавляем первый график с тренировочным балансом.
        fig.add_trace(go.Scatter(name="Train", mode='lines', legendgroup="1",
                                 line={"color": px.colors.qualitative.G10[3], "width": 1}),
                      row=1, col=1)

        # Добавляем первый график с тестовым балансом.
        fig.add_trace(go.Scatter(name="Test", mode='lines', legendgroup="1",
                                 line={"color": px.colors.qualitative.G10[2], "width": 1}),
                      row=1, col=1)
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        # Добавляем график с количество штрафов на тренировочном датасете.
        fig.add_trace(go.Scatter(name="Train", mode='lines', legendgroup="2", 
                                 line={"color": px.colors.qualitative.D3[3], "width": 1}),
                      row=2, col=1, secondary_y=False)

        # Добавляем график с количество штрафов на Тестовом датасете.
        fig.add_trace(go.Scatter(name="Test", mode='lines', legendgroup="2",
                                 line={"color": px.colors.qualitative.G10[2], "width": 1}),
                      row=2, col=1, secondary_y=False)

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # Добавляем график с количеством положительных и отрицательных торговых операций.
        fig.add_trace(go.Scatter(name="PosTrades", mode='lines', legendgroup="3",
                                 line={"color": px.colors.qualitative.Dark24[19], "width": 1, "dash" :'dot'}),
                      row=3, col=1, secondary_y=False)

        # Добавляем график с количеством положительных и отрицательных торговых операций.
        fig.add_trace(go.Scatter(name="NegTrades", mode='lines', legendgroup="3",
                                 line={"color": px.colors.qualitative.Dark24[3], "width": 1, "dash" :'dot'}),
                      row=3, col=1)

        # Добавляем график с оценкой разреженности.
        fig.add_trace(go.Scatter(name="Sparsity", mode='lines', legendgroup="3",
                                 line={"color": px.colors.qualitative.Dark2[7], "width": 1}),
                      row=3, col=1, secondary_y=True)

        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        # fig.update_layout(legend=dict(
        #    yanchor="top",
        #    y=0.99,
        #    xanchor="right",
        #    x=0.01,
        #    bgcolor='rgba(255,255,255,0)'
        # ))

        fig.layout.yaxis6.showgrid = False
        fig.update_layout(yaxis6=dict(range=[0, 1]))

        fig.update_layout(
            autosize=False,
            paper_bgcolor='rgba(255,255,255,0)',
            plot_bgcolor='rgba(245,245,245,245)',
            legend_tracegroupgap=180
        )

        self.fig = go.FigureWidget(fig)
        display(self.fig)

    def get_data(self, history, domain, feature):
        x, y = [], []
        for step in history:
            if domain in step:
                x.append(step.get("episode"))
                y.append(step.get(domain, {}).get(feature, 0))
        return x, y

    def update_plot(self, history):
        self.fig.data[0].x, self.fig.data[0].y = history.get_data("train", "Balance")
        self.fig.data[1].x, self.fig.data[1].y = history.get_data("test", "Balance")

        self.fig.data[2].x, self.fig.data[2].y = history.get_data("train", "Penalties")
        self.fig.data[3].x, self.fig.data[3].y = history.get_data("test", "Penalties")

        self.fig.data[4].x, self.fig.data[4].y = history.get_data("train", "PosTrades")
        self.fig.data[5].x, self.fig.data[5].y = history.get_data("train", "NegTrades")

        steps_0_x, steps_0_y = history.get_data("train", "StepsClosed")
        steps_1_x, steps_1_y = history.get_data("train", "StepsOpened")
        data = np.array(steps_0_y) / (np.array(steps_1_y) + np.array(steps_0_y))
        self.fig.data[6].x, self.fig.data[6].y = steps_0_x, data

        self.fig.update_traces()
