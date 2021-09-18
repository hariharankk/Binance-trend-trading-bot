import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
def vis(df,Crypto):
  df.dropna(subset = ["lower_bound"], inplace=True)
  df.dropna(subset = ["atr"], inplace=True)
  fig = go.Figure()
  fig.add_trace(go.Candlestick(x=df['timestamp'],
                  open=df['Open'],
                  high=df['High'],
                  low=df['Low'],
                  close=df['Close'],increasing_line_color= 'black', decreasing_line_color= 'yellow'))
  fig.update_layout(
      title= {
          'text': Crypto,
          'y':0.9,
          'x':0.5,
          'xanchor': 'center',
          'yanchor': 'top'},
          font=dict(
            family="Courier New, monospace",
            size=20,
            color="#7f7f7f"
          )
      )
  fig.add_trace(go.Scatter(x=df['timestamp'],y=df['lower_bound'],name='lowerbound'))
  fig.add_trace(go.Scatter(x=df['timestamp'],y=df['atr'],name='atr'))
  fig.show()
