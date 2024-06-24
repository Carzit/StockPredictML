import plotly.graph_objs as go
import plotly.offline as py

def plot_plotly(df_test, model_name="Linear Regression"):
    forecast = df_test[['Date']].copy()
    forecast['yhat'] = df_test['predict']
    forecast['yhat_lower'] = forecast['yhat'] - 10
    forecast['yhat_upper'] = forecast['yhat'] + 10

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_test['Date'], y=df_test["Close"], mode='markers', name='Actual'))
    fig.add_trace(go.Scatter(x=forecast['Date'], y=forecast['yhat'], mode='lines', name='Predicted'))

    fig.add_trace(go.Scatter(
        x=forecast['Date'].tolist() + forecast['Date'].tolist()[::-1],
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name='Confidence Interval'
    ))

    fig.update_layout(
        title=f"{model_name} Forecast with Confidence Interval",
        xaxis_title="Date",
        yaxis_title="Value",
        showlegend=True
    )

    py.iplot(fig)