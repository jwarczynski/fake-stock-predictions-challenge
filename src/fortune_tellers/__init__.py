import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_preds_dataframes(df, stocks, train_size, predict_size, predictor):
    times_train = df.Time[:train_size]
    times_all = df.Time

    preds_all_df = pd.DataFrame(index=np.arange(len(df.Time), len(df.Time) + predict_size))
    preds_train_df = pd.DataFrame(index=np.arange(train_size, train_size + predict_size))

    for stock in stocks:
        stock_hist_train = df[stock][:train_size]
        stock_hist_all = df[stock]

        stock_pred_train = predictor(times_train, stock_hist_train)
        stock_pred_all = predictor(times_all, stock_hist_all)

        preds_train_df[stock] = stock_pred_train
        preds_all_df[stock] = stock_pred_all

    return preds_train_df, preds_all_df


def plot_predicitons_with_train_data(axs, stocks, df, preds_all_df, train_size, predict_size, preds_train_df=None):

    for ax, stock in zip(axs, stocks):
        ax.plot(df.Time, df[stock], label="Real")
        if preds_train_df is not None:
            ax.plot(preds_train_df.index, preds_train_df[stock], label="Predicted Train")
            ax.vlines([train_size], 0, df[stock].max(), colors='r', linestyles='dashed', label="Train-Test Split")
        ax.plot(preds_all_df.index, preds_all_df[stock], label="Future prediction")
        ax.vlines([train_size + predict_size], 0, df[stock].max(), colors='g', linestyles='dashed',
                      label="Prediction Start")
        ax.set_title(stock)
        # axs[i].legend()


