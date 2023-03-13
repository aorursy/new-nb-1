from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
market_train_df.head()
market_train_df.tail(2)
news_train_df.head(2)
#Load the data
market_train_df.describe()
news_train_df.describe()
coca = market_train_df.loc[market_train_df['assetCode'] == "KO.N"]
plt.plot(coca["time"], coca['close'])
plt.show()
occur = market_train_df.groupby('assetCode').size()
occur = occur.sort_values(ascending= False)
occur
import matplotlib.pyplot as plt

occur[:1000].plot.barh()
plt.xlabel('Count')
plt.ylabel('Take sequence')
plt.title('Top 10 take sequence')
# You can only iterate through a result from `get_prediction_days()` once
# so be careful not to lose it once you start iterating.
days = env.get_prediction_days()
(market_obs_df, news_obs_df, predictions_template_df) = next(days)
market_obs_df.head()
news_obs_df.head()
predictions_template_df.head()
next(days)
import numpy as np
def make_random_predictions(predictions_df):
    predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0
make_random_predictions(predictions_template_df)
env.predict(predictions_template_df)
(market_obs_df, news_obs_df, predictions_template_df) = next(days)
market_obs_df.head()
news_obs_df.head()
predictions_template_df.head()
make_random_predictions(predictions_template_df)
env.predict(predictions_template_df)
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_random_predictions(predictions_template_df)
    env.predict(predictions_template_df)
print('Done!')
env.write_submission_file()
# We've got a submission file!
import os
print([filename for filename in os.listdir('.') if '.csv' in filename])