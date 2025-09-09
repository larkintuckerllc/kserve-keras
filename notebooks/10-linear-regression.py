# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Kserve Keras
#     language: python
#     name: kserve-keras
# ---

# %% [markdown]
# # imports

# %%
import os
import pandas as pd
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

# %% [markdown]
# # constants

# %%
BATCH_SIZE=10
EPOCHS = 50
LEARNING_RATE=0.2

# %% [markdown]
# # fetch

# %%
training_df = pd.read_csv("../data/mpg-pounds.csv")
training_df.head()

# %% [markdown]
# # train

# %%
inputs = keras.Input(shape=(1,))
outputs = keras.layers.Dense(1)(inputs)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
    loss=keras.losses.mean_squared_error,
    metrics=[keras.metrics.RootMeanSquaredError()])

feature = training_df["pounds"].values
label = training_df["mpg"].values
history = model.fit(x=feature,
            y=label,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS)
