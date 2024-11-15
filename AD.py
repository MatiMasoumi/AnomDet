import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Add
from tensorflow.keras.layers import Input, Embedding , GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.regularizers import l2
data = pd.read_csv('/content/drive/MyDrive/creditcard.csv')
counts = data['Class'].value_counts()
print(counts)
X = data.drop('Class', axis=1)
y = data['Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,test_size=0.2,random_state=42)
maxlen = X_train.shape[1]
vocab_size = maxlen
embed_dim = 128
num_heads = 16
ff_dim = 256
dropout_rate = 0.1
def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)
def transformer_block(inputs, embed_dim, num_heads, ff_dim, rate=0.1):
  attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
  attn_output = Dropout(rate)(attn_output)
  out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

  ffn_output = Dense(ff_dim, activation='relu')(out1)
  ffn_output = Dense(embed_dim)(ffn_output)
  ffn_output = Dropout(rate)(ffn_output)
  return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
def build_model(vocab_size, maxlen, embed_dim, num_heads, ff_dim):
  inputs = Input(shape=(maxlen,))
  embedding_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
  pos_encoding = positional_encoding(maxlen, embed_dim)
  x = embedding_layer + pos_encoding

  x = transformer_block(x, embed_dim, num_heads, ff_dim)
  x = GlobalAveragePooling1D()(x)
  x = Dropout(0.1)(x)
  x = Dense(64, activation='relu')(x)
  x = Dropout(0.1)(x)
  outputs = Dense(1, activation='sigmoid')(x)

  model = Model(inputs=inputs, outputs=outputs)

  model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy",
                metrics = [Precision(), Recall()])
  return model
model = build_model(vocab_size, maxlen, embed_dim, num_heads, ff_dim)
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    X_train[..., np.newaxis],
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    #,class_weight=class_weight_dict
)
y_pred = model.predict(X_test).flatten()
threshold = 0.5

y_pred_labels = (y_pred > 0.5).astype(int)

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred_labels, average='binary')
roc_auc = roc_auc_score(y_test, y_pred)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1_score:.2f}')
print(f'ROC AUC: {roc_auc:.2f}')