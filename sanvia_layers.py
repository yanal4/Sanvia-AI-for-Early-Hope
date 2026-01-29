@tf.keras.utils.register_keras_serializable(package="sanvia")
class TabularEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim: int, dropout_rate: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(embed_dim * 2, activation='gelu')
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dense2 = tf.keras.layers.Dense(embed_dim, activation='gelu')
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.norm1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.norm2(x, training=training)
        x = self.dropout2(x, training=training)
        return x

@tf.keras.utils.register_keras_serializable(package="sanvia")
class EfficientCrossAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads: int = 8, key_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.query_proj = tf.keras.layers.Conv2D(num_heads * key_dim, 1, activation='gelu')
        self.key_proj = tf.keras.layers.Conv2D(num_heads * key_dim, 1, activation='gelu')
        self.value_proj = tf.keras.layers.Conv2D(num_heads * key_dim, 1, activation='gelu')
        self.output_proj = tf.keras.layers.Conv2D(key_dim, 1)
        self.norm = tf.keras.layers.BatchNormalization()

    def call(self, query_features, key_value_features, training=False):
        batch_size = tf.shape(query_features)[0]
        H, W = tf.shape(query_features)[1], tf.shape(query_features)[2]

        queries = self.query_proj(query_features)
        keys = self.key_proj(key_value_features)
        values = self.value_proj(key_value_features)

        queries = tf.reshape(queries, [batch_size, H, W, self.num_heads, self.key_dim])
        keys = tf.reshape(keys, [batch_size, H, W, self.num_heads, self.key_dim])
        values = tf.reshape(values, [batch_size, H, W, self.num_heads, self.key_dim])

        attention_scores = tf.einsum('bhwnc,bhwnc->bhwn', queries, keys)
        attention_scores = attention_scores / tf.sqrt(tf.cast(self.key_dim, attention_scores.dtype))
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        attended = tf.einsum('bhwn,bhwnc->bhwnc', attention_probs, values)
        attended = tf.reshape(attended, [batch_size, H, W, self.num_heads * self.key_dim])

        output = self.output_proj(attended)
        output = self.norm(output, training=training)
        return output

@tf.keras.utils.register_keras_serializable(package="sanvia")
class GatedFusionLayer(tf.keras.layers.Layer):
    def __init__(self, feature_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.cc_gate = tf.keras.layers.Dense(feature_dim, activation='sigmoid')
        self.mlo_gate = tf.keras.layers.Dense(feature_dim, activation='sigmoid')
        self.cc_transform = tf.keras.layers.Dense(feature_dim)
        self.mlo_transform = tf.keras.layers.Dense(feature_dim)
        self.norm = tf.keras.layers.BatchNormalization()

    def call(self, cc_features, mlo_features, training=False):
        cc_transformed = self.cc_transform(cc_features)
        mlo_transformed = self.mlo_transform(mlo_features)
        cc_gate = self.cc_gate(cc_features)
        mlo_gate = self.mlo_gate(mlo_features)
        fused = self.norm(cc_transformed * cc_gate + mlo_transformed * mlo_gate, training=training)
        return fused


@tf.keras.utils.register_keras_serializable(package="sanvia")
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights, gamma=3, alpha=0.5, name="focal_loss"):
        super().__init__(name=name)
        self.class_weights = tf.constant(class_weights[None, :], dtype=tf.float32)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        ce = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = self.alpha * tf.pow(1 - pt, self.gamma)
        class_weight_vec = tf.reduce_sum(y_true * self.class_weights, axis=-1)
        loss = focal_weight * class_weight_vec * ce
        return tf.reduce_mean(loss)