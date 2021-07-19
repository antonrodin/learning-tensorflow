async function makePrediction(callback) {
    // 1. Make a Model
    const model = tf.sequential();

    // 2. Make Layers
    model.add(tf.layers.dense({ 
        inputShape: [1], 
        units: 1 
    }));

    // 3. Compile the models
    model.compile({
        optimizer: 'sgd',
        loss: 'meanSquaredError',
        metrics: [
            'accuracy'
        ]
    });

    // 4. Get a Training Set
    let data = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [8, 1]);
    let labels = tf.tensor2d([-1, -2, -3, -4, -5, -6, -7, -8], [8, 1]);

    // 5. Train the model
    await model.fit(data, labels, {
        epochs: 500
    });

    // 6. Make a predictions
    let prediction = await model.predict(tf.tensor2d([13], [1, 1]));

    callback(prediction)
}

document.getElementById('prediction').innerHTML = "please wait a second";

// Paint prediction
makePrediction((prediction) => {
    document.getElementById('prediction').innerHTML = prediction;
})