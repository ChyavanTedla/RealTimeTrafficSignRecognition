// Get references to the HTML elements
const video = document.getElementById('webcam');
const statusText = document.getElementById('status');
const predText = document.getElementById('prediction-text');
const confText = document.getElementById('confidence-text');

let model;

// --- CLASS NAMES ARRAY ---
// This list is based on your 'traffic_sign.csv' file (ClassId 0-58).
const CLASS_NAMES = [
    'Give way', 'No entry', 'One-way traffic', 'One-way traffic', 'No vehicles in both directions',
    'No entry for cycles', 'No entry for goods vehicles', 'No entry for pedestrians', 'No entry for bullock carts',
    'No entry for hand carts', 'No entry for motor vehicles', 'Height limit', 'Weight limit', 'Axle weight limit',
    'Length limit', 'No left turn', 'No right turn', 'No overtaking', 'Maximum speed limit (90 km/h)',
    'Maximum speed limit (110 km/h)', 'Horn prohibited', 'No parking', 'No stopping', 'Turn left',
    'Turn right', 'Steep descent', 'Steep ascent', 'Narrow road', 'Narrow bridge', 'Unprotected quay',
    'Road hump', 'Dip', 'Loose gravel', 'Falling rocks', 'Cattle', 'Crossroads', 'Side road junction',
    'Side road junction', 'Oblique side road junction', 'Oblique side road junction', 'T-junction',
    'Y-junction', 'Staggered side road junction', 'Staggered side road junction', 'Roundabout',
    'Guarded level crossing ahead', 'Unguarded level crossing ahead', 'Level crossing countdown marker',
    'Level crossing countdown marker', 'Level crossing countdown marker', 'Level crossing countdown marker',
    'Parking', 'Bus stop', 'First aid post', 'Telephone', 'Filling station', 'Hotel', 'Restaurant', 'Refreshments'
];

// --- 1. Load the Model ---
async function loadModel() {
    console.log("Loading model...");
    statusText.innerText = "Loading Model... (this may take a moment)";
    
    // Path to your model.json file
    const modelPath = 'web_model/model.json';
    
    try {
        // Use loadLayersModel for Keras models
        model = await tf.loadLayersModel(modelPath);
        console.log("Model loaded successfully.");
        statusText.innerText = "Model Loaded. Starting Webcam...";
    } catch (err) {
        console.error("Failed to load model: ", err);
        statusText.innerText = "Error: Could not load model.";
    }
}

// --- 2. Access the Webcam ---
async function setupWebcam() {
    return new Promise((resolve, reject) => {
        navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 },
            audio: false 
        })
        .then(stream => {
            video.srcObject = stream;
            video.addEventListener('loadeddata', () => {
                statusText.innerText = "Webcam active. Detecting...";
                resolve();
            }, false);
        })
        .catch(err => {
            console.error("Error accessing webcam: ", err);
            statusText.innerText = "Error: Could not access webcam.";
            reject(err);
        });
    });
}

// --- 3. Run the Prediction Loop ---
async function runPrediction() {
    if (!model) {
        setTimeout(runPrediction, 100);
        return;
    }

    // 1. Get frame from video
    const tensor = tf.browser.fromPixels(video);

    // 2. Preprocess the frame (This matches your original Python script)
    //    Resize the *entire frame* to 48x48, normalize, and add batch dimension
    const processedTensor = tensor
        .resizeBilinear([48, 48]) // Model expects 48x48
        .toFloat()
        .div(255.0)               // Normalize
        .expandDims(0);           // Add batch dimension

    // 3. Run prediction
    const prediction = model.predict(processedTensor);
    
    // 4. Get the results
    const predictionData = await prediction.data();
    const confidence = Math.max(...predictionData); // Get the highest confidence
    const classId = predictionData.indexOf(confidence); // Get the class ID for that confidence
    const label = CLASS_NAMES[classId];

    // 5. Display the results
    if (confidence > 0.75) { // Confidence threshold
        predText.innerText = label;
        confText.innerText = `Confidence: ${Math.round(confidence * 100)}%`;
    } else {
        predText.innerText = "---";
        confText.innerText = "(Hold sign close to camera)";
    }
    
    // 6. Clean up tensors
    tensor.dispose();
    processedTensor.dispose();
    prediction.dispose();
    
    // 7. Run this function again on the next frame
    requestAnimationFrame(runPrediction);
}

// --- Main function to start everything ---
async function main() {
    await loadModel();
    await setupWebcam();
    runPrediction();
}

main();
