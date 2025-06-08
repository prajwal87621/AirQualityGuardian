const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const cron = require('node-cron');

// Configuration
const config = {
    port: process.env.PORT || 3000,
    mongoUri: process.env.MONGO_URI || 'mongodb://localhost:27017/air_quality_db',
    mlScript: path.join(__dirname, 'ml_predictor.py'),
    dataExportPath: path.join(__dirname, 'sensor_data.csv'),
    predictionInterval: '*/30 * * * *', // Every 30 minutes
    dataRetentionDays: 365
};

// MongoDB Schema
const sensorDataSchema = new mongoose.Schema({
    timestamp: { type: Date, default: Date.now, index: true },
    deviceId: { type: String, default: 'ESP32_AQM_01' },
    
    // Environmental sensors
    temperature: { type: Number, required: true },
    humidity: { type: Number, required: true },
    pressure: { type: Number, required: true },
    
    // Air quality sensors
    pm25: { type: Number, required: true },
    voc: { type: Number, required: true },
    dustDensity: { type: Number, required: true },
    
    // Gas sensors
    co: { type: Number, required: true },
    no2: { type: Number, required: true },
    nh3: { type: Number, required: true },
    
    // Calculated values - USE ESP32's calculated AQI
    aqi: { type: Number, required: true },
    
    // GPS data
    gpsLat: { type: Number },
    gpsLon: { type: Number },
    gpsAlt: { type: Number },
    gpsValid: { type: Boolean, default: false },
    placeName: { type: String },
    
    // Device status
    fanState: { type: Boolean, default: false },
    deviceStatus: { type: String, default: 'online' },
    batteryLevel: { type: Number },
    signalStrength: { type: Number }
});

const predictionSchema = new mongoose.Schema({
    timestamp: { type: Date, default: Date.now, index: true },
    predictionDate: { type: Date, required: true },
    predictedAQI: { type: Number, required: true },
    predictedPM25: { type: Number, required: true },
    predictedTemp: { type: Number, required: true },
    confidence: { type: Number, default: 0.8 },
    modelAccuracy: { type: Number, default: 0.8 },
    dataPoints: { type: Number, default: 0 }
});

// Models
const SensorData = mongoose.model('SensorData', sensorDataSchema);
const Prediction = mongoose.model('Prediction', predictionSchema);

// Express app setup
const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Store active WebSocket connections
const clients = new Set();

// Utility Functions
function calculateAQI(pm25, co, no2, voc) {
    // BACKUP AQI calculation - only used if ESP32 doesn't send AQI
    // EPA AQI calculation for PM2.5 (primary pollutant)
    let aqi = 0;
    
    if (pm25 <= 12.0) {
        aqi = linearInterpolation(pm25, 0, 12.0, 0, 50);
    } else if (pm25 <= 35.4) {
        aqi = linearInterpolation(pm25, 12.1, 35.4, 51, 100);
    } else if (pm25 <= 55.4) {
        aqi = linearInterpolation(pm25, 35.5, 55.4, 101, 150);
    } else if (pm25 <= 150.4) {
        aqi = linearInterpolation(pm25, 55.5, 150.4, 151, 200);
    } else if (pm25 <= 250.4) {
        aqi = linearInterpolation(pm25, 150.5, 250.4, 201, 300);
    } else {
        aqi = linearInterpolation(pm25, 250.5, 500.4, 301, 500);
    }
    
    // Adjust for other pollutants
    if (co > 15) aqi += 10;
    if (no2 > 100) aqi += 15;
    if (voc > 220) aqi += 20;
    
    return Math.round(Math.max(0, Math.min(500, aqi)));
}

function linearInterpolation(value, x1, x2, y1, y2) {
    return ((value - x1) / (x2 - x1)) * (y2 - y1) + y1;
}

async function getLocationName(lat, lon) {
    try {
        // Use a free geocoding service (you might want to use a proper API key)
        const response = await fetch(
            `https://api.openweathermap.org/geo/1.0/reverse?lat=${lat}&lon=${lon}&limit=1&appid=your_api_key`
        );
        
        if (response.ok) {
            const data = await response.json();
            if (data && data.length > 0) {
                const location = data[0];
                return `${location.name}, ${location.country}`;
            }
        }
    } catch (error) {
        console.error('Geocoding error:', error);
    }
    
    // Fallback location names based on coordinates
    if (lat >= 28.5 && lat <= 28.7 && lon >= 77.1 && lon <= 77.3) {
        return 'New Delhi, India';
    } else if (lat >= 19.0 && lat <= 19.3 && lon >= 72.8 && lon <= 73.0) {
        return 'Mumbai, India';
    } else if (lat >= 12.9 && lat <= 13.1 && lon >= 77.5 && lon <= 77.7) {
        return 'Bangalore, India';
    }
    
    return `Location (${lat.toFixed(4)}, ${lon.toFixed(4)})`;
}

function shouldActivateFan(aqi, pm25, voc) {
    // Fan activation logic - use ESP32's AQI primarily
    return aqi > 100 || pm25 > 35 || voc > 150;
}

// WebSocket handling
wss.on('connection', (ws) => {
    console.log('Client connected');
    clients.add(ws);
    
    // Send initial data to new client
    sendInitialData(ws);
    
    ws.on('close', () => {
        console.log('Client disconnected');
        clients.delete(ws);
    });
    
    ws.on('error', (error) => {
        console.error('WebSocket error:', error);
        clients.delete(ws);
    });
});

async function sendInitialData(ws) {
    try {
        // Get latest sensor data
        const latestSensorData = await SensorData.find()
            .sort({ timestamp: -1 })
            .limit(10);
        
        // Get latest predictions
        const latestPredictions = await Prediction.find()
            .sort({ predictionDate: 1 })
            .limit(7);
        
        const initialData = {
            type: 'initial_data',
            sensorData: latestSensorData,
            predictions: latestPredictions
        };
        
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(initialData));
        }
    } catch (error) {
        console.error('Error sending initial data:', error);
    }
}

function broadcastToClients(data) {
    const message = JSON.stringify(data);
    clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(message);
        }
    });
}

// API Routes
app.get('/api/health', (req, res) => {
    res.json({ 
        status: 'ok', 
        timestamp: new Date().toISOString(),
        clients: clients.size,
        mongodb: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected'
    });
});

// Get latest sensor data
app.get('/api/sensor/latest', async (req, res) => {
    try {
        const latestData = await SensorData.findOne().sort({ timestamp: -1 });
        res.json(latestData);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Get sensor data history
app.get('/api/sensor/history', async (req, res) => {
    try {
        const { hours = 24, limit = 100 } = req.query;
        const startTime = new Date(Date.now() - hours * 60 * 60 * 1000);
        
        const data = await SensorData.find({ 
            timestamp: { $gte: startTime } 
        })
        .sort({ timestamp: -1 })
        .limit(parseInt(limit));
        
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Get predictions
app.get('/api/predictions', async (req, res) => {
    try {
        const predictions = await Prediction.find()
            .sort({ predictionDate: 1 })
            .limit(7);
        
        res.json(predictions);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Receive sensor data from ESP32
app.post('/api/sensor/data', async (req, res) => {
    try {
        const sensorData = req.body;
        
        // **FIXED**: Use ESP32's calculated AQI instead of recalculating
        let aqi = sensorData.aqi;
        
        // Only calculate AQI if ESP32 didn't send it or sent invalid value
        if (!aqi || aqi <= 0 || aqi > 500) {
            console.log('ESP32 AQI invalid or missing, calculating backup AQI');
            aqi = calculateAQI(
                sensorData.pm25 || 0,
                sensorData.co || 0,
                sensorData.no2 || 0,
                sensorData.voc || 0
            );
        } else {
            console.log(`Using ESP32 calculated AQI: ${aqi}`);
        }
        
        // Determine fan state based on ESP32's AQI
        const fanState = shouldActivateFan(aqi, sensorData.pm25 || 0, sensorData.voc || 0);
        
        // Get location name if GPS is valid
        let placeName = 'Unknown Location';
        if (sensorData.gpsValid && sensorData.gpsLat && sensorData.gpsLon) {
            placeName = await getLocationName(sensorData.gpsLat, sensorData.gpsLon);
        }
        
        // Create sensor data document with ESP32's AQI
        const newSensorData = new SensorData({
            ...sensorData,
            aqi,  // Use ESP32's AQI (or backup if invalid)
            fanState,
            placeName,
            timestamp: new Date()
        });
        
        // Save to database
        await newSensorData.save();
        
        // Broadcast to connected clients
        broadcastToClients({
            type: 'sensor_update',
            data: newSensorData
        });
        
        // Response to ESP32
        res.json({
            success: true,
            aqi,  // Return the AQI we're using
            fanState,
            timestamp: newSensorData.timestamp,
            message: 'Data received successfully'
        });
        
        console.log(`Sensor data received - AQI: ${aqi} (ESP32: ${sensorData.aqi}), Fan: ${fanState ? 'ON' : 'OFF'}`);
        
    } catch (error) {
        console.error('Error processing sensor data:', error);
        res.status(500).json({ error: error.message });
    }
});

// Manual prediction trigger
app.post('/api/predictions/generate', async (req, res) => {
    try {
        const predictions = await generatePredictions();
        res.json(predictions);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Export data as CSV
app.get('/api/export/csv', async (req, res) => {
    try {
        await exportDataToCSV();
        res.download(config.dataExportPath);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// ML Prediction Functions
async function exportDataToCSV() {
    try {
        const data = await SensorData.find()
            .sort({ timestamp: -1 })
            .limit(1000);
        
        if (data.length === 0) {
            throw new Error('No data to export');
        }
        
        // CSV headers
        const headers = [
            'Date', 'AQI', 'Temperature', 'Humidity', 'Pressure',
            'PM2.5', 'VOC', 'DustDensity', 'CO', 'NO2', 'NH3',
            'GPSLat', 'GPSLon', 'FanState'
        ];
        
        // Create target columns for ML training
        const csvData = data.reverse().map((record, index) => {
            const nextRecord = data[index + 1];
            return [
                record.timestamp.toISOString(),
                record.aqi,  // Use stored AQI (from ESP32)
                record.temperature,
                record.humidity,
                record.pressure,
                record.pm25,
                record.voc,
                record.dustDensity,
                record.co,
                record.no2,
                record.nh3,
                record.gpsLat || 0,
                record.gpsLon || 0,
                record.fanState ? 1 : 0,
                // Add next values for training
                nextRecord ? nextRecord.aqi : record.aqi,
                nextRecord ? nextRecord.pm25 : record.pm25,
                nextRecord ? nextRecord.temperature : record.temperature
            ];
        });
        
        // Add target headers
        headers.push('AQI_next', 'PM25_next', 'Temperature_next');
        
        // Write CSV
        const csvContent = [
            headers.join(','),
            ...csvData.map(row => row.join(','))
        ].join('\n');
        
        fs.writeFileSync(config.dataExportPath, csvContent);
        console.log(`Exported ${data.length} records to CSV`);
        
    } catch (error) {
        console.error('CSV export error:', error);
        throw error;
    }
}

async function generatePredictions() {
    try {
        // Export current data to CSV
        await exportDataToCSV();
        
        // Run ML prediction script
        const predictions = await runMLPrediction();
        
        // Save predictions to database
        const predictionDocs = [];
        const baseDate = new Date();
        
        for (let i = 0; i < predictions.predictions.length; i++) {
            const predData = predictions.predictions[i];
            const predDate = new Date(baseDate);
            predDate.setDate(predDate.getDate() + i + 1);
            
            predictionDocs.push(new Prediction({
                predictionDate: predDate,
                predictedAQI: Math.round(predData.aqi),
                predictedPM25: Math.round(predData.pm25 * 10) / 10,
                predictedTemp: Math.round(predData.temperature * 10) / 10,
                confidence: predictions.confidence || 0.8,
                modelAccuracy: predictions.accuracy || 0.8,
                dataPoints: predictions.data_points || 0
            }));
        }
        
        // Remove old predictions
        await Prediction.deleteMany({});
        
        // Insert new predictions
        await Prediction.insertMany(predictionDocs);
        
        // Broadcast to clients
        broadcastToClients({
            type: 'predictions_update',
            predictions: predictionDocs
        });
        
        console.log(`Generated ${predictionDocs.length} predictions`);
        return predictionDocs;
        
    } catch (error) {
        console.error('Prediction generation error:', error);
        throw error;
    }
}

function runMLPrediction() {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python3', [config.mlScript, config.dataExportPath]);
        
        let stdoutData = '';
        let stderrData = '';
        
        pythonProcess.stdout.on('data', (data) => {
            stdoutData += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
            stderrData += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
            if (code === 0) {
                try {
                    const result = JSON.parse(stdoutData);
                    resolve(result);
                } catch (parseError) {
                    reject(new Error('Failed to parse ML prediction result'));
                }
            } else {
                console.error('ML Script Error:', stderrData);
                reject(new Error(`ML prediction failed with code ${code}`));
            }
        });
        
        pythonProcess.on('error', (error) => {
            reject(new Error(`Failed to start ML script: ${error.message}`));
        });
        
        // Set timeout
        setTimeout(() => {
            pythonProcess.kill();
            reject(new Error('ML prediction timeout'));
        }, 60000); // 60 second timeout
    });
}

// Database cleanup
async function cleanupOldData() {
    try {
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - config.dataRetentionDays);
        
        const result = await SensorData.deleteMany({
            timestamp: { $lt: cutoffDate }
        });
        
        console.log(`Cleaned up ${result.deletedCount} old records`);
    } catch (error) {
        console.error('Data cleanup error:', error);
    }
}

// Scheduled tasks
cron.schedule(config.predictionInterval, async () => {
    console.log('Running scheduled prediction generation...');
    try {
        await generatePredictions();
    } catch (error) {
        console.error('Scheduled prediction error:', error);
    }
});

// Daily cleanup at 2 AM
cron.schedule('0 2 * * *', cleanupOldData);

// Database connection
mongoose.connect(config.mongoUri, {
    useNewUrlParser: true,
    useUnifiedTopology: true
})
.then(() => {
    console.log('Connected to MongoDB');
    
    // Create indexes for performance
    SensorData.createIndexes();
    Prediction.createIndexes();
})
.catch(err => {
    console.error('MongoDB connection error:', err);
    process.exit(1);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
    console.log('Shutting down gracefully...');
    
    // Close WebSocket connections
    clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
            client.close();
        }
    });
    
    // Close MongoDB connection
    await mongoose.connection.close();
    
    // Close server
    server.close(() => {
        console.log('Server closed');
        process.exit(0);
    });
});

// Start server
server.listen(config.port, () => {
    console.log(`Air Quality Guardian Backend running on port ${config.port}`);
    console.log(`WebSocket server ready`);
    console.log(`MongoDB: ${config.mongoUri}`);
    
    // Generate initial predictions after startup
    setTimeout(async () => {
        try {
            const dataCount = await SensorData.countDocuments();
            if (dataCount > 10) {
                console.log('Generating initial predictions...');
                await generatePredictions();
            }
        } catch (error) {
            console.error('Initial prediction error:', error);
        }
    }, 5000);
});

module.exports = { app, server };