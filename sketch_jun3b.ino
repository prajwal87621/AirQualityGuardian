#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME680.h>
#include <TinyGPSPlus.h>
#include <HardwareSerial.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <WebSocketsClient.h>

// WiFi credentials (update as needed)
const char* ssid = "Praj";  // Replace with your WiFi SSID
const char* password = "1234567890";  // Replace with your WiFi password
const char* serverUrl = "http://10.20.11.189:3000/api/sensor/data";  // Updated to match backend
const char* wsServerUrl = "10.20.11.189";  // WebSocket server IP
const int wsServerPort = 3000;
const char* wsPath = "/";  // Backend uses default WebSocket upgrade path

// Device identification (matches backend schema)
const char* deviceId = "ESP32_AQM_01";  // Default device ID from backend

// Sensor objects
Adafruit_BME680 bme;
TinyGPSPlus gps;
HardwareSerial gpsSerial(1);
WebSocketsClient webSocket;

// Pin definitions
#define MQ9_PIN       34    // MQ9 CO sensor
#define MQ135_PIN     35    // MQ135 air quality sensor
#define RELAY_PIN     4     // Fan relay
#define DUST_VO_PIN   32    // Dust sensor output
#define LED_CONTROL   25    // Dust sensor LED control
#define GPS_RX_PIN    16    // GPS RX
#define GPS_TX_PIN    17    // GPS TX

// Calibration constants
#define MQ9_R0        10.0
#define MQ135_R0      20.0
#define DUST_VOC      0.6
#define DUST_K        0.5

// Variables for sensor data (matching backend schema)
float temperature = 0, humidity = 0, pressure = 0, gasResistance = 0;
float pm25 = 0, voc = 0, dustDensity = 0;
float co = 0, no2 = 0, nh3 = 0;
int aqi = 0;
bool fanState = true;
double gpsLat = 0, gpsLon = 0;
float gpsAlt = 0;
bool gpsValid = true;
String placeName = "Unknown Location";
String deviceStatus = "online";
int batteryLevel = 100;  // Simulated battery level
int signalStrength = 0;

// Timers
unsigned long lastSensorRead = 0;
unsigned long lastHttpPost = 0;
unsigned long lastSerialPrint = 0;
unsigned long lastWsReconnect = 0;
const unsigned long SENSOR_INTERVAL = 2000;
const unsigned long HTTP_POST_INTERVAL = 30000;
const unsigned long SERIAL_PRINT_INTERVAL = 5000;
const unsigned long WS_RECONNECT_INTERVAL = 5000;

// Moving average for noise reduction
#define SAMPLES_COUNT 10
float pm25_samples[SAMPLES_COUNT] = {0};
float co_samples[SAMPLES_COUNT] = {0};
float no2_samples[SAMPLES_COUNT] = {0};
float nh3_samples[SAMPLES_COUNT] = {0};
float voc_samples[SAMPLES_COUNT] = {0};
int sample_index = 0;

// WebSocket connection status
bool wsConnected = false;

// AQI calculation functions (EPA standard, aligned with server.js and ml_predictor.py)
int calculate_aqi_pm25(float C) {
    if (C < 0) C = 0;
    if (C <= 12.0) return (50.0 * C / 12.0);
    if (C <= 35.4) return (49.0 * (C - 12.1) / 23.3 + 51.0);
    if (C <= 55.4) return (49.0 * (C - 35.5) / 19.9 + 101.0);
    if (C <= 150.4) return (49.0 * (C - 55.5) / 94.9 + 151.0);
    if (C <= 250.4) return (99.0 * (C - 150.5) / 99.9 + 201.0);
    if (C <= 350.4) return (99.0 * (C - 250.5) / 99.9 + 301.0);
    if (C <= 500.4) return (99.0 * (C - 350.5) / 149.9 + 401.0);
    return 500;
}

int calculate_aqi_co(float C) {
    if (C < 0) C = 0;
    if (C <= 4.4) return (50.0 * C / 4.4);
    if (C <= 9.4) return (49.0 * (C - 4.5) / 4.9 + 51.0);
    if (C <= 12.4) return (49.0 * (C - 12.5) / 2.9 + 101.0);
    if (C <= 15.4) return (49.0 * (C - 15.5) / 2.9 + 151.0);
    if (C <= 30.4) return (99.0 * (C - 30.5) / 14.9 + 201.0);
    if (C <= 40.4) return (99.0 * (C - 40.5) / 9.9 + 301.0);
    if (C <= 50.4) return (99.0 * (C - 50.5) / 9.9 + 401.0);
    return 500;
}

int calculate_aqi_no2(float C) {
    if (C < 0) C = 0;
    if (C <= 53) return (50.0 * C / 53.0);
    if (C <= 100) return (49.0 * (C - 54.0) / 46.0 + 51.0);
    if (C <= 360) return (49.0 * (C - 101.0) / 259.0 + 101.0);
    if (C <= 649) return (49.0 * (C - 361.0) / 288.0 + 151.0);
    if (C <= 1249) return (99.0 * (C - 650.0) / 599.0 + 201.0);
    if (C <= 1649) return (99.0 * (C - 1250.0) / 399.0 + 301.0);
    if (C <= 2049) return (99.0 * (C - 1650.0) / 399.0 + 401.0);
    return 500;
}

int calculate_aqi_voc(float C) {
    if (C < 0) C = 0;
    if (C < 50) return (int)(50.0 * C / 50.0);
    if (C < 100) return (int)(49.0 * (C - 50) / 50.0 + 51.0);
    if (C < 200) return (int)(49.0 * (C - 100) / 100.0 + 101.0);
    if (C < 300) return (int)(49.0 * (C - 200) / 100.0 + 151.0);
    return (int)(99.0 * (C - 300) / 100.0 + 201.0);
}

// WebSocket event handler
void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
    switch(type) {
        case WStype_DISCONNECTED:
            Serial.println("[WS] Disconnected");
            wsConnected = false;
            break;
        case WStype_CONNECTED:
            Serial.printf("[WS] Connected to: %s\n", payload);
            wsConnected = true;
            break;
        case WStype_TEXT:
            Serial.printf("[WS] Received: %s\n", payload);
            handleWebSocketMessage((char*)payload);
            break;
        case WStype_ERROR:
            Serial.println("[WS] Error");
            wsConnected = false;
            break;
        default:
            break;
    }
}

// Handle incoming WebSocket messages
void handleWebSocketMessage(const char* message) {
    DynamicJsonDocument doc(1024);
    deserializeJson(doc, message);
    
    String msgType = doc["type"];
    if (msgType == "sensor_update") {
        // Handle sensor update from server (e.g., fan control override)
        if (doc["data"]["fanState"].is<bool>()) {
            bool serverFanState = doc["data"]["fanState"];
            if (serverFanState != fanState) {
                fanState = serverFanState;
                digitalWrite(RELAY_PIN, fanState ? HIGH : LOW);
                Serial.printf("[WS] Fan state updated to: %s\n", fanState ? "ON" : "OFF");
            }
        }
    }
}

// Reconnect WebSocket
void reconnectWebSocket() {
    if (!wsConnected && millis() - lastWsReconnect >= WS_RECONNECT_INTERVAL) {
        Serial.println("[WS] Attempting to reconnect...");
        webSocket.begin(wsServerUrl, wsServerPort, wsPath);
        lastWsReconnect = millis();
    }
}

// Calculate moving average
float calculateMovingAverage(float samples[], float new_reading) {
    samples[sample_index] = new_reading;
    float sum = 0;
    for (int i = 0; i < SAMPLES_COUNT; i++) {
        sum += samples[i];
    }
    sample_index = (sample_index + 1) % SAMPLES_COUNT;
    return sum / SAMPLES_COUNT;
}

// Calculate sensor resistance
float calculateRs(int rawAdc, float Rl) {
    float voltage = rawAdc * (3.3 / 4095.0);
    if (voltage < 0.01) return 1000.0;
    return Rl * (3.3 - voltage) / voltage;
}

// Map float values
float mapFloat(float x, float in_min, float in_max, float out_min, float out_max) {
    if (in_max == in_min) return out_min;
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

void readSensors() {
    // Read BME680
    if (bme.performReading()) {
        temperature = bme.temperature;
        humidity = bme.humidity;
        pressure = bme.pressure / 100.0;  // Convert Pa to hPa
        gasResistance = bme.gas_resistance / 1000.0;  // Convert to kOhms
    } else {
        Serial.println("[BME680] Read failed");
        return;
    }

    // Read MQ9 (CO sensor)
    int mq9Raw = analogRead(MQ9_PIN);
    float rs_mq9 = calculateRs(mq9Raw, 10.0);
    float ratio_mq9 = rs_mq9 / MQ9_R0;
    co = calculateMovingAverage(co_samples, 100 * pow(ratio_mq9, -1.52));
    co = constrain(co, 0, 100);

    // Read MQ135 (Air quality - NO2 and NH3)
    int mq135Raw = analogRead(MQ135_PIN);
    float rs_mq135 = calculateRs(mq135Raw, 10.0);
    float ratio_mq135 = rs_mq135 / MQ135_R0;
    no2 = calculateMovingAverage(no2_samples, mapFloat(ratio_mq135, 5.0, 0.5, 0, 1000));
    no2 = constrain(no2, 0, 1000);
    nh3 = calculateMovingAverage(nh3_samples, mapFloat(ratio_mq135, 5.0, 0.1, 0, 1000));
    nh3 = constrain(nh3, 0, 1000);

    // Read dust sensor (PM2.5)
    digitalWrite(LED_CONTROL, LOW);
    delayMicroseconds(280);
    int dustRaw = analogRead(DUST_VO_PIN);
    float dustVoltage = dustRaw * (3.3 / 4095.0);
    delayMicroseconds(40);
    digitalWrite(LED_CONTROL, HIGH);
    delayMicroseconds(9680);

    dustDensity = (dustVoltage - DUST_VOC) * DUST_K * 1000;
    if (dustDensity < 0) dustDensity = 0;
    pm25 = calculateMovingAverage(pm25_samples, dustDensity);
    pm25 = constrain(pm25, 0, 1000);

    // Calculate VOC from BME680 gas resistance
    float log_gas = log10(gasResistance);
    voc = calculateMovingAverage(voc_samples, mapFloat(log_gas, 0.7, 2.7, 400, 0));
    voc = constrain(voc, 0, 1000);

    // Calculate AQI (use max of pollutant AQIs, aligned with server.js and ml_predictor.py)
    int aqi_pm25 = calculate_aqi_pm25(pm25);
    int aqi_co = calculate_aqi_co(co);
    int aqi_no2 = calculate_aqi_no2(no2);
    int aqi_voc = calculate_aqi_voc(voc);
    aqi = max(max(max(aqi_pm25, aqi_co), aqi_no2), aqi_voc);
    aqi = constrain(aqi, 0, 500);  // Align with server.js and ml_predictor.py

    // Control fan based on AQI (threshold aligned with ml_predictor.py)
    if (aqi > 20 && !fanState) {
        digitalWrite(RELAY_PIN, HIGH);
        fanState = true;
        Serial.println("[RELAY] Fan ON");
    } else if (aqi <= 20 && fanState) {
        digitalWrite(RELAY_PIN, LOW);
        fanState = false;
        Serial.println("[RELAY] Fan OFF");
    }

    // Get WiFi signal strength
    signalStrength = WiFi.RSSI();
    
    // Simulate battery level (you can connect actual battery monitoring)
    batteryLevel = random(85, 100);
}

void connectWiFi() {
    if (WiFi.status() == WL_CONNECTED) return;

    Serial.print("[WiFi] Connecting to ");
    Serial.print(ssid);
    WiFi.begin(ssid, password);
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n[WiFi] Connected");
        Serial.print("[WiFi] IP: ");
        Serial.println(WiFi.localIP());
        deviceStatus = "online";
    } else {
        Serial.println("\n[WiFi] Connection failed");
        deviceStatus = "offline";
    }
}

void postSensorData() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[HTTP] WiFi not connected, reconnecting...");
        connectWiFi();
        if (WiFi.status() != WL_CONNECTED) return;
    }

    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    // Create JSON payload matching backend schema
    DynamicJsonDocument doc(2048);
    doc["deviceId"] = deviceId;
    
    // Environmental sensors
    doc["temperature"] = temperature;
    doc["humidity"] = humidity;
    doc["pressure"] = pressure;
    
    // Air quality sensors
    doc["pm25"] = pm25;
    doc["voc"] = voc;
    doc["dustDensity"] = dustDensity;
    
    // Gas sensors
    doc["co"] = co;
    doc["no2"] = no2;
    doc["nh3"] = nh3;
    
    // AQI (calculated locally using standardized EPA formulas)
    doc["aqi"] = aqi;
    
    // GPS data
    doc["gpsLat"] = gpsLat;
    doc["gpsLon"] = gpsLon;
    doc["gpsAlt"] = gpsAlt;
    doc["gpsValid"] = gpsValid;
    doc["placeName"] = placeName;
    
    // Device status
    doc["fanState"] = fanState;
    doc["deviceStatus"] = deviceStatus;
    doc["batteryLevel"] = batteryLevel;
    doc["signalStrength"] = signalStrength;

    String payload;
    serializeJson(doc, payload);

    int httpCode = http.POST(payload);
    if (httpCode > 0) {
        Serial.printf("[HTTP] POST Response: %d\n", httpCode);
        if (httpCode == HTTP_CODE_OK) {
            String response = http.getString();
            Serial.println("[HTTP] Response: " + response);
            
            // Parse server response for any overrides
            DynamicJsonDocument responseDoc(1024);
            if (deserializeJson(responseDoc, response) == DeserializationError::Ok) {
                if (responseDoc["success"] == true) {
                    // Server can still override fan state if needed
                    if (responseDoc.containsKey("fanState")) {
                        bool serverFanState = responseDoc["fanState"];
                        if (serverFanState != fanState) {
                            fanState = serverFanState;
                            digitalWrite(RELAY_PIN, fanState ? HIGH : LOW);
                            Serial.printf("[HTTP] Server overrode fan state to: %s\n", fanState ? "ON" : "OFF");
                        }
                    }
                }
            }
        } else {
            Serial.println("[HTTP] Error: " + http.getString());
            deviceStatus = "error";
        }
    } else {
        Serial.println("[HTTP] POST failed: " + http.errorToString(httpCode));
        deviceStatus = "offline";
    }
    http.end();
}

void printSerialData() {
    Serial.println("========== AIR QUALITY GUARDIAN ==========");
    Serial.printf("Device ID: %s\n", deviceId);
    Serial.printf("Status: %s\n", deviceStatus.c_str());
    Serial.printf("Temperature: %.1f °C\n", temperature);
    Serial.printf("Humidity: %.1f %%\n", humidity);
    Serial.printf("Pressure: %.1f hPa\n", pressure);
    Serial.printf("Gas Resistance: %.1f kOhms\n", gasResistance);
    Serial.println("------ Air Quality ------");
    Serial.printf("PM2.5: %.1f μg/m³\n", pm25);
    Serial.printf("VOC: %.1f ppb\n", voc);
    Serial.printf("Dust Density: %.1f μg/m³\n", dustDensity);
    Serial.printf("CO: %.1f ppm\n", co);
    Serial.printf("NO2: %.1f ppb\n", no2);
    Serial.printf("NH3: %.1f ppb\n", nh3);
    Serial.printf("AQI: %d (EPA Standard)\n", aqi);
    Serial.printf("Fan: %s\n", fanState ? "ON" : "OFF");
    Serial.println("------ Location ------");
    if (gpsValid) {
        Serial.printf("GPS: %.6f, %.6f (%.1fm)\n", gpsLat, gpsLon, gpsAlt);
        Serial.printf("Place: %s\n", placeName.c_str());
    } else {
        Serial.println("GPS: No Fix");
    }
    Serial.println("------ Device Status ------");
    Serial.printf("Battery: %d%%\n", batteryLevel);
    Serial.printf("WiFi Signal: %d dBm\n", signalStrength);
    Serial.printf("WebSocket: %s\n", wsConnected ? "Connected" : "Disconnected");
    Serial.println("==========================================");
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("[SETUP] Air Quality Guardian Initializing...");

    // Initialize BME680
    if (!bme.begin()) {
        Serial.println("[ERROR] BME680 sensor not found!");
        while (1) {
            delay(1000);
            Serial.println("[ERROR] Check BME680 wiring...");
        }
    }
    
    // Configure BME680
    bme.setTemperatureOversampling(BME680_OS_8X);
    bme.setHumidityOversampling(BME680_OS_2X);
    bme.setPressureOversampling(BME680_OS_4X);
    bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
    bme.setGasHeater(320, 150);
    Serial.println("[BME680] Configured successfully");

    // Initialize pins
    pinMode(RELAY_PIN, OUTPUT);
    digitalWrite(RELAY_PIN, LOW);
    pinMode(LED_CONTROL, OUTPUT);
    digitalWrite(LED_CONTROL, HIGH);
    Serial.println("[GPIO] Pins configured");

    // Initialize GPS
    gpsSerial.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
    Serial.println("[GPS] Serial initialized");

    // Initialize moving average samples
    for (int i = 0; i < SAMPLES_COUNT; i++) {
        pm25_samples[i] = 0;
        co_samples[i] = 0;
        no2_samples[i] = 0;
        nh3_samples[i] = 0;
        voc_samples[i] = 0;
    }

    // Connect to WiFi
    connectWiFi();

    // Initialize WebSocket
    webSocket.begin(wsServerUrl, wsServerPort, wsPath);
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(5000);
    Serial.println("[WebSocket] Initialized");

    Serial.println("[SETUP] Complete! Warming up sensors...");
    delay(5000);  // Allow sensors to stabilize
}

void loop() {
    // Handle WebSocket
    webSocket.loop();
    reconnectWebSocket();

    // Read sensors periodically
    if (millis() - lastSensorRead >= SENSOR_INTERVAL) {
        readSensors();
        lastSensorRead = millis();
    }

    // Read GPS data
    while (gpsSerial.available() > 0) {
        if (gps.encode(gpsSerial.read())) {
            if (gps.location.isValid() && gps.location.age() < 2000) {
                gpsLat = gps.location.lat();
                gpsLon = gps.location.lng();
                gpsValid = true;
                
                if (gps.altitude.isValid()) {
                    gpsAlt = gps.altitude.meters();
                }
                
                // Update place name based on coordinates (simplified)
                if (gpsLat >= 28.5 && gpsLat <= 28.7 && gpsLon >= 77.1 && gpsLon <= 77.3) {
                    placeName = "New Delhi, India";
                } else if (gpsLat >= 19.0 && gpsLat <= 19.3 && gpsLon >= 72.8 && gpsLon <= 73.0) {
                    placeName = "Mumbai, India";
                } else if (gpsLat >= 12.9 && gpsLat <= 13.1 && gpsLon >= 77.5 && gpsLon <= 77.7) {
                    placeName = "Bangalore, India";
                } else {
                    placeName = String("Location (") + String(gpsLat, 4) + ", " + String(gpsLon, 4) + ")";
                }
            } else {
                gpsValid = false;
                placeName = "Unknown Location";
            }
        }
    }

    // Print to serial
    if (millis() - lastSerialPrint >= SERIAL_PRINT_INTERVAL) {
        printSerialData();
        lastSerialPrint = millis();
    }

    // Post sensor data to backend
    if (millis() - lastHttpPost >= HTTP_POST_INTERVAL) {
        postSensorData();
        lastHttpPost = millis();
    }

    // Check WiFi connection
    if (WiFi.status() != WL_CONNECTED) {
        deviceStatus = "offline";
        if (millis() - lastHttpPost > 10000) {
            connectWiFi();
        }
    } else {
        deviceStatus = "online";
    }

    delay(50);  // Small delay to prevent watchdog timeout
}