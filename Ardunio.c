#include <LiquidCrystal_I2C.h>
#include <Keypad.h>
#include <Wire.h>
#include <avr/pgmspace.h>

// ============= HARDWARE CONFIGURATION =============
LiquidCrystal_I2C lcd(0x27, 16, 2);

const byte ROWS = 4;
const byte COLS = 4;
char keys[ROWS][COLS] = {
  {'1', '2', '3', 'A'},
  {'4', '5', '6', 'B'},
  {'7', '8', '9', 'C'},
  {'*', '0', '#', 'D'}
};

byte rowPins[ROWS] = {9, 8, 7, 6};
byte colPins[COLS] = {5, 4, 3, 2};
Keypad keypad = Keypad(makeKeymap(keys), rowPins, colPins, ROWS, COLS);

// ============= HERB DATABASE =============
struct HerbProfile {
  char name[12];
  float values[9];
};

const HerbProfile herbDB[] PROGMEM = {
  {"Ashwagandha", {8.2, 1.1, 0.8, 1.5, 2.3, 2.8, 7.5, 8.1, 4.2}},
  {"Shatavari", {7.8, 1.5, 0.5, 0.8, 1.9, 3.2, 6.2, 7.8, 3.8}},
  {"Yashtimadhu", {9.2, 0.8, 0.3, 0.5, 1.2, 1.8, 8.5, 9.1, 5.5}},
  {"Amalaki", {3.2, 8.5, 1.1, 2.1, 2.8, 7.2, 4.5, 6.2, 8.9}},
  {"Kokam", {1.8, 9.1, 0.9, 1.2, 1.5, 4.8, 3.2, 4.8, 7.5}},
  {"Maricha", {1.2, 1.8, 0.8, 9.2, 2.1, 1.5, 7.8, 3.2, 6.5}},
  {"Shunthi", {2.1, 1.5, 0.5, 8.8, 1.8, 2.2, 6.5, 4.1, 5.8}},
  {"Neem", {0.8, 1.2, 0.5, 1.8, 9.5, 3.8, 8.9, 5.5, 7.8}},
  {"Arjuna", {1.5, 2.1, 0.5, 1.2, 3.8, 8.9, 4.5, 6.8, 8.2}}
};

const int NUM_HERBS = sizeof(herbDB) / sizeof(herbDB[0]);

// Parameter names
const char* paramNames[9] = {
  "Sweet", "Sour", "Salty", "Pungent", "Bitter", 
  "Astringent", "Alkaloid", "Glycoside", "Phenolic"
};

// ============= SYSTEM VARIABLES =============
float sensorReadings[9];
int currentParam = 0;
String inputBuffer = "";
bool systemStarted = false;
bool analysisComplete = false;
unsigned long lastKeyTime = 0;

// ============= SYSTEM STATES =============
enum SystemState {
  WELCOME_STATE,
  INPUT_STATE,
  ANALYSIS_STATE,
  RESULTS_STATE
};

SystemState currentState = WELCOME_STATE;

// ============= ANALYSIS FUNCTIONS =============
struct AnalysisResult {
  int herbIndex;
  float confidence;
};

float calculateDistance(float* readings, int herbIndex) {
  HerbProfile herb;
  memcpy_P(&herb, &herbDB[herbIndex], sizeof(HerbProfile));
  
  float distance = 0;
  for (int i = 0; i < 9; i++) {
    float diff = readings[i] - herb.values[i];
    distance += diff * diff;
  }
  return sqrt(distance);
}

AnalysisResult analyzeHerb(float* readings) {
  AnalysisResult result;
  float minDistance = 999999;
  int bestMatch = 0;
  
  for (int i = 0; i < NUM_HERBS; i++) {
    float distance = calculateDistance(readings, i);
    if (distance < minDistance) {
      minDistance = distance;
      bestMatch = i;
    }
  }
  
  float confidence = max(0, (30.0 - minDistance) / 30.0 * 100.0);
  result.herbIndex = bestMatch;
  result.confidence = confidence;
  
  return result;
}

// ============= DISPLAY FUNCTIONS =============
void displayWelcome() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("E-Tongue System");
  lcd.setCursor(0, 1);
  lcd.print("Press # to start");
  Serial.println("WELCOME: Press # to start");
}

void displayParameterInput() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(paramNames[currentParam]);
  lcd.print(" (");
  lcd.print(currentParam + 1);
  lcd.print("/9)");
  
  lcd.setCursor(0, 1);
  lcd.print("Value: ");
  lcd.print(inputBuffer);
  
  Serial.print("INPUT: ");
  Serial.print(paramNames[currentParam]);
  Serial.print(" - Current: ");
  Serial.println(inputBuffer);
}

void displayProgress() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Progress: ");
  lcd.print(currentParam);
  lcd.print("/9");
  
  lcd.setCursor(0, 1);
  int bars = (currentParam * 16) / 9;
  for (int i = 0; i < bars; i++) {
    lcd.print("=");
  }
  
  Serial.print("PROGRESS: ");
  Serial.print(currentParam);
  Serial.println("/9");
}

void displayAnalyzing() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Analyzing...");
  lcd.setCursor(0, 1);
  lcd.print("Please wait");
  
  Serial.println("ANALYZING: Processing data...");
}

void displayResults(AnalysisResult result) {
  HerbProfile herb;
  memcpy_P(&herb, &herbDB[result.herbIndex], sizeof(HerbProfile));
  
  // Show herb name
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Herb Found:");
  lcd.setCursor(0, 1);
  lcd.print(herb.name);
  delay(3000);
  
  // Show confidence
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Confidence:");
  lcd.setCursor(0, 1);
  lcd.print(result.confidence, 1);
  lcd.print("%");
  
  if (result.confidence >= 80) {
    lcd.print(" HIGH");
  } else if (result.confidence >= 60) {
    lcd.print(" MED");
  } else {
    lcd.print(" LOW");
  }
  delay(3000);
  
  // Show quality assessment
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Quality:");
  lcd.setCursor(0, 1);
  if (result.confidence >= 80) {
    lcd.print("AUTHENTIC");
  } else if (result.confidence >= 60) {
    lcd.print("CHECK NEEDED");
  } else {
    lcd.print("SUSPECT");
  }
  delay(3000);
  
  // Final message
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Analysis Done");
  lcd.setCursor(0, 1);
  lcd.print("Press D to reset");
  
  Serial.println("=== RESULTS ===");
  Serial.print("Herb: ");
  Serial.println(herb.name);
  Serial.print("Confidence: ");
  Serial.print(result.confidence);
  Serial.println("%");
  Serial.println("Press D to reset");
}

// ============= INPUT PROCESSING =============
void processInput() {
  char key = keypad.getKey();
  
  if (key == NO_KEY) return;
  
  // Debouncing
  if (millis() - lastKeyTime < 200) return;
  lastKeyTime = millis();
  
  Serial.print("KEY PRESSED: ");
  Serial.println(key);
  
  switch (currentState) {
    case WELCOME_STATE:
      handleWelcomeState(key);
      break;
      
    case INPUT_STATE:
      handleInputState(key);
      break;
      
    case RESULTS_STATE:
      handleResultsState(key);
      break;
      
    default:
      break;
  }
}

void handleWelcomeState(char key) {
  if (key == '#') {
    currentState = INPUT_STATE;
    currentParam = 0;
    inputBuffer = "";
    displayParameterInput();
    Serial.println("STATE: Switched to INPUT_STATE");
  }
}

void handleInputState(char key) {
  if (key >= '0' && key <= '9') {
    // Add digit
    if (inputBuffer.length() < 4) {
      inputBuffer += key;
      displayParameterInput();
    }
  }
  else if (key == '*') {
    // Backspace
    if (inputBuffer.length() > 0) {
      inputBuffer.remove(inputBuffer.length() - 1);
      displayParameterInput();
    }
  }
  else if (key == '#') {
    // Confirm entry
    if (inputBuffer.length() > 0) {
      float value = inputBuffer.toFloat();
      if (value >= 0 && value <= 10) {
        sensorReadings[currentParam] = value;
        Serial.print("STORED: ");
        Serial.print(paramNames[currentParam]);
        Serial.print(" = ");
        Serial.println(value);
        
        currentParam++;
        inputBuffer = "";
        
        if (currentParam >= 9) {
          // All parameters entered - start analysis
          currentState = ANALYSIS_STATE;
          displayAnalyzing();
          delay(2000);
          
          AnalysisResult result = analyzeHerb(sensorReadings);
          displayResults(result);
          currentState = RESULTS_STATE;
        } else {
          // Show progress and move to next parameter
          displayProgress();
          delay(1000);
          displayParameterInput();
        }
      } else {
        // Invalid value
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print("Error!");
        lcd.setCursor(0, 1);
        lcd.print("Enter 0-10 only");
        delay(2000);
        inputBuffer = "";
        displayParameterInput();
      }
    }
  }
  else if (key == 'A') {
    // Clear current input
    inputBuffer = "";
    displayParameterInput();
  }
  else if (key == 'B') {
    // Previous parameter
    if (currentParam > 0) {
      currentParam--;
      inputBuffer = "";
      displayParameterInput();
    }
  }
  else if (key == 'C') {
    // Skip parameter (set to 0)
    sensorReadings[currentParam] = 0;
    Serial.print("SKIPPED: ");
    Serial.print(paramNames[currentParam]);
    Serial.println(" = 0");
    
    currentParam++;
    inputBuffer = "";
    
    if (currentParam >= 9) {
      currentState = ANALYSIS_STATE;
      displayAnalyzing();
      delay(2000);
      
      AnalysisResult result = analyzeHerb(sensorReadings);
      displayResults(result);
      currentState = RESULTS_STATE;
    } else {
      displayProgress();
      delay(1000);
      displayParameterInput();
    }
  }
  else if (key == 'D') {
    // Reset to welcome
    resetSystem();
  }
}

void handleResultsState(char key) {
  if (key == 'D') {
    resetSystem();
  }
}

void resetSystem() {
  currentState = WELCOME_STATE;
  currentParam = 0;
  inputBuffer = "";
  analysisComplete = false;
  
  for (int i = 0; i < 9; i++) {
    sensorReadings[i] = 0;
  }
  
  displayWelcome();
  Serial.println("SYSTEM RESET");
}

// ============= SETUP =============
void setup() {
  lcd.init();
  lcd.backlight();
  Serial.begin(9600);
  
  displayWelcome();
  
  Serial.println("=== E-TONGUE SYSTEM INITIALIZED ===");
  Serial.println("System ready. Follow LCD instructions.");
  Serial.println("Serial monitor will show all key presses and system state.");
}

// ============= MAIN LOOP =============
void loop() {
  processInput();
  delay(50); // Small delay to prevent overwhelming the system
}
