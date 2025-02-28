''''Voice Based Medical Report Dictation & Summerization'''

import os
import speech_recognition as sr
import pyttsx3
import json
from datetime import datetime
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize speech recognizer
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Base directory for patient data storage
BASE_DIR = "E:\\PatientDatabase"

def speak(text):
    """Convert text to speech"""
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")
        print(text)  # Fallback to printing if TTS fails

def listen():
    """Listen for speech and convert to text"""
    with microphone as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Recognition service error: {e}")
            return None
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None

def extract_patient_info(text):
    """Extract patient name, disease, and medication from text using simple parsing"""
    if not text:
        return None
    
    # Initialize data structure
    patient_info = {
        "name": None,
        "disease": None,
        "medication": None,
        "raw_text": text,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Simplified extraction logic
    words = text.split()
    
    # First try to find a full name (usually first two words if they're capitalized)
    for i in range(len(words) - 1):
        if (words[i][0].isupper() and 
            words[i+1][0].isupper() and 
            len(words[i]) > 1 and 
            len(words[i+1]) > 1):
            patient_info["name"] = f"{words[i]} {words[i+1]}"
            break
    
    # If no name found, take the first word that starts with uppercase
    if not patient_info["name"]:
        for word in words:
            if word[0].isupper() and len(word) > 1:
                patient_info["name"] = word
                break
    
    # Find disease
    disease_keywords = ["has", "suffering", "diagnosed", "diabetes", "cancer", "fever", "cold", 
                       "flu", "hypertension", "disease", "condition", "illness", "infection"]
    
    for keyword in disease_keywords:
        if keyword.lower() in text.lower():
            idx = text.lower().find(keyword.lower())
            # Get the next few words after the keyword
            disease_part = text[idx:idx+30].split(",")[0].split(".")[0]
            patient_info["disease"] = disease_part.strip()
            break
    
    # Find medication
    med_keywords = ["taken", "prescribed", "given", "taking", "medicine", "medication", 
                   "drug", "pill", "tablet", "injection", "insulin", "antibiotic"]
    
    for keyword in med_keywords:
        if keyword.lower() in text.lower():
            idx = text.lower().find(keyword.lower())
            # Get the next few words after the keyword
            med_part = text[idx:idx+30].split(",")[0].split(".")[0]
            patient_info["medication"] = med_part.strip()
            break
    
    # If we couldn't extract information, make a best effort
    if not patient_info["name"]:
        patient_info["name"] = "Unknown_Patient_" + datetime.now().strftime("%Y%m%d%H%M%S")
    
    if not patient_info["disease"]:
        patient_info["disease"] = "Not specified"
    
    if not patient_info["medication"]:
        patient_info["medication"] = "Not specified"
    
    return patient_info

def save_patient_data(patient_info):
    """Save patient data to a file in a folder named after the patient"""
    if not patient_info:
        print("No valid patient information to save")
        return False
    
    try:
        # Make sure base directory exists
        if not os.path.exists(BASE_DIR):
            os.makedirs(BASE_DIR)
            print(f"Created base directory: {BASE_DIR}")
        
        # Create patient directory
        patient_name = patient_info["name"].replace(" ", "_")
        patient_dir = os.path.join(BASE_DIR, patient_name)
        
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)
            print(f"Created directory for patient: {patient_name}")
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visit_{timestamp}.json"
        filepath = os.path.join(patient_dir, filename)
        
        # Save data to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(patient_info, f, indent=4)
        
        # Also create a readable text summary file
        summary_file = os.path.join(patient_dir, f"summary_{timestamp}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Patient: {patient_info['name']}\n")
            f.write(f"Disease/Condition: {patient_info['disease']}\n")
            f.write(f"Medication/Treatment: {patient_info['medication']}\n")
            f.write(f"Date and Time: {patient_info['timestamp']}\n")
            f.write(f"Original Statement: {patient_info['raw_text']}\n")
        
        print(f"Saved patient data to {filepath}")
        print(f"Saved summary to {summary_file}")
        return True
    
    except Exception as e:
        print(f"Error saving patient data: {e}")
        # Try to create the directory with simplified name if there's an issue
        try:
            simple_dir = os.path.join(BASE_DIR, "Patient_" + datetime.now().strftime("%Y%m%d%H%M%S"))
            os.makedirs(simple_dir)
            simple_file = os.path.join(simple_dir, "patient_data.json")
            with open(simple_file, 'w', encoding='utf-8') as f:
                json.dump(patient_info, f, indent=4)
            print(f"Saved to alternative location: {simple_file}")
            return True
        except Exception as e2:
            print(f"Failed to save even to alternative location: {e2}")
            return False

def main():
    global BASE_DIR  # Declare global before using it
    
    try:
        print("Medical Voice Assistant started")
        print("--------------------------------")
        print(f"Patient data will be saved to: {BASE_DIR}")
        
        # Create base directory if it doesn't exist
        if not os.path.exists(BASE_DIR):
            try:
                os.makedirs(BASE_DIR)
                print(f"Created base directory: {BASE_DIR}")
            except Exception as e:
                print(f"Error creating base directory: {e}")
                print("Will attempt to save to current directory instead.")
                BASE_DIR = "."
        
        while True:
            # Prompt for patient information
            speak("Doctor please share patient's Information")
            
            # Get doctor's speech input
            text = listen()
            
            if text:
                # Process the text to extract patient information
                patient_info = extract_patient_info(text)
                
                # Always try to save whatever information we have
                save_result = save_patient_data(patient_info)
                
                if save_result:
                    # Confirm the data extraction
                    confirmation = (f"Information saved for patient {patient_info['name']}. "
                                    f"Disease: {patient_info['disease']}. "
                                    f"Medication: {patient_info['medication']}.")
                    speak(confirmation)
                    print(confirmation)
                else:
                    speak("There was an issue saving the patient information. Please try again.")
            else:
                speak("I didn't catch that. Please try again.")
            
            # Ask if the doctor wants to continue
            speak("Do you want to add another patient? Say yes or no.")
            response = listen()
            
            if response and "no" in response.lower():
                speak("Shutting down Medical Voice Assistant. Goodbye.")
                break
            
    except Exception as e:
        print(f"An error occurred in the main program: {e}")
        speak("An error occurred. The program will now exit.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Critical error: {e}")
        speak("A critical error has occurred. The program will now exit.")