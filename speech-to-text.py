import pyautogui as auto
import speech_recognition as sr  # speechRecognition Library to detect

while True:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak Anything: ")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print(f"you said: {text} ")
            auto.typewrite(f'{text}\n')

        except:
            print("Sorry. The system is unable to detect your speech. Try again.")
