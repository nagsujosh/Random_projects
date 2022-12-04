import pyttsx3

book = open(r"book.pdf")
book_text = book.readlines()
engine = pyttsx3.init()
for line in book_text:
    engine.say(line)
    engine.runAndWait()