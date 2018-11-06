import random
import time

import speech_recognition as sr


def recognize_speech_from_mic(recognizer,microphone):
    
    if not isinstance(recognizer,sr.Recognizer):
        raise TypeError("recognizer must be a Recognizer instance")
    if not isinstance(microphone, sr.Microphone):
        raise TypeError("microphone must be a Microphone instance")
    
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    response = {
        'success':True,
        'error':None,
        'transcription':None
    }
    
    try:
        response['transcription'] = recognizer.recognize_google(audio)
    except sr.RequestError:
        response['success'] = False
        response['error'] = "API unavailable"
    except sr.UnknownValueError:
        response['error'] = "unable to recognize speech"
    
    return response


WORDS = ["apple", "banana", "grape", "orange", "mango", "lemon"]
num_guesses = 3
prompt_limit = 5

#create instances
recognizer = sr.Recognizer()
microphone = sr.Microphone()

word = random.choice(WORDS)

instructions = (
        "I'm thinking of one of these words:\n"
        "{words}\n"
        "You have {n} tries to guess which one.\n".format(words=', '.join(WORDS), n=num_guesses))


for i in range(num_guesses):
    for j in range(PROMPT_LIMIT):
        
        print('Guess {}. Speak!'.format(i+1))
        guess = recognize_speech_from_mic(recognizer,microphone)
        if guess['transcription']:
            break
        if not guess["success"]:
            break
        print("I didnt catch that, what did you say?")
    
    if guess["error"]:
        print("Error {}".format(guess["error"]))
        break
    
    print("You said {}".format(guess["transcription"]))
    guess_is_correct = guess["transcription"].lower() == word.lower()
    user_has_more_attempts = i < NUM_GUESSES - 1
    
    if guess_is_correct:
        print("Correct! You win!".format(word))
        break
    elif user_has_more_attempts:
        print("Incorrect. Try again.\n")
    else:
        print("Sorry, you lose!\nI was thinking of '{}'.".format(word))
        break