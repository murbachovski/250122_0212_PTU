import pyfiglet

def call_message(sentence):
    say = pyfiglet.figlet_format(sentence)
    print(say)
    
call_message("HELLO")