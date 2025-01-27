from termcolor import colored
import pyfiglet

def color_nice_message(sentence, color):
    say_hello = pyfiglet.figlet_format(sentence)
    color_say_hello = colored(say_hello, color)
    return color_say_hello
    
# color_nice_message("HELLO, EVERYONE~", 'blue')
# return으로 출력시키려면???
a = color_nice_message("HELLO, EVERYONE~", 'blue')
print(a)