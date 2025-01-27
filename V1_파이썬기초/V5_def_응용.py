# import pyfiglet
# from termcolor import colored

# hello = pyfiglet.figlet_format("HELLO")
# color_hello = colored(hello, "red")
# print(hello)
# print(color_hello)

# => 함수로 만들어 주세요~
from termcolor import colored
import pyfiglet

def color_nice_message(sentence):
    say_hello = pyfiglet.figlet_format(sentence)
    color_say_hello = colored(say_hello, 'red')
    print(color_say_hello)
    
color_nice_message("HELLO, EVERYONE~")