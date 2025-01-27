import pyfiglet
from termcolor import colored

hello = pyfiglet.figlet_format("HELLO")
color_hello = colored(hello, "red")
print(hello)
print(color_hello)