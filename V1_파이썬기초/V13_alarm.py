import pygame

pygame.mixer.init()
pygame.mixer.music.load("./V1_파이썬기초/alarm.mp3")
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    continue
