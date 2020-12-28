from options import *


class Menu(object):

    def __init__(self):
        self.variable = ""

    def show_main_menu(self):
        print("------------------------------------")
        print("MENU")
        print("1.- Reg. Datos nuevo usuario")
        print("2.- Reg. Rostro")
        print("3.- Reconocer Rostro")
        print("4.- Registrar Entrada")
        print("5 Registrar Salida")
        print("------------------------------------")

    def process(self):
        self.show_main_menu()
        self.variable = input()
        should_exit = False
        options(self.variable)
        return should_exit



if __name__ == "__main__":
    Menu.process()