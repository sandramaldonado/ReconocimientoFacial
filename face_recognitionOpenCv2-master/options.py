def uno():
    return "uno"

def dos():
    return "dos"

def tres():
    return "tres"


def options(op):
    switcher ={
        1: uno(),
        2: dos(),
        3: tres()
    }

    return switcher.get(op)


def menu():
    print("HOLA")
    var = input("numeroo: ")
    options(var)


if __name__ == "__main__":
    menu()



