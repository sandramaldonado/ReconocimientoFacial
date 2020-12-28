
import pyrebase
import cv2
import os
import numpy as np
from PIL import Image
import json
import pickle
import time
import datetime
firebaseConfig = {
    'apiKey': "AIzaSyBzlrZ4nkqGFzDMhMapRTZciay5EeFLF2Y",
    'authDomain': "seguridad-84d91.firebaseapp.com",
    'databaseURL': "https://seguridad-84d91.firebaseio.com",
    'projectId': "seguridad-84d91",
    'storageBucket': "seguridad-84d91.appspot.com",
    'messagingSenderId': "354553334075",
    'appId': "1:354553334075:web:eba988109987277a0d4f1e",
    'measurementId': "G-4BR9632568"
}


class DatetimeEncoder(json.JSONEncoder):
    def default(self, obj):
     try:
      return super(DatetimeEncoder, obj).default(obj)
     except TypeError:
      return str(obj)


firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()
db = firebase.database()
cascaderef = storage.child('IA/Cascades')
eyeref = cascaderef.child('haarcascade_frontalface_default.xml')
entrenamiento = storage.child('IA/entremamiento')
entrenamientoRef = entrenamiento.child('entrenamiento.yml')
web_cam = cv2.VideoCapture(0)
carpeta = 'images/sandra/'
cascPath = "D:/Sandra/UNIVERSIDAD/Seguridad Informatica/reconocimiento/Reconocimiento facial-Maldonado-Valencia-Camacho-Arancibia/face_recognitionOpenCv2-master/Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
f = eyeref.path

def crear_datos():
    count = 0
    #se crea las imagenes con el nombre respectivo de la persona es importante que la imagen contenga los tres rasgos
    #de rostro para el aprendizaje, se crea un marco donde intenta reconocer un rostro en cada imagen, posteriormente
    #se sube a la nube las imagenes dentro del storage
    nombre = input("Ingrese el nombre")
    while(True):
        _, imagen_marco = web_cam.read()

        grises = cv2.cvtColor(imagen_marco, cv2.COLOR_BGR2GRAY)

        rostro = faceCascade.detectMultiScale(grises, 1.5, 5)

        for(x,y,w,h) in rostro:
            cv2.rectangle(imagen_marco, (x,y), (x+w, y+h), (255,0,0), 4)
            count += 1
            #cv2.imwrite("images/{0}/{0}_".format(nombre)+str(count)+".jpg", grises[y:y+h, x:x+w])
            cv2.imwrite("images/sandra/sandra_"+str(count)+".jpg", grises[y:y+h, x:x+w])
            cv2.imshow("Creando Dataset", imagen_marco)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elif count >= 400:
            break


    # Cuando todo está hecho, liberamos la captura
    web_cam.release()
    cv2.destroyAllWindows()
    subir_nube(nombre)


def subir_nube(nombre):
    #Con este metodo se sube a la nube los datos obtenidos al obtener un nuevo rostro dentro del storage del firebase
    #con el nombre respectivo de la persona y su carpeta de archivos

    with os.scandir(carpeta) as ficheros:
        ficheros = [fichero.name for fichero in ficheros if fichero.is_file()]
    print(ficheros)

    for x in range(0,len(ficheros)):
        print(ficheros[x])

        storage.child('image/{0}/{1}'.format(nombre,ficheros[x])).put('images/{0}/{1}'.format(nombre, ficheros[x]))

    print("Datos del Rostro Registrados correctamente")


def show_main_menu():
        print("------------------------------------")
        print("MENU")
        print("1.- Reg. Datos nuevo usuario")
        print("2.- Registrar Rostro")
        print("3.- Entrenar IA")
        print("4.- Registar Entrada")
        print("5 Registrar Salida")
        print("------------------------------------")

def entrenar_IA():
    #se entrena a la IA con las imagenes previamente obtenidas, si las imagenes no se detecta los rasgos de un rostro
    #la IA muestra error y no puede aprender el rostro, se usa la libreria de reconocimiento opencv, busca los archivos
    #que tenga terminacion png o jpg y las convierte en un array de datos con los cuales la IA aprende cada rostro
    cascPath = "D:/Sandra/UNIVERSIDAD/Seguridad Informatica/reconocimiento/Reconocimiento facial-Maldonado-Valencia-Camacho-Arancibia/face_recognitionOpenCv2-master/Cascades/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    # reconocimiento con opencv
    reconocimiento = cv2.face.LBPHFaceRecognizer_create()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "images")

    current_id = 0
    etiquetas_id = {}
    y_etiquetas = []
    x_entrenamiento = []

    for root, dirs, archivos in os.walk(image_dir):
        for archivo in archivos:
            if archivo.endswith("png") or archivo.endswith("jpg"):
                pathImagen = os.path.join(root, archivo)
                etiqueta = os.path.basename(root).replace(" ", "-")  # .lower()
                # print(etiqueta,pathImagen)

                # Creando las etiquetas
                if not etiqueta in etiquetas_id:
                    etiquetas_id[etiqueta] = current_id
                    current_id += 1
                id_ = etiquetas_id[etiqueta]
                # print(etiquetas_id)

                pil_image = Image.open(pathImagen).convert("L")
                tamanio = (550, 550)
                imagenFinal = pil_image.resize(tamanio, Image.ANTIALIAS)
                image_array = np.array(pil_image, "uint8")
                # print(image_array)

                rostros = faceCascade.detectMultiScale(image_array, 1.5, 5)

                for (x, y, w, h) in rostros:
                    roi = image_array[y:y + h, x:x + w]
                    x_entrenamiento.append(roi)
                    y_etiquetas.append(id_)

    # print(y_etiquetas)
    # print(x_entrenamiento)
    with open("labels.pickle", 'wb') as f:
        pickle.dump(etiquetas_id, f)

    reconocimiento.train(x_entrenamiento, np.array(y_etiquetas))
    reconocimiento.save("entrenamiento.yml")

def reconocer_rostro():
    #esta parte es para las rapberry en la cual la IA con los rostros aprendidos detecta las caracteristicas de la persona
    #mediante las librerias de opencv y si la aprendio bien muestra un texto con el nombre de la personas
    nombre= ""
    cascPath = "D:/Sandra/UNIVERSIDAD/Seguridad Informatica/reconocimiento/Reconocimiento facial-Maldonado-Valencia-Camacho-Arancibia/face_recognitionOpenCv2-master/Cascades/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    eyeCascade = cv2.CascadeClassifier(
        "D:/Sandra/UNIVERSIDAD/Seguridad Informatica/reconocimiento/Reconocimiento facial-Maldonado-Valencia-Camacho-Arancibia/face_recognitionOpenCv2-master/Cascades/haarcascade_eye.xml")
    smileCascade = cv2.CascadeClassifier(
        "D:/Sandra/UNIVERSIDAD/Seguridad Informatica/reconocimiento/Reconocimiento facial-Maldonado-Valencia-Camacho-Arancibia/face_recognitionOpenCv2-master/Cascades/haarcascade_smile.xml")

    reconocimiento = cv2.face.LBPHFaceRecognizer_create()
    reconocimiento.read(
        "D:/Sandra/UNIVERSIDAD/Seguridad Informatica/reconocimiento\Reconocimiento facial-Maldonado-Valencia-Camacho-Arancibia/face_recognitionOpenCv2-master/entrenamiento.yml")

    etiquetas = {"nombre_persona": 1}
    with open("labels.pickle", 'rb') as f:
        pre_etiquetas = pickle.load(f)
        etiquetas = {v: k for k, v in pre_etiquetas.items()}

    web_cam = cv2.VideoCapture(0)

    while True:
        # Capture el marco
        ret, marco = web_cam.read()
        grises = cv2.cvtColor(marco, cv2.COLOR_BGR2GRAY)
        rostros = faceCascade.detectMultiScale(grises, 1.5, 5)

        # Dibujar un rectángulo alrededor de las rostros
        for (x, y, w, h) in rostros:
            # print(x,y,w,h)
            roi_gray = grises[y:y + h, x:x + w]
            roi_color = marco[y:y + h, x:x + w]

            # reconocimiento
            id_, conf = reconocimiento.predict(roi_gray)
            if conf >= 4 and conf < 85:
                # print(id_)
                # print(etiquetas[id_])

                font = cv2.FONT_HERSHEY_SIMPLEX

                nombre = etiquetas[id_]



                if conf > 50:
                    # print(conf)
                    nombre = "Desconocido"

                color = (255, 255, 255)
                grosor = 2
                cv2.putText(marco, nombre, (x, y), font, 1, color, grosor, cv2.LINE_AA)

            img_item = "my-image.png"
            cv2.imwrite(img_item, roi_gray)

            cv2.rectangle(marco, (x, y), (x + w, y + h), (0, 255, 0), 2)

            rasgos = smileCascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in rasgos:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Display resize del marco
        marco_display = cv2.resize(marco, (1200, 650), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Detectando Rostros', marco_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            registrarEntrada(nombre)

            break

    # Cuando todo está hecho, liberamos la captura
    web_cam.release()
    cv2.destroyAllWindows()


def registrarEntrada(nombre):
        date_object = datetime.datetime.now()
        now_date = date_object.date()
        now_time = date_object.time()
        json_time = json.dumps(now_time, cls=DatetimeEncoder)
        json_date = json.dumps(now_date, cls=DatetimeEncoder)

        a = json_date.strip()
        b = json_date.strip('"')
        c = json_time.strip()
        d = json_time.strip('"')


        data = {'Empleado': nombre, 'fecha': b, 'Hora': d}

        db.child('Entrada').push(data)



def registrar_nuevo_usuario():
    print("Ingrese nombre del nuevo trabajador: ")
    nombre = input(">")
    print("Ingrese apellido del nuevo trabajador: ")
    apellido = input(">")
    print("Ingrese telefono del nuevo trabajador: ")
    telefono = input(">")

    data = {'nombre':nombre, 'apellido':apellido,'telefono':telefono}
    db.child('Personal').push(data)

    print("Registrado correctamente")

    pass


def process():
        show_main_menu()
        variable = int(input())
        if variable == 1:
            return registrar_nuevo_usuario()
        elif variable == 2:
            return crear_datos()
        elif variable == 3:
            return entrenar_IA()
        elif variable == 4:
            return reconocer_rostro()


if __name__ == "__main__":
    process()
