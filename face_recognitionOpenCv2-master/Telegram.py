import time, datetime
import telepot
from telepot.loop import MessageLoop
import pyrebase
import cv2
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
firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()
db = firebase.database()

now = datetime.datetime.now()

class DatetimeEncoder(json.JSONEncoder):
    def default(self, obj):
     try:
      return super(DatetimeEncoder, obj).default(obj)
     except TypeError:
      return str(obj)


def reconocer_rostro():

    nombre = ""
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
                print(nombre)
                if conf > 50:
                    # print(conf)
                    nombre = etiquetas[id_]

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
            web_cam.release()
            cv2.destroyAllWindows()
            return nombre


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


def registrarSalida(nombre):

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

    db.child('Salida').push(data)


def action(msg):
    chat_id = msg['chat']['id']
    command = msg['text']
    print('Received: %s' % command)
    message = ""
    if str(1) in command:
        message = "Entrada"
        persona = reconocer_rostro()
        registrarEntrada(persona)
    if str(2) in command:
        message = "Salida"
        persona = reconocer_rostro()
        registrarSalida(persona)

    telegram_bot.sendMessage(chat_id, message)


telegram_bot = telepot.Bot('1347740751:AAG9flSPKSCUdAioAt1Oyc36czDMa-R3Wmc')
print(telegram_bot.getMe())
MessageLoop(telegram_bot, action).run_as_thread()
print('Up and Running....')
while 1:
    time.sleep(10)
