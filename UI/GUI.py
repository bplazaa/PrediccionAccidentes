from faulthandler import disable
import tkinter as tk
from tkinter import ttk
import pronosticador as pr
import numpy as np
import pandas as pd

epocas = 10

X_train, X_test, y_train, y_test = pr.cargar_datos('2019_siniestros_de_transito_bdd.csv')
modelo = pr.entrenar_modelo(X_train, y_train, epocas)
y_pred = modelo.predict(X_test)
precision, exactitud = pr.obtener_estadisticas(y_test, y_pred)

WIN_WIDTH = 430
WIN_HEIGHT = 575


window = tk.Tk() 
window.title("Proyecto IA")  # to define the title
#window.config(cursor= 'heart')
canvas = tk.Canvas(window, width=WIN_WIDTH, height=WIN_HEIGHT)  # define the size
canvas.pack()
frame = tk.Frame(window)

#set window at the center of the screen
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x_win_coord = int((screen_width/2) - (WIN_WIDTH/2))
y_win_coord = int((screen_height/2) - (WIN_HEIGHT/2))
window.geometry("{}x{}+{}+{}".format(WIN_WIDTH, WIN_HEIGHT, x_win_coord, y_win_coord))


frame.place(relx=0.05, rely=0, relwidth=0.9, relheight=0.9)
menu = tk.Menu(window)
lexico = tk.IntVar() 
Sintactico = tk.IntVar() 
Semantico = tk.IntVar() 
window.config(menu=menu)


logo = tk.PhotoImage(file='ia.png')
logo = logo.subsample(12)
logo_label = tk.Label(frame, image=logo).grid(sticky='W',row=0,column=0,columnspan=1)
label = tk.Label(frame,text='Predicción de accidentes', bd='3',fg='black', font='Helvetica 12 bold').grid(row=0,column=0,columnspan=4)  # placing labels


#Canton field
label1 = tk.Label(frame,text='Lugar (cantón)', bd='1',fg='black', font='Helvetica 9 bold').grid(sticky = 'W',pady=10,row=1,column=0, columnspan=1)
#Combobox
cantones = {"CUENCA":"0101","GIRÓN":"0102","GUALACEO":"0103","NABÓN":"0104","PAUTE":"0105","PUCARÁ":"0106","SAN FERNANDO":"0107","SANTA ISABEL":"0108","SIGSIG":"0109","OÑA":"0110","SEVILLA DE ORO":"0113","GUACHAPALA":"0114","CAMILO PONCE ENRÍQUEZ":"0115","GUARANDA":"0201","CHILLANES":"0202","CHIMBO":"0203","ECHEANDIA":"0204","SAN MIGUEL":"0205","CALUMA":"0206","LAS NAVES":"0207","AZOGUES":"0301","BIBLIAN":"0302","CAÑAR":"0303","LA TRONCAL":"0304","EL TAMBO":"0305","SUSCAL":"0307","TULCÁN":"0401","BOLÍVAR":"0402","ESPEJO":"0403","MIRA":"0404","MONTÚFAR":"0405","SAN PEDRO DE HUACA":"0406","LATACUNGA":"0501","LA MANÁ":"0502","PUJILÍ":"0504","SALCEDO":"0505","SAQUISILÍ":"0506","SIGCHOS":"0507","RIOBAMBA":"0601","ALAUSÍ":"0602","COLTA":"0603","CHAMBO":"0604","CHUNCHI":"0605","GUAMOTE":"0606","GUANO":"0607","PALLATANGA":"0608","PENÍPE":"0609","CUMANDÁ":"0610","MACHALA":"0701","ARENILLAS":"0702","ATAHUALPA":"0703","BALSAS":"0704","EL GUABO":"0706","HUAQUILLAS":"0707","MARCABELÍ":"0708","PASAJE":"0709","PIÑAS":"0710","PORTOVELO":"0711","SANTA ROSA":"0712","ZARUMA":"0713","LAS LAJAS":"0714","ESMERALDAS":"0801","ELOY ALFARO":"0802","QUININDE":"0804","SAN LORENZO":"0805","ATACAMES":"0806","RÍOVERDE":"0807","GUAYAQUIL":"0901","ALFREDO BAQUERIZO MORENO (JUJA":"0902","BALAO":"0903","BALZAR":"0904","COLIMES":"0905","DAULE":"0906","DURÁN":"0907","EL EMPALME":"0908","EL TRIUNFO":"0909","MILAGRO":"0910","NARANJAL":"0911","NARANJITO":"0912","PALESTINA":"0913","PEDRO CARBO":"0914","SAMBORONDÓN":"0916","SANTA LUCIA":"0918","SALITRE (URBINA JADO)":"0919","YAGUACHI":"0920","PLAYAS":"0921","SIMÓN BOLÍVAR":"0922","CORONEL MARCELINO MARIDUEÑA":"0923","LOMAS DE SARGENTILLO":"0924","NOBOL":"0925","GENERAL ANTONIO ELIZALDE":"0927","ISIDRO AYORA":"0928","IBARRA":"1001","ANTONIO ANTE":"1002","COTACACHI":"1003","OTAVALO":"1004","PIMAMPIRO":"1005","SAN MIGUEL DE URCUQUI":"1006","LOJA":"1101","CALVAS":"1102","CATAMAYO":"1103","CELICA":"1104","ESPíNDOLA":"1106","GONZANAMÁ":"1107","MACARA":"1108","PALTAS":"1109","SARAGURO":"1111","ZAPOTILLO":"1113","OLMEDO":"1116","BABAHOYO":"1201","BABA":"1202","MONTALVO":"1203","PUEBLOVIEJO":"1204","QUEVEDO":"1205","URDANETA":"1206","VENTANAS":"1207","VINCES":"1208","PALENQUE":"1209","BUENA FÉ":"1210","VALENCIA":"1211","MOCACHE":"1212","QUINSALOMA":"1213","PORTOVIEJO":"1301","BOLÍVAR":"1302","CHONE":"1303","EL CARMEN":"1304","FLAVIO ALFARO":"1305","JIPIJAPA":"1306","JUNÍN":"1307","MANTA":"1308","MONTECRISTI":"1309","PAJÁN":"1310","PICHINCHA":"1311","ROCAFUERTE":"1312","SANTA ANA":"1313","SUCRE":"1314","TOSAGUA":"1315","24 DE MAYO":"1316","OLMEDO":"1318","PUERTO LÓPEZ":"1319","JARAMIJÓ":"1321","SAN VICENTE":"1322","MORONA":"1401","GUALAQUIZA":"1402","LIMÓN INDANZA":"1403","PALORA":"1404","SANTIAGO DE MENDEZ":"1405","SUCÚA":"1406","HUAMBOYA":"1407","SAN JUAN BOSCO":"1408","TAISHA":"1409","TIWINTZA":"1412","TENA":"1501","ARCHIDONA":"1503","EL CHACO":"1504","QUIJOS":"1507","CARLOS JULIO ARROSEMENA TOLA":"1509","PASTAZA":"1601","MERA":"1602","SANTA CLARA":"1603","DISTRITO METROPOLITANO DE QUIT":"1701","CAYAMBE":"1702","MEJIA":"1703","PEDRO MONCAYO":"1704","RUMIÑAHUI":"1705","SAN MIGUEL DE LOS BANCOS":"1707","AMBATO":"1801","BAÑOS DE AGUA SANTA":"1802","CEVALLOS":"1803","MOCHA":"1804","PATATE":"1805","QUERO":"1806","PELILEO":"1807","PILLARO":"1808","TISALEO":"1809","ZAMORA":"1901","CHINCHIPE":"1902","YACUAMBÍ":"1904","YANTZAZA":"1905","EL PANGUI":"1906","CENTINELA DEL CÓNDOR":"1907","PALANDA":"1908","PAQUISHA":"1909","SAN CRISTOBAL":"2001","SANTA CRUZ":"2003","LAGO AGRIO":"2101","GONZALO PIZARRO":"2102","PUTUMAYO":"2103","SHUSHUFINDI":"2104","SUCUMBIOS":"2105","ORELLANA":"2201","LA JOYA DE LOS SACHAS":"2203","LORETO":"2204","SANTO DOMINGO":"2301","LA CONCORDIA":"2302","SANTA ELENA":"2401","LA LIBERTAD":"2402","SALINAS":"2403"}
listCanton = ttk.Combobox(frame, width = 35)  
# Adding combobox drop down list
listCanton['values'] = tuple(cantones.keys())
listCanton.grid(column = 3, row = 1)
listCanton.current()

label2 = tk.Label(frame,text='Mes', bd='1',fg='black', font='Helvetica 9 bold').grid(sticky = 'W',pady=10,row=3,column=0, columnspan=1)
meses = {1:"ENERO",2:"FEBRERO",3:"MARZO",4:"ABRIL",5:"MAYO",6:"JUNIO",7:"JULIO",8:"AGOSTO",9:"SEPTIEMBRE",10:"OCTUBRE",11:"NOVIEMBRE",12:"DICIEMBRE"}
listMes = ttk.Combobox(frame, width = 35)  
# Adding combobox drop down list
listMes['values'] = tuple(meses.values())
listMes.grid(column = 3, row = 3)
listMes.current()

label2 = tk.Label(frame,text='Día', bd='1',fg='black', font='Helvetica 9 bold').grid(sticky = 'W',pady=10,row=5,column=0, columnspan=1)
dias = {1:"LUNES",2:"MARTES",3:"MIÉRCOLES",4:"JUEVES",5:"VIERNES",6:"SÁBADO",7:"DOMINGO"}
listDia = ttk.Combobox(frame, width = 35)  
# Adding combobox drop down list
listDia['values'] = tuple(dias.values())
listDia.grid(column = 3, row = 5)
listDia.current()

label2 = tk.Label(frame,text='Hora', bd='1',fg='black', font='Helvetica 9 bold').grid(sticky = 'W',pady=10,row=7,column=0, columnspan=1)
horas = {0:"00:00 A 00:59", 1:"01:00 A 01:59", 2:"02:00 A 02:59",3:"03:00 A 03:59",4:"04:00 A 04:59",5:"05:00 A 05:59",6:"06:00 A 06:59",7:"07:00 A 07:59",8:"08:00 A 08:59",9:"09:00 A 09:59", 10:"10:00 A 10:59",11:"11:00 A 11:59",12:"12:00 A 12:59",13:"13:00 A 13:59",14:"14:00 A 14:59",15:"15:00 A 15:59",16:"16:00 A 16:59",17:"17:00 A 17:59",18:"18:00 A 18:59",19:"19:00 A 19:59",20:"20:00 A 20:59",21:"21:00 A 21:59",22:"22:00 A 22:59",23:"23:00 A 23:59"}
listHora = ttk.Combobox(frame, width = 35)  
# Adding combobox drop down list
listHora['values'] = tuple(horas.values())
listHora.grid(column = 3, row = 7)
listHora.current()

#botones
analizar = tk.Button(frame, text="Ejecutar predicción", command=lambda: getInfo()).grid(sticky = 'W', padx=100, pady=10, row=10, column=1,columnspan=4)
clear_button = tk.Button(frame, text="Agregar", command=lambda: delInfo()).grid(sticky = 'E',pady=10, padx=3, row=10, column=3,columnspan=1)


#output field
label2 = tk.Label(frame,text='Output', bd='1',fg='black', font='Helvetica 9 bold').grid(sticky = 'W',pady=10,row=11, column=0,columnspan=1)
p1 = tk.StringVar(frame, "")
v = tk.Scrollbar(frame, orient='vertical')
t = tk.Text(frame, height = 10, width=48, bd='5', wrap=tk.WORD, yscrollcommand=v.set, borderwidth=2, relief="groove")
t.insert(tk.END, p1.get())
t.grid(sticky='W',row=12, column=0, columnspan=4)
v.config(command=t.yview)

#modelo field
label2 = tk.Label(frame,text='Datos del modelo', bd='1',fg='black', font='Helvetica 9 bold').grid(sticky = 'W',pady=10,row=13, column=0,columnspan=1)
v = tk.Scrollbar(frame, orient='vertical')
t2 = tk.Text(frame, height = 1, width=48, bd='5', wrap=tk.WORD, yscrollcommand=v.set, borderwidth=2, relief="groove")
msgt2 = "Precisión: " + str(round(precision*100,2)) +  "% Exactitud: " + str(round(exactitud*100,2)) + "%"
t2.insert(tk.END, msgt2)
t2.grid(sticky='W',row=14, column=0, columnspan=4)
t2.configure(state="disabled")

def delInfo():
    listCanton.current(0)
    listMes.current(0)
    listDia.current(0)
    listHora.current(0)
def getInfo():
    t.delete("1.0", "end")
    inCanton = int(cantones.get(listCanton.get()))
    inMes = int(listMes.current()+1)
    inDia = int(listDia.current()+1)
    inHora = int(listHora.current())
    x_input = pd.DataFrame(np.array([[inMes , inDia, inHora , inCanton, 1]]), columns=["MES", "DIA", "HORA", "CANTON", "INTERCEPT"])
    y_pred = modelo.predict_proba(x_input)
    probabilidad = round(y_pred[:, 1][0]*100, 2)
    message=""
    if(probabilidad>50):
        message = "La probabilidad de que este accidente ocurra por exceso de velocidad es de: {}%".format(probabilidad)
    else:
        message = "No es probable que este accidente por exceso de velocidad ocurra"
    t.insert(tk.END, message)
window.mainloop()

