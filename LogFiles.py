import os
import datetime
import pathlib


class log:
    def __init__(self, path, nom=None, create=False):

        if nom is None:
            # Si on ne definit pas de nom => log avec la date
            dd = datetime.datetime.now()
            self.nom = str(dd.year) + "-" + str(dd.month) + "-" + str(dd.day) + ".txt"
            self.logMode = True
        else:
            # Si on definit de nom => fichier de sortie
            self.nom = nom + ".txt"
            self.logMode = False

        self.mypath = path

        if not os.path.exists(self.mypath):
            pathlib.Path(self.mypath).mkdir(parents=True, exist_ok=True)

        if create is True:
            text_file = open(self.mypath + self.nom, "w")
            text_file.close()

    def write(self, msg):
        text_file = open(self.mypath + self.nom, "a")
        if self.logMode is True:
            dd = datetime.datetime.now().time()
            prfx = str(dd.hour) + "h " + str(dd.minute) +"m "+ str(dd.second) +"s"
            text_file.write(prfx + " -----> " + msg)
            text_file.write('\n')
        else:
            text_file.write(msg)
            text_file.write('\n')
        text_file.close()
