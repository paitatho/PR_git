######### Makefile ##############

- "make":
	execute le fichier appelé Makefile présent dans le dossier dans lequel est exécutée la cmd

- "make .PHONY" :
	nettoie le projet

- "make depend" :
	génére les dépendances du projet (les includes)

pour chaque modification executez make pour recompiler le projet

lorsque les dépendances sont modifiées (include) nettoyez le projet et reconstruisez les dépendances 