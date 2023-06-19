carpeta grafos: grafos_w o weighted son los primeros n archivos  mas pesados(con n el nombre del archivo entre _) formados como Lkvec
		k_ : varuando la cantidad de palabras consecutivas que tienen que tener para unir las frases.
		_2 : uno frases que puedan tener igual cantidad de palabras no solo menor

meme_tracking_funciones.py: version definitiva para hacer un grafo segun Lkvec
saving_data_frame.py: guardado de todo en un dataframe
removing_edges: prueba de remosion de enlaces para ver si se obtiene lo mismo que Lkvec (probando si borrando se borra una ruta, no funciona)
analisis_componente_gigante.py: prueba de separacioon con k chico de la componente gigante en memes

community test.py: deteccion de comunidades con infomap en las distintas compoentntes obtenidas con Lkvec

componente_gigante.pickle: solo la componente gigante con k = 3
cluster_comp_gig_06.pk: clsuterizacion de la compoentnete gigtante con un threshold de 06
sorted_files.pickle: guardado de los nombres de los archivos ordenados por pesos