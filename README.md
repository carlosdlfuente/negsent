# negsent

En este trabajo se implementa un sistema para la detección de partículas y ámbitos de la negación en textos escritos en idioma español y su evaluación extrínseca en un sistema de clasificación de opiniones. El sistema verifica que la integración de un detector de la negación en un sistema de clasificación de opiniones impacta positivamente y mejora la eficacia de dicha clasificación. La experimentación que se desarrolla contrasta dos métodos alternativos de clasificación secuencial de atributos sobre los comentarios recogidos en el conjunto de datos SFU ReviewSP-NEG. Por un lado, se utiliza el método de clasificación secuencial Conditional Random Fields (CRF) que se asienta en la eficiente ingeniería de características lingüísticas y es el modelo que mejores resultados ha alcanzado hasta el momento, y por otro lado, una aproximación Transfer Learning basada en reconocimiento de entidades de las claves de negación que aun no se ha explorado para la detección de negación en español. Para abordar el alcance descrito se estudia la evolución del tratamiento computacional del fenómeno de la negación, se analiza la disponibilidad de colecciones de opiniones en español con anotaciones de negación y se examina el estado del arte sobre diferentes métodos para la clasificación secuencial. Los resultados de la fase experimental son comparables a los obtenidos en los talleres de trabajo de referencia en el dominio de estudio, como NEGES y TASS, resultando superiores los obtenidos con la aplicación del método basado en CRF dado el limitado tamaño de los datos de entrenamiento.
