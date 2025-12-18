Com este projeto foi possível desenvolver um sistema de deteção de quedas em tempo real através de visão computacional.
Como unidade de processamento foi usado um RaspberryPi 3, para a captura de vídeo e imagem uma RaspberryPiCam 2.
Como ferramentas/tecnologias estão presentes, OpenCV para processamento da imagem, HOG+SVM para deteção da pessoa, HaarCascade para a deteção da face, GaussianBlur para privacidade da face e um algoritmo tendo como ponto de referência o eixo do y (com um ponto central) para fazer a deteção da queda.
O PushSafer também está implementado a fim de enviar uma notificação com um screenshot em anexo para os dispositivos associados para permitir uma resposta mais rápida em caso de queda.
