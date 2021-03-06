## Проект от СКБ ЛАБ: Сервис для обработки медицинских анализов

### Команда:
  *Структурный момент*
![image](https://user-images.githubusercontent.com/71917550/150393700-8d45eb96-49fc-4eee-a668-63df4cd3c1a7.png)

### Описание:
  Сервис, способный распозновать медицинские анализы в формате *PDF*, а после выдавать данные, содержащиеся в анализах, в формате JSON.
  
### Цель проекта:
  Разработать сервис с REST API, принимающий на вход файл в графическом формате и выдающий массив с параметрами анализов, такими как наименование вещества, показатели и единицы измерения. Сервис должен быть доступен для интеграции, а также иметь документацию в формате OpenAPI.

### Как запустить API на локальном хосте:
  - Установить Ghostscript
    - Cкачать установщик на Windows можно [здесь](https://ghostscript.com/releases/gsdnld.html)
  - Добавить Ghostscript в PATH
  - Установить Poppler
    - Cкачать установщик на Windows можно [здесь](https://github.com/oschwartz10612/poppler-windows/releases/)
  - Добавить Poppler bin/папку в PATH
  - Установить Tesseract
    - Cкачать установщик на Windows можно [здесь](https://github.com/UB-Mannheim/tesseract/wiki)
  - Добавить путь к Tesseract в файле main_parsers.py
  - Установить все необходимые пакеты из [requirements.txt](requirements.txt)
  - Перейти в директорию SKB-LAB-Project\restapi (можно сделать как через терминал операционной системы, так и среды разработки)
  - Ввести поочередно следующие команды:
    - `python manage.py migrate` - запускаем миграцию БД
    - `python manage.py makemigrations` - создаем файлы миграций для БД
    - `python manage.py migrate` - заносим новые миграции в БД
    - `python manage.py createsuperuser` - регистрируем пользователя для доступа к админке
      - Пройти регистрацию
    - `python manage.py runserver` - запускаем локальный сервер

### Способы обращения к API:
  Возможные запросы описаны в [документации](https://app.swaggerhub.com/apis-docs/AniWooV/skb-lab-api/1.0.0#/)
