# Проект по классификации пола цыплят

Это реализация обучения сверточной нейронной сети для решения задачи классификации пола цыплят в первые два-три дня после вылупления. Для задачи были взяты модели семейства ResNet18, предобученные на открытом датасете ImageNet. Для реализации также были протестированы другие модели семейства MobileNet и EfficientNet. но они оказалиись малоэффективны для решения задачи.

Кроме основных файлов вы можете увидеть converting.py, который нужен для вырезания куриц из фона изображений и добавления белого шума. Эта необходимость была вызвана специфичностью полученного датасета.

## Установка

Для работы с проектом вам понадобятся следующие библиотеки:

- Python 3.8+
- PyTorch
- Torchvision
- Imgaug
- Matplotlib
- Pycocotools
- OpenCV

Установите зависимости с помощью команды:

```bash
pip install -r requirements.txt

## Запуск файлов

### Обучение модели 

```bash
python chicken_train.py --train-dataset path_to_train_data --val-dataset path_to_val_data --test-dataset path_to_test_data --output-dir path_to_output_dir --model resnet50 --batch-size 16 --epochs 50 --device cuda --pretrained